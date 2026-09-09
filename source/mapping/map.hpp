/*
Copyright (C) 2026 Geoffrey Daniels. https://gpdaniels.com/

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License only.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#ifndef ZEROSLAM_MAPPING_MAP_HPP
#define ZEROSLAM_MAPPING_MAP_HPP

#include "core/logger.hpp"
#include "mapping/frame.hpp"
#include "mapping/point.hpp"
#include "math/lie.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/reprojection.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/loss.hpp"
#include "optimisation/losses/huber.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/point.hpp"
#include "optimisation/vertices/pose.hpp"
#include "sensor/camera/model.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mapping {
    class map {
    public:
        struct observation final {
            int frame_id;
            size_t kp_index;
            math::matrix<double, 2, 1> point;
        };

    public:
        std::unordered_map<int, mapping::frame> frames;
        std::unordered_map<int, mapping::point> landmarks;
        std::unordered_map<decltype(mapping::point::id), std::vector<observation>> observations;

        int next_frame_id = 0;
        int next_landmark_id = 0;

        int allocate_frame_id() {
            return this->next_frame_id++;
        }

        int allocate_landmark_id() {
            return this->next_landmark_id++;
        }

        int gauge_frame_id = 0;
        int gauge_frame_secondary_id = 1;

    public:
        void add_frame(const mapping::frame& frame) {
            frames[frame.id] = frame;
        }

        void add_landmark(const mapping::point& landmark) {
            this->landmarks[landmark.id] = landmark;
        }

        void add_observation(const mapping::frame& frame, const mapping::point& landmark, size_t kp_index) {
            const feature::point& keypoint = frame.keypoints[kp_index];
            this->observations[landmark.id].push_back(observation{ frame.id, kp_index, math::matrix<double, 2, 1>{ { static_cast<double>(keypoint.x), static_cast<double>(keypoint.y) } } });
        }

        void add_observation(int frame_id, const mapping::point& landmark, double x, double y) {
            this->observations[landmark.id].push_back(observation{ frame_id, static_cast<size_t>(-1), math::matrix<double, 2, 1>{ { x, y } } });
        }

        void optimise(int local_window, bool fix_landmarks, int rounds, bool use_relative_convergence = false) {
            if (this->frames.empty()) {
                return;
            }
            std::unordered_map<int, optimisation::vertex*> camera_vertexes;
            std::unordered_map<int, optimisation::vertex*> landmark_vertexes;
            // Get camera parameters.
            double camera_parameters[sensor::model::parameter_count];
            this->frames.begin()->second.camera.get_parameters(camera_parameters, sensor::model::parameter_count);
            const sensor::model camera_model(camera_parameters, sensor::model::parameter_count);
            // Setup ba.
            const optimisation::loss lossfunction(optimisation::losses::huber(math::sqrt(5.991)));
            optimisation::factor_graph ba;
            // Add frames.
            const int local_window_below = this->next_frame_id - 1 - local_window;
            const int local_window_fixed_below = local_window_below + 1;
            std::unordered_set<int> relevant_frame_ids;
            std::unordered_set<int> active_landmark_ids;
            std::unordered_set<int> active_frame_ids;

            if (local_window > 0) {
                int horizon_threshold;
                for (const auto& [frame_id, frame] : this->frames) {
                    const bool fixed = ((frame_id == this->gauge_frame_id) || (frame_id == this->gauge_frame_secondary_id && this->frames.size() > 2) || (frame_id < local_window_fixed_below));
                    if (!fixed) {
                        active_frame_ids.insert(frame_id);
                    }
                }
                const int current_frame_id = this->next_frame_id - 1;
                const int fixed_horizon_limit = 2 * local_window;
                horizon_threshold = current_frame_id - fixed_horizon_limit;

                for (const auto& [landmark_id, landmark_obs] : this->observations) {
                    bool seen_by_active = false;
                    for (const auto& obs : landmark_obs) {
                        if (active_frame_ids.count(obs.frame_id)) {
                            seen_by_active = true;
                            break;
                        }
                    }
                    if (seen_by_active) {
                        active_landmark_ids.insert(static_cast<int>(landmark_id));
                        for (const auto& obs : landmark_obs) {
                            if (obs.frame_id >= horizon_threshold) {
                                relevant_frame_ids.insert(obs.frame_id);
                            }
                        }
                    }
                }
                // Also ensure active frames are in relevant_frame_ids.
                relevant_frame_ids.insert(active_frame_ids.begin(), active_frame_ids.end());
            }
            else {
                // Global BA or window not specified: include everything.
                for (const auto& [frame_id, _] : this->frames) {
                    relevant_frame_ids.insert(frame_id);
                }
                for (const auto& [landmark_id, _] : this->landmarks) {
                    active_landmark_ids.insert(static_cast<int>(landmark_id));
                }
            }

            // Add frames.
            int non_fixed_poses = 0;
            for (const auto& frame_id : relevant_frame_ids) {
                const auto& frame = this->frames.at(frame_id);
                const math::se3<double> v_se3(frame.rotation, frame.translation);
                const bool fixed = ((frame_id == this->gauge_frame_id) || (frame_id == this->gauge_frame_secondary_id && this->frames.size() > 2) || ((local_window > 0) && (active_frame_ids.count(frame_id) == 0)));
                optimisation::vertex c{ optimisation::vertices::pose() };
                const double pose_parameters[7] = { v_se3.translation()[0], v_se3.translation()[1], v_se3.translation()[2], v_se3.rotation().get_quaternion()[1], v_se3.rotation().get_quaternion()[2], v_se3.rotation().get_quaternion()[3], v_se3.rotation().get_quaternion()[0] };
                c.set_parameters(&pose_parameters[0], 7);
                c.set_fixed(fixed);
                camera_vertexes[frame_id] = ba.add_vertex(static_cast<optimisation::vertex&&>(c));
                non_fixed_poses += (fixed == false);
            }
            // Add landmarks.
            int non_fixed_landmarks = 0;
            int non_fixed_edges = 0;
            for (const auto& landmark_id : active_landmark_ids) {
                // Only add the landmark if it is in a frame. Initially assume it is not.
                bool landmark_added = false;
                // Add edges.
                for (const auto& obs : this->observations.at(landmark_id)) {
                    const int frame_id = obs.frame_id;
                    // Only add the edge if it will do something.
                    if (camera_vertexes.count(frame_id) == 0)
                        continue;
                    if (camera_vertexes.at(frame_id)->is_fixed() && fix_landmarks) {
                        continue;
                    }
                    // Given that we're adding at least one edge, make sure the landmark has been added.
                    const mapping::point& landmark_record = this->landmarks.at(landmark_id);
                    if (!landmark_added) {
                        landmark_added = true;
                        optimisation::vertex l{ optimisation::vertices::point() };
                        l.set_parameters(landmark_record.location.data(), 3);
                        l.set_fixed(fix_landmarks);
                        l.set_marginalised(true);
                        landmark_vertexes[landmark_id] = ba.add_vertex(static_cast<optimisation::vertex&&>(l));
                        non_fixed_landmarks += (fix_landmarks == false);
                    }
                    optimisation::edge m{ optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)) };
                    m.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { obs.point[0], obs.point[1] } }.data()));
                    m.add_vertex(camera_vertexes[frame_id]);
                    m.add_vertex(landmark_vertexes[landmark_id]);
                    m.set_loss(lossfunction);
                    ba.add_edge(static_cast<optimisation::edge&&>(m));
                    ++non_fixed_edges;
                }
            }

            // Remove vertexes not attached to any edges.
            for (const auto& v : camera_vertexes) {
                if (ba.get_connected_edges(v.second).empty()) {
                    non_fixed_poses -= !v.second->is_fixed();
                    ba.remove_vertex(v.second);
                }
            }
            for (const auto& v : landmark_vertexes) {
                if (ba.get_connected_edges(v.second).empty()) {
                    non_fixed_landmarks -= !v.second->is_fixed();
                    ba.remove_vertex(v.second);
                }
            }

            // Check for some invalid optimiser states.
            if (non_fixed_poses == 0 && non_fixed_landmarks == 0) {
                core::logger::log(core::logger::level::note, "Optimised: No non fixed poses or landmarks [frames: %d landmarks: %d edges: %d]", non_fixed_poses, non_fixed_landmarks, non_fixed_edges);
                return;
            }
            if (camera_vertexes.empty() || landmark_vertexes.empty()) {
                core::logger::log(core::logger::level::note, "Optimised: Nothing to optimise [frames: %zu, landmarks: %zu]", camera_vertexes.size(), landmark_vertexes.size());
                return;
            }

            // Run the optimisation.
            double initial_chi = ba.get_current_chi();
            int number_of_accepted_rounds = ba.solve(rounds, use_relative_convergence);
            core::logger::log(core::logger::level::info, "Optimised: %f to %f error [frames: %d landmarks: %d edges: %d] [%d/%d valid rounds]", initial_chi, ba.get_current_chi(), non_fixed_poses, non_fixed_landmarks, non_fixed_edges, number_of_accepted_rounds, rounds);
            // Apply optimised vertices to frames and landmarks.
            for (const auto& [frame_id, vertex] : camera_vertexes) {
                mapping::frame& frame = frames.at(frame_id);
                const double* const p = vertex->get_parameters();
                const math::se3<double> v_se3 = math::se3<double>(math::so3<double>(p[6], p[3], p[4], p[5]), { { p[0], p[1], p[2] } });
                frame.rotation = v_se3.rotation().get_matrix();
                frame.translation = v_se3.translation();
            }
            for (const auto& [landmark_id, vertex] : landmark_vertexes) {
                mapping::point& landmark = this->landmarks.at(landmark_id);
                const double* const p = vertex->get_parameters();
                landmark.location[0] = p[0];
                landmark.location[1] = p[1];
                landmark.location[2] = p[2];
            }
        }

        void cull() {
            const size_t landmarks_before_cull = this->landmarks.size();

            constexpr static const double outlier_scene_scale_multiple = 200.0;
            constexpr static const size_t outlier_minimum_frames = 10;
            bool outlier_bound_valid = false;
            double centroid_x = 0.0;
            double centroid_y = 0.0;
            double centroid_z = 0.0;
            double outlier_distance_squared_maximum = 0.0;
            if (this->frames.size() >= outlier_minimum_frames) {
                std::vector<double> cx;
                std::vector<double> cy;
                std::vector<double> cz;
                cx.reserve(this->frames.size());
                cy.reserve(this->frames.size());
                cz.reserve(this->frames.size());
                for (const auto& [frame_id, frame] : this->frames) {
                    static_cast<void>(frame_id);
                    const math::matrix<double, 3, 1> centre = -math::transpose(frame.rotation) * frame.translation;
                    cx.push_back(centre[0]);
                    cy.push_back(centre[1]);
                    cz.push_back(centre[2]);
                }
                const auto median = [](std::vector<double>& v) -> double {
                    std::sort(v.begin(), v.end());
                    return v[v.size() / 2];
                };
                centroid_x = median(cx);
                centroid_y = median(cy);
                centroid_z = median(cz);
                std::vector<double> distances;
                distances.reserve(this->frames.size());
                for (size_t i = 0; i < cx.size(); ++i) {
                    const double dx = cx[i] - centroid_x;
                    const double dy = cy[i] - centroid_y;
                    const double dz = cz[i] - centroid_z;
                    distances.push_back(math::sqrt((dx * dx) + (dy * dy) + (dz * dz)));
                }
                const double robust_scale = median(distances);
                if (robust_scale > 1.0e-9) {
                    const double bound = outlier_scene_scale_multiple * robust_scale;
                    outlier_distance_squared_maximum = bound * bound;
                    outlier_bound_valid = true;
                }
            }

            size_t outliers_culled = 0;
            for (std::unordered_map<int, mapping::point>::iterator it = this->landmarks.begin(); it != this->landmarks.end();) {
                if (outlier_bound_valid) {
                    const double offset_x = it->second.location[0] - centroid_x;
                    const double offset_y = it->second.location[1] - centroid_y;
                    const double offset_z = it->second.location[2] - centroid_z;
                    const double offset_squared = (offset_x * offset_x) + (offset_y * offset_y) + (offset_z * offset_z);
                    if (!(offset_squared <= outlier_distance_squared_maximum)) {
                        this->observations.erase(it->first);
                        it = this->landmarks.erase(it);
                        ++outliers_culled;
                        continue;
                    }
                }
                if ((it->first + 500) < this->next_landmark_id) {
                    ++it;
                    continue;
                }
                const std::unordered_map<int, std::vector<observation>>::const_iterator observations_it = this->observations.find(it->first);
                if (observations_it == this->observations.end()) {
                    it = this->landmarks.erase(it);
                    continue;
                }
                const std::vector<observation>& landmark_observations = observations_it->second;
                const bool not_seen_in_many_frames = landmark_observations.size() <= 4;
                const bool not_seen_recently = landmark_observations.empty() || ((landmark_observations.back().frame_id + 7) < this->next_frame_id);
                if (not_seen_in_many_frames && not_seen_recently) {
                    this->observations.erase(it->first);
                    it = this->landmarks.erase(it);
                    continue;
                }
                float reprojection_error = 0.0f;
                size_t processed_observations = 0;
                for (const auto& obs : landmark_observations) {
                    const int frame_id = obs.frame_id;
                    const std::unordered_map<int, mapping::frame>::const_iterator frame_it = this->frames.find(frame_id);
                    if (frame_it == this->frames.end()) {
                        continue;
                    }
                    const mapping::frame& frame = frame_it->second;
                    const math::matrix<double, 2, 1> measured = obs.point;
                    math::matrix<double, 3, 1> mapped;
                    mapped = (frame.rotation * it->second.location) + frame.translation;
                    math::matrix<double, 2, 1> reprojected;
                    if (!frame.camera.project(mapped.data(), reprojected.data())) {
                        reprojection_error += 5.991f;
                    }
                    else {
                        reprojection_error += static_cast<float>(math::sqrt((measured - reprojected).get_length_squared()));
                    }
                    ++processed_observations;
                }
                if (processed_observations == 0) {
                    this->observations.erase(it->first);
                    it = this->landmarks.erase(it);
                    continue;
                }
                reprojection_error /= static_cast<float>(processed_observations);
                if (reprojection_error >= 5.991f) {
                    this->observations.erase(it->first);
                    it = this->landmarks.erase(it);
                    continue;
                }
                ++it;
            }
            const size_t landmarks_after_cull = this->landmarks.size();
            core::logger::log(core::logger::level::info, "Culled: %zu points (%zu scale-outliers)", landmarks_before_cull - landmarks_after_cull, outliers_culled);
        }
    };
}

#endif // ZEROSLAM_MAPPING_MAP_HPP
