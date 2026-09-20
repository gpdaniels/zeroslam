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
#include "mapping/line.hpp"
#include "mapping/point.hpp"
#include "math/lie.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/baseline.hpp"
#include "optimisation/edges/line_reprojection.hpp"
#include "optimisation/edges/reprojection.hpp"
#include "optimisation/edges/reprojection_inverse_depth.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/loss.hpp"
#include "optimisation/losses/huber.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/line.hpp"
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
            int octave;
        };

        struct line_observation final {
            int frame_id;
            double x1;
            double y1;
            double x2;
            double y2;
        };

    public:
        std::unordered_map<int, mapping::frame> frames;
        std::unordered_map<int, mapping::point> landmarks;
        std::unordered_map<decltype(mapping::point::id), std::vector<observation>> observations;
        std::unordered_map<int, mapping::line> line_landmarks;
        std::unordered_map<decltype(mapping::line::id), std::vector<line_observation>> line_observations;

        int next_frame_id = 0;
        int next_landmark_id = 0;
        int next_line_landmark_id = 0;

        int allocate_frame_id() {
            return this->next_frame_id++;
        }

        int allocate_landmark_id() {
            return this->next_landmark_id++;
        }

        int allocate_line_landmark_id() {
            return this->next_line_landmark_id++;
        }

        int gauge_frame_id = 0;

        std::unordered_set<int> anchor_frame_ids;

        constexpr static const double gauge_baseline_information = 1.0e6;

        constexpr static const int gauge_minimum_shared_landmarks = 20;
        constexpr static const size_t scale_links_minimum = 30;
        constexpr static const double scale_link_parallax_degrees_minimum = 1.0;

        constexpr static const int covisible_free_minimum = 20;
        constexpr static const size_t covisible_free_maximum = 10;
        constexpr static const size_t window_anchor_links_minimum = 50;

        constexpr static const double inlier_bound_squared = 5.991;
        constexpr static const float cull_mean_normalised_distance_maximum = 5.991f;

        constexpr static const int first_pass_rounds = 5;

        constexpr static const size_t repose_observations_minimum = 10;
        constexpr static const size_t repose_inlier_fraction_denominator = 2;
        constexpr static const double repose_wide_bound_pixels = 20.0;
        constexpr static const int repose_rounds = 10;
        constexpr static const int repose_recent_frames = 90;

        constexpr static const double gross_error_squared = 25.0;
        constexpr static const size_t divergence_gross_fraction_denominator = 4;
        constexpr static const size_t divergence_minimum_observations = 50;
        constexpr static const size_t frame_divergence_minimum_observations = 20;

        std::vector<int> corrected_frame_ids;

        optimisation::factor_graph::strategy solver_strategy = optimisation::factor_graph::strategy::dense_schur;
        optimisation::factor_graph::precision solver_precision = optimisation::factor_graph::precision::double_precision;

        constexpr static const double line_minimum_ray_angle_degrees = 10.0;
        constexpr static const double line_minimum_camera_distance = 1.0e-2;
        constexpr static const int line_minimum_observations = 3;

        double line_ray_angle_degrees = map::line_minimum_ray_angle_degrees;

        static bool line_observation_usable(const geometry::plucker& line_camera, const math::matrix<double, 3, 1>& end_camera_1, const math::matrix<double, 3, 1>& end_camera_2, const double minimum_ray_angle_degrees = map::line_minimum_ray_angle_degrees) {
            const double distance = line_camera.distance_to_origin();
            if (distance < map::line_minimum_camera_distance) {
                return false;
            }
            const double limit = math::cos(minimum_ray_angle_degrees * math::pi<double>() / 180.0);
            const math::matrix<double, 3, 1>* const ends[2] = { &end_camera_1, &end_camera_2 };
            for (int end = 0; end < 2; ++end) {
                const double length = math::sqrt(ends[end]->get_length_squared());
                if (length < 1.0e-12) {
                    return false;
                }
                if (math::abs(geometry::plucker::dot(line_camera.direction, *ends[end]) / length) > limit) {
                    return false;
                }
            }
            return true;
        }

    public:
        void add_frame(const mapping::frame& frame) {
            frames[frame.id] = frame;
        }

        void add_landmark(const mapping::point& landmark) {
            this->landmarks[landmark.id] = landmark;
        }

        void add_line_landmark(const mapping::line& landmark) {
            this->line_landmarks[landmark.id] = landmark;
        }

        void add_line_observation(int frame_id, const mapping::line& landmark, double x1, double y1, double x2, double y2) {
            this->line_observations[landmark.id].push_back(line_observation{ frame_id, x1, y1, x2, y2 });
        }

        void add_observation(const mapping::frame& frame, const mapping::point& landmark, size_t kp_index) {
            const feature::point& keypoint = frame.keypoints[kp_index];
            this->observations[landmark.id].push_back(observation{ frame.id, kp_index, math::matrix<double, 2, 1>{ { static_cast<double>(keypoint.x), static_cast<double>(keypoint.y) } }, keypoint.octave });
        }

        bool merge_landmark(const int keep_id, const int drop_id) {
            if ((keep_id == drop_id) || (this->landmarks.count(keep_id) == 0) || (this->landmarks.count(drop_id) == 0)) {
                return false;
            }
            const std::unordered_map<decltype(mapping::point::id), std::vector<observation>>::iterator dropped = this->observations.find(drop_id);
            if (dropped != this->observations.end()) {
                std::vector<observation>& kept = this->observations[keep_id];
                for (const observation& candidate : dropped->second) {
                    bool present = false;
                    for (const observation& existing : kept) {
                        present = present || (existing.frame_id == candidate.frame_id);
                    }
                    if (!present) {
                        kept.push_back(candidate);
                    }
                }
                std::sort(kept.begin(), kept.end(), [](const observation& lhs, const observation& rhs) {
                    return lhs.frame_id < rhs.frame_id;
                });
                this->observations.erase(dropped);
            }
            this->landmarks.erase(drop_id);
            return true;
        }

        void add_observation(int frame_id, const mapping::point& landmark, double x, double y, int octave = 0) {
            this->observations[landmark.id].push_back(observation{ frame_id, static_cast<size_t>(-1), math::matrix<double, 2, 1>{ { x, y } }, octave });
        }

        math::matrix<double, 3, 1> depth_direction_of(const mapping::point& landmark) const {
            if (landmark.inverse_depth) {
                return math::matrix<double, 3, 1>({ 0.0, 0.0, 1.0 });
            }
            const std::unordered_map<decltype(mapping::point::id), std::vector<observation>>::const_iterator found = this->observations.find(landmark.id);
            if ((found != this->observations.end()) && !found->second.empty()) {
                const std::unordered_map<int, mapping::frame>::const_iterator frame_it = this->frames.find(found->second.front().frame_id);
                if (frame_it != this->frames.end()) {
                    const math::matrix<double, 3, 1> centre = -(math::transpose(frame_it->second.rotation) * frame_it->second.translation);
                    const math::matrix<double, 3, 1> ray = landmark.location - centre;
                    const double length = math::sqrt(ray.get_length_squared());
                    if (length > 1.0e-12) {
                        return ray * (1.0 / length);
                    }
                }
            }
            return math::matrix<double, 3, 1>({ 0.0, 0.0, 1.0 });
        }

        void estimate_landmark_uncertainty(mapping::point& landmark) {
            const std::unordered_map<decltype(mapping::point::id), std::vector<observation>>::const_iterator found = this->observations.find(landmark.id);
            if (found == this->observations.end()) {
                return;
            }
            const optimisation::loss lossfunction(optimisation::losses::huber(math::sqrt(map::inlier_bound_squared)));
            math::matrix<double, 3, 3> information = math::matrix<double, 3, 3>::zero();
            for (const observation& obs : found->second) {
                const std::unordered_map<int, mapping::frame>::const_iterator frame_it = this->frames.find(obs.frame_id);
                if (frame_it == this->frames.end()) {
                    continue;
                }
                const mapping::frame& frame = frame_it->second;
                optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
                const math::se3<double> pose(frame.rotation, frame.translation);
                const double pose_parameters[7] = { pose.translation()[0], pose.translation()[1], pose.translation()[2], pose.rotation().get_quaternion()[1], pose.rotation().get_quaternion()[2], pose.rotation().get_quaternion()[3], pose.rotation().get_quaternion()[0] };
                pose_vertex.set_parameters(&pose_parameters[0], 7);
                optimisation::vertex point_vertex{ optimisation::vertices::point() };
                point_vertex.set_parameters(landmark.inverse_depth ? landmark.inverse_parameters.data() : landmark.location.data(), 3);
                optimisation::edge factor = map::make_point_edge(frame.camera, landmark, obs, lossfunction, frame.measurement_sigma);
                factor.add_vertex(&pose_vertex);
                factor.add_vertex(&point_vertex);
                factor.compute_jacobians();
                const math::matrix<double, 0, 0>& jacobian = factor.get_jacobians()[1];
                const double weight = 1.0 / (frame.measurement_sigma * frame.measurement_sigma);
                for (size_t a = 0; a < 3; ++a) {
                    for (size_t b = 0; b < 3; ++b) {
                        double sum = 0.0;
                        for (size_t r = 0; r < 2; ++r) {
                            sum += jacobian[r][a] * jacobian[r][b];
                        }
                        information[a][b] += weight * sum;
                    }
                }
            }
            landmark.set_information(information, this->depth_direction_of(landmark));
        }

        static optimisation::edge make_point_edge(const sensor::model& camera_model, const mapping::point& landmark, const observation& obs, const optimisation::loss& lossfunction, const double sigma) {
            optimisation::edge m = landmark.inverse_depth
                                       ? optimisation::edge{ optimisation::edges::reprojection_inverse_depth(sensor::camera::model<double>(camera_model), landmark.anchor_rotation, landmark.anchor_translation) }
                                       : optimisation::edge{ optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)) };
            m.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { obs.point[0], obs.point[1] } }.data()));
            if (sigma != 1.0) {
                m.set_information(math::matrix<double, 0, 0>::identity(2, 2) * (1.0 / (sigma * sigma)));
            }
            m.set_loss(lossfunction);
            return m;
        }

        bool repose_frame(const int frame_id, const std::vector<std::pair<int, const observation*>>& frame_observations, const sensor::model& camera_model) {
            const std::unordered_map<int, mapping::frame>::iterator frame_it = this->frames.find(frame_id);
            if ((frame_it == this->frames.end()) || (frame_observations.size() < map::repose_observations_minimum)) {
                return false;
            }
            mapping::frame& frame = frame_it->second;
            const auto count_inliers = [this, &frame, &frame_observations]() {
                size_t inliers = 0;
                for (const std::pair<int, const observation*>& entry : frame_observations) {
                    math::matrix<double, 2, 1> reprojected;
                    inliers += map::project_landmark(frame, this->landmarks.at(entry.first), reprojected) && ((reprojected - entry.second->point).get_length_squared() <= map::inlier_bound_squared * frame.measurement_sigma * frame.measurement_sigma);
                }
                return inliers;
            };
            const size_t inliers_before = count_inliers();
            if (inliers_before * map::repose_inlier_fraction_denominator >= frame_observations.size()) {
                return false;
            }
            const optimisation::loss wide_loss{ optimisation::losses::huber(map::repose_wide_bound_pixels) };
            const optimisation::loss tight_loss(optimisation::losses::huber(math::sqrt(map::inlier_bound_squared)));
            optimisation::factor_graph solver;
            const math::se3<double> prior(frame.rotation, frame.translation);
            optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
            const double pose_parameters[7] = { prior.translation()[0], prior.translation()[1], prior.translation()[2], prior.rotation().get_quaternion()[1], prior.rotation().get_quaternion()[2], prior.rotation().get_quaternion()[3], prior.rotation().get_quaternion()[0] };
            pose_vertex.set_parameters(&pose_parameters[0], 7);
            optimisation::vertex* const pose_node = solver.add_vertex(static_cast<optimisation::vertex&&>(pose_vertex));
            std::vector<optimisation::edge*> edges;
            edges.reserve(frame_observations.size());
            for (const std::pair<int, const observation*>& entry : frame_observations) {
                const mapping::point& landmark = this->landmarks.at(entry.first);
                optimisation::vertex point_vertex{ optimisation::vertices::point() };
                point_vertex.set_parameters(landmark.inverse_depth ? landmark.inverse_parameters.data() : landmark.location.data(), 3);
                point_vertex.set_fixed(true);
                point_vertex.set_marginalised(true);
                optimisation::vertex* const point_node = solver.add_vertex(static_cast<optimisation::vertex&&>(point_vertex));
                optimisation::edge factor = map::make_point_edge(camera_model, landmark, *entry.second, wide_loss, frame.measurement_sigma);
                factor.add_vertex(pose_node);
                factor.add_vertex(point_node);
                edges.push_back(solver.add_edge(static_cast<optimisation::edge&&>(factor)));
            }
            solver.solve(map::repose_rounds, true);
            for (optimisation::edge* const factor : edges) {
                factor->set_loss(tight_loss);
            }
            solver.solve(map::repose_rounds, true);
            const double* const p = pose_node->get_parameters();
            const math::se3<double> refined(math::so3<double>(p[6], p[3], p[4], p[5]), { { p[0], p[1], p[2] } });
            const math::matrix<double, 3, 3> rotation_before = frame.rotation;
            const math::matrix<double, 3, 1> translation_before = frame.translation;
            frame.rotation = refined.rotation().get_matrix();
            frame.translation = refined.translation();
            const size_t inliers_after = count_inliers();
            if ((inliers_after <= inliers_before) || (inliers_after * map::repose_inlier_fraction_denominator < frame_observations.size())) {
                frame.rotation = rotation_before;
                frame.translation = translation_before;
                return false;
            }
            core::logger::log(core::logger::level::info, "Re-posed frame %d before the adjustment: %zu of %zu observations within the bound, was %zu.", frame_id, inliers_after, frame_observations.size(), inliers_before);
            this->corrected_frame_ids.push_back(frame_id);
            return true;
        }

        void optimise(int local_window, bool fix_landmarks, int rounds, bool use_relative_convergence = false) {
            this->corrected_frame_ids.clear();
            if (this->frames.empty()) {
                return;
            }
            std::unordered_map<int, optimisation::vertex*> camera_vertexes;
            std::unordered_map<int, optimisation::vertex*> landmark_vertexes;
            std::unordered_map<int, optimisation::vertex*> line_vertexes;
            // Get camera parameters.
            double camera_parameters[sensor::model::parameter_count];
            this->frames.begin()->second.camera.get_parameters(camera_parameters, sensor::model::parameter_count);
            const sensor::model camera_model(camera_parameters, sensor::model::parameter_count);
            // Setup ba.
            const optimisation::loss lossfunction(optimisation::losses::huber(math::sqrt(5.991)));
            optimisation::factor_graph ba;
            ba.set_strategy(this->solver_strategy);
            ba.set_precision(this->solver_precision);
            // Add frames.
            const int local_window_below = this->next_frame_id - 1 - local_window;
            const int local_window_fixed_below = local_window_below + 1;
            std::unordered_set<int> relevant_frame_ids;
            std::unordered_set<int> active_landmark_ids;
            std::unordered_set<int> active_line_landmark_ids;
            std::unordered_set<int> active_frame_ids;

            if (local_window > 0) {
                for (const auto& [frame_id, frame] : this->frames) {
                    const bool fixed = ((frame_id == this->gauge_frame_id) || (this->anchor_frame_ids.count(frame_id) != 0) || (frame_id < local_window_fixed_below));
                    if (!fixed) {
                        active_frame_ids.insert(frame_id);
                    }
                }
                if (!fix_landmarks) {
                    std::unordered_map<int, int> shared_with_window;
                    for (const auto& [landmark_id, landmark_obs] : this->observations) {
                        static_cast<void>(landmark_id);
                        bool seen_by_active = false;
                        for (const auto& obs : landmark_obs) {
                            seen_by_active = seen_by_active || (active_frame_ids.count(obs.frame_id) != 0);
                        }
                        if (!seen_by_active) {
                            continue;
                        }
                        for (const auto& obs : landmark_obs) {
                            if ((active_frame_ids.count(obs.frame_id) == 0) && (obs.frame_id != this->gauge_frame_id)) {
                                ++shared_with_window[obs.frame_id];
                            }
                        }
                    }
                    std::vector<std::pair<int, int>> covisible;
                    for (const auto& [frame_id, shared] : shared_with_window) {
                        if ((shared >= map::covisible_free_minimum) && (this->frames.count(frame_id) != 0)) {
                            covisible.push_back({ frame_id, shared });
                        }
                    }
                    std::sort(covisible.begin(), covisible.end(), [](const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) {
                        return (lhs.second != rhs.second) ? (lhs.second > rhs.second) : (lhs.first > rhs.first);
                    });
                    for (size_t i = 0; (i < covisible.size()) && (i < map::covisible_free_maximum); ++i) {
                        active_frame_ids.insert(covisible[i].first);
                    }
                }
                const int horizon_threshold = (this->next_frame_id - 1) - (2 * local_window);
                size_t recent_anchor_links = 0;
                for (const auto& [landmark_id, landmark_obs] : this->observations) {
                    bool seen_by_active = false;
                    for (const auto& obs : landmark_obs) {
                        seen_by_active = seen_by_active || (active_frame_ids.count(obs.frame_id) != 0);
                    }
                    if (!seen_by_active) {
                        continue;
                    }
                    active_landmark_ids.insert(static_cast<int>(landmark_id));
                    for (const auto& obs : landmark_obs) {
                        recent_anchor_links += (active_frame_ids.count(obs.frame_id) == 0) && (obs.frame_id >= horizon_threshold);
                    }
                }
                const bool anchor_everywhere = recent_anchor_links < map::window_anchor_links_minimum;
                core::logger::log(core::logger::level::debug, "Window: %zu free frames, %zu landmarks, %zu recent anchor links%s.", active_frame_ids.size(), active_landmark_ids.size(), recent_anchor_links, anchor_everywhere ? ", anchoring everywhere" : "");
                for (const int landmark_id : active_landmark_ids) {
                    for (const auto& obs : this->observations.at(landmark_id)) {
                        if (anchor_everywhere || (obs.frame_id >= horizon_threshold)) {
                            relevant_frame_ids.insert(obs.frame_id);
                        }
                    }
                }
                for (const auto& [landmark_id, landmark_obs] : this->line_observations) {
                    bool seen_by_active = false;
                    for (const line_observation& obs : landmark_obs) {
                        seen_by_active = seen_by_active || (active_frame_ids.count(obs.frame_id) != 0);
                    }
                    if (seen_by_active) {
                        active_line_landmark_ids.insert(static_cast<int>(landmark_id));
                        for (const line_observation& obs : landmark_obs) {
                            if (anchor_everywhere || (obs.frame_id >= horizon_threshold)) {
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
                for (const auto& [landmark_id, _] : this->line_landmarks) {
                    active_line_landmark_ids.insert(static_cast<int>(landmark_id));
                }
            }

            if (!fix_landmarks) {
                std::unordered_map<int, std::vector<std::pair<int, const observation*>>> observations_by_frame;
                for (const int landmark_id : active_landmark_ids) {
                    for (const observation& obs : this->observations.at(landmark_id)) {
                        if ((relevant_frame_ids.count(obs.frame_id) != 0) && (obs.frame_id != this->gauge_frame_id)) {
                            observations_by_frame[obs.frame_id].push_back({ landmark_id, &obs });
                        }
                    }
                }
                std::vector<int> frame_ids_to_check;
                frame_ids_to_check.reserve(observations_by_frame.size());
                for (const auto& [frame_id, frame_observations] : observations_by_frame) {
                    static_cast<void>(frame_observations);
                    frame_ids_to_check.push_back(frame_id);
                }
                std::sort(frame_ids_to_check.begin(), frame_ids_to_check.end());
                for (const int frame_id : frame_ids_to_check) {
                    if ((local_window > 0) && (frame_id < local_window_fixed_below)) {
                        continue;
                    }
                    if ((local_window == 0) && (frame_id < (this->next_frame_id - 1 - (2 * map::repose_recent_frames)))) {
                        continue;
                    }
                    this->repose_frame(frame_id, observations_by_frame.at(frame_id), camera_model);
                }
            }

            // Add frames.
            int non_fixed_poses = 0;
            for (const auto& frame_id : relevant_frame_ids) {
                const auto& frame = this->frames.at(frame_id);
                const math::se3<double> v_se3(frame.rotation, frame.translation);
                const bool fixed = ((frame_id == this->gauge_frame_id) || (this->anchor_frame_ids.count(frame_id) != 0) || ((local_window > 0) && (active_frame_ids.count(frame_id) == 0)));
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

            struct point_edge final {
                optimisation::edge* factor;
                int landmark_id;
                int frame_id;
            };

            std::vector<point_edge> point_edges;
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
                        l.set_parameters(landmark_record.inverse_depth ? landmark_record.inverse_parameters.data() : landmark_record.location.data(), 3);
                        l.set_fixed(fix_landmarks);
                        l.set_marginalised(true);
                        landmark_vertexes[landmark_id] = ba.add_vertex(static_cast<optimisation::vertex&&>(l));
                        non_fixed_landmarks += (fix_landmarks == false);
                    }
                    optimisation::edge m = map::make_point_edge(camera_model, landmark_record, obs, lossfunction, this->frames.at(frame_id).measurement_sigma);
                    m.add_vertex(camera_vertexes[frame_id]);
                    m.add_vertex(landmark_vertexes[landmark_id]);
                    if (fix_landmarks && (landmark_record.uncertainty == mapping::point::uncertainty_kind::unbounded)) {
                        m.compute_jacobians();
                        m.set_information(math::matrix<double, 0, 0>(2, 2, landmark_record.observation_information(m.get_jacobians()[1], this->frames.at(frame_id).measurement_sigma).data()));
                    }
                    point_edges.push_back({ ba.add_edge(static_cast<optimisation::edge&&>(m)), landmark_id, frame_id });
                    ++non_fixed_edges;
                }
            }

            int non_fixed_lines = 0;
            for (const int landmark_id : active_line_landmark_ids) {
                const std::unordered_map<int, std::vector<line_observation>>::const_iterator observations_it = this->line_observations.find(landmark_id);
                if ((observations_it == this->line_observations.end()) || (static_cast<int>(observations_it->second.size()) < map::line_minimum_observations) || (this->line_landmarks.count(landmark_id) == 0)) {
                    continue;
                }
                const mapping::line& landmark_record = this->line_landmarks.at(landmark_id);
                size_t usable_observations = 0;
                for (const line_observation& obs : observations_it->second) {
                    if (camera_vertexes.count(obs.frame_id) == 0) {
                        continue;
                    }
                    if (camera_vertexes.at(obs.frame_id)->is_fixed() && fix_landmarks) {
                        continue;
                    }
                    const mapping::frame& observing_frame = this->frames.at(obs.frame_id);
                    if (!map::line_observation_usable(landmark_record.plucker_line.transformed(observing_frame.rotation, observing_frame.translation), (observing_frame.rotation * landmark_record.locations[0]) + observing_frame.translation, (observing_frame.rotation * landmark_record.locations[1]) + observing_frame.translation, this->line_ray_angle_degrees)) {
                        continue;
                    }
                    ++usable_observations;
                }
                if (!fix_landmarks && (usable_observations < 2)) {
                    continue;
                }
                bool landmark_added = false;
                for (const line_observation& obs : observations_it->second) {
                    if (camera_vertexes.count(obs.frame_id) == 0) {
                        continue;
                    }
                    if (camera_vertexes.at(obs.frame_id)->is_fixed() && fix_landmarks) {
                        continue;
                    }
                    const mapping::frame& observing_frame = this->frames.at(obs.frame_id);
                    if (!map::line_observation_usable(landmark_record.plucker_line.transformed(observing_frame.rotation, observing_frame.translation), (observing_frame.rotation * landmark_record.locations[0]) + observing_frame.translation, (observing_frame.rotation * landmark_record.locations[1]) + observing_frame.translation, this->line_ray_angle_degrees)) {
                        continue;
                    }
                    optimisation::edges::line_reprojection factor{ sensor::camera::model<double>(camera_model) };
                    if (!factor.set_measured_segment(obs.x1, obs.y1, obs.x2, obs.y2)) {
                        continue;
                    }
                    if (!landmark_added) {
                        landmark_added = true;
                        optimisation::vertex l{ optimisation::vertices::line() };
                        const double line_parameters[6] = { landmark_record.plucker_line.moment[0], landmark_record.plucker_line.moment[1], landmark_record.plucker_line.moment[2], landmark_record.plucker_line.direction[0], landmark_record.plucker_line.direction[1], landmark_record.plucker_line.direction[2] };
                        l.set_parameters(&line_parameters[0], 6);
                        l.set_fixed(fix_landmarks);
                        l.set_marginalised(true);
                        line_vertexes[landmark_id] = ba.add_vertex(static_cast<optimisation::vertex&&>(l));
                        non_fixed_lines += (fix_landmarks == false);
                    }
                    optimisation::edge m{ factor };
                    m.add_vertex(camera_vertexes[obs.frame_id]);
                    m.add_vertex(line_vertexes[landmark_id]);
                    m.set_loss(lossfunction);
                    ba.add_edge(static_cast<optimisation::edge&&>(m));
                    ++non_fixed_edges;
                }
            }

            if (!fix_landmarks && (camera_vertexes.count(this->gauge_frame_id) > 0)) {
                const auto camera_centre = [this](const int frame_id) {
                    const mapping::frame& frame = this->frames.at(frame_id);
                    return math::matrix<double, 3, 1>(-(math::transpose(frame.rotation) * frame.translation));
                };
                const math::matrix<double, 3, 1> centre_gauge = camera_centre(this->gauge_frame_id);
                std::unordered_map<int, std::vector<int>> shared_landmarks;
                for (const auto& [landmark_id, landmark_obs] : this->observations) {
                    if (landmark_vertexes.count(static_cast<int>(landmark_id)) == 0) {
                        continue;
                    }
                    bool at_gauge = false;
                    for (const auto& obs : landmark_obs) {
                        if (obs.frame_id == this->gauge_frame_id) {
                            at_gauge = true;
                            break;
                        }
                    }
                    if (!at_gauge) {
                        continue;
                    }
                    for (const auto& obs : landmark_obs) {
                        if (obs.frame_id != this->gauge_frame_id) {
                            shared_landmarks[obs.frame_id].push_back(static_cast<int>(landmark_id));
                        }
                    }
                }
                std::vector<int> candidate_frame_ids;
                for (const auto& [frame_id, vertex] : camera_vertexes) {
                    if ((!vertex->is_fixed()) && (!ba.get_connected_edges(vertex).empty())) {
                        candidate_frame_ids.push_back(frame_id);
                    }
                }
                std::sort(candidate_frame_ids.begin(), candidate_frame_ids.end());
                int best_frame_id = -1;
                double best_parallax = -1.0;
                int fallback_frame_id = -1;
                double fallback_distance = -1.0;
                std::vector<double> parallaxes;
                for (const int frame_id : candidate_frame_ids) {
                    const math::matrix<double, 3, 1> centre_candidate = camera_centre(frame_id);
                    const double distance = math::sqrt((centre_candidate - centre_gauge).get_length_squared());
                    if ((distance > fallback_distance) || ((distance == fallback_distance) && (frame_id > fallback_frame_id))) {
                        fallback_distance = distance;
                        fallback_frame_id = frame_id;
                    }
                    const std::unordered_map<int, std::vector<int>>::const_iterator shared = shared_landmarks.find(frame_id);
                    if ((shared == shared_landmarks.end()) || (static_cast<int>(shared->second.size()) < map::gauge_minimum_shared_landmarks)) {
                        continue;
                    }
                    parallaxes.clear();
                    for (const int landmark_id : shared->second) {
                        const math::matrix<double, 3, 1>& location = this->landmarks.at(landmark_id).location;
                        const math::matrix<double, 3, 1> ray_gauge = location - centre_gauge;
                        const math::matrix<double, 3, 1> ray_candidate = location - centre_candidate;
                        const double norms = math::sqrt(ray_gauge.get_length_squared()) * math::sqrt(ray_candidate.get_length_squared());
                        if (norms < 1.0e-24) {
                            continue;
                        }
                        const double cosine = math::max(-1.0, math::min(1.0, ((ray_gauge[0] * ray_candidate[0]) + (ray_gauge[1] * ray_candidate[1]) + (ray_gauge[2] * ray_candidate[2])) / norms));
                        parallaxes.push_back(math::acos(cosine));
                    }
                    if (parallaxes.empty()) {
                        continue;
                    }
                    std::sort(parallaxes.begin(), parallaxes.end());
                    const double median_parallax = parallaxes[parallaxes.size() / 2];
                    if ((median_parallax > best_parallax) || ((median_parallax == best_parallax) && (frame_id > best_frame_id))) {
                        best_parallax = median_parallax;
                        best_frame_id = frame_id;
                    }
                }
                const int partner_frame_id = (best_frame_id >= 0) ? best_frame_id : fallback_frame_id;
                if (partner_frame_id >= 0) {
                    const double target_distance = math::sqrt((camera_centre(partner_frame_id) - centre_gauge).get_length_squared());
                    if (target_distance > 1.0e-9) {
                        optimisation::edge gauge_edge{ optimisation::edges::baseline(target_distance) };
                        gauge_edge.add_vertex(camera_vertexes.at(partner_frame_id));
                        gauge_edge.add_vertex(camera_vertexes.at(this->gauge_frame_id));
                        gauge_edge.set_information(math::matrix<double, 0, 0>::identity(1, 1) * map::gauge_baseline_information);
                        ba.add_edge(static_cast<optimisation::edge&&>(gauge_edge));
                    }
                }
            }
            if (!fix_landmarks && (local_window > 0) && (camera_vertexes.count(this->gauge_frame_id) == 0)) {
                const auto camera_centre = [this](const int frame_id) {
                    const mapping::frame& frame = this->frames.at(frame_id);
                    return math::matrix<double, 3, 1>(-(math::transpose(frame.rotation) * frame.translation));
                };
                const double cosine_maximum = math::cos(map::scale_link_parallax_degrees_minimum * 3.14159265358979323846 / 180.0);
                size_t scale_links = 0;
                size_t shared = 0;
                for (const auto& [landmark_id, vertex] : landmark_vertexes) {
                    static_cast<void>(vertex);
                    int fixed_frame_id = -1;
                    int free_frame_id = -1;
                    for (const observation& obs : this->observations.at(landmark_id)) {
                        const std::unordered_map<int, optimisation::vertex*>::const_iterator camera = camera_vertexes.find(obs.frame_id);
                        if (camera == camera_vertexes.end()) {
                            continue;
                        }
                        if (camera->second->is_fixed()) {
                            fixed_frame_id = (fixed_frame_id < 0) ? obs.frame_id : fixed_frame_id;
                        }
                        else {
                            free_frame_id = obs.frame_id;
                        }
                    }
                    if ((fixed_frame_id < 0) || (free_frame_id < 0)) {
                        continue;
                    }
                    ++shared;
                    const math::matrix<double, 3, 1>& location = this->landmarks.at(landmark_id).location;
                    const math::matrix<double, 3, 1> ray_fixed = location - camera_centre(fixed_frame_id);
                    const math::matrix<double, 3, 1> ray_free = location - camera_centre(free_frame_id);
                    const double norms = math::sqrt(ray_fixed.get_length_squared()) * math::sqrt(ray_free.get_length_squared());
                    if (norms < 1.0e-24) {
                        continue;
                    }
                    const double cosine = ((ray_fixed[0] * ray_free[0]) + (ray_fixed[1] * ray_free[1]) + (ray_fixed[2] * ray_free[2])) / norms;
                    scale_links += (cosine < cosine_maximum) ? 1 : 0;
                }
                core::logger::log(core::logger::level::debug, "Window scale: %zu of %zu landmarks shared with the fixed frames carry parallax.", scale_links, shared);
                if (scale_links < map::scale_links_minimum) {
                    int oldest_frame_id = -1;
                    int newest_frame_id = -1;
                    for (const auto& [frame_id, vertex] : camera_vertexes) {
                        if (vertex->is_fixed() || (frame_id < local_window_fixed_below) || ba.get_connected_edges(vertex).empty()) {
                            continue;
                        }
                        oldest_frame_id = ((oldest_frame_id < 0) || (frame_id < oldest_frame_id)) ? frame_id : oldest_frame_id;
                        newest_frame_id = (frame_id > newest_frame_id) ? frame_id : newest_frame_id;
                    }
                    if ((oldest_frame_id >= 0) && (newest_frame_id > oldest_frame_id)) {
                        const double target_distance = math::sqrt((camera_centre(newest_frame_id) - camera_centre(oldest_frame_id)).get_length_squared());
                        if (target_distance > 1.0e-9) {
                            optimisation::edge gauge_edge{ optimisation::edges::baseline(target_distance) };
                            gauge_edge.add_vertex(camera_vertexes.at(newest_frame_id));
                            gauge_edge.add_vertex(camera_vertexes.at(oldest_frame_id));
                            gauge_edge.set_information(math::matrix<double, 0, 0>::identity(1, 1) * map::gauge_baseline_information);
                            ba.add_edge(static_cast<optimisation::edge&&>(gauge_edge));
                            core::logger::log(core::logger::level::info, "Window scale weakly observed (%zu of %zu shared landmarks carry parallax): holding the distance between frames %d and %d.", scale_links, shared, oldest_frame_id, newest_frame_id);
                        }
                    }
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
            for (const auto& v : line_vertexes) {
                if (ba.get_connected_edges(v.second).empty()) {
                    non_fixed_lines -= !v.second->is_fixed();
                    ba.remove_vertex(v.second);
                }
            }

            // Check for some invalid optimiser states.
            if (non_fixed_poses == 0 && non_fixed_landmarks == 0 && non_fixed_lines == 0) {
                core::logger::log(core::logger::level::note, "Optimised: No non fixed poses or landmarks [frames: %d landmarks: %d edges: %d]", non_fixed_poses, non_fixed_landmarks, non_fixed_edges);
                return;
            }
            if (camera_vertexes.empty() || landmark_vertexes.empty()) {
                core::logger::log(core::logger::level::note, "Optimised: Nothing to optimise [frames: %zu, landmarks: %zu]", camera_vertexes.size(), landmark_vertexes.size());
                return;
            }

            std::unordered_map<int, mapping::frame> frames_before;
            std::unordered_map<int, mapping::point> landmarks_before;
            std::unordered_map<int, mapping::line> lines_before;
            for (const auto& [frame_id, vertex] : camera_vertexes) {
                static_cast<void>(vertex);
                frames_before[frame_id] = this->frames.at(frame_id);
            }
            for (const auto& [landmark_id, vertex] : landmark_vertexes) {
                static_cast<void>(vertex);
                landmarks_before[landmark_id] = this->landmarks.at(landmark_id);
            }
            for (const auto& [landmark_id, vertex] : line_vertexes) {
                static_cast<void>(vertex);
                lines_before[landmark_id] = this->line_landmarks.at(landmark_id);
            }

            double initial_chi = ba.get_current_chi();
            std::vector<std::pair<int, int>> dropped_observations;
            int number_of_accepted_rounds = 0;
            const bool two_pass = !fix_landmarks && (rounds > map::first_pass_rounds);
            if (two_pass) {
                number_of_accepted_rounds = ba.solve(map::first_pass_rounds, use_relative_convergence);
                static_cast<void>(ba.get_current_chi(true));
                std::vector<point_edge> kept_edges;
                kept_edges.reserve(point_edges.size());
                for (const point_edge& entry : point_edges) {
                    if (entry.factor->chi2() > map::inlier_bound_squared) {
                        ba.remove_edge(entry.factor);
                        dropped_observations.push_back({ entry.landmark_id, entry.frame_id });
                    }
                    else {
                        kept_edges.push_back(entry);
                    }
                }
                for (std::unordered_map<int, optimisation::vertex*>::iterator it = landmark_vertexes.begin(); it != landmark_vertexes.end();) {
                    const std::vector<optimisation::edge*> connected = ba.get_connected_edges(it->second);
                    if (connected.size() < 2) {
                        for (optimisation::edge* const factor : connected) {
                            ba.remove_edge(factor);
                        }
                        ba.remove_vertex(it->second);
                        it = landmark_vertexes.erase(it);
                    }
                    else {
                        ++it;
                    }
                }
                for (std::unordered_map<int, optimisation::vertex*>::iterator it = camera_vertexes.begin(); it != camera_vertexes.end();) {
                    if (ba.get_connected_edges(it->second).empty()) {
                        ba.remove_vertex(it->second);
                        it = camera_vertexes.erase(it);
                    }
                    else {
                        ++it;
                    }
                }
                if (!dropped_observations.empty()) {
                    core::logger::log(core::logger::level::info, "Adjustment first pass: %zu of %zu observations beyond the bound leave the problem.", dropped_observations.size(), point_edges.size());
                }
                point_edges = kept_edges;
                number_of_accepted_rounds += ba.solve(rounds - map::first_pass_rounds, use_relative_convergence);
            }
            else {
                number_of_accepted_rounds = ba.solve(rounds, use_relative_convergence);
            }
            core::logger::log(core::logger::level::info, "Optimised: %f to %f error [frames: %d landmarks: %d lines: %d edges: %d] [%d/%d valid rounds]", initial_chi, ba.get_current_chi(), non_fixed_poses, non_fixed_landmarks, non_fixed_lines, non_fixed_edges, number_of_accepted_rounds, rounds);
            const optimisation::factor_graph::diagnostics& solver_diagnostics = ba.get_diagnostics();
            core::logger::log(core::logger::level::debug, "Solver: %s%s, %d landmark blocks, %d reduced solves, %d conjugate gradient iterations, %d unconverged, %d rejected attempts.", solver_diagnostics.used_square_root ? "square root" : "dense schur", solver_diagnostics.used_single_precision ? " single" : "", solver_diagnostics.landmark_blocks, solver_diagnostics.reduced_solves, solver_diagnostics.reduced_iterations, solver_diagnostics.reduced_failures, solver_diagnostics.rejected_attempts);
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
                if (landmark.inverse_depth) {
                    landmark.inverse_parameters = math::matrix<double, 3, 1>({ p[0], p[1], math::max(0.0, p[2]) });
                    landmark.update_location_from_inverse_depth();
                }
                else {
                    landmark.location[0] = p[0];
                    landmark.location[1] = p[1];
                    landmark.location[2] = p[2];
                }
            }
            if (!fix_landmarks) {
                std::unordered_map<int, math::matrix<double, 3, 3>> informations;
                for (const point_edge& entry : point_edges) {
                    if ((landmark_vertexes.count(entry.landmark_id) == 0) || landmark_vertexes.at(entry.landmark_id)->is_fixed()) {
                        continue;
                    }
                    entry.factor->compute_residual();
                    const double drho = entry.factor->robust_weight();
                    const math::matrix<double, 0, 0>& weight = entry.factor->get_information();
                    const math::matrix<double, 0, 0>& jacobian = entry.factor->get_jacobians()[1];
                    math::matrix<double, 3, 3>& information = informations[entry.landmark_id];
                    for (size_t a = 0; a < 3; ++a) {
                        for (size_t b = 0; b < 3; ++b) {
                            double sum = 0.0;
                            for (size_t r = 0; r < 2; ++r) {
                                for (size_t c = 0; c < 2; ++c) {
                                    sum += jacobian[r][a] * weight[r][c] * jacobian[c][b];
                                }
                            }
                            information[a][b] += drho * sum;
                        }
                    }
                }
                for (auto& [landmark_id, information] : informations) {
                    mapping::point& landmark = this->landmarks.at(landmark_id);
                    landmark.set_information(information, this->depth_direction_of(landmark));
                }
            }
            for (const auto& [landmark_id, vertex] : line_vertexes) {
                if (vertex->is_fixed()) {
                    continue;
                }
                mapping::line& landmark = this->line_landmarks.at(landmark_id);
                const double* const p = vertex->get_parameters();
                landmark.plucker_line.moment = math::matrix<double, 3, 1>({ p[0], p[1], p[2] });
                landmark.plucker_line.direction = math::matrix<double, 3, 1>({ p[3], p[4], p[5] });
                landmark.resynchronise_endpoints();
            }
            if (!fix_landmarks) {
                std::unordered_map<int, std::pair<size_t, size_t>> gross_after_by_frame;
                std::unordered_map<int, size_t> inliers_before_by_frame;
                for (const int landmark_id : active_landmark_ids) {
                    const std::unordered_map<int, mapping::point>::const_iterator landmark_it = this->landmarks.find(landmark_id);
                    const std::unordered_map<int, mapping::point>::const_iterator before_it = landmarks_before.find(landmark_id);
                    const std::unordered_map<int, std::vector<observation>>::const_iterator observations_it = this->observations.find(landmark_id);
                    if ((landmark_it == this->landmarks.end()) || (before_it == landmarks_before.end()) || (observations_it == this->observations.end())) {
                        continue;
                    }
                    for (const observation& obs : observations_it->second) {
                        if (active_frame_ids.count(obs.frame_id) == 0) {
                            continue;
                        }
                        const std::unordered_map<int, mapping::frame>::const_iterator frame_before_it = frames_before.find(obs.frame_id);
                        const std::unordered_map<int, mapping::frame>::const_iterator frame_it = this->frames.find(obs.frame_id);
                        if ((frame_before_it == frames_before.end()) || (frame_it == this->frames.end())) {
                            continue;
                        }
                        math::matrix<double, 2, 1> reprojected;
                        const double sigma_squared = frame_it->second.measurement_sigma * frame_it->second.measurement_sigma;
                        inliers_before_by_frame[obs.frame_id] += map::project_landmark(frame_before_it->second, before_it->second, reprojected) && ((reprojected - obs.point).get_length_squared() <= map::inlier_bound_squared * sigma_squared);
                        std::pair<size_t, size_t>& after = gross_after_by_frame[obs.frame_id];
                        ++after.second;
                        after.first += !map::project_landmark(frame_it->second, landmark_it->second, reprojected) || ((reprojected - obs.point).get_length_squared() > map::gross_error_squared * sigma_squared);
                    }
                }
                for (const auto& [frame_id, after] : gross_after_by_frame) {
                    const size_t inliers_before = inliers_before_by_frame[frame_id];
                    if ((after.second >= map::frame_divergence_minimum_observations) && (inliers_before * 2 >= after.second) && (after.first * 2 > after.second)) {
                        core::logger::log(core::logger::level::warn, "Frame %d diverged in the adjustment: %zu of %zu observations beyond %.0f px (%zu within the bound before); its pose before the solve is kept.", frame_id, after.first, after.second, math::sqrt(map::gross_error_squared), inliers_before);
                        this->frames.at(frame_id) = frames_before.at(frame_id);
                        this->corrected_frame_ids.push_back(frame_id);
                    }
                }
            }
            if (!fix_landmarks) {
                size_t examined = 0;
                size_t gross = 0;
                for (const int landmark_id : active_landmark_ids) {
                    const std::unordered_map<int, mapping::point>::const_iterator landmark_it = this->landmarks.find(landmark_id);
                    const std::unordered_map<int, std::vector<observation>>::const_iterator observations_it = this->observations.find(landmark_id);
                    if ((landmark_it == this->landmarks.end()) || (observations_it == this->observations.end())) {
                        continue;
                    }
                    for (const observation& obs : observations_it->second) {
                        const std::unordered_map<int, mapping::frame>::const_iterator frame_it = this->frames.find(obs.frame_id);
                        if ((frame_it == this->frames.end()) || (relevant_frame_ids.count(obs.frame_id) == 0)) {
                            continue;
                        }
                        math::matrix<double, 2, 1> reprojected;
                        ++examined;
                        gross += !map::project_landmark(frame_it->second, landmark_it->second, reprojected) || ((reprojected - obs.point).get_length_squared() > map::gross_error_squared * frame_it->second.measurement_sigma * frame_it->second.measurement_sigma);
                    }
                }
                if ((examined >= map::divergence_minimum_observations) && (gross * map::divergence_gross_fraction_denominator > examined)) {
                    core::logger::log(core::logger::level::warn, "Adjustment diverged: %zu of %zu observations beyond %.0f px; the state before it is kept.", gross, examined, math::sqrt(map::gross_error_squared));
                    for (const auto& [frame_id, frame] : frames_before) {
                        this->frames.at(frame_id) = frame;
                    }
                    for (const auto& [landmark_id, landmark] : landmarks_before) {
                        this->landmarks.at(landmark_id) = landmark;
                    }
                    for (const auto& [landmark_id, landmark] : lines_before) {
                        this->line_landmarks.at(landmark_id) = landmark;
                    }
                    return;
                }
            }
            if (!fix_landmarks) {
                for (const std::pair<int, int>& dropped : dropped_observations) {
                    const std::unordered_map<int, std::vector<observation>>::iterator observations_it = this->observations.find(dropped.first);
                    if (observations_it == this->observations.end()) {
                        continue;
                    }
                    std::vector<observation>& landmark_observations = observations_it->second;
                    for (size_t index = 0; index < landmark_observations.size(); ++index) {
                        if (landmark_observations[index].frame_id == dropped.second) {
                            landmark_observations.erase(landmark_observations.begin() + static_cast<std::ptrdiff_t>(index));
                            break;
                        }
                    }
                }
                this->remove_outlier_observations(active_landmark_ids, relevant_frame_ids);
            }
        }

        static bool project_landmark(const mapping::frame& frame, const mapping::point& landmark, math::matrix<double, 2, 1>& pixel) {
            math::matrix<double, 3, 1> mapped;
            if (landmark.inverse_depth) {
                const math::matrix<double, 3, 1> bearing({ landmark.inverse_parameters[0], landmark.inverse_parameters[1], 1.0 });
                mapped = (frame.rotation * (landmark.anchor_rotation * bearing)) + (((frame.rotation * landmark.anchor_translation) + frame.translation) * landmark.inverse_parameters[2]);
            }
            else {
                mapped = (frame.rotation * landmark.location) + frame.translation;
            }
            return (mapped[2] > 0.0) && frame.camera.project(mapped.data(), pixel.data());
        }

        size_t remove_outlier_observations(const std::unordered_set<int>& landmark_ids, const std::unordered_set<int>& frame_ids) {
            size_t removed = 0;
            size_t examined = 0;
            size_t beyond_5 = 0;
            size_t beyond_10 = 0;
            for (const int landmark_id : landmark_ids) {
                const std::unordered_map<int, mapping::point>::const_iterator landmark_it = this->landmarks.find(landmark_id);
                const std::unordered_map<int, std::vector<observation>>::iterator observations_it = this->observations.find(landmark_id);
                if ((landmark_it == this->landmarks.end()) || (observations_it == this->observations.end())) {
                    continue;
                }
                std::vector<observation>& landmark_observations = observations_it->second;
                size_t write = 0;
                for (size_t read = 0; read < landmark_observations.size(); ++read) {
                    const observation& obs = landmark_observations[read];
                    const std::unordered_map<int, mapping::frame>::const_iterator frame_it = this->frames.find(obs.frame_id);
                    bool keep = true;
                    if ((frame_it != this->frames.end()) && (frame_ids.count(obs.frame_id) != 0)) {
                        math::matrix<double, 2, 1> reprojected;
                        const double error_squared = map::project_landmark(frame_it->second, landmark_it->second, reprojected) ? ((reprojected - obs.point).get_length_squared() / (frame_it->second.measurement_sigma * frame_it->second.measurement_sigma)) : 1.0e300;
                        keep = error_squared <= map::inlier_bound_squared;
                        ++examined;
                        beyond_5 += (error_squared > 25.0);
                        beyond_10 += (error_squared > 100.0);
                    }
                    if (keep) {
                        landmark_observations[write++] = obs;
                    }
                    else {
                        ++removed;
                    }
                }
                landmark_observations.resize(write);
                if (landmark_observations.size() < 2) {
                    this->observations.erase(observations_it);
                    this->landmarks.erase(landmark_id);
                }
            }
            if (removed > 0) {
                core::logger::log(core::logger::level::info, "Removed %zu outlier observations after the adjustment (%zu examined, %zu beyond 5 px, %zu beyond 10 px).", removed, examined, beyond_5, beyond_10);
            }
            return removed;
        }

        static double line_reprojection_error(const mapping::frame& frame, const mapping::line& landmark, const line_observation& obs) {
            constexpr static const double failure = 5.991;
            const geometry::plucker line_camera = landmark.plucker_line.transformed(frame.rotation, frame.translation);
            const double line_norm = math::sqrt((line_camera.moment[0] * line_camera.moment[0]) + (line_camera.moment[1] * line_camera.moment[1]));
            if (line_norm < 1.0e-9) {
                return failure;
            }
            double camera_parameters[sensor::model::parameter_count];
            frame.camera.get_parameters(camera_parameters, sensor::model::parameter_count);
            const double focal = camera_parameters[0];
            const double pixels[2][2] = { { obs.x1, obs.y1 }, { obs.x2, obs.y2 } };
            double error = 0.0;
            for (int i = 0; i < 2; ++i) {
                double ray[3];
                if (!frame.camera.unproject(&pixels[i][0], &ray[0]) || !(ray[2] > 1.0e-12)) {
                    return failure;
                }
                const double alignment = (line_camera.moment[0] * ray[0] / ray[2]) + (line_camera.moment[1] * ray[1] / ray[2]) + line_camera.moment[2];
                error += math::abs(focal * alignment / line_norm);
            }
            return error / 2.0;
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
                if (outlier_bound_valid && !it->second.at_infinity()) {
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
                    math::matrix<double, 2, 1> reprojected;
                    if (!map::project_landmark(frame, it->second, reprojected)) {
                        reprojection_error += map::cull_mean_normalised_distance_maximum;
                    }
                    else {
                        reprojection_error += static_cast<float>(math::sqrt((measured - reprojected).get_length_squared()) / frame.measurement_sigma);
                    }
                    ++processed_observations;
                }
                if (processed_observations == 0) {
                    this->observations.erase(it->first);
                    it = this->landmarks.erase(it);
                    continue;
                }
                reprojection_error /= static_cast<float>(processed_observations);
                if (reprojection_error >= map::cull_mean_normalised_distance_maximum) {
                    this->observations.erase(it->first);
                    it = this->landmarks.erase(it);
                    continue;
                }
                ++it;
            }
            const size_t landmarks_after_cull = this->landmarks.size();

            const size_t lines_before_cull = this->line_landmarks.size();
            for (std::unordered_map<int, mapping::line>::iterator it = this->line_landmarks.begin(); it != this->line_landmarks.end();) {
                bool finite = true;
                for (size_t i = 0; i < 3; ++i) {
                    finite = finite && (math::abs(it->second.plucker_line.moment[i]) <= 1.0e12) && (math::abs(it->second.plucker_line.direction[i]) <= 1.0e12);
                }
                if (!finite) {
                    this->line_observations.erase(it->first);
                    it = this->line_landmarks.erase(it);
                    continue;
                }
                if ((it->first + 500) < this->next_line_landmark_id) {
                    ++it;
                    continue;
                }
                const std::unordered_map<int, std::vector<line_observation>>::const_iterator observations_it = this->line_observations.find(it->first);
                if ((observations_it == this->line_observations.end()) || observations_it->second.empty()) {
                    this->line_observations.erase(it->first);
                    it = this->line_landmarks.erase(it);
                    continue;
                }
                const std::vector<line_observation>& landmark_observations = observations_it->second;
                const bool not_seen_in_many_frames = landmark_observations.size() <= 4;
                const bool not_seen_recently = (landmark_observations.back().frame_id + 7) < this->next_frame_id;
                if (not_seen_in_many_frames && not_seen_recently) {
                    this->line_observations.erase(it->first);
                    it = this->line_landmarks.erase(it);
                    continue;
                }
                double reprojection_error = 0.0;
                size_t processed_observations = 0;
                for (const line_observation& obs : landmark_observations) {
                    const std::unordered_map<int, mapping::frame>::const_iterator frame_it = this->frames.find(obs.frame_id);
                    if (frame_it == this->frames.end()) {
                        continue;
                    }
                    reprojection_error += map::line_reprojection_error(frame_it->second, it->second, obs);
                    ++processed_observations;
                }
                if ((processed_observations == 0) || ((reprojection_error / static_cast<double>(processed_observations)) >= 5.991)) {
                    this->line_observations.erase(it->first);
                    it = this->line_landmarks.erase(it);
                    continue;
                }
                ++it;
            }
            core::logger::log(core::logger::level::info, "Culled: %zu points (%zu scale-outliers), %zu lines", landmarks_before_cull - landmarks_after_cull, outliers_culled, lines_before_cull - this->line_landmarks.size());
        }
    };
}

#endif // ZEROSLAM_MAPPING_MAP_HPP
