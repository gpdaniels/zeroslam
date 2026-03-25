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
#ifndef MAP_HPP
#define MAP_HPP

#include "factor_graph.hpp"
#include "frame.hpp"
#include "landmark.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace map {
    class map {
    public:
        template <class type_lhs, class type_rhs>
        class observation {
        public:
            type_lhs first;
            type_rhs second;
        };

    public:
        std::unordered_map<int, frame::frame> frames;
        std::unordered_map<int, landmark::point> landmarks;
        std::unordered_map<decltype(landmark::point::id), std::vector<observation<decltype(frame::frame::id), size_t>>> observations;

    public:
        void add_frame(const frame::frame& frame) {
            frames[frame.id] = frame;
        }

        void add_landmark(const landmark::point& landmark) {
            this->landmarks[landmark.id] = landmark;
        }

        void add_observation(const frame::frame& frame, const landmark::point& landmark, size_t kp_index) {
            this->observations[landmark.id].push_back({ frame.id, kp_index });
        }

        void optimise(int local_window, bool fix_landmarks, int rounds) {
            std::unordered_map<int, factor_graph::vertex_base*> camera_vertexes;
            std::unordered_map<int, factor_graph::vertex_base*> landmark_vertexes;
            std::unordered_map<int, factor_graph::edge_base*> observation_edges;
            // Get camera parameters.
            double camera_parameters[4];
            this->frames.begin()->second.camera.get_parameters(camera_parameters, 4);
            // Setup ba.
            factor_graph::loss_function_base* lossfunction = new factor_graph::loss_huber(math::sqrt(5.991));
            factor_graph::factor_graph ba(false);
            // Add frames.
            const int local_window_below = frame::frame::id_generator - 1 - local_window;
            const int local_window_fixed_below = local_window_below + 1;
            // Determine relevant frames and landmarks.
            std::unordered_set<int> relevant_frame_ids;
            std::unordered_set<int> active_landmark_ids;
            if (local_window > 0) {
                // Find non-fixed (active) frames.
                std::unordered_set<int> active_frame_ids;
                for (const auto& [frame_id, frame] : this->frames) {
                    const bool fixed = ((frame_id == 0) || (frame_id == 1 && this->frames.size() > 2) || (frame_id < local_window_fixed_below));
                    if (!fixed) {
                        active_frame_ids.insert(frame_id);
                    }
                }

                // Find landmarks seen by active frames and identify all frames that see them within a fixed horizon.
                const int current_frame_id = frame::frame::id_generator - 1;
                const int fixed_horizon_limit = 2 * local_window;
                const int horizon_threshold = current_frame_id - fixed_horizon_limit;

                for (const auto& [landmark_id, landmark_obs] : this->observations) {
                    bool seen_by_active = false;
                    for (const auto& obs : landmark_obs) {
                        if (active_frame_ids.count(obs.first)) {
                            seen_by_active = true;
                            break;
                        }
                    }
                    if (seen_by_active) {
                        active_landmark_ids.insert(static_cast<int>(landmark_id));
                        for (const auto& obs : landmark_obs) {
                            if (obs.first >= horizon_threshold) {
                                relevant_frame_ids.insert(obs.first);
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
                const lie::se3<double> v_se3(frame.rotation, frame.translation);
                const bool fixed = ((frame_id == 0) || (frame_id == 1 && this->frames.size() > 2) || ((local_window > 0) && (frame_id < local_window_fixed_below)));
                factor_graph::vertex_base* c = new factor_graph::vertex_pose();
                c->set_parameters(matrix::matrix<double, 0, 0>(7, 1, matrix::matrix<double, 7, 1>{ { v_se3.translation()[0], v_se3.translation()[1], v_se3.translation()[2], v_se3.rotation().get_quaternion()[1], v_se3.rotation().get_quaternion()[2], v_se3.rotation().get_quaternion()[3], v_se3.rotation().get_quaternion()[0] } }.data()));
                c->set_fixed(fixed);
                ba.add_vertex(c);
                camera_vertexes[frame_id] = c;
                non_fixed_poses += (fixed == false);
            }
            // Add landmarks.
            int non_fixed_landmarks = 0;
            int non_fixed_edges = 0;
            for (const auto& landmark_id : active_landmark_ids) {
                // Only add the landmark if it is in a frame. Initially assume it is not.
                bool landmark_added = false;
                // Add edges.
                for (const auto& [frame_id, kp_index] : this->observations.at(landmark_id)) {
                    // Only add the edge if it will do something.
                    if (camera_vertexes.count(frame_id) == 0) continue;
                    if (camera_vertexes.at(frame_id)->is_fixed() && fix_landmarks) {
                        continue;
                    }
                    // Given that we're adding at least one edge, make sure the landmark has been added.
                    if (!landmark_added) {
                        landmark_added = true;
                        factor_graph::vertex_base* l = new factor_graph::vertex_point_xyz();
                        l->set_parameters(matrix::matrix<double, 0, 0>(3, 1, this->landmarks.at(landmark_id).location.data()));
                        l->set_fixed(fix_landmarks);
                        l->set_marginalised(true);
                        ba.add_vertex(l);
                        landmark_vertexes[landmark_id] = l;
                        non_fixed_landmarks += (fix_landmarks == false);
                    }
                    // Now add the edge.
                    const frame::frame& frame = frames.at(frame_id);
                    camera::pinhole camera_model(camera_parameters, 4);
                    factor_graph::edge_base* m = new factor_graph::edge_reprojection<camera::pinhole>(camera_model);
                    m->set_observation(matrix::matrix<double, 0, 0>(2, 1, matrix::matrix<double, 2, 1>{ { static_cast<double>(frame.keypoint_pyramid[0][static_cast<size_t>(kp_index)].x), static_cast<double>(frame.keypoint_pyramid[0][static_cast<size_t>(kp_index)].y) } }.data()));
                    m->add_vertex(camera_vertexes[frame_id]);
                    m->add_vertex(landmark_vertexes[landmark_id]);
                    m->set_loss_function(lossfunction);
                    ba.add_edge(m);
                    observation_edges[static_cast<int>(observation_edges.size())] = m;
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
                std::printf("Optimised: No non fixed poses or landmarks [frames: %d landmarks: %d edges: %d]\n", non_fixed_poses, non_fixed_landmarks, non_fixed_edges);
                // Cleanup
                for (const auto& [frame_id, vertex] : camera_vertexes) {
                    static_cast<void>(frame_id);
                    delete vertex;
                }
                for (const auto& [landmark_id, vertex] : landmark_vertexes) {
                    static_cast<void>(landmark_id);
                    delete vertex;
                }
                for (const auto& [observation_id, edge] : observation_edges) {
                    static_cast<void>(observation_id);
                    delete edge;
                }
                delete lossfunction;
                // Exit.
                return;
            }
            if (camera_vertexes.empty() || landmark_vertexes.empty()) {
                std::printf("Optimised: Nothing to optimise [frames: %zu, landmarks: %zu]\n", camera_vertexes.size(), landmark_vertexes.size());
                // Cleanup
                for (const auto& [frame_id, vertex] : camera_vertexes) {
                    static_cast<void>(frame_id);
                    delete vertex;
                }
                for (const auto& [landmark_id, vertex] : landmark_vertexes) {
                    static_cast<void>(landmark_id);
                    delete vertex;
                }
                for (const auto& [observation_id, edge] : observation_edges) {
                    static_cast<void>(observation_id);
                    delete edge;
                }
                delete lossfunction;
                // Exit.
                return;
            }

            // Run the optimisation.
            double initial_chi = ba.get_current_chi();
            int number_of_accepted_rounds = ba.solve(rounds);
            std::printf("Optimised: %f to %f error [frames: %d landmarks: %d edges: %d] [%d/%d valid rounds]\n", initial_chi, ba.get_current_chi(), non_fixed_poses, non_fixed_landmarks, non_fixed_edges, number_of_accepted_rounds, rounds);
            // Apply optimised vertices to frames and landmarks.
            for (const auto& [frame_id, vertex] : camera_vertexes) {
                frame::frame& frame = frames.at(frame_id);
                matrix::matrix<double, 0, 0> p = vertex->get_parameters();
                const lie::se3 v_se3 = lie::se3(lie::so3(p[6][0], p[3][0], p[4][0], p[5][0]), { { p[0][0], p[1][0], p[2][0] } });
                frame.rotation = v_se3.rotation().get_matrix();
                frame.translation = v_se3.translation();
            }
            for (const auto& [landmark_id, vertex] : landmark_vertexes) {
                landmark::point& landmark = this->landmarks.at(landmark_id);
                matrix::matrix<double, 0, 0> p = vertex->get_parameters();
                landmark.location[0] = p[0][0];
                landmark.location[1] = p[1][0];
                landmark.location[2] = p[2][0];
            }
            // Cleanup
            for (const auto& [frame_id, vertex] : camera_vertexes) {
                static_cast<void>(frame_id);
                delete vertex;
            }
            for (const auto& [landmark_id, vertex] : landmark_vertexes) {
                static_cast<void>(landmark_id);
                delete vertex;
            }
            for (const auto& [observation_id, edge] : observation_edges) {
                static_cast<void>(observation_id);
                delete edge;
            }
            delete lossfunction;
        }

        void cull() {
            const size_t landmarks_before_cull = this->landmarks.size();
            for (std::unordered_map<int, landmark::point>::iterator it = this->landmarks.begin(); it != this->landmarks.end();) {
                if ((it->first + 500) < landmark::point::id_generator) {
                    ++it;
                    continue;
                }
                const std::vector<observation<int, size_t>>& landmark_observations = this->observations.at(it->first);
                const bool not_seen_in_many_frames = landmark_observations.size() <= 4;
                const bool not_seen_recently = landmark_observations.empty() || ((landmark_observations.back().first + 7) < frame::frame::id_generator);
                if (not_seen_in_many_frames && not_seen_recently) {
                    this->observations.erase(it->first);
                    it = this->landmarks.erase(it);
                    continue;
                }
                float reprojection_error = 0.0f;
                for (const auto& [frame_id, kp_index] : landmark_observations) {
                    const frame::frame& frame = this->frames.at(frame_id);
                    const matrix::matrix<double, 2, 1> measured = { { static_cast<double>(frame.keypoint_pyramid[0][static_cast<size_t>(kp_index)].x), static_cast<double>(frame.keypoint_pyramid[0][static_cast<size_t>(kp_index)].y) } };
                    const matrix::matrix<double, 3, 1> mapped = (frame.rotation * it->second.location) + frame.translation;
                    matrix::matrix<double, 2, 1> reprojected;
                    if (!frame.camera.project(mapped.data(), reprojected.data())) {
                        reprojection_error += 5.991f;
                    }
                    else {
                        reprojection_error += static_cast<float>(math::sqrt((measured - reprojected).get_length_squared()));
                    }
                }
                reprojection_error /= static_cast<float>(landmark_observations.size());
                if (reprojection_error >= 5.991f) {
                    this->observations.erase(it->first);
                    it = this->landmarks.erase(it);
                    continue;
                }
                ++it;
            }
            const size_t landmarks_after_cull = this->landmarks.size();
            std::printf("Culled: %zu points\n", landmarks_before_cull - landmarks_after_cull);
        }
    };
}

#endif // MAP_HPP