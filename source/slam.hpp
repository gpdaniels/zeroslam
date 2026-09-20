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
#ifndef ZEROSLAM_SLAM_HPP
#define ZEROSLAM_SLAM_HPP

#include "core/coordinates.hpp"
#include "core/logger.hpp"
#include "estimation/correspondence_2d_2d.hpp"
#include "estimation/correspondence_2d_3d.hpp"
#include "estimation/pose/essential.hpp"
#include "estimation/pose/homography.hpp"
#include "estimation/robust/solver/essential.hpp"
#include "estimation/robust/solver/homography.hpp"
#include "estimation/robust/solver/p3p.hpp"
#include "feature/angle/orb.hpp"
#include "feature/descriptor/binary.hpp"
#include "feature/descriptor/orb.hpp"
#include "feature/descriptor/sift.hpp"
#include "feature/detector/elsed.hpp"
#include "feature/point.hpp"
#include "feature/tracker/extrema.hpp"
#include "feature/tracker/line.hpp"
#include "feature/tracker/tracker.hpp"
#include "geometry/triangulation/linear_least_squares.hpp"
#include "image/image.hpp"
#include "mapping/covisibility.hpp"
#include "mapping/frame.hpp"
#include "mapping/line.hpp"
#include "mapping/loop_closure.hpp"
#include "mapping/map.hpp"
#include "mapping/point.hpp"
#include "match/distance/hamming.hpp"
#include "match/matcher/bruteforce.hpp"
#include "match/matcher/epipolar.hpp"
#include "match/matcher/gms.hpp"
#include "match/pair.hpp"
#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/line_reprojection.hpp"
#include "optimisation/edges/relative_similarity.hpp"
#include "optimisation/edges/reprojection.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/loss.hpp"
#include "optimisation/losses/huber.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/line.hpp"
#include "optimisation/vertices/point.hpp"
#include "optimisation/vertices/pose.hpp"
#include "optimisation/vertices/similarity.hpp"
#include "sensor/camera.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

class slam final {
public:
    enum class tracking_state {
        initialising,
        tracking
    };

    mapping::map reconstruction;
    mapping::frame::settings frontend;

    size_t reacquired_by_epipolar = 0;
    size_t admitted_at_infinity = 0;
    size_t admitted_unbounded = 0;
    size_t matched_by_projection = 0;
    size_t culled_keyframes = 0;

    struct verified_loop final {
        int keyframe_id;
        mapping::loop_closure::result loop;
        size_t fused;
    };

    std::vector<verified_loop> verified_loops;

    tracking_state state() const {
        return this->state_;
    }

    const mapping::covisibility& covisibility() const {
        return this->covisibility_;
    }

    const std::unordered_set<int>& keyframe_ids() const {
        return this->keyframe_ids_;
    }

    struct submap final {
        int start_id;
        int second_id;
        bool joined;
    };

    const std::vector<submap>& submaps() const {
        return this->submaps_;
    }

private:
    tracking_state state_ = tracking_state::initialising;
    int blind_frames_ = 0;
    static constexpr int blind_frames_tolerated = 10;
    std::vector<int> blind_frame_ids_;
    int blind_run_origin_id_ = -1;
    static constexpr size_t pose_inliers_minimum = 10;
    static constexpr size_t relocalise_inliers_minimum = 15;
    static constexpr size_t relocalise_candidates = 3;
    static constexpr double relocalise_distance_spacings = 4.0;
    static constexpr int relocalisation_probation_frames = 5;
    int relocalisation_probation_ = 0;
    std::vector<int> probation_frame_ids_;
    std::vector<std::pair<int, int>> probation_track_links_;
    static constexpr size_t blur_history_frames = 30;
    static constexpr double blur_sigma_maximum = 3.0;
    std::vector<double> laplacian_history_;
    int last_tracked_frame_id_ = 0;
    int previous_tracked_frame_id_ = -1;

    feature::tracker::tracker tracker_;
    feature::tracker::extrema extrema_tracker_;

    feature::tracker::line line_tracker_;

    mapping::loop_closure loop_closure_;
    mapping::covisibility covisibility_;

    bool close_loop(const int keyframe_id, const mapping::loop_closure::result& loop, std::vector<mapping::loop_closure::record>& records) {
        std::vector<int> ordered_keyframe_ids;
        for (const int id : this->keyframe_ids_) {
            if (this->reconstruction.frames.count(id) != 0) {
                ordered_keyframe_ids.push_back(id);
            }
        }
        std::sort(ordered_keyframe_ids.begin(), ordered_keyframe_ids.end());
        if ((ordered_keyframe_ids.size() < 2) || (this->reconstruction.frames.count(loop.keyframe_id) == 0) || (this->reconstruction.frames.count(keyframe_id) == 0)) {
            return false;
        }

        const auto similarity_parameters = [](const math::sim3<double>& similarity, double (&parameters)[8]) {
            parameters[0] = similarity.transformation().translation()[0];
            parameters[1] = similarity.transformation().translation()[1];
            parameters[2] = similarity.transformation().translation()[2];
            parameters[3] = similarity.transformation().rotation().get_quaternion()[1];
            parameters[4] = similarity.transformation().rotation().get_quaternion()[2];
            parameters[5] = similarity.transformation().rotation().get_quaternion()[3];
            parameters[6] = similarity.transformation().rotation().get_quaternion()[0];
            parameters[7] = similarity.scale();
        };

        const size_t current_submap = this->submap_of(keyframe_id);
        const bool joining = current_submap != this->submap_of(loop.keyframe_id);
        optimisation::factor_graph graph;
        std::unordered_map<int, optimisation::vertex*> vertices;
        std::unordered_map<int, math::se3<double>> previous_poses;
        std::unordered_map<int, math::sim3<double>> camera_to_world;
        for (const int id : ordered_keyframe_ids) {
            const mapping::frame& frame = this->reconstruction.frames.at(id);
            const math::se3<double> pose(frame.rotation, frame.translation);
            previous_poses[id] = pose;
            camera_to_world[id] = math::sim3<double>(pose.inverse(), 1.0);
            optimisation::vertex vertex{ optimisation::vertices::similarity() };
            double parameters[8];
            similarity_parameters(camera_to_world[id], parameters);
            vertex.set_parameters(&parameters[0], 8);
            const bool foreign_anchor = (this->reconstruction.anchor_frame_ids.count(id) != 0) && (this->submap_of(id) != current_submap);
            vertex.set_fixed((id == this->reconstruction.gauge_frame_id) || (id == loop.keyframe_id) || foreign_anchor);
            vertices[id] = graph.add_vertex(static_cast<optimisation::vertex&&>(vertex));
        }
        const auto add_edge = [&graph, &vertices, &similarity_parameters](const int first_id, const int second_id, const math::sim3<double>& measurement) {
            optimisation::edge edge{ optimisation::edges::relative_similarity() };
            double parameters[8];
            similarity_parameters(measurement, parameters);
            edge.set_observation(math::matrix<double, 0, 0>(8, 1, &parameters[0]));
            edge.add_vertex(vertices.at(first_id));
            edge.add_vertex(vertices.at(second_id));
            graph.add_edge(static_cast<optimisation::edge&&>(edge));
        };
        for (size_t i = 1; i < ordered_keyframe_ids.size(); ++i) {
            const int previous_id = ordered_keyframe_ids[i - 1];
            const int current_id = ordered_keyframe_ids[i];
            if (this->submap_of(previous_id) != this->submap_of(current_id)) {
                continue;
            }
            add_edge(previous_id, current_id, camera_to_world.at(previous_id).inverse() * camera_to_world.at(current_id));
        }
        add_edge(loop.keyframe_id, keyframe_id, loop.relative);
        for (const verified_loop& earlier : this->verified_loops) {
            if ((vertices.count(earlier.keyframe_id) != 0) && (vertices.count(earlier.loop.keyframe_id) != 0)) {
                add_edge(earlier.loop.keyframe_id, earlier.keyframe_id, camera_to_world.at(earlier.loop.keyframe_id).inverse() * camera_to_world.at(earlier.keyframe_id));
            }
        }

        if (graph.solve(50, true) == 0) {
            core::logger::log(core::logger::level::warn, "Loop %d -> %d not closed, the pose graph did not converge.", keyframe_id, loop.keyframe_id);
            return false;
        }

        std::unordered_map<int, math::sim3<double>> corrected;
        for (const auto& [id, vertex] : vertices) {
            const double* const p = vertex->get_parameters();
            corrected[id] = math::sim3<double>(math::se3<double>(math::so3<double>(p[6], p[3], p[4], p[5]), { { p[0], p[1], p[2] } }), p[7]);
        }
        for (const auto& [id, similarity] : corrected) {
            if (joining && (this->submap_of(id) == current_submap)) {
                continue;
            }
            if ((similarity.scale() > mapping::loop_closure::max_scale_ratio) || (similarity.scale() < 1.0 / mapping::loop_closure::max_scale_ratio)) {
                core::logger::log(core::logger::level::warn, "Loop %d -> %d not closed, the pose graph rescales keyframe %d by %.3f.", keyframe_id, loop.keyframe_id, id, similarity.scale());
                return false;
            }
        }

        size_t fused = 0;
        for (const mapping::loop_closure::correspondence& match : loop.matches) {
            if ((match.landmark_id == match.recorded_landmark_id) || !this->reconstruction.merge_landmark(match.recorded_landmark_id, match.landmark_id)) {
                continue;
            }
            ++fused;
            for (feature::tracker::tracker::track* const t : this->all_point_tracks()) {
                if (t->landmark_id == match.landmark_id) {
                    t->landmark_id = match.recorded_landmark_id;
                }
            }
            for (mapping::loop_closure::record& record : records) {
                if (record.landmark_id == match.landmark_id) {
                    record.landmark_id = match.recorded_landmark_id;
                }
            }
        }
        this->verified_loops.push_back({ keyframe_id, loop, fused });
        if (joining) {
            this->join_submap(keyframe_id);
        }

        for (const auto& [id, similarity] : corrected) {
            const math::se3<double> pose = similarity.transformation().inverse();
            mapping::frame& frame = this->reconstruction.frames.at(id);
            frame.rotation = pose.rotation().get_matrix();
            frame.translation = pose.translation();
        }
        for (const auto& [landmark_id, landmark_observations] : this->reconstruction.observations) {
            if (landmark_observations.empty()) {
                continue;
            }
            const std::unordered_map<int, math::se3<double>>::const_iterator previous = previous_poses.find(landmark_observations.front().frame_id);
            const std::unordered_map<int, math::sim3<double>>::const_iterator current = corrected.find(landmark_observations.front().frame_id);
            const std::unordered_map<int, mapping::point>::iterator landmark = this->reconstruction.landmarks.find(landmark_id);
            if ((previous == previous_poses.end()) || (current == corrected.end()) || (landmark == this->reconstruction.landmarks.end())) {
                continue;
            }
            landmark->second.location = current->second.transformation() * (previous->second * landmark->second.location);
            if (landmark->second.inverse_depth) {
                landmark->second.anchor_rotation = current->second.transformation().rotation().get_matrix() * (previous->second.rotation().get_matrix() * landmark->second.anchor_rotation);
                landmark->second.anchor_translation = current->second.transformation() * (previous->second * landmark->second.anchor_translation);
                landmark->second.update_location_from_inverse_depth();
            }
        }
        for (const auto& [landmark_id, landmark_observations] : this->reconstruction.line_observations) {
            if (landmark_observations.empty()) {
                continue;
            }
            const std::unordered_map<int, math::se3<double>>::const_iterator previous = previous_poses.find(landmark_observations.front().frame_id);
            const std::unordered_map<int, math::sim3<double>>::const_iterator current = corrected.find(landmark_observations.front().frame_id);
            const std::unordered_map<int, mapping::line>::iterator landmark = this->reconstruction.line_landmarks.find(landmark_id);
            if ((previous == previous_poses.end()) || (current == corrected.end()) || (landmark == this->reconstruction.line_landmarks.end())) {
                continue;
            }
            landmark->second.locations[0] = current->second.transformation() * (previous->second * landmark->second.locations[0]);
            landmark->second.locations[1] = current->second.transformation() * (previous->second * landmark->second.locations[1]);
            geometry::plucker corrected_line;
            if (geometry::plucker::from_points(landmark->second.locations[0], landmark->second.locations[1], corrected_line)) {
                landmark->second.plucker_line = corrected_line;
            }
        }
        this->update_anchored_frames();
        this->reconstruction.optimise(0, false, 50, true);
        this->reanchor_corrected();
        this->update_anchored_frames();
        this->refit_anchored_frames();
        core::logger::log(core::logger::level::note, "Loop %d -> %d closed over %zu keyframes, %zu landmarks fused, scale at the keyframe %.4f.", keyframe_id, loop.keyframe_id, ordered_keyframe_ids.size(), fused, corrected.at(keyframe_id).scale());
        return true;
    }

    int triangulate_lines(const mapping::frame& frame_current) {
        constexpr static const double minimum_plane_angle_sine = 0.017452406437283512;
        constexpr static const double reprojection_maximum = 2.0;
        int created = 0;
        std::vector<feature::tracker::line::track*> candidates = this->line_tracker_.active_tracks();
        std::stable_sort(candidates.begin(), candidates.end(), [](const feature::tracker::line::track* const lhs, const feature::tracker::line::track* const rhs) {
            const float lhs_length = ((lhs->x2 - lhs->x1) * (lhs->x2 - lhs->x1)) + ((lhs->y2 - lhs->y1) * (lhs->y2 - lhs->y1));
            const float rhs_length = ((rhs->x2 - rhs->x1) * (rhs->x2 - rhs->x1)) + ((rhs->y2 - rhs->y1) * (rhs->y2 - rhs->y1));
            return (lhs_length != rhs_length) ? (lhs_length > rhs_length) : (lhs->id < rhs->id);
        });
        for (feature::tracker::line::track* const t : candidates) {
            if (static_cast<size_t>(created) >= slam::line_landmarks_per_keyframe_maximum) {
                break;
            }
            if ((t->landmark_id != -1) || (t->length < 2)) {
                continue;
            }
            const feature::tracker::line::observation* ref_obs = nullptr;
            for (const feature::tracker::line::observation& obs : t->history) {
                if ((obs.frame_id != frame_current.id) && (this->keyframe_ids_.count(obs.frame_id) != 0) && (this->reconstruction.frames.count(obs.frame_id) != 0)) {
                    ref_obs = &obs;
                    break;
                }
            }
            if (ref_obs == nullptr) {
                continue;
            }
            const mapping::frame& frame_ref = this->reconstruction.frames.at(ref_obs->frame_id);
            const double cur_pixels[2][2] = { { static_cast<double>(t->x1), static_cast<double>(t->y1) }, { static_cast<double>(t->x2), static_cast<double>(t->y2) } };
            const double ref_pixels[2][2] = { { static_cast<double>(ref_obs->x1), static_cast<double>(ref_obs->y1) }, { static_cast<double>(ref_obs->x2), static_cast<double>(ref_obs->y2) } };
            math::matrix<double, 3, 1> ray_cur[2];
            math::matrix<double, 3, 1> ray_ref[2];
            bool unprojected = true;
            for (int i = 0; i < 2; ++i) {
                unprojected = unprojected && frame_current.camera.unproject(&cur_pixels[i][0], ray_cur[i].data());
                unprojected = unprojected && frame_ref.camera.unproject(&ref_pixels[i][0], ray_ref[i].data());
            }
            if (!unprojected) {
                continue;
            }
            const math::matrix<double, 3, 1> normal_camera_cur = geometry::plucker::cross(ray_cur[0], ray_cur[1]);
            const math::matrix<double, 3, 1> normal_camera_ref = geometry::plucker::cross(ray_ref[0], ray_ref[1]);
            const math::matrix<double, 3, 1> normal_world_cur = math::transpose(frame_current.rotation) * normal_camera_cur;
            const math::matrix<double, 3, 1> normal_world_ref = math::transpose(frame_ref.rotation) * normal_camera_ref;
            const double offset_cur = geometry::plucker::dot(normal_camera_cur, frame_current.translation);
            const double offset_ref = geometry::plucker::dot(normal_camera_ref, frame_ref.translation);
            geometry::plucker line_world;
            if (!geometry::plucker::from_planes(normal_world_cur, offset_cur, normal_world_ref, offset_ref, minimum_plane_angle_sine, line_world)) {
                continue;
            }
            const math::matrix<double, 3, 1> centre_cur = -(math::transpose(frame_current.rotation) * frame_current.translation);
            const math::matrix<double, 3, 1> centre_ref = -(math::transpose(frame_ref.rotation) * frame_ref.translation);
            math::matrix<double, 3, 1> endpoints[2];
            if (!line_world.closest_point_to_ray(centre_cur, math::transpose(frame_current.rotation) * ray_cur[0], endpoints[0]) || !line_world.closest_point_to_ray(centre_cur, math::transpose(frame_current.rotation) * ray_cur[1], endpoints[1])) {
                continue;
            }
            if ((endpoints[1] - endpoints[0]).get_length_squared() < 1.0e-12) {
                continue;
            }
            {
                const math::matrix<double, 3, 1> end_cur[2] = { (frame_current.rotation * endpoints[0]) + frame_current.translation, (frame_current.rotation * endpoints[1]) + frame_current.translation };
                const math::matrix<double, 3, 1> end_ref[2] = { (frame_ref.rotation * endpoints[0]) + frame_ref.translation, (frame_ref.rotation * endpoints[1]) + frame_ref.translation };
                if (!mapping::map::line_observation_usable(line_world.transformed(frame_current.rotation, frame_current.translation), end_cur[0], end_cur[1], this->frontend.line_angle) || !mapping::map::line_observation_usable(line_world.transformed(frame_ref.rotation, frame_ref.translation), end_ref[0], end_ref[1], this->frontend.line_angle)) {
                    continue;
                }
            }
            const math::matrix<double, 3, 1> midpoint = (endpoints[0] + endpoints[1]) * 0.5;
            if (parallax_angle(midpoint, centre_cur, centre_ref) * (180.0 / 3.14159265358979323846) < triangulation_parallax_degrees_minimum) {
                continue;
            }
            const auto perpendicular = [](const double* const pixel, const double (&segment)[2][2]) {
                const double dx = segment[1][0] - segment[0][0];
                const double dy = segment[1][1] - segment[0][1];
                const double length_squared = (dx * dx) + (dy * dy);
                if (length_squared < 1.0e-12) {
                    return 1.0e300;
                }
                const double cross = ((pixel[0] - segment[0][0]) * dy) - ((pixel[1] - segment[0][1]) * dx);
                return math::abs(cross) / math::sqrt(length_squared);
            };
            bool consistent = true;
            for (int i = 0; (i < 2) && consistent; ++i) {
                const math::matrix<double, 3, 1> mapped_cur = (frame_current.rotation * endpoints[i]) + frame_current.translation;
                const math::matrix<double, 3, 1> mapped_ref = (frame_ref.rotation * endpoints[i]) + frame_ref.translation;
                double reprojected_cur[2];
                double reprojected_ref[2];
                consistent = (mapped_cur[2] > 0.0) && (mapped_ref[2] > 0.0);
                consistent = consistent && frame_current.camera.project(mapped_cur.data(), &reprojected_cur[0]) && frame_ref.camera.project(mapped_ref.data(), &reprojected_ref[0]);
                consistent = consistent && (perpendicular(&reprojected_cur[0], cur_pixels) <= reprojection_maximum) && (perpendicular(&reprojected_ref[0], ref_pixels) <= reprojection_maximum);
            }
            if (!consistent) {
                continue;
            }
            const mapping::line landmark(this->reconstruction.allocate_line_landmark_id(), line_world, endpoints[0], endpoints[1]);
            this->reconstruction.add_line_landmark(landmark);
            t->landmark_id = landmark.id;
            for (const feature::tracker::line::observation& obs : t->history) {
                if (this->keyframe_ids_.count(obs.frame_id) != 0) {
                    this->reconstruction.add_line_observation(obs.frame_id, landmark, static_cast<double>(obs.x1), static_cast<double>(obs.y1), static_cast<double>(obs.x2), static_cast<double>(obs.y2));
                }
            }
            ++created;
        }
        return created;
    }

    struct pose_correspondence final {
        double pixel[2];
        math::matrix<double, 3, 1> location;
        feature::tracker::tracker::track* source;
        const mapping::point* landmark;
    };

    struct line_correspondence final {
        double pixel[2][2];
        geometry::plucker line;
    };

    static bool prior_sees(const mapping::frame& frame, const double (&intrinsics)[4], const math::matrix<double, 3, 1>& location) {
        const double image_columns = static_cast<double>(frame.image_pyramid[0].get_cols());
        const double image_rows = static_cast<double>(frame.image_pyramid[0].get_rows());
        const math::matrix<double, 3, 1> seen = (frame.rotation * location) + frame.translation;
        if (seen[2] <= 1.0e-6) {
            return false;
        }
        const double pinhole_x = (intrinsics[0] * seen[0] / seen[2]) + intrinsics[2];
        const double pinhole_y = (intrinsics[1] * seen[1] / seen[2]) + intrinsics[3];
        return (pinhole_x >= -0.5 * image_columns) && (pinhole_x <= 1.5 * image_columns) && (pinhole_y >= -0.5 * image_rows) && (pinhole_y <= 1.5 * image_rows);
    }

    bool refine_pose_from_prior(mapping::frame& frame_current, size_t& inliers) {
        inliers = 0;

        std::vector<pose_correspondence> correspondences;
        double intrinsics[4] = { 1.0, 1.0, 0.0, 0.0 };
        frame_current.camera.get_parameters(&intrinsics[0], 4);
        for (feature::tracker::tracker::track* const t : this->active_point_tracks()) {
            if ((t->landmark_id < 0) || (this->reconstruction.landmarks.count(t->landmark_id) == 0)) {
                continue;
            }
            const mapping::point& landmark = this->reconstruction.landmarks.at(t->landmark_id);
            if (!slam::prior_sees(frame_current, intrinsics, landmark.location)) {
                continue;
            }
            correspondences.push_back({ { static_cast<double>(t->x), static_cast<double>(t->y) }, landmark.location, t, &landmark });
        }

        std::vector<line_correspondence> line_correspondences;
        if (this->frontend.lines && this->frontend.line_pose) {
            for (feature::tracker::line::track* const t : this->line_tracker_.active_tracks()) {
                if ((t->landmark_id < 0) || (this->reconstruction.line_landmarks.count(t->landmark_id) == 0)) {
                    continue;
                }
                const mapping::line& line_record = this->reconstruction.line_landmarks.at(t->landmark_id);
                const geometry::plucker& world_line = line_record.plucker_line;
                if (!mapping::map::line_observation_usable(world_line.transformed(frame_current.rotation, frame_current.translation), (frame_current.rotation * line_record.locations[0]) + frame_current.translation, (frame_current.rotation * line_record.locations[1]) + frame_current.translation, this->frontend.line_angle)) {
                    continue;
                }
                line_correspondences.push_back({ { { static_cast<double>(t->x1), static_cast<double>(t->y1) }, { static_cast<double>(t->x2), static_cast<double>(t->y2) } }, world_line });
            }
        }
        std::vector<unsigned char> agreed;
        if (!this->refine_pose(frame_current, correspondences, line_correspondences, inliers, agreed)) {
            return false;
        }
        if (this->frontend.pose_outlier_limit > 0) {
            size_t unlinked = 0;
            for (size_t c = 0; c < correspondences.size(); ++c) {
                feature::tracker::tracker::track* const source = correspondences[c].source;
                if (agreed[c]) {
                    source->outliers = 0;
                    source->outlier_frame_id = frame_current.id;
                    continue;
                }
                if (source->outlier_frame_id == frame_current.id) {
                    continue;
                }
                source->outlier_frame_id = frame_current.id;
                source->outliers += 1;
                if (source->outliers > this->frontend.pose_outlier_limit) {
                    size_t keep = 0;
                    for (size_t h = 0; h < source->history.size(); ++h) {
                        if (source->history[h].frame_id >= frame_current.id) {
                            source->history[keep] = source->history[h];
                            ++keep;
                        }
                    }
                    source->history.resize(keep);
                    source->landmark_id = -1;
                    source->outliers = 0;
                    ++unlinked;
                }
            }
            if (unlinked > 0) {
                core::logger::log(core::logger::level::debug, "Inliers: %zu tracks unlinked from their landmarks after %d outlier frames.", unlinked, this->frontend.pose_outlier_limit);
            }
        }
        return true;
    }

    bool refine_pose(mapping::frame& frame_current, const std::vector<pose_correspondence>& correspondences, const std::vector<line_correspondence>& line_correspondences, size_t& inliers, std::vector<unsigned char>& agreed) {
        inliers = 0;
        agreed.assign(correspondences.size(), static_cast<unsigned char>(0));
        if (correspondences.size() + line_correspondences.size() < slam::pose_inliers_minimum) {
            return false;
        }
        double intrinsics[4] = { 1.0, 1.0, 0.0, 0.0 };
        frame_current.camera.get_parameters(&intrinsics[0], 4);
        const math::se3<double> prior(frame_current.rotation, frame_current.translation);
        const optimisation::loss lossfunction(optimisation::losses::huber(math::sqrt(5.991)));
        optimisation::factor_graph ba;
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        const double pose_parameters[7] = { prior.translation()[0], prior.translation()[1], prior.translation()[2], prior.rotation().get_quaternion()[1], prior.rotation().get_quaternion()[2], prior.rotation().get_quaternion()[3], prior.rotation().get_quaternion()[0] };
        pose_vertex.set_parameters(&pose_parameters[0], 7);
        optimisation::vertex* const pose_node = ba.add_vertex(static_cast<optimisation::vertex&&>(pose_vertex));
        const sensor::camera::model<double> camera_model(frame_current.camera);
        std::vector<optimisation::edge*> weighted_edges(correspondences.size(), nullptr);
        for (size_t c = 0; c < correspondences.size(); ++c) {
            const pose_correspondence& match = correspondences[c];
            const bool weighted = (match.landmark != nullptr) && (match.landmark->uncertainty == mapping::point::uncertainty_kind::unbounded);
            optimisation::vertex point_vertex{ optimisation::vertices::point() };
            point_vertex.set_parameters((weighted && match.landmark->inverse_depth) ? match.landmark->inverse_parameters.data() : match.location.data(), 3);
            point_vertex.set_fixed(true);
            point_vertex.set_marginalised(true);
            optimisation::vertex* const point_node = ba.add_vertex(static_cast<optimisation::vertex&&>(point_vertex));
            if (weighted) {
                const mapping::map::observation recorded{ frame_current.id, static_cast<size_t>(-1), math::matrix<double, 2, 1>({ match.pixel[0], match.pixel[1] }), 0 };
                optimisation::edge factor = mapping::map::make_point_edge(frame_current.camera, *match.landmark, recorded, lossfunction, 1.0);
                factor.add_vertex(pose_node);
                factor.add_vertex(point_node);
                factor.compute_jacobians();
                factor.set_information(math::matrix<double, 0, 0>(2, 2, match.landmark->observation_information(factor.get_jacobians()[1], frame_current.measurement_sigma).data()));
                weighted_edges[c] = ba.add_edge(static_cast<optimisation::edge&&>(factor));
                continue;
            }
            optimisation::edge factor{ optimisation::edges::reprojection(camera_model) };
            factor.set_observation(math::matrix<double, 0, 0>(2, 1, &match.pixel[0]));
            factor.add_vertex(pose_node);
            factor.add_vertex(point_node);
            factor.set_loss(lossfunction);
            ba.add_edge(static_cast<optimisation::edge&&>(factor));
        }
        for (const line_correspondence& match : line_correspondences) {
            optimisation::edges::line_reprojection line_factor{ camera_model };
            if (!line_factor.set_measured_segment(match.pixel[0][0], match.pixel[0][1], match.pixel[1][0], match.pixel[1][1])) {
                continue;
            }
            optimisation::vertex line_vertex{ optimisation::vertices::line() };
            const double line_parameters[6] = { match.line.moment[0], match.line.moment[1], match.line.moment[2], match.line.direction[0], match.line.direction[1], match.line.direction[2] };
            line_vertex.set_parameters(&line_parameters[0], 6);
            line_vertex.set_fixed(true);
            line_vertex.set_marginalised(true);
            optimisation::vertex* const line_node = ba.add_vertex(static_cast<optimisation::vertex&&>(line_vertex));
            optimisation::edge factor{ line_factor };
            factor.add_vertex(pose_node);
            factor.add_vertex(line_node);
            factor.set_loss(lossfunction);
            ba.add_edge(static_cast<optimisation::edge&&>(factor));
        }
        ba.solve(20, true);
        const double* const p = pose_node->get_parameters();
        const math::se3<double> refined(math::so3<double>(p[6], p[3], p[4], p[5]), { { p[0], p[1], p[2] } });
        const math::matrix<double, 3, 3> rotation = refined.rotation().get_matrix();
        size_t bounded = 0;
        for (const pose_correspondence& match : correspondences) {
            bounded += ((match.landmark == nullptr) || (match.landmark->uncertainty != mapping::point::uncertainty_kind::unbounded)) ? 1 : 0;
        }
        const bool translation_observed = (bounded >= slam::pose_bounded_minimum) || !line_correspondences.empty();
        math::matrix<double, 3, 1> translation = refined.translation();
        if (!translation_observed) {
            const math::matrix<double, 3, 1> centre = -(math::transpose(prior.rotation().get_matrix()) * prior.translation());
            translation = -(rotation * centre);
        }
        {
            const double final_parameters[7] = { translation[0], translation[1], translation[2], refined.rotation().get_quaternion()[1], refined.rotation().get_quaternion()[2], refined.rotation().get_quaternion()[3], refined.rotation().get_quaternion()[0] };
            pose_node->set_parameters(&final_parameters[0], 7);
        }
        for (size_t c = 0; c < correspondences.size(); ++c) {
            const pose_correspondence& match = correspondences[c];
            if (weighted_edges[c] != nullptr) {
                optimisation::edge& factor = *weighted_edges[c];
                factor.compute_jacobians();
                factor.set_information(math::matrix<double, 0, 0>(2, 2, match.landmark->observation_information(factor.get_jacobians()[1], frame_current.measurement_sigma).data()));
                factor.compute_residual();
                if (factor.chi2() <= 5.991) {
                    agreed[c] = static_cast<unsigned char>(1);
                    inliers += 1;
                }
                continue;
            }
            const math::matrix<double, 3, 1> mapped = (rotation * match.location) + translation;
            double projected[2];
            if ((mapped[2] <= 0.0) || !frame_current.camera.project(mapped.data(), &projected[0])) {
                continue;
            }
            const double dx = projected[0] - match.pixel[0];
            const double dy = projected[1] - match.pixel[1];
            if (((dx * dx) + (dy * dy)) <= 5.991 * frame_current.measurement_sigma * frame_current.measurement_sigma) {
                agreed[c] = static_cast<unsigned char>(1);
                inliers += 1;
            }
        }
        size_t line_inliers = 0;
        if (!line_correspondences.empty()) {
            const double focal = intrinsics[0];
            for (const line_correspondence& match : line_correspondences) {
                const geometry::plucker line_camera = match.line.transformed(rotation, translation);
                const math::matrix<double, 3, 1>& image_line = line_camera.moment;
                const double norm = math::sqrt((image_line[0] * image_line[0]) + (image_line[1] * image_line[1]));
                if (norm < 1.0e-9) {
                    continue;
                }
                bool agrees = true;
                for (int end = 0; end < 2; ++end) {
                    double ray[3] = { 0.0, 0.0, 0.0 };
                    if (!frame_current.camera.unproject(&match.pixel[end][0], &ray[0], nullptr) || (ray[2] <= 0.0)) {
                        agrees = false;
                        break;
                    }
                    const double distance = focal * ((image_line[0] * (ray[0] / ray[2])) + (image_line[1] * (ray[1] / ray[2])) + image_line[2]) / norm;
                    if ((distance * distance) > 5.991 * frame_current.measurement_sigma * frame_current.measurement_sigma) {
                        agrees = false;
                        break;
                    }
                }
                line_inliers += agrees ? 1 : 0;
            }
        }
        if (!line_correspondences.empty()) {
            core::logger::log(core::logger::level::debug, "Pose refinement (frame %d): %zu of %zu points and %zu of %zu lines agree.", frame_current.id, inliers, correspondences.size(), line_inliers, line_correspondences.size());
        }
        if ((inliers + line_inliers < slam::pose_inliers_minimum) || (2 * inliers < correspondences.size())) {
            return false;
        }
        inliers += line_inliers;
        frame_current.rotation = rotation;
        frame_current.translation = translation;
        return true;
    }

    static constexpr size_t pose_bounded_minimum = 20;

    bool pose_from_tracks(mapping::frame& frame_current, size_t& correspondences) {
        std::vector<estimation::correspondence_2d_3d<double>> pnp_correspondencies;
        {
            const std::vector<feature::tracker::tracker::track*> active = this->active_point_tracks();
            size_t bounded = 0;
            for (const feature::tracker::tracker::track* const t : active) {
                if ((t->landmark_id >= 0) && (this->reconstruction.landmarks.count(t->landmark_id) != 0) && (this->reconstruction.landmarks.at(t->landmark_id).uncertainty != mapping::point::uncertainty_kind::unbounded)) {
                    ++bounded;
                }
            }
            if (bounded < slam::pose_bounded_minimum) {
                correspondences = bounded;
                core::logger::log(core::logger::level::debug, "Pose from tracks: %zu landmarks with a depth, too few for a perspective solve.", bounded);
                return false;
            }
            for (feature::tracker::tracker::track* const t : active) {
                if (t->landmark_id < 0) {
                    continue;
                }
                if (this->reconstruction.landmarks.count(t->landmark_id) == 0) {
                    continue;
                }
                if (this->reconstruction.landmarks.at(t->landmark_id).uncertainty == mapping::point::uncertainty_kind::unbounded) {
                    continue;
                }
                const double lhs_point[2] = { static_cast<double>(t->x), static_cast<double>(t->y) };
                double lhs_ray[3];
                if (!frame_current.camera.unproject(&lhs_point[0], &lhs_ray[0])) {
                    continue;
                }
                estimation::correspondence_2d_3d<double> corr;
                corr.lhs[0] = lhs_ray[0] / lhs_ray[2];
                corr.lhs[1] = lhs_ray[1] / lhs_ray[2];
                const mapping::point& landmark = this->reconstruction.landmarks.at(t->landmark_id);
                corr.rhs[0] = landmark.location[0];
                corr.rhs[1] = landmark.location[1];
                corr.rhs[2] = landmark.location[2];
                pnp_correspondencies.push_back(corr);
            }
        }
        correspondences = pnp_correspondencies.size();
        if (pnp_correspondencies.size() < 3) {
            return false;
        }
        estimation::robust::estimate::p3p<double>::model model;
        std::vector<float> pnp_residuals(pnp_correspondencies.size());
        std::vector<size_t> pnp_inliers(pnp_correspondencies.size());
        size_t inliers_size = 0;
        if (!estimation::robust::solver::p3p<double>::solve(pnp_correspondencies.data(), pnp_correspondencies.size(), pnp_residuals.data(), pnp_inliers.data(), inliers_size, model) || (inliers_size < slam::pose_inliers_minimum)) {
            return false;
        }
        frame_current.rotation = math::matrix<double, 3, 3>({ { model.rotation[0][0], model.rotation[0][1], model.rotation[0][2] },
                                                              { model.rotation[1][0], model.rotation[1][1], model.rotation[1][2] },
                                                              { model.rotation[2][0], model.rotation[2][1], model.rotation[2][2] } });
        frame_current.translation = math::matrix<double, 3, 1>({ { model.translation[0] },
                                                                 { model.translation[1] },
                                                                 { model.translation[2] } });
        const math::matrix<double, 3, 3> consensus_rotation = frame_current.rotation;
        const math::matrix<double, 3, 1> consensus_translation = frame_current.translation;
        size_t refined_inliers = 0;
        if (!this->refine_pose_from_prior(frame_current, refined_inliers)) {
            frame_current.rotation = consensus_rotation;
            frame_current.translation = consensus_translation;
        }
        core::logger::log(core::logger::level::info, "Inliers: %zu inliers in PnP pose estimation out of %zu correspondencies, %zu after refinement.", inliers_size, pnp_correspondencies.size(), refined_inliers);
        return true;
    }

    bool relocalise(mapping::frame& frame_current) {
        if (frame_current.descriptors.empty()) {
            return false;
        }
        const std::vector<int> recalled = this->loop_closure_.recall(frame_current.descriptors.data(), frame_current.descriptors.size(), slam::relocalise_candidates);
        for (const int keyframe_id : recalled) {
            size_t records_size = 0;
            const mapping::loop_closure::record* const records = this->loop_closure_.records_of(keyframe_id, records_size);
            if ((records == nullptr) || (records_size < slam::relocalise_inliers_minimum)) {
                continue;
            }
            std::vector<feature::descriptor::binary<256>> record_descriptors(records_size);
            for (size_t i = 0; i < records_size; ++i) {
                record_descriptors[i] = records[i].descriptor;
            }
            std::vector<match::pair> forward(2 * frame_current.descriptors.size());
            const size_t forward_count = match::matcher::bruteforce::find_matches(frame_current.descriptors.data(), frame_current.descriptors.size(), record_descriptors.data(), record_descriptors.size(), mapping::loop_closure::match_hamming_maximum * this->frontend.descriptor_distance_scale(), 2, forward.data(), forward.size());
            std::vector<match::pair> backward(records_size);
            const size_t backward_count = match::matcher::bruteforce::find_matches(record_descriptors.data(), record_descriptors.size(), frame_current.descriptors.data(), frame_current.descriptors.size(), mapping::loop_closure::match_hamming_maximum * this->frontend.descriptor_distance_scale(), 1, backward.data(), backward.size());
            std::vector<size_t> best_keypoint_of_record(records_size, static_cast<size_t>(-1));
            for (size_t i = 0; i < backward_count; ++i) {
                best_keypoint_of_record[backward[i].lhs_index] = backward[i].rhs_index;
            }
            std::vector<match::pair> descriptor_pairs;
            for (size_t i = 0; i < forward_count; ++i) {
                const match::pair& best = forward[i];
                const bool has_second = (i + 1 < forward_count) && (forward[i + 1].lhs_index == best.lhs_index);
                if (has_second) {
                    ++i;
                    if (!(best.score < mapping::loop_closure::match_ratio * forward[i].score)) {
                        continue;
                    }
                }
                if (best_keypoint_of_record[best.rhs_index] != best.lhs_index) {
                    continue;
                }
                descriptor_pairs.push_back(best);
            }
            if (descriptor_pairs.size() >= mapping::loop_closure::gms_minimum_pairs) {
                std::vector<feature::point> record_points(records_size);
                float record_extent[2] = { 1.0f, 1.0f };
                for (size_t i = 0; i < records_size; ++i) {
                    record_points[i] = feature::point{ records[i].pixel_x, records[i].pixel_y, 0.0f, 0.0f, 0 };
                    record_extent[0] = math::max(record_extent[0], records[i].pixel_x + 1.0f);
                    record_extent[1] = math::max(record_extent[1], records[i].pixel_y + 1.0f);
                }
                const size_t before = descriptor_pairs.size();
                descriptor_pairs.resize(match::matcher::gms::filter(frame_current.keypoints.data(), static_cast<float>(frame_current.image_pyramid[0].get_cols()), static_cast<float>(frame_current.image_pyramid[0].get_rows()), record_points.data(), record_extent[0], record_extent[1], descriptor_pairs.data(), descriptor_pairs.size()));
                core::logger::log(core::logger::level::debug, "Relocalisation of frame %d against keyframe %d: %zu of %zu descriptor pairs pass the motion statistics.", frame_current.id, keyframe_id, descriptor_pairs.size(), before);
            }
            std::vector<estimation::correspondence_2d_3d<double>> correspondences;
            std::vector<size_t> keypoint_of_correspondence;
            std::vector<int> landmark_of_correspondence;
            for (const match::pair& best : descriptor_pairs) {
                const std::unordered_map<int, mapping::point>::const_iterator landmark = this->reconstruction.landmarks.find(records[best.rhs_index].landmark_id);
                if (landmark == this->reconstruction.landmarks.end()) {
                    continue;
                }
                const feature::point& keypoint = frame_current.keypoints[best.lhs_index];
                const double pixel[2] = { static_cast<double>(keypoint.x), static_cast<double>(keypoint.y) };
                double ray[3];
                if (!frame_current.camera.unproject(&pixel[0], &ray[0]) || !(ray[2] > 1.0e-12)) {
                    continue;
                }
                estimation::correspondence_2d_3d<double> correspondence;
                correspondence.lhs[0] = ray[0] / ray[2];
                correspondence.lhs[1] = ray[1] / ray[2];
                correspondence.rhs = landmark->second.location;
                correspondences.push_back(correspondence);
                keypoint_of_correspondence.push_back(best.lhs_index);
                landmark_of_correspondence.push_back(landmark->first);
            }
            if (correspondences.size() < slam::relocalise_inliers_minimum) {
                core::logger::log(core::logger::level::debug, "Relocalisation of frame %d against keyframe %d: %zu pairs, too few.", frame_current.id, keyframe_id, correspondences.size());
                continue;
            }
            estimation::robust::estimate::p3p<double>::model model;
            std::vector<float> residuals(correspondences.size());
            std::vector<size_t> inliers(correspondences.size());
            size_t inliers_size = 0;
            if (!estimation::robust::solver::p3p<double>::solve(correspondences.data(), correspondences.size(), residuals.data(), inliers.data(), inliers_size, model) || (inliers_size < slam::relocalise_inliers_minimum)) {
                core::logger::log(core::logger::level::debug, "Relocalisation of frame %d against keyframe %d: %zu inliers of %zu pairs, rejected.", frame_current.id, keyframe_id, inliers_size, correspondences.size());
                continue;
            }
            const math::matrix<double, 3, 3> rotation({ { model.rotation[0][0], model.rotation[0][1], model.rotation[0][2] }, { model.rotation[1][0], model.rotation[1][1], model.rotation[1][2] }, { model.rotation[2][0], model.rotation[2][1], model.rotation[2][2] } });
            const math::matrix<double, 3, 1> translation({ { model.translation[0] }, { model.translation[1] }, { model.translation[2] } });
            const std::unordered_map<int, mapping::frame>::const_iterator recalled_frame = this->reconstruction.frames.find(keyframe_id);
            if (recalled_frame != this->reconstruction.frames.end()) {
                const math::matrix<double, 3, 1> centre = -(math::transpose(rotation) * translation);
                const math::matrix<double, 3, 1> recalled_centre = -(math::transpose(recalled_frame->second.rotation) * recalled_frame->second.translation);
                const double distance = math::sqrt((centre - recalled_centre).get_length_squared());
                const double spacing = this->median_keyframe_spacing();
                if ((spacing > 0.0) && (distance > slam::relocalise_distance_spacings * spacing)) {
                    core::logger::log(core::logger::level::debug, "Relocalisation of frame %d against keyframe %d: %zu inliers but %.3f from the keyframe (%.3f spacing), an alias.", frame_current.id, keyframe_id, inliers_size, distance, spacing);
                    continue;
                }
            }
            frame_current.rotation = rotation;
            frame_current.translation = translation;
            std::unordered_set<int> linked;
            for (feature::tracker::tracker::track* const t : this->all_point_tracks()) {
                if (t->landmark_id >= 0) {
                    linked.insert(t->landmark_id);
                }
            }
            size_t relinked = 0;
            const std::vector<feature::tracker::tracker::track*> active = this->active_point_tracks();
            for (size_t i = 0; i < inliers_size; ++i) {
                const feature::point& keypoint = frame_current.keypoints[keypoint_of_correspondence[inliers[i]]];
                const int landmark_id = landmark_of_correspondence[inliers[i]];
                if (linked.count(landmark_id) != 0) {
                    continue;
                }
                for (feature::tracker::tracker::track* const t : active) {
                    if ((t->landmark_id < 0) && (math::abs(t->x - keypoint.x) < 0.5f) && (math::abs(t->y - keypoint.y) < 0.5f)) {
                        t->landmark_id = landmark_id;
                        this->probation_track_links_.push_back({ t->id, landmark_id });
                        linked.insert(landmark_id);
                        ++relinked;
                        break;
                    }
                }
            }
            const math::matrix<double, 3, 1> relocalised_centre = -(math::transpose(rotation) * translation);
            core::logger::log(core::logger::level::note, "Relocalised frame %d against keyframe %d: %zu inliers of %zu pairs, %zu tracks relinked, centre (%.3f, %.3f, %.3f).", frame_current.id, keyframe_id, inliers_size, correspondences.size(), relinked, relocalised_centre[0], relocalised_centre[1], relocalised_centre[2]);
            return true;
        }
        return false;
    }

    double median_keyframe_spacing() const {
        std::vector<int> ordered;
        for (const int id : this->keyframe_ids_) {
            if (this->reconstruction.frames.count(id) != 0) {
                ordered.push_back(id);
            }
        }
        std::sort(ordered.begin(), ordered.end());
        std::vector<double> spacings;
        for (size_t i = 1; i < ordered.size(); ++i) {
            const mapping::frame& previous = this->reconstruction.frames.at(ordered[i - 1]);
            const mapping::frame& current = this->reconstruction.frames.at(ordered[i]);
            const math::matrix<double, 3, 1> previous_centre = -(math::transpose(previous.rotation) * previous.translation);
            const math::matrix<double, 3, 1> current_centre = -(math::transpose(current.rotation) * current.translation);
            spacings.push_back(math::sqrt((current_centre - previous_centre).get_length_squared()));
        }
        if (spacings.empty()) {
            return 0.0;
        }
        std::sort(spacings.begin(), spacings.end());
        return spacings[spacings.size() / 2];
    }

    int init_anchor_frame_id_ = 0;

    std::vector<submap> submaps_;
    bool reinitialising_ = false;
    int lost_frames_ = 0;
    static constexpr int reinitialise_after_frames = 5;
    static constexpr size_t reinitialise_tracks_minimum = 50;

    size_t submap_of(const int frame_id) const {
        size_t index = 0;
        while ((index < this->submaps_.size()) && (this->submaps_[index].start_id <= frame_id)) {
            ++index;
        }
        return index;
    }

    int submap_start_of(const int frame_id) const {
        const size_t index = this->submap_of(frame_id);
        return (index == 0) ? 0 : this->submaps_[index - 1].start_id;
    }

    void begin_submap(mapping::frame& frame_current) {
        const std::unordered_map<int, mapping::frame>::const_iterator last = this->reconstruction.frames.find(this->last_tracked_frame_id_);
        if (last != this->reconstruction.frames.end()) {
            frame_current.rotation = last->second.rotation;
            frame_current.translation = last->second.translation;
        }
        this->init_anchor_frame_id_ = frame_current.id;
        this->reinitialising_ = true;
        this->last_tracked_frame_id_ = frame_current.id;
        this->previous_tracked_frame_id_ = -1;
        core::logger::log(core::logger::level::note, "Lost for %d frames: frame %d anchors a new submap, initialising again.", this->lost_frames_, frame_current.id);
    }

    void start_submap(const mapping::frame& frame_current) {
        this->submaps_.push_back({ this->init_anchor_frame_id_, frame_current.id, false });
        this->reconstruction.anchor_frame_ids.insert(this->init_anchor_frame_id_);
        this->reconstruction.anchor_frame_ids.insert(frame_current.id);
        for (feature::tracker::tracker::track* const t : this->all_point_tracks()) {
            t->landmark_id = -1;
            t->outliers = 0;
            size_t keep = 0;
            for (size_t h = 0; h < t->history.size(); ++h) {
                if (t->history[h].frame_id >= this->init_anchor_frame_id_) {
                    t->history[keep] = t->history[h];
                    ++keep;
                }
            }
            t->history.resize(keep);
        }
        for (feature::tracker::line::track* const t : this->line_tracker_.all_tracks()) {
            t->landmark_id = -1;
        }
        this->scale_sentinel_depths_.clear();
        this->reinitialising_ = false;
        this->lost_frames_ = 0;
        core::logger::log(core::logger::level::note, "Submap %zu started at keyframes %d and %d.", this->submaps_.size(), this->init_anchor_frame_id_, frame_current.id);
    }

    void join_submap(const int keyframe_id) {
        const size_t index = this->submap_of(keyframe_id);
        if ((index == 0) || this->submaps_[index - 1].joined) {
            return;
        }
        submap& joined = this->submaps_[index - 1];
        joined.joined = true;
        this->reconstruction.anchor_frame_ids.erase(joined.start_id);
        this->reconstruction.anchor_frame_ids.erase(joined.second_id);
        core::logger::log(core::logger::level::note, "Submap %zu joined to the map through keyframe %d.", index, keyframe_id);
    }

    std::vector<feature::tracker::tracker::track*> active_point_tracks() {
        if (this->frontend.tracker == mapping::frame::settings::tracker_kind::extrema) {
            return this->extrema_tracker_.active_tracks();
        }
        return this->tracker_.active_tracks();
    }

    std::vector<feature::tracker::tracker::track*> all_point_tracks() {
        if (this->frontend.tracker == mapping::frame::settings::tracker_kind::extrema) {
            return this->extrema_tracker_.all_tracks();
        }
        return this->tracker_.all_tracks();
    }

    static constexpr double triangulation_parallax_degrees_minimum = 1.0;

    static constexpr size_t line_segments_maximum = 300;
    static constexpr size_t line_landmarks_per_keyframe_maximum = 40;

    std::unordered_set<int> keyframe_ids_;
    int last_keyframe_id_ = 0;

    struct anchored_observation final {
        int landmark_id;
        float pixel[2];
    };

    struct pose_anchor final {
        int reference_keyframe_id;
        math::se3<double> frame_from_keyframe;
        std::vector<anchored_observation> observations;
    };

    std::unordered_map<int, pose_anchor> non_keyframe_anchors_;

    constexpr static const size_t scale_sentinel_history = 10;
    std::vector<double> scale_sentinel_depths_;

    void report_scale_sentinel(const int frame_id, std::vector<double> depths) {
        if (depths.size() < 5) {
            return;
        }
        std::sort(depths.begin(), depths.end());
        const double median = depths[depths.size() / 2];
        double reference = median;
        if (!this->scale_sentinel_depths_.empty()) {
            std::vector<double> history(this->scale_sentinel_depths_);
            std::sort(history.begin(), history.end());
            reference = history[history.size() / 2];
        }
        core::logger::log(core::logger::level::info, "Scale sentinel keyframe %d: median new landmark depth %.4f map units over %zu, %.3f of the recent median.", frame_id, median, depths.size(), (reference > 0.0) ? (median / reference) : 0.0);
        this->scale_sentinel_depths_.push_back(median);
        if (this->scale_sentinel_depths_.size() > slam::scale_sentinel_history) {
            this->scale_sentinel_depths_.erase(this->scale_sentinel_depths_.begin());
        }
    }

    void anchor_non_keyframe(mapping::frame& frame_current) {
        const std::unordered_map<int, mapping::frame>::const_iterator keyframe_it = this->reconstruction.frames.find(this->last_keyframe_id_);
        if (keyframe_it == this->reconstruction.frames.end()) {
            return;
        }
        const math::se3<double> frame_pose(frame_current.rotation, frame_current.translation);
        const math::se3<double> keyframe_pose(keyframe_it->second.rotation, keyframe_it->second.translation);
        pose_anchor anchor{ this->last_keyframe_id_, frame_pose * keyframe_pose.inverse(), {} };
        for (const feature::tracker::tracker::track* const t : this->active_point_tracks()) {
            if ((t->landmark_id >= 0) && (this->reconstruction.landmarks.count(t->landmark_id) != 0)) {
                anchor.observations.push_back({ t->landmark_id, { static_cast<float>(t->x), static_cast<float>(t->y) } });
            }
        }
        this->non_keyframe_anchors_[frame_current.id] = static_cast<pose_anchor&&>(anchor);
    }

    static void update_anchored_frames_in(mapping::map& reconstruction, const std::unordered_map<int, pose_anchor>& anchors) {
        for (const auto& [frame_id, anchor] : anchors) {
            const std::unordered_map<int, mapping::frame>::iterator frame_it = reconstruction.frames.find(frame_id);
            const std::unordered_map<int, mapping::frame>::const_iterator keyframe_it = reconstruction.frames.find(anchor.reference_keyframe_id);
            if ((frame_it == reconstruction.frames.end()) || (keyframe_it == reconstruction.frames.end())) {
                continue;
            }
            const math::se3<double> keyframe_pose(keyframe_it->second.rotation, keyframe_it->second.translation);
            const math::se3<double> frame_pose = anchor.frame_from_keyframe * keyframe_pose;
            frame_it->second.rotation = frame_pose.rotation().get_matrix();
            frame_it->second.translation = frame_pose.translation();
        }
    }

    void update_anchored_frames() {
        slam::update_anchored_frames_in(this->reconstruction, this->non_keyframe_anchors_);
    }

    static constexpr double refit_agreement_fraction = 0.9;

    void refit_anchored_frames() {
        std::vector<int> frame_ids;
        frame_ids.reserve(this->non_keyframe_anchors_.size());
        for (const auto& [frame_id, anchor] : this->non_keyframe_anchors_) {
            static_cast<void>(anchor);
            frame_ids.push_back(frame_id);
        }
        std::sort(frame_ids.begin(), frame_ids.end());
        size_t refitted = 0;
        size_t agreed_already = 0;
        std::vector<pose_correspondence> correspondences;
        std::vector<unsigned char> agreed;
        for (const int frame_id : frame_ids) {
            pose_anchor& anchor = this->non_keyframe_anchors_.at(frame_id);
            const std::unordered_map<int, mapping::frame>::iterator frame_it = this->reconstruction.frames.find(frame_id);
            const std::unordered_map<int, mapping::frame>::const_iterator keyframe_it = this->reconstruction.frames.find(anchor.reference_keyframe_id);
            if ((frame_it == this->reconstruction.frames.end()) || (keyframe_it == this->reconstruction.frames.end()) || anchor.observations.empty()) {
                continue;
            }
            mapping::frame& frame = frame_it->second;
            double intrinsics[4] = { 1.0, 1.0, 0.0, 0.0 };
            frame.camera.get_parameters(&intrinsics[0], 4);
            correspondences.clear();
            for (const anchored_observation& recorded : anchor.observations) {
                const std::unordered_map<decltype(mapping::point::id), mapping::point>::const_iterator landmark_it = this->reconstruction.landmarks.find(recorded.landmark_id);
                if ((landmark_it == this->reconstruction.landmarks.end()) || !slam::prior_sees(frame, intrinsics, landmark_it->second.location)) {
                    continue;
                }
                correspondences.push_back({ { static_cast<double>(recorded.pixel[0]), static_cast<double>(recorded.pixel[1]) }, landmark_it->second.location, nullptr, &landmark_it->second });
            }
            if (correspondences.size() < slam::pose_inliers_minimum) {
                continue;
            }
            size_t agreeing = 0;
            for (const pose_correspondence& match : correspondences) {
                const math::matrix<double, 3, 1> mapped = (frame.rotation * match.location) + frame.translation;
                double projected[2];
                if ((mapped[2] <= 0.0) || !frame.camera.project(mapped.data(), &projected[0])) {
                    continue;
                }
                const double dx = projected[0] - match.pixel[0];
                const double dy = projected[1] - match.pixel[1];
                agreeing += (((dx * dx) + (dy * dy)) <= 5.991 * frame.measurement_sigma * frame.measurement_sigma) ? 1 : 0;
            }
            if (static_cast<double>(agreeing) >= slam::refit_agreement_fraction * static_cast<double>(correspondences.size())) {
                ++agreed_already;
                continue;
            }
            size_t inliers = 0;
            if (!this->refine_pose(frame, correspondences, std::vector<line_correspondence>(), inliers, agreed)) {
                continue;
            }
            const math::se3<double> frame_pose(frame.rotation, frame.translation);
            const math::se3<double> keyframe_pose(keyframe_it->second.rotation, keyframe_it->second.translation);
            anchor.frame_from_keyframe = frame_pose * keyframe_pose.inverse();
            ++refitted;
        }
        core::logger::log(core::logger::level::info, "Refitted %zu of %zu anchored frames to the adjusted map, %zu already agreed with it.", refitted, frame_ids.size(), agreed_already);
    }

    void reanchor_corrected() {
        for (const int keyframe_id : this->reconstruction.corrected_frame_ids) {
            const std::unordered_map<int, mapping::frame>::const_iterator keyframe_it = this->reconstruction.frames.find(keyframe_id);
            if (keyframe_it == this->reconstruction.frames.end()) {
                continue;
            }
            const math::se3<double> keyframe_pose(keyframe_it->second.rotation, keyframe_it->second.translation);
            for (auto& [frame_id, anchor] : this->non_keyframe_anchors_) {
                if (anchor.reference_keyframe_id != keyframe_id) {
                    continue;
                }
                const std::unordered_map<int, mapping::frame>::const_iterator frame_it = this->reconstruction.frames.find(frame_id);
                if (frame_it == this->reconstruction.frames.end()) {
                    continue;
                }
                const math::se3<double> frame_pose(frame_it->second.rotation, frame_it->second.translation);
                anchor.frame_from_keyframe = frame_pose * keyframe_pose.inverse();
            }
        }
    }

    static constexpr int keyframe_interval = 5;
    static constexpr int keyframe_min_interval = 2;
    static constexpr double keyframe_tracked_ratio = 0.5;

    static constexpr int local_map_covisibility_minimum = 15;
    static constexpr double local_map_projection_radius = 8.0;
    static constexpr double local_map_border = 8.0;
    static constexpr int local_map_octave_window = 1;
    static constexpr float local_map_match_ratio = 0.8f;
    static constexpr unsigned int local_map_match_hamming = 64;
    static constexpr unsigned int local_map_match_hamming_both = 50;
    static constexpr double local_map_relink_distance = 2.0;
    static constexpr double local_map_recovery_radius = 20.0;
    std::vector<int> local_landmark_ids_;

    static constexpr size_t keyframe_cull_other_observers = 3;
    static constexpr double keyframe_cull_redundant_fraction = 0.9;

    static constexpr int reacquire_window_frames = 50;

    unsigned int hamming_bound(const unsigned int orb_bound) const {
        return static_cast<unsigned int>(static_cast<float>(orb_bound) * this->frontend.descriptor_distance_scale() + 0.5f);
    }

    static constexpr unsigned int reacquire_hamming_maximum = 50;

    void add_landmark_descriptors(mapping::point& landmark, const mapping::frame& frame, const float x, const float y, const int octave, const feature::descriptor::binary<256>& track_descriptor) {
        landmark.add_descriptor(&track_descriptor.data[0]);
        if (!this->frontend.affine || (this->frontend.descriptor != mapping::frame::settings::descriptor_kind::bsift) || (octave < 0) || (static_cast<size_t>(octave) >= frame.image_pyramid.size())) {
            return;
        }
        const image::image& level = frame.image_pyramid[static_cast<size_t>(octave)];
        const int columns = static_cast<int>(level.get_cols());
        const int rows = static_cast<int>(level.get_rows());
        const int level_x = core::to_pixel_index(core::to_pixel_centre(core::to_pixel_index_position(x) / frame.image_pyramid.scale_x(static_cast<size_t>(octave))));
        const int level_y = core::to_pixel_index(core::to_pixel_centre(core::to_pixel_index_position(y) / frame.image_pyramid.scale_y(static_cast<size_t>(octave))));
        constexpr static const int border = feature::descriptor::sift::radius + 2;
        if ((level_x < border) || (level_y < border) || (level_x >= columns - border) || (level_y >= rows - border)) {
            return;
        }
        const unsigned char* const data = level.get_data() + (static_cast<size_t>(level_y) * level.get_cols()) + static_cast<size_t>(level_x);
        const float angle = feature::angle::orb::dominant_angle(data, columns);
        const float tilts[2] = { 1.41421356f, 2.0f };
        for (const float tilt : tilts) {
            for (int direction = 0; direction < 4; ++direction) {
                const float phi = static_cast<float>(direction) * 0.78539816f;
                const float c = math::cos(phi);
                const float s = math::sin(phi);
                const float affine[4] = { (c * c) + ((s * s) / tilt), -(c * s) * (1.0f - (1.0f / tilt)), -(c * s) * (1.0f - (1.0f / tilt)), (s * s) + ((c * c) / tilt) };
                float vector[feature::descriptor::sift::dimensions];
                feature::descriptor::sift::describe_float(data, columns, angle, &affine[0], vector);
                feature::descriptor::binary<256> tilted;
                feature::descriptor::sift::binarise(vector, tilted);
                landmark.add_descriptor(&tilted.data[0]);
            }
        }
    }

    static constexpr double reacquire_epipolar_tolerance = 3.0;

    bool holds_loop(const int frame_id) const {
        for (const verified_loop& loop : this->verified_loops) {
            if ((loop.keyframe_id == frame_id) || (loop.loop.keyframe_id == frame_id)) {
                return true;
            }
        }
        return false;
    }

    void demote_keyframe(const int candidate_id) {
        this->keyframe_ids_.erase(candidate_id);
        this->loop_closure_.remove_keyframe(candidate_id);
        int reference_id = -1;
        for (const int keyframe_id : this->keyframe_ids_) {
            if (this->reconstruction.frames.count(keyframe_id) == 0) {
                continue;
            }
            const bool better = (reference_id < 0) || ((keyframe_id < candidate_id) && ((reference_id > candidate_id) || (keyframe_id > reference_id))) || ((keyframe_id > candidate_id) && (reference_id > candidate_id) && (keyframe_id < reference_id));
            if (better) {
                reference_id = keyframe_id;
            }
        }
        if (reference_id < 0) {
            ++this->culled_keyframes;
            return;
        }
        const mapping::frame& culled = this->reconstruction.frames.at(candidate_id);
        const mapping::frame& reference = this->reconstruction.frames.at(reference_id);
        const math::se3<double> culled_pose(culled.rotation, culled.translation);
        const math::se3<double> reference_pose(reference.rotation, reference.translation);
        const math::se3<double> culled_from_reference = culled_pose * reference_pose.inverse();
        for (auto& [frame_id, anchor] : this->non_keyframe_anchors_) {
            static_cast<void>(frame_id);
            if (anchor.reference_keyframe_id == candidate_id) {
                anchor.reference_keyframe_id = reference_id;
                anchor.frame_from_keyframe = anchor.frame_from_keyframe * culled_from_reference;
            }
        }
        this->non_keyframe_anchors_[candidate_id] = pose_anchor{ reference_id, culled_from_reference, {} };
        ++this->culled_keyframes;
    }

    void cull_keyframes(const int current_keyframe_id, const int recent_frames) {
        std::unordered_map<int, int> shared_with_current;
        for (const auto& [landmark_id, landmark_observations] : this->reconstruction.observations) {
            static_cast<void>(landmark_id);
            bool observed_by_current = false;
            for (const mapping::map::observation& obs : landmark_observations) {
                observed_by_current = observed_by_current || (obs.frame_id == current_keyframe_id);
            }
            if (!observed_by_current) {
                continue;
            }
            for (const mapping::map::observation& obs : landmark_observations) {
                if (obs.frame_id != current_keyframe_id) {
                    ++shared_with_current[obs.frame_id];
                }
            }
        }
        std::vector<int> candidates;
        for (const auto& [frame_id, shared] : shared_with_current) {
            if ((shared >= slam::local_map_covisibility_minimum) && (frame_id != this->reconstruction.gauge_frame_id) && (frame_id != this->init_anchor_frame_id_) && (this->reconstruction.anchor_frame_ids.count(frame_id) == 0) && (this->keyframe_ids_.count(frame_id) != 0) && (this->reconstruction.frames.count(frame_id) != 0) && (frame_id < current_keyframe_id - recent_frames) && !this->holds_loop(frame_id)) {
                candidates.push_back(frame_id);
            }
        }
        std::sort(candidates.begin(), candidates.end());
        for (const int candidate_id : candidates) {
            size_t total = 0;
            size_t redundant = 0;
            for (const auto& [landmark_id, landmark_observations] : this->reconstruction.observations) {
                static_cast<void>(landmark_id);
                bool observed = false;
                size_t other_keyframes = 0;
                for (const mapping::map::observation& obs : landmark_observations) {
                    if (obs.frame_id == candidate_id) {
                        observed = true;
                    }
                    else if ((this->keyframe_ids_.count(obs.frame_id) != 0) && (this->reconstruction.frames.count(obs.frame_id) != 0)) {
                        ++other_keyframes;
                    }
                }
                if (observed) {
                    ++total;
                    redundant += (other_keyframes >= slam::keyframe_cull_other_observers);
                }
            }
            if (static_cast<double>(redundant) < slam::keyframe_cull_redundant_fraction * static_cast<double>(total)) {
                continue;
            }
            if (total == 0) {
                for (std::unordered_map<int, pose_anchor>::iterator it = this->non_keyframe_anchors_.begin(); it != this->non_keyframe_anchors_.end();) {
                    if (it->second.reference_keyframe_id == candidate_id) {
                        this->reconstruction.frames.erase(it->first);
                        it = this->non_keyframe_anchors_.erase(it);
                    }
                    else {
                        ++it;
                    }
                }
                this->reconstruction.frames.erase(candidate_id);
                this->keyframe_ids_.erase(candidate_id);
                this->loop_closure_.remove_keyframe(candidate_id);
                ++this->culled_keyframes;
                core::logger::log(core::logger::level::warn, "Dropped keyframe %d and the frames anchored to it: no observation supports its pose.", candidate_id);
                continue;
            }
            std::vector<anchored_observation> kept_observations;
            for (std::unordered_map<int, std::vector<mapping::map::observation>>::iterator it = this->reconstruction.observations.begin(); it != this->reconstruction.observations.end();) {
                std::vector<mapping::map::observation>& landmark_observations = it->second;
                size_t write = 0;
                for (size_t read = 0; read < landmark_observations.size(); ++read) {
                    if (landmark_observations[read].frame_id != candidate_id) {
                        landmark_observations[write++] = landmark_observations[read];
                    }
                    else {
                        kept_observations.push_back({ it->first, { static_cast<float>(landmark_observations[read].point[0]), static_cast<float>(landmark_observations[read].point[1]) } });
                    }
                }
                landmark_observations.resize(write);
                if (write < 2) {
                    this->reconstruction.landmarks.erase(it->first);
                    it = this->reconstruction.observations.erase(it);
                }
                else {
                    ++it;
                }
            }
            for (auto& [line_id, line_observations] : this->reconstruction.line_observations) {
                static_cast<void>(line_id);
                size_t write = 0;
                for (size_t read = 0; read < line_observations.size(); ++read) {
                    if (line_observations[read].frame_id != candidate_id) {
                        line_observations[write++] = line_observations[read];
                    }
                }
                line_observations.resize(write);
            }
            this->demote_keyframe(candidate_id);
            const std::unordered_map<int, pose_anchor>::iterator culled_anchor = this->non_keyframe_anchors_.find(candidate_id);
            if (culled_anchor != this->non_keyframe_anchors_.end()) {
                std::sort(kept_observations.begin(), kept_observations.end(), [](const anchored_observation& lhs, const anchored_observation& rhs) {
                    return lhs.landmark_id < rhs.landmark_id;
                });
                culled_anchor->second.observations = static_cast<std::vector<anchored_observation>&&>(kept_observations);
            }
            core::logger::log(core::logger::level::info, "Culled keyframe %d: %zu of %zu landmarks observed by %zu other keyframes or more.", candidate_id, redundant, total, slam::keyframe_cull_other_observers);
        }

        std::unordered_map<int, size_t> observation_counts;
        for (const auto& [landmark_id, landmark_observations] : this->reconstruction.observations) {
            static_cast<void>(landmark_id);
            for (const mapping::map::observation& obs : landmark_observations) {
                ++observation_counts[obs.frame_id];
            }
        }
        std::vector<int> unobserved;
        for (const int keyframe_id : this->keyframe_ids_) {
            if ((keyframe_id != this->reconstruction.gauge_frame_id) && (keyframe_id != this->init_anchor_frame_id_) && (this->reconstruction.anchor_frame_ids.count(keyframe_id) == 0) && (keyframe_id != current_keyframe_id) && (this->reconstruction.frames.count(keyframe_id) != 0) && !this->holds_loop(keyframe_id) && (observation_counts.count(keyframe_id) == 0)) {
                unobserved.push_back(keyframe_id);
            }
        }
        std::sort(unobserved.begin(), unobserved.end());
        for (const int keyframe_id : unobserved) {
            this->demote_keyframe(keyframe_id);
            core::logger::log(core::logger::level::info, "Demoted keyframe %d: no observation is left to constrain its pose.", keyframe_id);
        }
    }

    void rebuild_local_map(const int keyframe_id) {
        std::unordered_set<int> local_keyframe_ids;
        local_keyframe_ids.insert(keyframe_id);
        for (const int neighbour_id : this->covisibility_.neighbours(keyframe_id, slam::local_map_covisibility_minimum)) {
            local_keyframe_ids.insert(neighbour_id);
        }
        this->local_landmark_ids_.clear();
        for (const auto& [landmark_id, landmark_observations] : this->reconstruction.observations) {
            if (this->reconstruction.landmarks.count(landmark_id) == 0) {
                continue;
            }
            for (const mapping::map::observation& obs : landmark_observations) {
                if (local_keyframe_ids.count(obs.frame_id) != 0) {
                    this->local_landmark_ids_.push_back(landmark_id);
                    break;
                }
            }
        }
        std::sort(this->local_landmark_ids_.begin(), this->local_landmark_ids_.end());
    }

    size_t track_local_map(mapping::frame& frame_current, const double radius = slam::local_map_projection_radius, const bool refine_pose = true) {
        if (this->local_landmark_ids_.empty()) {
            return 0;
        }
        const std::vector<feature::tracker::tracker::track*> active = this->active_point_tracks();
        std::unordered_set<int> linked_landmark_ids;
        for (const feature::tracker::tracker::track* const t : this->all_point_tracks()) {
            if (t->landmark_id >= 0) {
                linked_landmark_ids.insert(t->landmark_id);
            }
        }
        const auto cell_key = [radius](const double x, const double y) {
            const long long cell_x = static_cast<long long>(math::floor(x / radius));
            const long long cell_y = static_cast<long long>(math::floor(y / radius));
            return (cell_x * 1000003LL) + cell_y;
        };
        std::unordered_map<long long, std::vector<size_t>> track_grid;
        for (size_t index = 0; index < active.size(); ++index) {
            if (active[index]->landmark_id < 0) {
                track_grid[cell_key(static_cast<double>(active[index]->x), static_cast<double>(active[index]->y))].push_back(index);
            }
        }
        const bool match_detections = (this->frontend.association != mapping::frame::settings::association_kind::klt) && (this->frontend.tracker == mapping::frame::settings::tracker_kind::klt) && (frame_current.keypoints.size() == frame_current.descriptors.size());
        std::unordered_map<long long, std::vector<size_t>> detection_grid;
        if (match_detections) {
            for (size_t index = 0; index < frame_current.keypoints.size(); ++index) {
                detection_grid[cell_key(static_cast<double>(frame_current.keypoints[index].x), static_cast<double>(frame_current.keypoints[index].y))].push_back(index);
            }
        }
        std::vector<unsigned char> claimed(active.size(), static_cast<unsigned char>(0));
        std::vector<unsigned char> detection_claimed(frame_current.keypoints.size(), static_cast<unsigned char>(0));
        const double columns = static_cast<double>(frame_current.image_pyramid[0].get_cols());
        const double rows = static_cast<double>(frame_current.image_pyramid[0].get_rows());
        const double radius_squared = radius * radius;

        struct candidate final {
            size_t index;
            unsigned int distance;
        };

        const auto search = [&](const std::unordered_map<long long, std::vector<size_t>>& grid, const math::matrix<double, 2, 1>& projected, const std::vector<feature::descriptor::binary<256>>& queries, const int query_octave, const auto& position_x, const auto& position_y, const auto& octave_of, const auto& descriptor_of, const auto& skip, candidate& best, candidate& second) {
            best = candidate{ static_cast<size_t>(-1), 256u };
            second = candidate{ static_cast<size_t>(-1), 256u };
            for (int offset_y = -1; offset_y <= 1; ++offset_y) {
                for (int offset_x = -1; offset_x <= 1; ++offset_x) {
                    const std::unordered_map<long long, std::vector<size_t>>::const_iterator cell_it = grid.find(cell_key(projected[0] + (static_cast<double>(offset_x) * radius), projected[1] + (static_cast<double>(offset_y) * radius)));
                    if (cell_it == grid.end()) {
                        continue;
                    }
                    for (const size_t index : cell_it->second) {
                        if (skip(index)) {
                            continue;
                        }
                        const double dx = position_x(index) - projected[0];
                        const double dy = position_y(index) - projected[1];
                        if (((dx * dx) + (dy * dy)) > radius_squared) {
                            continue;
                        }
                        const int octave_offset = octave_of(index) - query_octave;
                        if (((octave_offset < 0) ? -octave_offset : octave_offset) > slam::local_map_octave_window) {
                            continue;
                        }
                        unsigned int distance = 256u;
                        for (const feature::descriptor::binary<256>& query : queries) {
                            distance = math::min(distance, match::distance::hamming::distance(query, descriptor_of(index)));
                        }
                        if (distance < best.distance) {
                            second = best;
                            best = candidate{ index, distance };
                        }
                        else if (distance < second.distance) {
                            second = candidate{ index, distance };
                        }
                    }
                }
            }
        };
        const unsigned int match_bound = this->hamming_bound((this->frontend.association == mapping::frame::settings::association_kind::match) ? slam::local_map_match_hamming : slam::local_map_match_hamming_both);
        const unsigned int plain_bound = this->hamming_bound(slam::reacquire_hamming_maximum);
        const auto accepted = [match_detections, match_bound, plain_bound](const candidate& best, const candidate& second) {
            if (!match_detections) {
                return (best.index != static_cast<size_t>(-1)) && (best.distance <= plain_bound);
            }
            return (best.index != static_cast<size_t>(-1)) && (best.distance <= match_bound) && ((second.index == static_cast<size_t>(-1)) || (static_cast<float>(best.distance) < slam::local_map_match_ratio * static_cast<float>(second.distance)));
        };
        size_t in_view = 0;
        size_t matched = 0;
        std::vector<std::pair<size_t, int>> spawns;
        for (const int landmark_id : this->local_landmark_ids_) {
            const std::unordered_map<int, mapping::point>::iterator landmark_it = this->reconstruction.landmarks.find(landmark_id);
            if (landmark_it == this->reconstruction.landmarks.end()) {
                continue;
            }
            const mapping::point& landmark = landmark_it->second;
            if (linked_landmark_ids.count(landmark_id) != 0) {
                continue;
            }
            math::matrix<double, 2, 1> projected;
            if (!mapping::map::project_landmark(frame_current, landmark, projected)) {
                continue;
            }
            if ((projected[0] < slam::local_map_border) || (projected[0] >= columns - slam::local_map_border) || (projected[1] < slam::local_map_border) || (projected[1] >= rows - slam::local_map_border)) {
                continue;
            }
            ++in_view;
            std::vector<feature::descriptor::binary<256>> landmark_descriptor(1);
            for (size_t descriptor_byte = 0; descriptor_byte < 32; ++descriptor_byte) {
                landmark_descriptor[0].data[descriptor_byte] = landmark.descriptor[descriptor_byte];
            }
            if (this->frontend.affine) {
                landmark_descriptor.resize(1 + landmark.descriptor_history.size());
                for (size_t h = 0; h < landmark.descriptor_history.size(); ++h) {
                    for (size_t descriptor_byte = 0; descriptor_byte < 32; ++descriptor_byte) {
                        landmark_descriptor[1 + h].data[descriptor_byte] = landmark.descriptor_history[h][descriptor_byte];
                    }
                }
            }
            const int landmark_octave = this->reconstruction.observations.at(landmark_id).back().octave;
            candidate best;
            candidate second;
            search(
                track_grid,
                projected,
                landmark_descriptor,
                landmark_octave,
                [&](const size_t index) {
                    return static_cast<double>(active[index]->x);
                },
                [&](const size_t index) {
                    return static_cast<double>(active[index]->y);
                },
                [&](const size_t index) {
                    return active[index]->octave;
                },
                [&](const size_t index) -> const feature::descriptor::binary<256>& {
                    return active[index]->descriptor;
                },
                [&](const size_t index) {
                    return (claimed[index] != 0) || (active[index]->landmark_id >= 0);
                },
                best,
                second
            );
            if (accepted(best, second)) {
                claimed[best.index] = static_cast<unsigned char>(1);
                active[best.index]->landmark_id = landmark_id;
                ++matched;
                continue;
            }
            if (!match_detections) {
                continue;
            }
            search(
                detection_grid,
                projected,
                landmark_descriptor,
                landmark_octave,
                [&](const size_t index) {
                    return static_cast<double>(frame_current.keypoints[index].x);
                },
                [&](const size_t index) {
                    return static_cast<double>(frame_current.keypoints[index].y);
                },
                [&](const size_t index) {
                    return frame_current.keypoints[index].octave;
                },
                [&](const size_t index) -> const feature::descriptor::binary<256>& {
                    return frame_current.descriptors[index];
                },
                [&](const size_t index) {
                    return detection_claimed[index] != 0;
                },
                best,
                second
            );
            if (!accepted(best, second)) {
                continue;
            }
            detection_claimed[best.index] = static_cast<unsigned char>(1);
            const feature::point& detection = frame_current.keypoints[best.index];
            size_t nearest_track = active.size();
            double nearest_distance_squared = slam::local_map_relink_distance * slam::local_map_relink_distance;
            const std::unordered_map<long long, std::vector<size_t>>::const_iterator cell_it = track_grid.find(cell_key(static_cast<double>(detection.x), static_cast<double>(detection.y)));
            if (cell_it != track_grid.end()) {
                for (const size_t index : cell_it->second) {
                    if ((claimed[index] != 0) || (active[index]->landmark_id >= 0)) {
                        continue;
                    }
                    const double dx = static_cast<double>(active[index]->x) - static_cast<double>(detection.x);
                    const double dy = static_cast<double>(active[index]->y) - static_cast<double>(detection.y);
                    if (((dx * dx) + (dy * dy)) < nearest_distance_squared) {
                        nearest_distance_squared = (dx * dx) + (dy * dy);
                        nearest_track = index;
                    }
                }
            }
            if (nearest_track != active.size()) {
                claimed[nearest_track] = static_cast<unsigned char>(1);
                active[nearest_track]->landmark_id = landmark_id;
            }
            else {
                spawns.push_back({ best.index, landmark_id });
            }
            ++matched;
        }
        for (const std::pair<size_t, int>& spawn : spawns) {
            feature::tracker::tracker::track& created = this->tracker_.spawn(frame_current.id, frame_current.keypoints[spawn.first], frame_current.descriptors[spawn.first]);
            created.landmark_id = spawn.second;
        }
        this->matched_by_projection += matched;
        if ((matched > 0) && refine_pose) {
            size_t refined_inliers = 0;
            const bool refined = this->refine_pose_from_prior(frame_current, refined_inliers);
            core::logger::log(core::logger::level::info, "Local map (frame %d): %zu of %zu landmarks in view without a track matched by projection (%zu tracks started), pose %s with %zu inliers.", frame_current.id, matched, in_view, spawns.size(), refined ? "refined" : "kept", refined_inliers);
        }
        return matched;
    }

    void reacquire(mapping::frame& frame_current) {
        const std::vector<feature::tracker::tracker::track*> active = this->active_point_tracks();
        const std::vector<feature::tracker::tracker::track*> all_tracks = this->all_point_tracks();

        std::unordered_set<int> linked_landmark_ids;
        for (const feature::tracker::tracker::track* const t : all_tracks) {
            if (t->landmark_id >= 0) {
                linked_landmark_ids.insert(t->landmark_id);
            }
        }

        std::vector<int> orphan_ids;
        for (const auto& [landmark_id, landmark_record] : this->reconstruction.landmarks) {
            static_cast<void>(landmark_record);
            if (linked_landmark_ids.count(landmark_id) != 0) {
                continue;
            }
            const auto observations_it = this->reconstruction.observations.find(landmark_id);
            if ((observations_it == this->reconstruction.observations.end()) || (observations_it->second.empty())) {
                continue;
            }
            const int last_seen_frame_id = observations_it->second.back().frame_id;
            if (((last_seen_frame_id + reacquire_window_frames) < frame_current.id) || (last_seen_frame_id >= frame_current.id)) {
                continue;
            }
            orphan_ids.push_back(landmark_id);
        }
        std::sort(orphan_ids.begin(), orphan_ids.end());
        std::vector<unsigned char> claimed(active.size(), static_cast<unsigned char>(0));

        const auto reattach = [this, &frame_current](mapping::point& landmark_record, feature::tracker::tracker::track* const t) {
            const int newest_frame_id = this->reconstruction.observations.at(landmark_record.id).back().frame_id;
            for (const feature::tracker::tracker::observation& history_obs : t->history) {
                if ((history_obs.frame_id <= newest_frame_id) || (history_obs.frame_id >= frame_current.id)) {
                    continue;
                }
                if (this->keyframe_ids_.count(history_obs.frame_id) == 0) {
                    continue;
                }
                if (this->reconstruction.frames.count(history_obs.frame_id) == 0) {
                    continue;
                }
                this->reconstruction.add_observation(history_obs.frame_id, landmark_record, static_cast<double>(history_obs.x), static_cast<double>(history_obs.y), t->octave);
            }
            this->reconstruction.add_observation(frame_current.id, landmark_record, static_cast<double>(t->x), static_cast<double>(t->y), t->octave);
            t->landmark_id = landmark_record.id;
        };
        size_t reacquired_epipolar_now = 0;
        double camera_parameters[sensor::model::parameter_count];
        frame_current.camera.get_parameters(&camera_parameters[0], sensor::model::parameter_count);
        const double focal_x = camera_parameters[0];
        const double focal_y = camera_parameters[1];
        const double centre_x = camera_parameters[2];
        const double centre_y = camera_parameters[3];
        const auto undistort = [&](const sensor::model& camera_model, const double pixel_x, const double pixel_y, float& virtual_x, float& virtual_y) -> bool {
            const double pixel[2] = { pixel_x, pixel_y };
            double ray[3];
            if (!camera_model.unproject(&pixel[0], &ray[0])) {
                return false;
            }
            if (!(ray[2] > 1.0e-12)) {
                return false;
            }
            virtual_x = static_cast<float>(focal_x * (ray[0] / ray[2]) + centre_x);
            virtual_y = static_cast<float>(focal_y * (ray[1] / ray[2]) + centre_y);
            return true;
        };

        std::vector<feature::point> rhs_points;
        std::vector<feature::descriptor::binary<256>> rhs_descriptors;
        std::vector<size_t> rhs_track_indices;
        for (size_t track_index = 0; track_index < active.size(); ++track_index) {
            const feature::tracker::tracker::track* const t = active[track_index];
            if ((t->landmark_id >= 0) || (claimed[track_index] != 0)) {
                continue;
            }
            float virtual_x = 0.0f;
            float virtual_y = 0.0f;
            if (!undistort(frame_current.camera, static_cast<double>(t->x), static_cast<double>(t->y), virtual_x, virtual_y)) {
                continue;
            }
            rhs_points.push_back(feature::point{ virtual_x, virtual_y, 0.0f, 0.0f, 0 });
            rhs_descriptors.push_back(t->descriptor);
            rhs_track_indices.push_back(track_index);
        }

        std::vector<std::pair<int, int>> orphan_by_source;
        for (const int orphan_id : orphan_ids) {
            const auto landmark_it = this->reconstruction.landmarks.find(orphan_id);
            if (landmark_it == this->reconstruction.landmarks.end()) {
                continue;
            }
            if (linked_landmark_ids.count(orphan_id) != 0) {
                continue;
            }
            const std::vector<mapping::map::observation>& landmark_observations = this->reconstruction.observations.at(orphan_id);
            if (landmark_observations.back().frame_id >= frame_current.id) {
                continue;
            }
            int source_frame_id = -1;
            for (auto obs_it = landmark_observations.rbegin(); obs_it != landmark_observations.rend(); ++obs_it) {
                if ((this->keyframe_ids_.count(obs_it->frame_id) != 0) && (this->reconstruction.frames.count(obs_it->frame_id) != 0)) {
                    source_frame_id = obs_it->frame_id;
                    break;
                }
            }
            if (source_frame_id < 0) {
                continue;
            }
            orphan_by_source.push_back({ source_frame_id, orphan_id });
        }
        std::sort(orphan_by_source.begin(), orphan_by_source.end());

        const math::matrix<double, 3, 3> intrinsics_inverse({ { 1.0 / focal_x, 0.0, -centre_x / focal_x }, { 0.0, 1.0 / focal_y, -centre_y / focal_y }, { 0.0, 0.0, 1.0 } });
        const math::matrix<double, 3, 3> intrinsics_inverse_transposed = math::transpose(intrinsics_inverse);

        size_t group_begin = 0;
        while ((group_begin < orphan_by_source.size()) && (!rhs_points.empty())) {
            size_t group_end = group_begin;
            while ((group_end < orphan_by_source.size()) && (orphan_by_source[group_end].first == orphan_by_source[group_begin].first)) {
                ++group_end;
            }
            const int source_frame_id = orphan_by_source[group_begin].first;
            const mapping::frame& frame_source = this->reconstruction.frames.at(source_frame_id);

            const math::matrix<double, 3, 3> relative_rotation = frame_current.rotation * math::transpose(frame_source.rotation);
            const math::matrix<double, 3, 1> relative_translation = frame_current.translation - (relative_rotation * frame_source.translation);
            const math::matrix<double, 3, 3> translation_skew({ { 0.0, -relative_translation[2], relative_translation[1] }, { relative_translation[2], 0.0, -relative_translation[0] }, { -relative_translation[1], relative_translation[0], 0.0 } });
            const math::matrix<double, 3, 3> fundamental = intrinsics_inverse_transposed * (translation_skew * relative_rotation) * intrinsics_inverse;

            std::vector<feature::point> lhs_points;
            std::vector<feature::descriptor::binary<256>> lhs_descriptors;
            std::vector<int> lhs_landmark_ids;
            for (size_t group_index = group_begin; group_index < group_end; ++group_index) {
                const int orphan_id = orphan_by_source[group_index].second;
                const std::vector<mapping::map::observation>& orphan_observations = this->reconstruction.observations.at(orphan_id);
                const mapping::map::observation* source_observation = nullptr;
                for (auto obs_it = orphan_observations.rbegin(); obs_it != orphan_observations.rend(); ++obs_it) {
                    if (obs_it->frame_id == source_frame_id) {
                        source_observation = &(*obs_it);
                        break;
                    }
                }
                if (source_observation == nullptr) {
                    continue;
                }
                const mapping::map::observation& last_observation = *source_observation;
                float virtual_x = 0.0f;
                float virtual_y = 0.0f;
                if (!undistort(frame_source.camera, last_observation.point[0], last_observation.point[1], virtual_x, virtual_y)) {
                    continue;
                }
                lhs_points.push_back(feature::point{ virtual_x, virtual_y, 0.0f, 0.0f, 0 });
                feature::descriptor::binary<256> landmark_descriptor;
                for (size_t descriptor_byte = 0; descriptor_byte < 32; ++descriptor_byte) {
                    landmark_descriptor.data[descriptor_byte] = this->reconstruction.landmarks.at(orphan_id).descriptor[descriptor_byte];
                }
                lhs_descriptors.push_back(landmark_descriptor);
                lhs_landmark_ids.push_back(orphan_id);
            }

            if (!lhs_points.empty()) {
                std::vector<match::pair> matches(lhs_points.size());
                const size_t match_count = match::matcher::epipolar::find_matches(
                    lhs_points.data(),
                    lhs_descriptors.data(),
                    lhs_points.size(),
                    rhs_points.data(),
                    rhs_descriptors.data(),
                    rhs_points.size(),
                    fundamental,
                    static_cast<float>(reacquire_epipolar_tolerance),
                    nullptr,
                    static_cast<float>(this->hamming_bound(slam::reacquire_hamming_maximum)) + 0.5f,
                    1,
                    matches.data(),
                    matches.size()
                );
                for (size_t match_index = 0; match_index < match_count; ++match_index) {
                    const size_t track_index = rhs_track_indices[matches[match_index].rhs_index];
                    if (claimed[track_index] != 0) {
                        continue;
                    }
                    feature::tracker::tracker::track* const t = active[track_index];
                    if (t->landmark_id >= 0) {
                        continue;
                    }
                    const int orphan_id = lhs_landmark_ids[matches[match_index].lhs_index];
                    reattach(this->reconstruction.landmarks.at(orphan_id), t);
                    claimed[track_index] = static_cast<unsigned char>(1);
                    ++reacquired_epipolar_now;
                }
            }
            group_begin = group_end;
        }
        this->reacquired_by_epipolar += reacquired_epipolar_now;
        if (reacquired_epipolar_now > 0) {
            core::logger::log(core::logger::level::info, "Map re-acquisition (frame %d): %zu landmarks by epipolar matching.", frame_current.id, reacquired_epipolar_now);
        }
    }

public:
    static double model_selection_score(const float* const residuals, const size_t count, const float threshold) {
        double score = 0.0;
        for (size_t i = 0; i < count; ++i) {
            if (residuals[i] < threshold) {
                score += static_cast<double>(threshold) - static_cast<double>(residuals[i]);
            }
        }
        return score;
    }

    static double parallax_angle(
        const math::matrix<double, 3, 1>& point,
        const math::matrix<double, 3, 1>& camera_centre_lhs,
        const math::matrix<double, 3, 1>& camera_centre_rhs
    ) {
        const math::matrix<double, 3, 1> ray_lhs = camera_centre_lhs - point;
        const math::matrix<double, 3, 1> ray_rhs = camera_centre_rhs - point;
        const double length_lhs = math::sqrt(ray_lhs.get_length_squared());
        const double length_rhs = math::sqrt(ray_rhs.get_length_squared());
        if ((length_lhs < 1.0e-12) || (length_rhs < 1.0e-12)) {
            return 0.0;
        }
        const double dot = ray_lhs[0] * ray_rhs[0] + ray_lhs[1] * ray_rhs[1] + ray_lhs[2] * ray_rhs[2];
        double cosine = dot / (length_lhs * length_rhs);
        if (cosine > 1.0) {
            cosine = 1.0;
        }
        else if (cosine < -1.0) {
            cosine = -1.0;
        }
        return math::acos(cosine);
    }

    static void predict_constant_velocity(
        const math::matrix<double, 3, 3>& rotation_previous,
        const math::matrix<double, 3, 1>& translation_previous,
        const math::matrix<double, 3, 3>& rotation_last,
        const math::matrix<double, 3, 1>& translation_last,
        math::matrix<double, 3, 3>& rotation_predicted,
        math::matrix<double, 3, 1>& translation_predicted
    ) {
        const math::se3<double> pose_previous(rotation_previous, translation_previous);
        const math::se3<double> pose_last(rotation_last, translation_last);
        const math::se3<double> pose_predicted = pose_last * pose_previous.inverse() * pose_last;
        rotation_predicted = pose_predicted.rotation().get_matrix();
        translation_predicted = pose_predicted.translation();
    }

    static void ratio_test(std::vector<match::pair>& matches) {
        std::vector<match::pair>::iterator keep = matches.begin();
        for (std::vector<match::pair>::iterator i = matches.begin(); i != matches.end();) {
            std::vector<match::pair>::iterator group_end = i + 1;
            while ((group_end != matches.end()) && (group_end->lhs_index == i->lhs_index)) {
                ++group_end;
            }
            if (group_end - i == 1) {
                *keep++ = *i;
            }
            else if (group_end - i == 2) {
                const std::vector<match::pair>::iterator best = (i->score <= (i + 1)->score) ? i : (i + 1);
                const std::vector<match::pair>::iterator worst = (i->score <= (i + 1)->score) ? (i + 1) : i;
                if ((worst->score > 0.0f) && (best->score <= 0.75f * worst->score)) {
                    *keep++ = *best;
                }
            }
            i = group_end;
        }
        matches.erase(keep, matches.end());
    }

    void finalise() {
        this->reconstruction.optimise(0, false, 200, true);
        this->reanchor_corrected();
        this->update_anchored_frames();
        this->refit_anchored_frames();
    }

    void process_frame(const math::matrix<double, 3, 3>& intrinsics, const image::image& image_grey) {
        const double parameters[sensor::model::parameter_count] = {
            intrinsics[0][0],
            intrinsics[1][1],
            intrinsics[0][2],
            intrinsics[1][2],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        };
        this->process_frame(sensor::model(&parameters[0], sensor::model::parameter_count), image_grey);
    }

    void process_frame(const sensor::model& camera_intrinsics, const image::image& image_grey) {
        mapping::frame frame(this->reconstruction.allocate_frame_id(), camera_intrinsics, image_grey, this->frontend);
        if (this->frontend.blur_weighting) {
            this->laplacian_history_.push_back(frame.laplacian_variance);
            if (this->laplacian_history_.size() > slam::blur_history_frames) {
                this->laplacian_history_.erase(this->laplacian_history_.begin());
            }
            std::vector<double> sorted(this->laplacian_history_);
            std::sort(sorted.begin(), sorted.end());
            const double median = sorted[sorted.size() / 2];
            if ((median > 0.0) && (frame.laplacian_variance > 0.0)) {
                frame.measurement_sigma = math::max(1.0, math::min(slam::blur_sigma_maximum, math::sqrt(median / frame.laplacian_variance)));
            }
            core::logger::log(core::logger::level::debug, "Frame %d sharpness %.1f (median %.1f), measurement sigma %.2f.", frame.id, frame.laplacian_variance, median, frame.measurement_sigma);
        }

        if (core::logger::enabled(core::logger::level::info)) {
            char counts[256] = "";
            size_t offset = 0;
            for (size_t i = 0; (i < frame.keypoint_pyramid.size()) && (offset < sizeof(counts)); ++i) {
                const int length = std::snprintf(counts + offset, sizeof(counts) - offset, "%zu ", frame.keypoint_pyramid[i].size());
                offset += (length > 0) ? static_cast<size_t>(length) : 0;
            }
            core::logger::log(core::logger::level::info, "Detected Features: %sfeatures (at each pyramid level), %zu consumed.", counts, frame.keypoints.size());
        }

        // Add the frame to the reconstruction.
        this->reconstruction.add_frame(frame);

        this->loop_closure_.set_hamming_scale(this->frontend.descriptor_distance_scale());
        this->reconstruction.line_ray_angle_degrees = this->frontend.line_angle;
        this->reconstruction.solver_strategy = this->frontend.solver;
        this->reconstruction.solver_precision = this->frontend.solver_precision;
        if (this->frontend.tracker == mapping::frame::settings::tracker_kind::extrema) {
            this->extrema_tracker_.update(frame.id, image_grey);
            if (core::logger::enabled(core::logger::level::info)) {
                const feature::tracker::extrema::diagnostics& diagnostics = this->extrema_tracker_.get_diagnostics();
                core::logger::log(core::logger::level::info, "Extrema: %zu detected, %zu flow matches%s, %zu tracked, %zu lost, %zu spawned.", diagnostics.detected_extrema, diagnostics.flow_matches, diagnostics.flow_fallback ? " (fallback flow)" : "", diagnostics.tracked, diagnostics.lost, diagnostics.spawned);
            }
        }
        else {
            {
                feature::tracker::tracker::options options = this->tracker_.get_options();
                const feature::tracker::tracker::association_kind wanted = (this->frontend.association == mapping::frame::settings::association_kind::match) ? feature::tracker::tracker::association_kind::descriptor : ((this->frontend.association == mapping::frame::settings::association_kind::both) ? feature::tracker::tracker::association_kind::both : feature::tracker::tracker::association_kind::optical_flow);
                const feature::tracker::tracker::options defaults;
                const unsigned int match_bound_wanted = this->hamming_bound(defaults.match_hamming);
                const unsigned int reacquire_bound_wanted = this->hamming_bound(defaults.reacquire_hamming);
                if ((options.association != wanted) || (options.match_hamming != match_bound_wanted) || (options.reacquire_hamming != reacquire_bound_wanted) || (options.klt_damped_steps != this->frontend.klt_damped_steps) || (options.collision_distance != this->frontend.track_collision_distance) || (options.anchored_patches != this->frontend.anchored_patches) || (options.anchor_model != this->frontend.anchor_model) || (options.anchor_refresh_error != this->frontend.anchor_refresh_error) || (options.flow != this->frontend.flow) || (options.wavelet_half_window != this->frontend.wavelet_half_window) || (options.wavelet_levels != this->frontend.wavelet_levels) || (options.wavelet_robust != this->frontend.wavelet_robust) || (options.wavelet_undecimated != this->frontend.wavelet_undecimated) || (options.wavelet_seed != this->frontend.wavelet_seed)) {
                    options.association = wanted;
                    options.match_hamming = match_bound_wanted;
                    options.reacquire_hamming = reacquire_bound_wanted;
                    options.klt_damped_steps = this->frontend.klt_damped_steps;
                    options.collision_distance = this->frontend.track_collision_distance;
                    options.anchored_patches = this->frontend.anchored_patches;
                    options.anchor_model = this->frontend.anchor_model;
                    options.anchor_refresh_error = this->frontend.anchor_refresh_error;
                    options.flow = this->frontend.flow;
                    options.wavelet_half_window = this->frontend.wavelet_half_window;
                    options.wavelet_levels = this->frontend.wavelet_levels;
                    options.wavelet_robust = this->frontend.wavelet_robust;
                    options.wavelet_undecimated = this->frontend.wavelet_undecimated;
                    options.wavelet_seed = this->frontend.wavelet_seed;
                    this->tracker_.set_options(options);
                }
            }
            this->tracker_.update(frame.id, frame.image_pyramid, frame.keypoints, frame.descriptors);
        }

        core::logger::log(core::logger::level::info, "Tracked: %zu active tracks.", this->active_point_tracks().size());

        if (this->frontend.lines) {
            feature::detector::elsed::options elsed_options;
            elsed_options.minimum_length = 10.0f;
            std::vector<feature::detector::elsed::segment> segments(8192);
            const size_t segment_count = feature::detector::elsed::detect(image_grey.get_data(), static_cast<int>(image_grey.get_cols()), static_cast<int>(image_grey.get_rows()), static_cast<int>(image_grey.get_cols()), elsed_options, segments.data(), segments.size());
            segments.resize(segment_count);
            if (segments.size() > slam::line_segments_maximum) {
                std::stable_sort(segments.begin(), segments.end(), [](const feature::detector::elsed::segment& lhs, const feature::detector::elsed::segment& rhs) {
                    return lhs.length > rhs.length;
                });
                segments.resize(slam::line_segments_maximum);
            }
            this->line_tracker_.update(frame.id, segments);
            core::logger::log(core::logger::level::info, "Detected Lines: %zu segments, %zu line tracks.", segment_count, this->line_tracker_.active_tracks().size());
        }

        // Nothing to do for the first frame.
        if (frame.id == 0) {
            return;
        }

        mapping::frame& frame_current = this->reconstruction.frames.at(frame.id);

        const auto reset_dangling = [this]() {
            const std::vector<feature::tracker::tracker::track*> active = this->active_point_tracks();
            for (feature::tracker::tracker::track* const t : active) {
                if ((t->landmark_id >= 0) && (this->reconstruction.landmarks.count(t->landmark_id) == 0)) {
                    t->landmark_id = -1;
                }
            }
            for (feature::tracker::line::track* const t : this->line_tracker_.all_tracks()) {
                if ((t->landmark_id >= 0) && (this->reconstruction.line_landmarks.count(t->landmark_id) == 0)) {
                    t->landmark_id = -1;
                }
            }
        };
        reset_dangling();

        bool is_keyframe = false;
        bool posed_from_tracks = false;
        bool relocalised = false;
        bool initialise = (this->state_ != tracking_state::tracking) && this->reconstruction.landmarks.empty();
        if ((this->state_ != tracking_state::tracking) && !this->reconstruction.landmarks.empty()) {
            size_t correspondences = 0;
            posed_from_tracks = this->pose_from_tracks(frame_current, correspondences);
            relocalised = !posed_from_tracks && this->relocalise(frame_current);
            if (!posed_from_tracks && !relocalised) {
                if (this->reinitialising_) {
                    initialise = true;
                }
                else {
                    ++this->lost_frames_;
                    if ((this->lost_frames_ > slam::reinitialise_after_frames) && (this->active_point_tracks().size() >= slam::reinitialise_tracks_minimum)) {
                        this->begin_submap(frame_current);
                    }
                    else {
                        this->reconstruction.frames.erase(frame_current.id);
                    }
                    return;
                }
            }
        }
        if ((this->state_ != tracking_state::tracking) && (posed_from_tracks || relocalised)) {
            if (this->reinitialising_) {
                this->reconstruction.frames.erase(this->init_anchor_frame_id_);
                this->reinitialising_ = false;
            }
            this->lost_frames_ = 0;
            core::logger::log(core::logger::level::note, "Tracking resumed at frame %d %s.", frame_current.id, posed_from_tracks ? "from its tracks" : "by relocalisation");
            this->state_ = tracking_state::tracking;
            this->blind_frames_ = 0;
            this->relocalisation_probation_ = slam::relocalisation_probation_frames;
            this->probation_frame_ids_.clear();
            this->probation_frame_ids_.push_back(frame_current.id);
            this->previous_tracked_frame_id_ = -1;
        }
        else if (initialise) {
            const mapping::frame& frame_anchor = this->reconstruction.frames.at(this->init_anchor_frame_id_);

            std::vector<estimation::correspondence_2d_2d<double>> init_correspondences;
            {
                const std::vector<feature::tracker::tracker::track*> active = this->active_point_tracks();
                for (feature::tracker::tracker::track* const t : active) {
                    const feature::tracker::tracker::observation* anchor_observation = nullptr;
                    for (const feature::tracker::tracker::observation& observation : t->history) {
                        if (observation.frame_id == this->init_anchor_frame_id_) {
                            anchor_observation = &observation;
                            break;
                        }
                        if (observation.frame_id > this->init_anchor_frame_id_) {
                            break;
                        }
                    }
                    if (anchor_observation == nullptr) {
                        continue;
                    }
                    const double lhs_point[2] = { static_cast<double>(t->x), static_cast<double>(t->y) };
                    double lhs_ray[3];
                    if (!frame_current.camera.unproject(&lhs_point[0], &lhs_ray[0])) {
                        continue;
                    }
                    const double rhs_point[2] = { static_cast<double>(anchor_observation->x), static_cast<double>(anchor_observation->y) };
                    double rhs_ray[3];
                    if (!frame_anchor.camera.unproject(&rhs_point[0], &rhs_ray[0])) {
                        continue;
                    }
                    init_correspondences.push_back({ { { lhs_ray[0] / lhs_ray[2], lhs_ray[1] / lhs_ray[2] } }, { { rhs_ray[0] / rhs_ray[2], rhs_ray[1] / rhs_ray[2] } } });
                }
            }
            const size_t correspondence_count = init_correspondences.size();

            const float essential_inlier_threshold = 1.0e-5f;

            const float model_selection_threshold = 1.0e-5f;

            estimation::robust::estimate::essential<double>::model model_essential{};
            std::vector<float> essential_residuals(correspondence_count);
            std::vector<size_t> essential_inlier_indices(correspondence_count);
            size_t essential_inliers = correspondence_count;
            const bool essential_ok = estimation::robust::solver::essential<double>::solve(init_correspondences.data(), correspondence_count, essential_residuals.data(), essential_inlier_indices.data(), essential_inliers, model_essential);

            estimation::robust::estimate::homography<double>::model model_homography{};
            std::vector<float> homography_residuals(correspondence_count);
            std::vector<size_t> homography_inlier_indices(correspondence_count);
            size_t homography_inliers = correspondence_count;
            const bool homography_ok = estimation::robust::solver::homography<double>::solve(init_correspondences.data(), correspondence_count, homography_residuals.data(), homography_inlier_indices.data(), homography_inliers, model_homography);

            if (!essential_ok && !homography_ok) {
                core::logger::log(core::logger::level::info, "Init REJECTED (frame %d vs anchor %d): neither an essential matrix nor a homography could be estimated.", frame_current.id, frame_anchor.id);
                this->reconstruction.frames.erase(frame_current.id);
                return;
            }

            const double score_essential = essential_ok ? model_selection_score(essential_residuals.data(), correspondence_count, model_selection_threshold) : 0.0;
            const double score_homography = homography_ok ? model_selection_score(homography_residuals.data(), correspondence_count, model_selection_threshold) : 0.0;
            const double score_total = score_essential + score_homography;
            const double ratio_homography = (score_total > 0.0) ? (score_homography / score_total) : 0.0;
            const bool use_homography = homography_ok && (!essential_ok || (ratio_homography > 0.45));
            core::logger::log(core::logger::level::debug, "Init model selection: E score %.6g (%zu inliers), H score %.6g (%zu inliers), R_H = %.4f -> selected %s.", score_essential, essential_inliers, score_homography, homography_inliers, ratio_homography, use_homography ? "HOMOGRAPHY" : "ESSENTIAL");

            const size_t* const chosen_inlier_indices = use_homography ? homography_inlier_indices.data() : essential_inlier_indices.data();
            const size_t inliers = use_homography ? homography_inliers : essential_inliers;
            std::vector<math::matrix<double, 2, 1>> match_point_current_inlier;
            std::vector<math::matrix<double, 2, 1>> match_point_previous_inlier;
            match_point_current_inlier.reserve(inliers);
            match_point_previous_inlier.reserve(inliers);
            for (size_t inlier_index = 0; inlier_index < inliers; ++inlier_index) {
                match_point_current_inlier.push_back({ { init_correspondences[chosen_inlier_indices[inlier_index]].lhs[0], init_correspondences[chosen_inlier_indices[inlier_index]].lhs[1] } });
                match_point_previous_inlier.push_back({ { init_correspondences[chosen_inlier_indices[inlier_index]].rhs[0], init_correspondences[chosen_inlier_indices[inlier_index]].rhs[1] } });
            }

            math::matrix<double, 3, 3> rotation;
            math::matrix<double, 3, 1> translation;
            std::vector<math::matrix<double, 3, 1>> match_point_triangulated_inlier(inliers);
            size_t recover_pose_support = 0;
            bool recover_ok;
            if (use_homography) {
                recover_ok = estimation::pose::homography<double>::recover(&model_homography.homography[0][0], match_point_current_inlier.data()->data(), match_point_previous_inlier.data()->data(), inliers, rotation.data(), translation.data(), match_point_triangulated_inlier.data()->data(), &recover_pose_support);
            }
            else {
                recover_ok = estimation::pose::essential<double>::recover(&model_essential.essential[0][0], match_point_current_inlier.data()->data(), match_point_previous_inlier.data()->data(), inliers, rotation.data(), translation.data(), match_point_triangulated_inlier.data()->data(), &recover_pose_support);
            }
            core::logger::log(core::logger::level::debug, "Support: %zu of %zu inliers support the recovered pose.", recover_pose_support, inliers);

            const size_t inlier_floor = 50;
            const double inlier_fraction_minimum = 0.5;
            const double parallax_degrees_minimum = 2.0;
            const double reprojection_fraction_minimum = 0.7;
            const double reprojection_error_squared_maximum = 9.0 * static_cast<double>(essential_inlier_threshold);

            if ((correspondence_count < inlier_floor) && (frame_current.id != this->init_anchor_frame_id_)) {
                core::logger::log(core::logger::level::note, "Init anchor moved from frame %d to frame %d: %zu tracks span them.", this->init_anchor_frame_id_, frame_current.id, correspondence_count);
                const mapping::frame& anchor_previous = this->reconstruction.frames.at(this->init_anchor_frame_id_);
                frame_current.rotation = anchor_previous.rotation;
                frame_current.translation = anchor_previous.translation;
                this->reconstruction.frames.erase(this->init_anchor_frame_id_);
                this->init_anchor_frame_id_ = frame_current.id;
                if (this->keyframe_ids_.empty()) {
                    this->reconstruction.gauge_frame_id = frame_current.id;
                    this->last_keyframe_id_ = frame_current.id;
                }
                this->last_tracked_frame_id_ = frame_current.id;
                return;
            }

            const math::matrix<double, 3, 1> camera_centre_lhs = math::matrix<double, 3, 1>::zero();
            const math::matrix<double, 3, 1> camera_centre_rhs = -math::transpose(rotation) * translation;
            std::vector<double> parallax_angles;
            parallax_angles.reserve(inliers);
            size_t reprojection_valid = 0;
            for (size_t inlier_index = 0; inlier_index < inliers; ++inlier_index) {
                const math::matrix<double, 3, 1>& point = match_point_triangulated_inlier[inlier_index];
                const double depth_lhs = point[2];
                const math::matrix<double, 3, 1> point_rhs = (rotation * point) + translation;
                const double depth_rhs = point_rhs[2];
                if ((depth_lhs <= 0.0) || (depth_rhs <= 0.0)) {
                    continue;
                }
                parallax_angles.push_back(parallax_angle(point, camera_centre_lhs, camera_centre_rhs));
                const double error_lhs_x = (point[0] / depth_lhs) - match_point_current_inlier[inlier_index][0];
                const double error_lhs_y = (point[1] / depth_lhs) - match_point_current_inlier[inlier_index][1];
                const double error_rhs_x = (point_rhs[0] / depth_rhs) - match_point_previous_inlier[inlier_index][0];
                const double error_rhs_y = (point_rhs[1] / depth_rhs) - match_point_previous_inlier[inlier_index][1];
                const double error_lhs = error_lhs_x * error_lhs_x + error_lhs_y * error_lhs_y;
                const double error_rhs = error_rhs_x * error_rhs_x + error_rhs_y * error_rhs_y;
                if ((error_lhs < reprojection_error_squared_maximum) && (error_rhs < reprojection_error_squared_maximum)) {
                    ++reprojection_valid;
                }
            }
            double parallax_median_degrees = 0.0;
            if (!parallax_angles.empty()) {
                std::sort(parallax_angles.begin(), parallax_angles.end());
                const double parallax_median_radians = parallax_angles[parallax_angles.size() / 2];
                parallax_median_degrees = parallax_median_radians * (180.0 / 3.14159265358979323846);
            }
            const double reprojection_fraction = (inliers > 0) ? (static_cast<double>(reprojection_valid) / static_cast<double>(inliers)) : 0.0;

            const bool gate_recover = recover_ok;
            const bool gate_inliers = (inliers >= inlier_floor) && (static_cast<double>(inliers) >= inlier_fraction_minimum * static_cast<double>(correspondence_count));
            const bool gate_parallax = (parallax_median_degrees >= parallax_degrees_minimum);
            const bool gate_reprojection = (reprojection_fraction >= reprojection_fraction_minimum);
            const bool accept = gate_recover && gate_inliers && gate_parallax && gate_reprojection;
            core::logger::log(core::logger::level::info, "Init gate (frame %d vs anchor %d): recover=%s inliers=%zu (floor %zu, >= %.0f%% of %zu correspondences: %s) parallax_median=%.3f deg (min %.2f: %s) reproj_frac=%.3f (min %.2f: %s) -> %s", frame_current.id, frame_anchor.id, gate_recover ? "yes" : "NO", inliers, inlier_floor, inlier_fraction_minimum * 100.0, correspondence_count, gate_inliers ? "yes" : "NO", parallax_median_degrees, parallax_degrees_minimum, gate_parallax ? "yes" : "NO", reprojection_fraction, reprojection_fraction_minimum, gate_reprojection ? "yes" : "NO", accept ? "ACCEPTED" : "REJECTED");

            if (!accept) {
                this->reconstruction.frames.erase(frame_current.id);
                return;
            }

            // Recover pose returns pose 2 to pose 1 rather than pose 1 to pose 2, so invert it.
            rotation = math::transpose(rotation);
            translation = -rotation * translation;

            frame_current.rotation = rotation * frame_anchor.rotation;
            frame_current.translation = (rotation * frame_anchor.translation) + translation;

            this->state_ = tracking_state::tracking;
            this->blind_frames_ = 0;

            this->keyframe_ids_.insert(this->init_anchor_frame_id_);
            this->keyframe_ids_.insert(frame_current.id);
            this->last_keyframe_id_ = frame_current.id;
            is_keyframe = true;
            if (this->reinitialising_) {
                this->start_submap(frame_current);
            }

            core::logger::log(core::logger::level::info, "Inliers: %zu inliers in pose estimation.", inliers);
            core::logger::log(
                core::logger::level::debug,
                "%s:\n{\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f },\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f },\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f },\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f }\n}",
                "Initial pose",
                frame_current.rotation[0][0],
                frame_current.rotation[0][1],
                frame_current.rotation[0][2],
                frame_current.translation[0],
                frame_current.rotation[1][0],
                frame_current.rotation[1][1],
                frame_current.rotation[1][2],
                frame_current.translation[1],
                frame_current.rotation[2][0],
                frame_current.rotation[2][1],
                frame_current.rotation[2][2],
                frame_current.translation[2],
                0.0,
                0.0,
                0.0,
                1.0
            );
        }
        else {
            size_t correspondences = 0;
            bool pnp_success = this->pose_from_tracks(frame_current, correspondences);
            const size_t pnp_correspondencies_size = correspondences;
            if (this->relocalisation_probation_ > 0) {
                if (!pnp_success) {
                    core::logger::log(core::logger::level::warn, "Relocalisation revoked at frame %d: the tracks it relinked do not pose the frame; %zu frames dropped, relocalising again.", frame_current.id, this->probation_frame_ids_.size() + 1);
                    for (const int probation_id : this->probation_frame_ids_) {
                        this->reconstruction.frames.erase(probation_id);
                        this->non_keyframe_anchors_.erase(probation_id);
                    }
                    this->probation_frame_ids_.clear();
                    for (feature::tracker::tracker::track* const t : this->all_point_tracks()) {
                        for (const std::pair<int, int>& link : this->probation_track_links_) {
                            if ((t->id == link.first) && (t->landmark_id == link.second)) {
                                t->landmark_id = -1;
                            }
                        }
                    }
                    this->probation_track_links_.clear();
                    this->relocalisation_probation_ = 0;
                    this->reconstruction.frames.erase(frame_current.id);
                    this->state_ = tracking_state::initialising;
                    return;
                }
                --this->relocalisation_probation_;
                this->probation_frame_ids_.push_back(frame_current.id);
                if (this->relocalisation_probation_ == 0) {
                    core::logger::log(core::logger::level::note, "Recovery confirmed at frame %d.", frame_current.id);
                    this->probation_frame_ids_.clear();
                    this->probation_track_links_.clear();
                }
            }
            if (!pnp_success) {
                if (this->blind_frames_ >= slam::blind_frames_tolerated) {
                    core::logger::log(core::logger::level::warn, "Tracking lost at frame %d, no tracks own a landmark: relocalising; %zu frames the motion model posed are dropped.", frame_current.id, this->blind_frame_ids_.size());
                    this->state_ = tracking_state::initialising;
                    for (const int blind_id : this->blind_frame_ids_) {
                        this->reconstruction.frames.erase(blind_id);
                        this->non_keyframe_anchors_.erase(blind_id);
                    }
                    this->blind_frame_ids_.clear();
                    if (this->reconstruction.frames.count(this->blind_run_origin_id_) != 0) {
                        this->last_tracked_frame_id_ = this->blind_run_origin_id_;
                    }
                    this->previous_tracked_frame_id_ = -1;
                    return;
                }
                const int last_id = this->last_tracked_frame_id_;
                const int previous_id = this->previous_tracked_frame_id_;
                const auto last_it = this->reconstruction.frames.find(last_id);
                const auto previous_it = (previous_id >= 0) ? this->reconstruction.frames.find(previous_id) : this->reconstruction.frames.end();
                const bool have_two_priors = (last_it != this->reconstruction.frames.end()) && (previous_it != this->reconstruction.frames.end());
                if (have_two_priors) {
                    const mapping::frame& frame_previous = previous_it->second;
                    const mapping::frame& frame_last = last_it->second;
                    slam::predict_constant_velocity(
                        frame_previous.rotation,
                        frame_previous.translation,
                        frame_last.rotation,
                        frame_last.translation,
                        frame_current.rotation,
                        frame_current.translation
                    );
                    if (this->frontend.association != mapping::frame::settings::association_kind::klt) {
                        const size_t recovered = this->track_local_map(frame_current, slam::local_map_recovery_radius, false);
                        size_t recovered_correspondences = 0;
                        if ((recovered > 0) && this->pose_from_tracks(frame_current, recovered_correspondences)) {
                            pnp_success = true;
                            core::logger::log(core::logger::level::note, "PnP found no consensus (%zu correspondencies): %zu landmarks matched around the predicted pose, posed from %zu.", pnp_correspondencies_size, recovered, recovered_correspondences);
                        }
                    }
                    size_t refined_inliers = 0;
                    if (pnp_success) {
                    }
                    else if (this->refine_pose_from_prior(frame_current, refined_inliers)) {
                        pnp_success = true;
                        core::logger::log(core::logger::level::note, "PnP found no consensus (%zu correspondencies): pose refined from the constant-velocity prior with %zu inliers.", pnp_correspondencies_size, refined_inliers);
                    }
                    else {
                        core::logger::log(core::logger::level::warn, "PnP FAILED (%zu correspondencies, %zu agree with the refined prior): using constant-velocity predicted pose (from frames %d and %d).", pnp_correspondencies_size, refined_inliers, previous_id, last_id);
                    }
                }
                else if (last_it != this->reconstruction.frames.end()) {
                    const mapping::frame& frame_last_tracked = last_it->second;
                    frame_current.rotation = frame_last_tracked.rotation;
                    frame_current.translation = frame_last_tracked.translation;
                    core::logger::log(core::logger::level::warn, "PnP FAILED (%zu correspondencies): holding last tracked pose (frame %d).", pnp_correspondencies_size, last_id);
                }
                else {
                    core::logger::log(core::logger::level::warn, "PnP FAILED (%zu correspondencies): no prior tracked pose available (frame %d).", pnp_correspondencies_size, frame_current.id);
                }
            }

            if (pnp_success) {
                this->track_local_map(frame_current);
            }
            if (pnp_success) {
                this->blind_frames_ = 0;
                this->blind_frame_ids_.clear();
            }
            else {
                if (this->blind_frame_ids_.empty()) {
                    this->blind_run_origin_id_ = this->last_tracked_frame_id_;
                }
                ++this->blind_frames_;
                this->blind_frame_ids_.push_back(frame_current.id);
            }

            const int frames_since_last_kf = frame_current.id - this->last_keyframe_id_;
            size_t active_count = 0;
            size_t tracked_count = 0;
            {
                const std::vector<feature::tracker::tracker::track*> active = this->active_point_tracks();
                active_count = active.size();
                for (feature::tracker::tracker::track* const t : active) {
                    if ((t->landmark_id >= 0) && (this->reconstruction.landmarks.count(t->landmark_id) != 0)) {
                        ++tracked_count;
                    }
                }
            }
            const double tracked_ratio = static_cast<double>(tracked_count) / ((active_count > 0) ? static_cast<double>(active_count) : 1.0);
            is_keyframe = pnp_success && (tracked_count >= 3) && (frames_since_last_kf >= keyframe_min_interval) && ((frames_since_last_kf >= keyframe_interval) || (tracked_ratio < keyframe_tracked_ratio));
        }
        is_keyframe = is_keyframe && (this->relocalisation_probation_ == 0) && !relocalised && !posed_from_tracks;

        if (!is_keyframe) {
            core::logger::log(core::logger::level::info, "motion-only frame %d, pose from PnP.", frame_current.id);
            this->anchor_non_keyframe(frame_current);
            this->previous_tracked_frame_id_ = this->last_tracked_frame_id_;
            this->last_tracked_frame_id_ = frame_current.id;
            return;
        }

        this->keyframe_ids_.insert(frame_current.id);
        this->last_keyframe_id_ = frame_current.id;

        {
            const std::vector<feature::tracker::tracker::track*> active = this->active_point_tracks();
            for (feature::tracker::tracker::track* const t : active) {
                if (t->landmark_id < 0) {
                    continue;
                }
                if (this->reconstruction.landmarks.count(t->landmark_id) == 0) {
                    continue;
                }
                mapping::point& landmark = this->reconstruction.landmarks.at(t->landmark_id);
                this->reconstruction.add_observation(frame_current.id, landmark, static_cast<double>(t->x), static_cast<double>(t->y), t->octave);
                this->add_landmark_descriptors(landmark, frame_current, t->x, t->y, t->octave, t->descriptor);
            }
            for (feature::tracker::line::track* const t : this->line_tracker_.active_tracks()) {
                if ((t->landmark_id < 0) || (this->reconstruction.line_landmarks.count(t->landmark_id) == 0)) {
                    continue;
                }
                this->reconstruction.add_line_observation(frame_current.id, this->reconstruction.line_landmarks.at(t->landmark_id), static_cast<double>(t->x1), static_cast<double>(t->y1), static_cast<double>(t->x2), static_cast<double>(t->y2));
            }
        }
        this->reconstruction.optimise(1, true, 50);
        this->reacquire(frame_current);
        this->reconstruction.cull();
        reset_dangling();

        int new_landmarks = 0;
        std::vector<double> new_landmark_depths;
        size_t candidates = 0;
        size_t without_reference = 0;
        size_t at_infinity = 0;
        size_t below_parallax = 0;
        size_t failed_gates = 0;
        size_t demoted = 0;
        {
            const std::vector<feature::tracker::tracker::track*> active = this->active_point_tracks();
            for (feature::tracker::tracker::track* const t : active) {
                if (t->landmark_id != -1) {
                    continue;
                }
                if (t->length < 2) {
                    continue;
                }
                ++candidates;
                const feature::tracker::tracker::observation* ref_obs = nullptr;
                const int submap_start = this->submap_start_of(frame_current.id);
                for (const feature::tracker::tracker::observation& obs : t->history) {
                    if ((obs.frame_id != frame_current.id) && (obs.frame_id >= submap_start) && (this->keyframe_ids_.count(obs.frame_id) != 0)) {
                        ref_obs = &obs;
                        break;
                    }
                }
                if (ref_obs == nullptr) {
                    ++without_reference;
                    continue;
                }
                const mapping::frame& frame_ref = this->reconstruction.frames.at(ref_obs->frame_id);
                const double ref_point[2] = { static_cast<double>(ref_obs->x), static_cast<double>(ref_obs->y) };
                math::matrix<double, 3, 1> ray_ref;
                if (!frame_ref.camera.unproject(&ref_point[0], ray_ref.data())) {
                    continue;
                }
                const double cur_point[2] = { static_cast<double>(t->x), static_cast<double>(t->y) };
                math::matrix<double, 3, 1> ray_cur;
                if (!frame_current.camera.unproject(&cur_point[0], ray_cur.data())) {
                    continue;
                }
                const auto admit_at_infinity = [&]() {
                    if (!this->frontend.inverse_depth) {
                        return;
                    }
                    const math::matrix<double, 3, 1> direction_world = math::transpose(frame_ref.rotation) * ray_ref;
                    const math::matrix<double, 3, 1> direction_cur = frame_current.rotation * direction_world;
                    math::matrix<double, 2, 1> reprojected_cur;
                    if ((direction_cur[2] <= 0.0) || !frame_current.camera.project(direction_cur.data(), reprojected_cur.data())) {
                        return;
                    }
                    const math::matrix<double, 2, 1> measured_cur = { { static_cast<double>(t->x), static_cast<double>(t->y) } };
                    if ((measured_cur - reprojected_cur).get_length_squared() > (2 * 2)) {
                        return;
                    }
                    mapping::point landmark(this->reconstruction.allocate_landmark_id(), math::matrix<double, 3, 1>::zero(), math::matrix<double, 3, 1>{ { 0.5, 0.5, 0.5 } });
                    landmark.anchor_at_infinity(frame_ref.rotation, frame_ref.translation, ray_ref);
                    landmark.add_descriptor(&t->descriptor.data[0]);
                    this->reconstruction.add_landmark(landmark);
                    t->landmark_id = landmark.id;
                    for (const feature::tracker::tracker::observation& obs : t->history) {
                        if (this->keyframe_ids_.count(obs.frame_id) != 0) {
                            this->reconstruction.add_observation(obs.frame_id, landmark, static_cast<double>(obs.x), static_cast<double>(obs.y), t->octave);
                        }
                    }
                    ++new_landmarks;
                    ++this->admitted_at_infinity;
                };
                math::matrix<double, 3, 1> point;
                if (!geometry::triangulation::linear_least_squares<double>::triangulate(ray_cur, frame_current.get_pose(), ray_ref, frame_ref.get_pose(), point)) {
                    ++at_infinity;
                    admit_at_infinity();
                    continue;
                }
                const math::matrix<double, 3, 1> centre_cur = -math::transpose(frame_current.rotation) * frame_current.translation;
                const math::matrix<double, 3, 1> centre_ref = -math::transpose(frame_ref.rotation) * frame_ref.translation;
                const double parallax_degrees = parallax_angle(point, centre_cur, centre_ref) * (180.0 / 3.14159265358979323846);
                if (parallax_degrees < triangulation_parallax_degrees_minimum) {
                    ++below_parallax;
                    admit_at_infinity();
                    continue;
                }
                const math::matrix<double, 3, 1> mapped_cur = (frame_current.rotation * point) + frame_current.translation;
                if (mapped_cur[2] < 0) {
                    ++failed_gates;
                    continue;
                }
                const math::matrix<double, 3, 1> mapped_ref = (frame_ref.rotation * point) + frame_ref.translation;
                if (mapped_ref[2] < 0) {
                    ++failed_gates;
                    continue;
                }
                math::matrix<double, 2, 1> reprojected_cur;
                if (!frame_current.camera.project(mapped_cur.data(), reprojected_cur.data())) {
                    ++failed_gates;
                    continue;
                }
                const math::matrix<double, 2, 1> measured_cur = { { static_cast<double>(t->x), static_cast<double>(t->y) } };
                if ((measured_cur - reprojected_cur).get_length_squared() > (2 * 2)) {
                    ++failed_gates;
                    continue;
                }
                math::matrix<double, 2, 1> reprojected_ref;
                if (!frame_ref.camera.project(mapped_ref.data(), reprojected_ref.data())) {
                    ++failed_gates;
                    continue;
                }
                const math::matrix<double, 2, 1> measured_ref = { { static_cast<double>(ref_obs->x), static_cast<double>(ref_obs->y) } };
                if ((measured_ref - reprojected_ref).get_length_squared() > (2 * 2)) {
                    ++failed_gates;
                    continue;
                }
                const size_t colour_col = static_cast<size_t>(math::max(0, math::min(static_cast<int>(image_grey.get_cols()) - 1, static_cast<int>(t->x))));
                const size_t colour_row = static_cast<size_t>(math::max(0, math::min(static_cast<int>(image_grey.get_rows()) - 1, static_cast<int>(t->y))));
                const float colour = static_cast<float>(image_grey.get_data()[colour_row * image_grey.get_cols() + colour_col]) / 255.0f;
                mapping::point landmark(this->reconstruction.allocate_landmark_id(), point, math::matrix<double, 3, 1>{ { static_cast<double>(colour), static_cast<double>(colour), static_cast<double>(colour) } });
                if (this->frontend.inverse_depth && !landmark.anchor(frame_ref.rotation, frame_ref.translation)) {
                    continue;
                }
                this->add_landmark_descriptors(landmark, frame_current, t->x, t->y, t->octave, t->descriptor);
                this->reconstruction.add_landmark(landmark);
                t->landmark_id = landmark.id;
                new_landmark_depths.push_back(mapped_cur[2]);
                for (const feature::tracker::tracker::observation& obs : t->history) {
                    if (this->keyframe_ids_.count(obs.frame_id) != 0) {
                        this->reconstruction.add_observation(obs.frame_id, landmark, static_cast<double>(obs.x), static_cast<double>(obs.y), t->octave);
                    }
                }
                mapping::point& created = this->reconstruction.landmarks.at(landmark.id);
                this->reconstruction.estimate_landmark_uncertainty(created);
                if (created.inverse_depth && (created.uncertainty == mapping::point::uncertainty_kind::estimated)) {
                    const double rho = created.inverse_parameters[2];
                    const double sigma = math::sqrt(math::max(0.0, created.covariance[2][2]));
                    if (rho < mapping::point::depth_significance_minimum * sigma) {
                        created.set_unbounded(math::matrix<double, 3, 1>({ 0.0, 0.0, 1.0 }));
                        ++this->admitted_unbounded;
                        ++demoted;
                    }
                }
                ++new_landmarks;
            }
            core::logger::log(core::logger::level::debug, "Triangulation keyframe %d: %zu candidates, %d landmarks (%zu unbounded), %zu at infinity (%zu below the parallax gate), %zu failed the gates, %zu without a keyframe reference.", frame_current.id, candidates, new_landmarks, demoted, at_infinity + below_parallax, below_parallax, failed_gates, without_reference);
        }
        int new_line_landmarks = 0;
        if (this->frontend.lines) {
            new_line_landmarks = this->triangulate_lines(frame_current);
        }
        this->reconstruction.optimise(1, true, 50);
        this->reconstruction.cull();
        reset_dangling();

        constexpr static const int broad_ba_keyframes = 10;
        int broad_ba_window = 10;
        {
            std::vector<int> keyframe_ids_ascending(this->keyframe_ids_.begin(), this->keyframe_ids_.end());
            std::sort(keyframe_ids_ascending.begin(), keyframe_ids_ascending.end());
            if (static_cast<int>(keyframe_ids_ascending.size()) >= broad_ba_keyframes) {
                const int kf_oldest_in_window = keyframe_ids_ascending[keyframe_ids_ascending.size() - static_cast<size_t>(broad_ba_keyframes)];
                broad_ba_window = this->reconstruction.next_frame_id - kf_oldest_in_window;
            }
            else {
                broad_ba_window = this->reconstruction.next_frame_id;
            }
            broad_ba_window = math::max(broad_ba_window, 10);
        }
        this->reconstruction.optimise(broad_ba_window, false, 50, true);
        const bool globally_adjusted = (this->frontend.global_adjustment_keyframes > 0) && ((this->keyframe_ids_.size() % static_cast<size_t>(this->frontend.global_adjustment_keyframes)) == 0);
        if (globally_adjusted) {
            this->reconstruction.optimise(0, false, 20, true);
        }
        this->reanchor_corrected();
        this->reconstruction.cull();
        if (this->frontend.cull_keyframes) {
            this->cull_keyframes(frame_current.id, 2 * broad_ba_window);
        }

        {
            std::vector<mapping::loop_closure::record> records;
            const std::vector<feature::tracker::tracker::track*> active = this->active_point_tracks();
            records.reserve(active.size());
            for (const feature::tracker::tracker::track* const t : active) {
                if (t->landmark_id < 0) {
                    continue;
                }
                const std::unordered_map<int, mapping::point>::const_iterator landmark_it = this->reconstruction.landmarks.find(t->landmark_id);
                if (landmark_it == this->reconstruction.landmarks.end()) {
                    continue;
                }
                if (landmark_it->second.at_infinity()) {
                    continue;
                }
                mapping::loop_closure::record record;
                record.landmark_id = t->landmark_id;
                for (size_t descriptor_byte = 0; descriptor_byte < 32; ++descriptor_byte) {
                    record.descriptor[descriptor_byte] = landmark_it->second.descriptor[descriptor_byte];
                }
                record.location = landmark_it->second.location;
                record.pixel_x = t->x;
                record.pixel_y = t->y;
                records.push_back(record);
            }
            this->covisibility_.clear();
            std::vector<int> observing_frames;
            for (const auto& [landmark_id, landmark_observations] : this->reconstruction.observations) {
                static_cast<void>(landmark_id);
                observing_frames.clear();
                for (const mapping::map::observation& observation : landmark_observations) {
                    observing_frames.push_back(observation.frame_id);
                }
                this->covisibility_.add(observing_frames.data(), observing_frames.size());
            }
            this->rebuild_local_map(frame_current.id);
            const math::se3<double> pose(frame_current.rotation, frame_current.translation);
            const mapping::loop_closure::result loop = this->loop_closure_.detect(frame_current.id, pose, frame_current.camera, this->covisibility_, records.data(), records.size(), this->submap_start_of(frame_current.id));
            if (loop.found) {
                this->close_loop(frame_current.id, loop, records);
            }
            this->loop_closure_.add_keyframe(frame_current.id, pose, frame_current.camera, records.data(), records.size());
        }

        size_t active_track_count = 0;
        size_t tracks_with_landmarks = 0;
        {
            const std::vector<feature::tracker::tracker::track*> active = this->active_point_tracks();
            active_track_count = active.size();
            for (feature::tracker::tracker::track* const t : active) {
                if ((t->landmark_id >= 0) && (this->reconstruction.landmarks.count(t->landmark_id) != 0)) {
                    ++tracks_with_landmarks;
                }
            }
        }
        core::logger::log(core::logger::level::note, "keyframe %d: %zu keyframes, %zu landmarks, %d new landmarks, %zu active tracks, %zu lines, %d new lines.", frame_current.id, this->keyframe_ids_.size(), this->reconstruction.landmarks.size(), new_landmarks, active_track_count, this->reconstruction.line_landmarks.size(), new_line_landmarks);
        core::logger::log(core::logger::level::info, "Map status: %zu frames, %zu landmarks | active tracks: %zu, tracks with landmarks: %zu, new landmarks this frame: %d.", this->reconstruction.frames.size(), this->reconstruction.landmarks.size(), active_track_count, tracks_with_landmarks, new_landmarks);
        this->report_scale_sentinel(frame_current.id, new_landmark_depths);
        core::logger::log(
            core::logger::level::debug,
            "%s:\n{\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f },\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f },\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f },\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f }\n}",
            "Current pose",
            frame_current.rotation[0][0],
            frame_current.rotation[0][1],
            frame_current.rotation[0][2],
            frame_current.translation[0],
            frame_current.rotation[1][0],
            frame_current.rotation[1][1],
            frame_current.rotation[1][2],
            frame_current.translation[1],
            frame_current.rotation[2][0],
            frame_current.rotation[2][1],
            frame_current.rotation[2][2],
            frame_current.translation[2],
            0.0,
            0.0,
            0.0,
            1.0
        );
        this->update_anchored_frames();
        if (globally_adjusted) {
            this->refit_anchored_frames();
        }

        this->previous_tracked_frame_id_ = this->last_tracked_frame_id_;
        this->last_tracked_frame_id_ = frame_current.id;
    }
};

#endif // ZEROSLAM_SLAM_HPP
