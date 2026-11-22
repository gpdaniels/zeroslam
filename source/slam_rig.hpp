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
#ifndef ZEROSLAM_SLAM_RIG_HPP
#define ZEROSLAM_SLAM_RIG_HPP

// The part of the system that runs a rig of several cameras, declared with the rest of the system in slam.hpp, which
// includes this file at its end. The reference camera's frames go through the system as a single camera's do; the other
// cameras' images are views of those frames (mapping::frame::views), each camera tracking its own features with a tracker
// of its own, linking its tracks to the map's landmarks by projection, and adding its observations to the keyframes. The
// rig's pose is estimated from every camera's tracks, a metric rig (mapping::map::metric_scale) initialises from one frame
// whose cameras share their view, and new landmarks are triangulated across the rig's baseline and along each camera's own
// tracks. With one camera none of this is reached.

#include "slam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

inline void slam::set_rig(const mapping::rig& rig) {
    this->reconstruction.sensor_rig = rig;
    const size_t cameras = rig.camera_count();
    this->view_trackers_.clear();
    this->view_trackers_.resize((cameras > 1) ? (cameras - 1) : 0);
    this->view_laplacian_history_.assign(this->view_trackers_.size(), std::vector<double>());
    this->view_calibrations_.assign(this->view_trackers_.size(), mapping::calibration());
    this->declared_refined_.assign(rig.size(), static_cast<unsigned char>(0));
    for (size_t index = 0; index < rig.size(); ++index) {
        this->declared_refined_[index] = rig.at(index).refined ? 1 : 0;
    }
    this->apply_extrinsics_setting();
}

inline void slam::process_rig_frame(const std::vector<sensor::model>& cameras, const std::vector<image::image>& images) {
    if (cameras.empty() || (cameras.size() != images.size())) {
        return;
    }
    if (cameras.size() == 1) {
        this->process(cameras[0], images[0], nullptr, nullptr, 0);
        return;
    }
    this->process(cameras[0], images[0], &cameras[1], &images[1], cameras.size() - 1);
}

inline feature::tracker::tracker& slam::tracker_of(const int camera) {
    return (camera == 0) ? this->tracker_ : this->view_trackers_[static_cast<size_t>(camera - 1)];
}

inline std::vector<feature::tracker::tracker::track*> slam::point_tracks_of(const int camera) {
    return (camera == 0) ? this->active_point_tracks() : this->view_trackers_[static_cast<size_t>(camera - 1)].active_tracks();
}

inline std::vector<feature::tracker::tracker::track*> slam::all_point_tracks_of(const int camera) {
    return (camera == 0) ? this->all_point_tracks() : this->view_trackers_[static_cast<size_t>(camera - 1)].all_tracks();
}

inline size_t slam::tracked_views(const mapping::frame& frame) const {
    return math::min(frame.views.size(), this->view_trackers_.size());
}

inline void slam::apply_extrinsics_setting() {
    mapping::rig& rig = this->reconstruction.sensor_rig;
    for (size_t index = 0; index < rig.size(); ++index) {
        bool refined = (index < this->declared_refined_.size()) && (this->declared_refined_[index] != 0);
        if (this->frontend.extrinsics == mapping::frame::settings::extrinsics_kind::fixed) {
            refined = false;
        }
        else if (this->frontend.extrinsics == mapping::frame::settings::extrinsics_kind::refine) {
            refined = rig.at(index).kind == mapping::rig::sensor_kind::camera;
        }
        rig.set_refined(index, refined);
    }
}

inline void slam::add_views(mapping::frame& frame, const sensor::model* const view_cameras, const image::image* const view_images, const size_t view_count) {
    if ((view_cameras == nullptr) || (view_images == nullptr)) {
        return;
    }
    const size_t views = math::min(view_count, this->view_trackers_.size());
    frame.views.reserve(views);
    for (size_t index = 0; index < views; ++index) {
        // While the caller gives the camera it gave before, the view takes it as the adjustments have refined it.
        this->view_calibrations_[index].set_level(this->frontend.calibration);
        mapping::frame view(frame.id, this->view_calibrations_[index].camera_for(view_cameras[index]), view_images[index], this->frontend);
        if (this->frontend.blur_weighting) {
            std::vector<double>& history = this->view_laplacian_history_[index];
            history.push_back(view.laplacian_variance);
            if (history.size() > slam::blur_history_frames) {
                history.erase(history.begin());
            }
            std::vector<double> sorted(history);
            std::sort(sorted.begin(), sorted.end());
            const double median = sorted[sorted.size() / 2];
            if ((median > 0.0) && (view.laplacian_variance > 0.0)) {
                view.measurement_sigma = math::max(1.0, math::min(slam::blur_sigma_maximum, math::sqrt(median / view.laplacian_variance)));
            }
        }
        core::logger::log(core::logger::level::info, "Detected Features: camera %zu, %zu consumed.", index + 1, view.keypoints.size());
        frame.views.push_back(static_cast<mapping::frame&&>(view));
    }
    this->apply_extrinsics_setting();
}

inline std::vector<mapping::calibration::refinement> slam::view_refinement_requests() {
    std::vector<mapping::calibration::refinement> requests(this->view_calibrations_.size());
    for (size_t index = 0; index < this->view_calibrations_.size(); ++index) {
        // A camera not ready to refine gives an empty request, which its refinement in the adjustment leaves alone.
        if (!mapping::intrinsics_refinement::make_request(this->view_calibrations_[index], this->reconstruction.frames, this->keyframe_ids_, requests[index])) {
            requests[index] = mapping::calibration::refinement();
        }
    }
    return requests;
}

inline void slam::take_view_refinements(const std::vector<mapping::calibration::refinement>& requests) {
    for (size_t index = 0; (index < requests.size()) && (index < this->view_calibrations_.size()); ++index) {
        if (requests[index].refined) {
            this->view_calibrations_[index].update(requests[index]);
        }
    }
}

inline void slam::update_view_trackers(mapping::frame& frame) {
    for (size_t index = 0; index < this->tracked_views(frame); ++index) {
        feature::tracker::tracker::options options = this->tracker_.get_options();
        const feature::tracker::tracker::options defaults;
        options.association = (this->frontend.association == mapping::frame::settings::association_kind::match) ? feature::tracker::tracker::association_kind::descriptor : ((this->frontend.association == mapping::frame::settings::association_kind::both) ? feature::tracker::tracker::association_kind::both : feature::tracker::tracker::association_kind::optical_flow);
        options.match_hamming = this->hamming_bound(defaults.match_hamming);
        options.reacquire_hamming = this->hamming_bound(defaults.reacquire_hamming);
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
        this->view_trackers_[index].set_options(options);
        mapping::frame& view = frame.views[index];
        this->view_trackers_[index].update(frame.id, view.image_pyramid, view.keypoints, view.descriptors);
    }
}

inline void slam::unlink_dangling_view_tracks() {
    for (feature::tracker::tracker& view_tracker : this->view_trackers_) {
        for (feature::tracker::tracker::track* const t : view_tracker.active_tracks()) {
            if ((t->landmark_id >= 0) && (this->reconstruction.landmarks.count(t->landmark_id) == 0)) {
                t->landmark_id = -1;
            }
        }
    }
}

inline void slam::unlink_view_tracks_before(const int frame_id) {
    for (feature::tracker::tracker& view_tracker : this->view_trackers_) {
        for (feature::tracker::tracker::track* const t : view_tracker.all_tracks()) {
            t->landmark_id = -1;
            t->outliers = 0;
            size_t keep = 0;
            for (size_t h = 0; h < t->history.size(); ++h) {
                if (t->history[h].frame_id >= frame_id) {
                    t->history[keep] = t->history[h];
                    ++keep;
                }
            }
            t->history.resize(keep);
        }
    }
}

inline void slam::relink_view_tracks(const int from_landmark_id, const int to_landmark_id) {
    for (feature::tracker::tracker& view_tracker : this->view_trackers_) {
        for (feature::tracker::tracker::track* const t : view_tracker.all_tracks()) {
            if (t->landmark_id == from_landmark_id) {
                t->landmark_id = to_landmark_id;
            }
        }
    }
}

inline void slam::trim_view_histories(const int frame_id, const bool at_keyframe) {
    for (feature::tracker::tracker& view_tracker : this->view_trackers_) {
        for (feature::tracker::tracker::track* const t : (at_keyframe ? view_tracker.all_tracks() : view_tracker.active_tracks())) {
            size_t keep = 0;
            for (size_t h = 0; h < t->history.size(); ++h) {
                const int observed_id = t->history[h].frame_id;
                // As the reference camera's tracks keep theirs (trim_track_histories, and at a keyframe what is read back).
                const bool read_back = at_keyframe ? ((observed_id >= this->last_keyframe_id_) || (this->keyframe_ids_.count(observed_id) != 0)) : ((observed_id == frame_id) || (observed_id == this->init_anchor_frame_id_) || (observed_id == this->reckoning_.posed_frame_id) || (observed_id == this->reckoning_.frame_id) || (observed_id == this->probation_origin_reckoning_.posed_frame_id) || (observed_id == this->probation_origin_reckoning_.frame_id) || (this->keyframe_ids_.count(observed_id) != 0));
                if (read_back) {
                    t->history[keep] = t->history[h];
                    ++keep;
                }
            }
            t->history.resize(keep);
        }
    }
}

inline double slam::correspondence_sigma(const mapping::frame& frame_current, const pose_correspondence& match) {
    const mapping::frame* const view = mapping::map::view_of(frame_current, match.camera);
    return (view != nullptr) ? view->measurement_sigma : frame_current.measurement_sigma;
}

inline optimisation::edge* slam::add_view_pose_edge(optimisation::factor_graph& ba, optimisation::vertex* const pose_node, optimisation::vertex* const point_node, const mapping::frame& frame_current, const pose_correspondence& match, const bool weighted, const double scale, const optimisation::loss& lossfunction, std::vector<optimisation::vertex*>& extrinsic_nodes) {
    const mapping::frame* const view = mapping::map::view_of(frame_current, match.camera);
    const size_t camera = static_cast<size_t>(match.camera);
    if ((view == nullptr) || (camera >= this->reconstruction.sensor_rig.camera_count())) {
        return nullptr;
    }
    if (extrinsic_nodes.size() <= camera) {
        extrinsic_nodes.resize(camera + 1, nullptr);
    }
    if (extrinsic_nodes[camera] == nullptr) {
        optimisation::vertex extrinsic = this->reconstruction.sensor_rig.extrinsic_vertex(this->reconstruction.sensor_rig.camera(camera));
        extrinsic.set_fixed(true);
        extrinsic_nodes[camera] = ba.add_vertex(static_cast<optimisation::vertex&&>(extrinsic));
    }
    mapping::map::observation recorded{ frame_current.id, static_cast<size_t>(-1), math::matrix<double, 2, 1>({ match.pixel[0], match.pixel[1] }), 0 };
    recorded.camera = match.camera;
    optimisation::edge factor = weighted ? this->reconstruction.make_observation_edge(frame_current, *match.landmark, recorded, lossfunction, 1.0) : optimisation::edge{ optimisation::edges::rig_reprojection(sensor::camera::model<double>(view->camera)) };
    if (!weighted) {
        factor.set_observation(math::matrix<double, 0, 0>(2, 1, &match.pixel[0]));
        factor.set_information(math::matrix<double, 0, 0>::identity(2, 2) * (1.0 / (scale * scale)));
        factor.set_loss(lossfunction);
    }
    if (!mapping::map::add_observation_vertices(factor, pose_node, point_node, extrinsic_nodes, match.camera)) {
        return nullptr;
    }
    if (!weighted) {
        ba.add_edge(static_cast<optimisation::edge&&>(factor));
        return nullptr;
    }
    factor.compute_jacobians();
    factor.set_information(math::matrix<double, 0, 0>(2, 2, match.landmark->observation_information(factor.get_jacobians()[1], view->measurement_sigma * scale).data()));
    return ba.add_edge(static_cast<optimisation::edge&&>(factor));
}

inline bool slam::view_correspondence_agrees(const mapping::frame& frame_current, const pose_correspondence& match, const math::matrix<double, 3, 3>& rotation, const math::matrix<double, 3, 1>& translation, const double scale) const {
    const mapping::frame* const view = mapping::map::view_of(frame_current, match.camera);
    if (view == nullptr) {
        return false;
    }
    math::matrix<double, 3, 3> camera_rotation;
    math::matrix<double, 3, 1> camera_translation;
    this->reconstruction.camera_pose(match.camera, rotation, translation, camera_rotation, camera_translation);
    const math::matrix<double, 3, 1> seen = (camera_rotation * match.location) + camera_translation;
    double projected[2];
    if (!view->camera.project(seen.data(), &projected[0])) {
        return false;
    }
    const double dx = projected[0] - match.pixel[0];
    const double dy = projected[1] - match.pixel[1];
    const double sigma = view->measurement_sigma * scale;
    return ((dx * dx) + (dy * dy)) <= 5.991 * sigma * sigma;
}

inline void slam::append_view_correspondences(mapping::frame& frame_current, std::vector<pose_correspondence>& correspondences) {
    const size_t views = this->tracked_views(frame_current);
    if (views == 0) {
        return;
    }
    this->reconstruction.pose_views(frame_current);
    for (size_t index = 0; index < views; ++index) {
        const mapping::frame& view = frame_current.views[index];
        double intrinsics[4];
        slam::pinhole_intrinsics(view, intrinsics);
        for (feature::tracker::tracker::track* const t : this->view_trackers_[index].active_tracks()) {
            if (t->landmark_id < 0) {
                continue;
            }
            const std::unordered_map<int, mapping::point>::const_iterator landmark = this->reconstruction.landmarks.find(t->landmark_id);
            if ((landmark == this->reconstruction.landmarks.end()) || !slam::prior_sees(view, intrinsics, landmark->second.location)) {
                continue;
            }
            correspondences.push_back({ { static_cast<double>(t->x), static_cast<double>(t->y) }, landmark->second.location, t, &landmark->second, static_cast<int>(index + 1) });
        }
    }
}

inline bool slam::pose_from_rig_tracks(mapping::frame& frame_current, size_t& correspondences) {
    std::vector<estimation::robust::solver::generalised_p3p<double>::correspondence> matches;
    const size_t views = this->tracked_views(frame_current);
    std::vector<math::se3<double>> camera_from_reference(views + 1, math::se3<double>::identity());
    for (size_t camera = 1; camera <= views; ++camera) {
        camera_from_reference[camera] = this->reconstruction.sensor_rig.reference_from_sensor(this->reconstruction.sensor_rig.camera(camera)).inverse();
    }
    for (size_t camera = 0; camera <= views; ++camera) {
        const mapping::frame& view = (camera == 0) ? frame_current : frame_current.views[camera - 1];
        for (const feature::tracker::tracker::track* const t : this->point_tracks_of(static_cast<int>(camera))) {
            if (t->landmark_id < 0) {
                continue;
            }
            const std::unordered_map<int, mapping::point>::const_iterator landmark = this->reconstruction.landmarks.find(t->landmark_id);
            if ((landmark == this->reconstruction.landmarks.end()) || (landmark->second.uncertainty == mapping::point::uncertainty_kind::unbounded)) {
                continue;
            }
            const double pixel[2] = { static_cast<double>(t->x), static_cast<double>(t->y) };
            double ray[3];
            if (!view.camera.unproject(&pixel[0], &ray[0]) || !sensor::camera::on_image_plane(&ray[0])) {
                continue;
            }
            estimation::robust::solver::generalised_p3p<double>::correspondence match;
            match.bearing[0] = ray[0] / ray[2];
            match.bearing[1] = ray[1] / ray[2];
            match.point[0] = landmark->second.location[0];
            match.point[1] = landmark->second.location[1];
            match.point[2] = landmark->second.location[2];
            match.camera = camera;
            matches.push_back(match);
        }
    }
    correspondences = matches.size();
    if (matches.size() < slam::pose_bounded_minimum) {
        core::logger::log(core::logger::level::debug, "Pose from the rig's tracks: %zu landmarks with a depth, too few for a perspective solve.", matches.size());
        return false;
    }
    std::vector<size_t> inliers(matches.size());
    size_t inliers_size = 0;
    estimation::robust::solver::generalised_p3p<double>::model model{};
    if (!estimation::robust::solver::generalised_p3p<double>::solve(matches.data(), matches.size(), camera_from_reference.data(), camera_from_reference.size(), inliers.data(), inliers_size, model, slam::pose_residual_threshold(frame_current)) || (inliers_size < slam::pose_inliers_minimum)) {
        return false;
    }
    frame_current.rotation = math::matrix<double, 3, 3>({ { model.rotation[0][0], model.rotation[0][1], model.rotation[0][2] }, { model.rotation[1][0], model.rotation[1][1], model.rotation[1][2] }, { model.rotation[2][0], model.rotation[2][1], model.rotation[2][2] } });
    frame_current.translation = math::matrix<double, 3, 1>({ { model.translation[0] }, { model.translation[1] }, { model.translation[2] } });
    const math::matrix<double, 3, 3> consensus_rotation = frame_current.rotation;
    const math::matrix<double, 3, 1> consensus_translation = frame_current.translation;
    size_t refined_inliers = 0;
    if (!this->refine_pose_from_prior(frame_current, refined_inliers)) {
        frame_current.rotation = consensus_rotation;
        frame_current.translation = consensus_translation;
        this->reconstruction.pose_views(frame_current);
    }
    core::logger::log(core::logger::level::info, "Inliers: %zu inliers in the rig's pose estimation out of %zu correspondences (proposed by camera %zu), %zu after refinement.", inliers_size, matches.size(), model.camera, refined_inliers);
    return true;
}

inline void slam::track_view_local_maps(mapping::frame& frame_current) {
    const size_t views = this->tracked_views(frame_current);
    if (views == 0) {
        return;
    }
    this->reconstruction.pose_views(frame_current);
    for (size_t index = 0; index < views; ++index) {
        this->track_local_map_in(frame_current, static_cast<int>(index + 1), slam::local_map_projection_radius, false, nullptr, false);
    }
}

inline void slam::add_view_observations(mapping::frame& frame_current) {
    const size_t views = this->tracked_views(frame_current);
    this->reconstruction.pose_views(frame_current);
    for (size_t index = 0; index < views; ++index) {
        const int camera = static_cast<int>(index + 1);
        const size_t strays = this->unlink_strays(frame_current.views[index], camera);
        size_t observed = 0;
        for (feature::tracker::tracker::track* const t : this->view_trackers_[index].active_tracks()) {
            if (t->landmark_id < 0) {
                continue;
            }
            const std::unordered_map<int, mapping::point>::iterator landmark = this->reconstruction.landmarks.find(t->landmark_id);
            if (landmark == this->reconstruction.landmarks.end()) {
                continue;
            }
            this->reconstruction.add_observation(frame_current.id, landmark->second, static_cast<double>(t->x), static_cast<double>(t->y), t->octave, t->measurement_octave, camera);
            if (t->descriptor_valid) {
                this->add_landmark_descriptors(landmark->second, frame_current.views[index], t->x, t->y, t->octave, t->descriptor);
            }
            ++observed;
        }
        core::logger::log(core::logger::level::info, "Keyframe %d camera %d: %zu landmarks observed, %zu tracks the pose puts off their landmarks unlinked.", frame_current.id, camera, observed, strays);
    }
}

inline size_t slam::triangulate_across_cameras(mapping::frame& frame_current, const bool create) {
    const size_t views = this->tracked_views(frame_current);
    if (views == 0) {
        return 0;
    }
    this->reconstruction.pose_views(frame_current);
    const std::vector<feature::tracker::tracker::track*> primary = this->active_point_tracks();
    const unsigned int hamming_maximum = this->hamming_bound(slam::rig_match_hamming);
    const image::image& colours = frame_current.image_pyramid[0];
    std::vector<unsigned char> taken(primary.size(), static_cast<unsigned char>(0));
    size_t made = 0;
    for (size_t index = 0; index < views; ++index) {
        const int camera = static_cast<int>(index + 1);
        mapping::frame& view = frame_current.views[index];
        const size_t sensor = this->reconstruction.sensor_rig.camera(static_cast<size_t>(camera));
        const math::se3<double> camera_from_reference = this->reconstruction.sensor_rig.reference_from_sensor(sensor).inverse();
        const math::matrix<double, 3, 1> baseline = camera_from_reference.translation();
        if (math::sqrt(baseline.get_length_squared()) < mapping::rig::metric_baseline_minimum) {
            continue;
        }
        const math::matrix<double, 3, 3> rotation_cr = camera_from_reference.rotation().get_matrix();
        double view_intrinsics[4];
        slam::pinhole_intrinsics(view, view_intrinsics);
        // An extrinsic still being refined is not trusted to the pixel.
        const double band = slam::rig_epipolar_pixels * (this->reconstruction.sensor_rig.refines(sensor) ? slam::rig_epipolar_refined_factor : 1.0);
        const double tolerance = band / math::max(view_intrinsics[0], view_intrinsics[1]);
        std::vector<math::matrix<double, 3, 1>> rays(view.keypoints.size());
        std::vector<unsigned char> usable(view.keypoints.size(), static_cast<unsigned char>(0));
        for (size_t k = 0; k < view.keypoints.size(); ++k) {
            const double pixel[2] = { static_cast<double>(view.keypoints[k].x), static_cast<double>(view.keypoints[k].y) };
            double ray[3];
            if (!view.camera.unproject(&pixel[0], &ray[0])) {
                continue;
            }
            const double length = math::sqrt((ray[0] * ray[0]) + (ray[1] * ray[1]) + (ray[2] * ray[2]));
            rays[k] = math::matrix<double, 3, 1>({ ray[0] / length, ray[1] / length, ray[2] / length });
            usable[k] = static_cast<unsigned char>(1);
        }
        std::vector<unsigned char> claimed(view.keypoints.size(), static_cast<unsigned char>(0));
        const std::vector<feature::tracker::tracker::track*> view_tracks = this->view_trackers_[index].active_tracks();
        const math::matrix<double, 3, 1> centre_reference = -(math::transpose(frame_current.rotation) * frame_current.translation);
        const math::matrix<double, 3, 1> centre_view = -(math::transpose(view.rotation) * view.translation);
        for (size_t p = 0; p < primary.size(); ++p) {
            feature::tracker::tracker::track* const t = primary[p];
            if ((t->landmark_id >= 0) || !t->descriptor_valid || (taken[p] != 0)) {
                continue;
            }
            const double pixel[2] = { static_cast<double>(t->x), static_cast<double>(t->y) };
            math::matrix<double, 3, 1> ray_reference;
            if (!frame_current.camera.unproject(&pixel[0], ray_reference.data())) {
                continue;
            }
            // The plane through both camera centres and the point, in the other camera's frame.
            const math::matrix<double, 3, 1> direction = rotation_cr * ray_reference;
            math::matrix<double, 3, 1> normal({ (baseline[1] * direction[2]) - (baseline[2] * direction[1]), (baseline[2] * direction[0]) - (baseline[0] * direction[2]), (baseline[0] * direction[1]) - (baseline[1] * direction[0]) });
            const double normal_length = math::sqrt(normal.get_length_squared());
            if (normal_length < 1.0e-12) {
                continue;
            }
            normal = normal * (1.0 / normal_length);
            size_t best = static_cast<size_t>(-1);
            unsigned int best_distance = static_cast<unsigned int>(feature::descriptor::stored_bits) + 1u;
            unsigned int second_distance = best_distance;
            for (size_t k = 0; k < view.keypoints.size(); ++k) {
                if ((usable[k] == 0) || (claimed[k] != 0)) {
                    continue;
                }
                const double off_plane = (normal[0] * rays[k][0]) + (normal[1] * rays[k][1]) + (normal[2] * rays[k][2]);
                if (math::abs(off_plane) > tolerance) {
                    continue;
                }
                const int octave_offset = view.keypoints[k].octave - t->octave;
                if (((octave_offset < 0) ? -octave_offset : octave_offset) > slam::local_map_octave_window) {
                    continue;
                }
                const unsigned int distance = match::distance::hamming::distance(t->descriptor, view.descriptors[k]);
                if (distance < best_distance) {
                    second_distance = best_distance;
                    best_distance = distance;
                    best = k;
                }
                else if (distance < second_distance) {
                    second_distance = distance;
                }
            }
            if ((best == static_cast<size_t>(-1)) || (best_distance > hamming_maximum) || !(static_cast<float>(best_distance) < slam::rig_match_ratio * static_cast<float>(second_distance))) {
                continue;
            }
            const feature::point& detection = view.keypoints[best];
            const double detection_pixel[2] = { static_cast<double>(detection.x), static_cast<double>(detection.y) };
            math::matrix<double, 3, 1> ray_view;
            if (!view.camera.unproject(&detection_pixel[0], ray_view.data())) {
                continue;
            }
            math::matrix<double, 3, 1> point;
            if (!geometry::triangulation::linear_least_squares<double>::triangulate(ray_reference, frame_current.get_pose(), ray_view, view.get_pose(), point)) {
                continue;
            }
            if ((slam::parallax_angle(point, centre_reference, centre_view) * (180.0 / 3.14159265358979323846)) < slam::rig_parallax_degrees_minimum) {
                continue;
            }
            const math::matrix<double, 3, 1> mapped_reference = (frame_current.rotation * point) + frame_current.translation;
            const math::matrix<double, 3, 1> mapped_view = (view.rotation * point) + view.translation;
            math::matrix<double, 2, 1> reprojected_reference;
            math::matrix<double, 2, 1> reprojected_view;
            // Whether the point is in front of each camera is the camera model's valid region, which project tests.
            if (!frame_current.camera.project(mapped_reference.data(), reprojected_reference.data()) || !view.camera.project(mapped_view.data(), reprojected_view.data())) {
                continue;
            }
            const double bound_squared = band * band;
            if (((reprojected_reference - math::matrix<double, 2, 1>({ pixel[0], pixel[1] })).get_length_squared() > bound_squared) || ((reprojected_view - math::matrix<double, 2, 1>({ detection_pixel[0], detection_pixel[1] })).get_length_squared() > bound_squared)) {
                continue;
            }
            claimed[best] = static_cast<unsigned char>(1);
            taken[p] = static_cast<unsigned char>(1);
            if (!create) {
                ++made;
                continue;
            }
            const size_t colour_col = static_cast<size_t>(math::max(0, math::min(static_cast<int>(colours.get_cols()) - 1, static_cast<int>(t->x))));
            const size_t colour_row = static_cast<size_t>(math::max(0, math::min(static_cast<int>(colours.get_rows()) - 1, static_cast<int>(t->y))));
            const double colour = static_cast<double>(colours.get_data()[colour_row * colours.get_cols() + colour_col]) / 255.0;
            mapping::point landmark(this->reconstruction.allocate_landmark_id(), point, math::matrix<double, 3, 1>{ { colour, colour, colour } });
            if (this->frontend.inverse_depth && !landmark.anchor(frame_current.rotation, frame_current.translation)) {
                continue;
            }
            ++made;
            this->add_landmark_descriptors(landmark, frame_current, t->x, t->y, t->octave, t->descriptor);
            this->reconstruction.add_landmark(landmark);
            t->landmark_id = landmark.id;
            for (const feature::tracker::tracker::observation& obs : t->history) {
                if ((this->keyframe_ids_.count(obs.frame_id) != 0) && (this->reconstruction.frames.count(obs.frame_id) != 0)) {
                    this->reconstruction.add_observation(obs.frame_id, landmark, static_cast<double>(obs.x), static_cast<double>(obs.y), t->octave, obs.measurement_octave);
                }
            }
            this->reconstruction.add_observation(frame_current.id, landmark, detection_pixel[0], detection_pixel[1], detection.octave, detection.octave, camera);
            feature::tracker::tracker::track* follower = nullptr;
            double nearest = slam::rig_track_link_pixels * slam::rig_track_link_pixels;
            for (feature::tracker::tracker::track* const candidate : view_tracks) {
                if (candidate->landmark_id >= 0) {
                    continue;
                }
                const double dx = static_cast<double>(candidate->x) - detection_pixel[0];
                const double dy = static_cast<double>(candidate->y) - detection_pixel[1];
                if (((dx * dx) + (dy * dy)) < nearest) {
                    nearest = (dx * dx) + (dy * dy);
                    follower = candidate;
                }
            }
            if (follower != nullptr) {
                follower->landmark_id = landmark.id;
            }
            else {
                this->view_trackers_[index].spawn(frame_current.id, detection, view.descriptors[best]).landmark_id = landmark.id;
            }
            mapping::point& stored = this->reconstruction.landmarks.at(landmark.id);
            this->reconstruction.estimate_landmark_uncertainty(stored);
            if (stored.inverse_depth && (stored.uncertainty == mapping::point::uncertainty_kind::estimated)) {
                const double rho = stored.inverse_parameters[2];
                const double sigma = math::sqrt(math::max(0.0, stored.covariance[2][2]));
                if (rho < mapping::point::depth_significance_minimum * sigma) {
                    stored.set_unbounded(math::matrix<double, 3, 1>({ 0.0, 0.0, 1.0 }));
                    ++this->admitted_unbounded;
                }
            }
        }
    }
    return made;
}

inline size_t slam::triangulate_view_tracks(mapping::frame& frame_current) {
    const size_t views = this->tracked_views(frame_current);
    size_t made = 0;
    const size_t component = this->map_component(frame_current.id);
    for (size_t index = 0; index < views; ++index) {
        const int camera = static_cast<int>(index + 1);
        const mapping::frame& view = frame_current.views[index];
        for (feature::tracker::tracker::track* const t : this->view_trackers_[index].active_tracks()) {
            if ((t->landmark_id >= 0) || (t->length < 2)) {
                continue;
            }
            const feature::tracker::tracker::observation* reference = nullptr;
            const mapping::frame* reference_frame = nullptr;
            for (const feature::tracker::tracker::observation& obs : t->history) {
                if ((obs.frame_id == frame_current.id) || (this->keyframe_ids_.count(obs.frame_id) == 0) || (this->map_component(obs.frame_id) != component)) {
                    continue;
                }
                const std::unordered_map<int, mapping::frame>::const_iterator found = this->reconstruction.frames.find(obs.frame_id);
                if ((found == this->reconstruction.frames.end()) || (found->second.views.size() <= index)) {
                    continue;
                }
                reference = &obs;
                reference_frame = &found->second;
                break;
            }
            if (reference == nullptr) {
                continue;
            }
            mapping::frame reference_view_pose;
            this->reconstruction.pose_view(*reference_frame, reference_view_pose, camera);
            const sensor::model& reference_camera = reference_frame->views[index].camera;
            const double reference_pixel[2] = { static_cast<double>(reference->x), static_cast<double>(reference->y) };
            const double current_pixel[2] = { static_cast<double>(t->x), static_cast<double>(t->y) };
            math::matrix<double, 3, 1> ray_reference;
            math::matrix<double, 3, 1> ray_current;
            if (!reference_camera.unproject(&reference_pixel[0], ray_reference.data()) || !view.camera.unproject(&current_pixel[0], ray_current.data())) {
                continue;
            }
            math::matrix<double, 3, 1> point;
            if (!geometry::triangulation::linear_least_squares<double>::triangulate(ray_current, view.get_pose(), ray_reference, reference_view_pose.get_pose(), point)) {
                continue;
            }
            const math::matrix<double, 3, 1> centre_current = -(math::transpose(view.rotation) * view.translation);
            const math::matrix<double, 3, 1> centre_reference = -(math::transpose(reference_view_pose.rotation) * reference_view_pose.translation);
            if ((slam::parallax_angle(point, centre_current, centre_reference) * (180.0 / 3.14159265358979323846)) < slam::triangulation_parallax_degrees_minimum) {
                continue;
            }
            const math::matrix<double, 3, 1> mapped_current = (view.rotation * point) + view.translation;
            const math::matrix<double, 3, 1> mapped_reference = (reference_view_pose.rotation * point) + reference_view_pose.translation;
            math::matrix<double, 2, 1> reprojected_current;
            math::matrix<double, 2, 1> reprojected_reference;
            if (!view.camera.project(mapped_current.data(), reprojected_current.data()) || !reference_camera.project(mapped_reference.data(), reprojected_reference.data())) {
                continue;
            }
            const double bound_squared = slam::rig_epipolar_pixels * slam::rig_epipolar_pixels;
            if (((reprojected_current - math::matrix<double, 2, 1>({ current_pixel[0], current_pixel[1] })).get_length_squared() > bound_squared) || ((reprojected_reference - math::matrix<double, 2, 1>({ reference_pixel[0], reference_pixel[1] })).get_length_squared() > bound_squared)) {
                continue;
            }
            mapping::point landmark(this->reconstruction.allocate_landmark_id(), point, math::matrix<double, 3, 1>{ { 0.5, 0.5, 0.5 } });
            if (this->frontend.inverse_depth && !landmark.anchor(reference_view_pose.rotation, reference_view_pose.translation)) {
                continue;
            }
            if (t->descriptor_valid) {
                this->add_landmark_descriptors(landmark, view, t->x, t->y, t->octave, t->descriptor);
            }
            this->reconstruction.add_landmark(landmark);
            t->landmark_id = landmark.id;
            for (const feature::tracker::tracker::observation& obs : t->history) {
                const std::unordered_map<int, mapping::frame>::const_iterator observing = this->reconstruction.frames.find(obs.frame_id);
                if ((this->keyframe_ids_.count(obs.frame_id) != 0) && (observing != this->reconstruction.frames.end()) && (observing->second.views.size() > index)) {
                    this->reconstruction.add_observation(obs.frame_id, landmark, static_cast<double>(obs.x), static_cast<double>(obs.y), t->octave, obs.measurement_octave, camera);
                }
            }
            this->reconstruction.estimate_landmark_uncertainty(this->reconstruction.landmarks.at(landmark.id));
            ++made;
        }
    }
    return made;
}

inline void slam::count_frames_once(const std::vector<mapping::map::observation>& landmark_observations, std::vector<int>& observing_frames) {
    for (const mapping::map::observation& observation : landmark_observations) {
        if (observation.camera != 0) {
            std::sort(observing_frames.begin(), observing_frames.end());
            observing_frames.erase(std::unique(observing_frames.begin(), observing_frames.end()), observing_frames.end());
            return;
        }
    }
}

inline math::sim3<double> slam::loop_measurement(const math::sim3<double>& relative, const bool metric_scale) {
    return metric_scale ? math::sim3<double>(relative.transformation(), 1.0) : relative;
}

inline bool slam::initialise_from_rig(mapping::frame& frame_current) {
    if (this->reinitialising_) {
        const std::unordered_map<int, mapping::frame>::const_iterator anchor = this->reconstruction.frames.find(this->init_anchor_frame_id_);
        if ((anchor != this->reconstruction.frames.end()) && (anchor->first != frame_current.id)) {
            frame_current.rotation = anchor->second.rotation;
            frame_current.translation = anchor->second.translation;
        }
    }
    const size_t matched = this->triangulate_across_cameras(frame_current, false);
    const bool accept = matched >= slam::rig_initialisation_landmarks_minimum;
    core::logger::log(core::logger::level::info, "Rig init (frame %d): %zu points seen across the rig's baseline (minimum %zu) -> %s", frame_current.id, matched, slam::rig_initialisation_landmarks_minimum, accept ? "ACCEPTED" : "REJECTED");
    if (!accept) {
        return false;
    }
    // The frame is the map's first keyframe, its landmarks triangulated across the rig as at every keyframe.
    if (this->reinitialising_ && (this->init_anchor_frame_id_ != frame_current.id) && (this->keyframe_ids_.count(this->init_anchor_frame_id_) == 0)) {
        this->reconstruction.frames.erase(this->init_anchor_frame_id_);
    }
    this->init_anchor_frame_id_ = frame_current.id;
    // The map's scale is the rig's baseline from here on. A rig whose cameras share no view initialises as one camera does,
    // and its scale is then a monocular map's, held by the scale gauge: across cameras that share no view the baseline
    // measures the scale only through the rig's turns, and not at all while the rig moves without turning.
    this->reconstruction.metric_scale = true;
    if (this->keyframe_ids_.empty()) {
        this->reconstruction.gauge_frame_id = frame_current.id;
    }
    this->state_ = tracking_state::tracking;
    this->blind_frames_ = 0;
    this->reckon_from(frame_current);
    this->relocalised_keyframe_id_ = -1;
    if (this->reinitialising_) {
        this->start_submap(frame_current);
    }
    return true;
}

#endif // ZEROSLAM_SLAM_RIG_HPP
