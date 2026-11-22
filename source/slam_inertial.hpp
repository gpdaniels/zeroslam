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
#ifndef ZEROSLAM_SLAM_INERTIAL_HPP
#define ZEROSLAM_SLAM_INERTIAL_HPP

// The part of the system that runs an imu with the cameras, declared with the rest of the system in slam.hpp, which includes this
// file at its end. The imu is a sensor of the rig, posed on the body like the cameras, its samples buffered in time order with
// the frames (mapping::inertial). Each keyframe preintegrates the samples since the keyframe before it; once a window of them
// has given the direction of gravity, the scale where the map has none (a monocular map), the velocities and the biases
// (optimisation::inertial_initialisation), the map is made metric, and every adjustment holds consecutive keyframes with
// inertial and bias random walk edges, refined twice more as the window grows. The imu predicts each frame's pose where the
// tracks do not give it and carries the frames through a short loss of the tracks, and a loop on an inertial map keeps every
// keyframe's roll and pitch. Without an imu none of this is reached.

#include "optimisation/edges/gravity_alignment.hpp"
#include "optimisation/inertial_initialisation.hpp"
#include "slam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

inline void slam::set_imu(const int sensor_id, const sensor::imu::parameters& noise, const math::se3<double>& body_from_imu, const bool refined) {
    mapping::rig& rig = this->reconstruction.sensor_rig;
    size_t index = 0;
    if (!rig.find(sensor_id, index)) {
        index = rig.add(sensor_id, mapping::rig::sensor_kind::imu, body_from_imu, refined);
    }
    if (this->declared_refined_.size() < rig.size()) {
        this->declared_refined_.resize(rig.size(), static_cast<unsigned char>(0));
    }
    this->declared_refined_[index] = refined ? 1 : 0;
    this->apply_extrinsics_setting();
    this->reconstruction.inertial_state.declare(sensor_id, noise);
}

inline bool slam::add_imu_sample(const sensor::imu::sample& sample) {
    return this->reconstruction.inertial_state.declared() && this->reconstruction.inertial_state.add_sample(sample);
}

inline void slam::set_frame_timestamp(const long long timestamp) {
    this->next_frame_timestamp_ = timestamp;
    this->next_frame_timestamped_ = true;
}

inline bool slam::inertial_initialised() const {
    return this->reconstruction.inertial_state.initialised();
}

inline void slam::inertial_frame_added(const mapping::frame& frame) {
    if (this->next_frame_timestamped_ && this->reconstruction.inertial_state.declared()) {
        this->reconstruction.inertial_state.set_frame_time(frame.id, this->next_frame_timestamp_);
    }
    this->next_frame_timestamped_ = false;
}

inline bool slam::inertial_predict(mapping::frame& frame) {
    const mapping::inertial& state = this->reconstruction.inertial_state;
    long long timestamp = 0;
    if (!state.declared() || !state.frame_time(frame.id, timestamp)) {
        return false;
    }
    math::matrix<double, 3, 3> rotation;
    math::matrix<double, 3, 1> translation;
    math::matrix<double, 3, 1> velocity;
    if (!state.predict(timestamp, this->reconstruction.frames, this->reconstruction.sensor_rig, rotation, translation, velocity)) {
        return false;
    }
    frame.rotation = rotation;
    frame.translation = translation;
    return true;
}

inline int slam::blind_frames_allowed() const {
    // The imu carries a frame the tracks lose for a moment, as far as it can be trusted to without them.
    return this->inertial_initialised() ? slam::inertial_blind_frames_tolerated : slam::blind_frames_tolerated;
}

inline void slam::inertial_keyframes_added() {
    mapping::inertial& state = this->reconstruction.inertial_state;
    if (!state.declared() || this->relative_adjustment()) {
        return;
    }
    std::vector<int> fresh;
    for (const int id : this->keyframe_ids_) {
        if ((state.keyframes().count(id) == 0) && (this->reconstruction.frames.count(id) != 0)) {
            fresh.push_back(id);
        }
    }
    std::sort(fresh.begin(), fresh.end());
    for (const int id : fresh) {
        // A chain stays within one submap: keyframes posed in different submaps are in frames of their own until a loop joins them.
        if (!state.keyframes().empty()) {
            const int previous_id = state.keyframes().rbegin()->first;
            if ((previous_id < id) && (this->submap_of(previous_id) != this->submap_of(id))) {
                state.break_chain();
            }
        }
        state.add_keyframe(id, this->reconstruction.frames, this->reconstruction.sensor_rig);
    }
}

inline void slam::inertial_submap_started() {
    if (this->reconstruction.inertial_state.declared()) {
        this->reconstruction.inertial_state.break_chain();
    }
}

inline void slam::scale_component(const size_t component, const math::matrix<double, 3, 1>& centre, const double scale) {
    const auto scaled = [&centre, scale](const math::matrix<double, 3, 1>& point) {
        return math::matrix<double, 3, 1>(centre + (scale * (point - centre)));
    };
    for (auto& [frame_id, frame] : this->reconstruction.frames) {
        if (this->map_component(frame_id) != component) {
            continue;
        }
        const math::matrix<double, 3, 1> frame_centre = -(math::transpose(frame.rotation) * frame.translation);
        frame.translation = -(frame.rotation * scaled(frame_centre));
        this->reconstruction.pose_views(frame);
    }
    for (auto& [landmark_id, landmark] : this->reconstruction.landmarks) {
        const std::unordered_map<int, std::vector<mapping::map::observation>>::const_iterator observed = this->reconstruction.observations.find(landmark_id);
        if ((observed == this->reconstruction.observations.end()) || observed->second.empty() || (this->map_component(observed->second.front().frame_id) != component)) {
            continue;
        }
        if (landmark.inverse_depth) {
            landmark.anchor_translation = scaled(landmark.anchor_translation);
            landmark.inverse_parameters[2] /= scale;
            landmark.update_location_from_inverse_depth();
        }
        else {
            landmark.location = scaled(landmark.location);
        }
        if (landmark.uncertainty == mapping::point::uncertainty_kind::estimated) {
            this->reconstruction.estimate_landmark_uncertainty(landmark);
        }
    }
    for (auto& [landmark_id, landmark] : this->reconstruction.line_landmarks) {
        const std::unordered_map<int, std::vector<mapping::map::line_observation>>::const_iterator observed = this->reconstruction.line_observations.find(landmark_id);
        if ((observed == this->reconstruction.line_observations.end()) || observed->second.empty() || (this->map_component(observed->second.front().frame_id) != component)) {
            continue;
        }
        landmark.locations[0] = scaled(landmark.locations[0]);
        landmark.locations[1] = scaled(landmark.locations[1]);
        geometry::plucker line;
        if (geometry::plucker::from_points(landmark.locations[0], landmark.locations[1], line)) {
            landmark.plucker_line = line;
        }
    }
    for (auto& [frame_id, anchor] : this->non_keyframe_anchors_) {
        if (this->map_component(frame_id) == component) {
            anchor.frame_from_keyframe.translation() = scale * anchor.frame_from_keyframe.translation();
        }
    }
    this->reckoning_.centre = scaled(this->reckoning_.centre);
    this->probation_origin_reckoning_.centre = scaled(this->probation_origin_reckoning_.centre);
    this->submap_speed_ *= scale;
    for (double& depth : this->scale_sentinel_depths_) {
        depth *= scale;
    }
}

inline bool slam::inertial_estimate(const std::vector<int>& window, const bool fixed_scale, const double accelerometer_deviation, optimisation::inertial_initialisation::result& result) {
    mapping::inertial& state = this->reconstruction.inertial_state;
    size_t imu_index = 0;
    if (!state.index_in(this->reconstruction.sensor_rig, imu_index) || (window.size() < 4)) {
        return false;
    }
    std::vector<math::se3<double>> poses;
    std::vector<sensor::imu::preintegration> preintegrations;
    for (size_t k = 0; k < window.size(); ++k) {
        const mapping::frame& frame = this->reconstruction.frames.at(window[k]);
        poses.push_back(math::se3<double>(math::so3<double>(frame.rotation), frame.translation));
        if (k > 0) {
            preintegrations.push_back(state.keyframes().at(window[k]).from_previous);
        }
    }
    optimisation::inertial_initialisation::options settings;
    settings.body_from_imu = this->reconstruction.sensor_rig.reference_from_sensor(imu_index);
    settings.fixed_scale = fixed_scale;
    settings.accelerometer_bias_deviation = accelerometer_deviation;
    if (accelerometer_deviation > optimisation::inertial_initialisation::options().accelerometer_bias_deviation) {
        settings.gyroscope_bias_deviation = 1.0;
    }
    result = optimisation::inertial_initialisation::estimate(poses, preintegrations, settings);
    return result.success;
}

inline void slam::inertial_apply(const std::vector<int>& window, const optimisation::inertial_initialisation::result& result, const bool rescale, const bool set_gravity) {
    mapping::inertial& state = this->reconstruction.inertial_state;
    if (rescale && (result.scale != 1.0)) {
        // The part of the map with the gauge is scaled about the world's origin, where the gauge's camera sits, and another about
        // the window's first keyframe.
        const size_t component = this->map_component(window.front());
        const mapping::frame& first = this->reconstruction.frames.at(window.front());
        const bool gauged = (this->reconstruction.frames.count(this->reconstruction.gauge_frame_id) != 0) && (this->map_component(this->reconstruction.gauge_frame_id) == component);
        this->scale_component(component, gauged ? math::matrix<double, 3, 1>::zero() : math::matrix<double, 3, 1>(-(math::transpose(first.rotation) * first.translation)), result.scale);
    }
    if (set_gravity) {
        state.set_gravity(result.gravity);
    }
    for (size_t k = 0; k < window.size(); ++k) {
        mapping::inertial::keyframe_state& keyframe = state.keyframes().at(window[k]);
        keyframe.velocity = result.velocities[k];
        keyframe.bias = result.bias;
        keyframe.estimated = true;
        if (k > 0) {
            keyframe.from_previous = result.preintegrations[k - 1];
        }
    }
}

inline void slam::inertial_after_keyframe(const mapping::frame& frame_current) {
    mapping::inertial& state = this->reconstruction.inertial_state;
    if (!state.declared() || this->relative_adjustment()) {
        return;
    }
    state.synchronise(this->keyframe_ids_, this->reconstruction.frames);
    std::vector<int> window = state.chain_window();
    if (window.empty() || (window.back() != frame_current.id)) {
        return;
    }
    const size_t chain = state.chain();
    const double span = 1.0e-9 * static_cast<double>(state.keyframes().at(window.back()).timestamp - state.keyframes().at(window.front()).timestamp);
    if (!state.chain_initialised(chain)) {
        if ((window.size() < slam::inertial_initialisation_keyframes) || (span < slam::inertial_initialisation_seconds)) {
            return;
        }
        if (window.size() > slam::inertial_window_maximum) {
            window.erase(window.begin(), window.end() - static_cast<std::ptrdiff_t>(slam::inertial_window_maximum));
        }
        // A metric map, a stereo rig's or one an earlier chain made metric, keeps its scale.
        const bool fixed_scale = this->reconstruction.metric_scale;
        optimisation::inertial_initialisation::result result;
        const bool estimated = this->inertial_estimate(window, fixed_scale, optimisation::inertial_initialisation::options().accelerometer_bias_deviation, result);
        core::logger::log(core::logger::level::info, "Inertial initialisation over %zu keyframes (%.2f s): %s, scale %.4f (deviation %.4f), gravity deviation %.4f rad, gyroscope bias [%.4f %.4f %.4f].", window.size(), span, !estimated ? "failed" : (result.observable ? "observable" : "not observable"), result.scale, result.scale_deviation, result.gravity_deviation, result.bias.gyroscope[0], result.bias.gyroscope[1], result.bias.gyroscope[2]);
        if (!estimated || !result.observable) {
            return;
        }
        const bool first_chain = !state.initialised();
        this->inertial_apply(window, result, !fixed_scale, first_chain);
        state.set_chain_initialised(chain);
        this->reconstruction.metric_scale = true;
        this->inertial_refinements_.clear();
        const long long now = state.keyframes().at(window.back()).timestamp;
        for (const double delay : slam::inertial_refinement_seconds) {
            this->inertial_refinements_.push_back(now + static_cast<long long>(delay * 1.0e9));
        }
        core::logger::log(core::logger::level::note, "Inertial initialisation at keyframe %d: the map is metric (scale %.4f), gravity [%.3f %.3f %.3f] in the map's world.", frame_current.id, result.scale, state.gravity()[0], state.gravity()[1], state.gravity()[2]);
    }
    else {
        const long long now = state.keyframes().at(window.back()).timestamp;
        if (this->inertial_refinements_.empty() || (now < this->inertial_refinements_.front())) {
            return;
        }
        this->inertial_refinements_.erase(this->inertial_refinements_.begin());
        if (window.size() > slam::inertial_window_maximum) {
            window.erase(window.begin(), window.end() - static_cast<std::ptrdiff_t>(slam::inertial_window_maximum));
        }
        // The window is longer now, enough to tell the accelerometer bias from gravity's tilt.
        const bool fixed_scale = this->reconstruction.sensor_rig.metric();
        optimisation::inertial_initialisation::result result;
        const bool estimated = this->inertial_estimate(window, fixed_scale, slam::inertial_refinement_accelerometer_deviation, result);
        core::logger::log(core::logger::level::info, "Inertial refinement over %zu keyframes (%.2f s): %s, scale %.4f, accelerometer bias [%.4f %.4f %.4f].", window.size(), span, estimated ? "estimated" : "failed", result.scale, result.bias.accelerometer[0], result.bias.accelerometer[1], result.bias.accelerometer[2]);
        if (!estimated || (math::abs(result.scale - 1.0) > slam::inertial_refinement_scale_change)) {
            return;
        }
        this->inertial_apply(window, result, !fixed_scale, true);
    }
    this->reconstruction.optimise(0, false, 20, true);
    this->reanchor_corrected();
    this->update_anchored_frames();
    this->rebuild_maps_around(this->last_keyframe_id_);
}

inline void slam::inertial_loop_edges(optimisation::factor_graph& graph, const std::unordered_map<int, optimisation::vertex*>& vertices) {
    const mapping::inertial& state = this->reconstruction.inertial_state;
    if (!state.initialised()) {
        return;
    }
    std::vector<int> ids;
    for (const auto& [id, vertex] : vertices) {
        static_cast<void>(vertex);
        ids.push_back(id);
    }
    std::sort(ids.begin(), ids.end());
    // The loop's graph holds camera to world similarities: gravity as each camera sees it is the inverse of the rotation turning
    // the world's gravity.
    for (const int id : ids) {
        optimisation::vertex* const vertex = vertices.at(id);
        if (vertex->is_fixed()) {
            continue;
        }
        const optimisation::edges::gravity_alignment policy(state.gravity(), true);
        const math::matrix<double, 3, 1> held = policy.turned(vertex->get_parameters());
        optimisation::edge factor{ policy };
        factor.set_observation(math::matrix<double, 0, 0>(3, 1, held.data()));
        factor.set_information(math::matrix<double, 0, 0>::identity(3, 3) * slam::inertial_loop_gravity_information);
        static_cast<void>(factor.add_vertex(vertex));
        static_cast<void>(graph.add_edge(static_cast<optimisation::edge&&>(factor)));
    }
}

inline void slam::inertial_loop_corrected(const std::unordered_map<int, math::se3<double>>& previous_poses, const std::unordered_map<int, math::sim3<double>>& corrected) {
    mapping::inertial& state = this->reconstruction.inertial_state;
    for (auto& [id, keyframe] : state.keyframes()) {
        const std::unordered_map<int, math::se3<double>>::const_iterator previous = previous_poses.find(id);
        const std::unordered_map<int, math::sim3<double>>::const_iterator current = corrected.find(id);
        if ((previous == previous_poses.end()) || (current == corrected.end())) {
            continue;
        }
        // A velocity in the world turns with the keyframe that carries it.
        const math::so3<double> turn = (current->second.transformation().rotation() * previous->second.rotation()).normalised();
        keyframe.velocity = turn * keyframe.velocity;
    }
}

#endif // ZEROSLAM_SLAM_INERTIAL_HPP
