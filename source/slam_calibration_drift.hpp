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
#ifndef ZEROSLAM_SLAM_CALIBRATION_DRIFT_HPP
#define ZEROSLAM_SLAM_CALIBRATION_DRIFT_HPP

// The part of the system that follows the calibration over time (mapping::calibration_drift), declared with the rest of the
// system in slam.hpp, which includes this file at its end. A frame that comes with a time (slam::set_frame_timestamp) is put
// in its chunk of time and takes that chunk's cameras and, for the views of a rig's other cameras, their poses on the
// reference camera, which the map then uses for the frame in place of the rig's. The adjustments of the whole map solve the
// chunks; every other adjustment, and the tracking, holds them. The rig keeps the newest chunk's extrinsics, so what tracks
// the newest frame through the rig tracks it with its own chunk's. With the following off, or for frames that come without a
// time, none of this is reached and the calibration is the constant one.

#include "slam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

inline void slam::follow_calibration_drift(mapping::frame& frame) {
    this->calibration_drift_.set(this->frontend.calibration_drift);
    if (!this->frontend.calibration_drift.enabled || !this->next_frame_timestamped_) {
        return;
    }
    const mapping::rig& rig = this->reconstruction.sensor_rig;
    if (!frame.views.empty() && (frame.views.size() >= rig.camera_count())) {
        return;
    }
    std::vector<sensor::model> cameras(1, frame.camera);
    std::vector<math::se3<double>> extrinsics(1, math::se3<double>::identity());
    for (size_t index = 0; index < frame.views.size(); ++index) {
        cameras.push_back(frame.views[index].camera);
        extrinsics.push_back(rig.reference_from_sensor(rig.camera(index + 1)));
    }
    const long long chunk = this->calibration_drift_.take_frame(this->next_frame_timestamp_, cameras, extrinsics);
    if (chunk < 0) {
        return;
    }
    frame.calibration_chunk = chunk;
    frame.camera = cameras[0];
    for (size_t index = 0; index < frame.views.size(); ++index) {
        mapping::frame& view = frame.views[index];
        view.calibration_chunk = chunk;
        view.camera = cameras[index + 1];
        view.reference_from_camera = extrinsics[index + 1];
    }
}

inline bool slam::optimise_following_calibration_drift(const int rounds) {
    if (!this->frontend.calibration_drift.enabled) {
        return false;
    }
    // As a constant calibration waits, for the keyframes to have seen the scene turn (see mapping::calibration).
    std::vector<int> ordered(this->keyframe_ids_.begin(), this->keyframe_ids_.end());
    std::sort(ordered.begin(), ordered.end());
    std::vector<math::matrix<double, 3, 3>> rotations;
    rotations.reserve(ordered.size());
    for (const int keyframe_id : ordered) {
        const std::unordered_map<int, mapping::frame>::const_iterator found = this->reconstruction.frames.find(keyframe_id);
        if (found != this->reconstruction.frames.end()) {
            rotations.push_back(found->second.rotation);
        }
    }
    const double spread = mapping::calibration::view_spread_degrees(rotations);
    if (!this->calibration_drift_.ready(rotations.size(), spread)) {
        core::logger::log(core::logger::level::info, "Calibration is not followed yet: %zu keyframes spanning %.1f degrees, %zu and %.1f needed, and a frame with a time.", rotations.size(), spread, mapping::calibration::keyframes_minimum, mapping::calibration::view_spread_degrees_minimum);
        return false;
    }
    mapping::rig& rig = this->reconstruction.sensor_rig;
    std::vector<double> baselines;
    for (size_t camera = 0; camera < rig.camera_count(); ++camera) {
        baselines.push_back(rig.at(rig.camera(camera)).declared_distance);
    }
    mapping::calibration_drift::request following;
    if (!this->calibration_drift_.make_request(this->frontend.calibration, this->frontend.extrinsics == mapping::frame::settings::extrinsics_kind::fixed, baselines, following)) {
        return false;
    }
    if (!this->calibration_drift_held_warned_ && (this->frontend.calibration != mapping::calibration::level::off) && (this->frontend.calibration != mapping::calibration::level::focal)) {
        core::logger::log(core::logger::level::warn, "Calibration: the principal point and the distortion are held as given while the calibration is followed over time, which follows the focal lengths alone.");
        this->calibration_drift_held_warned_ = true;
    }
    this->reconstruction.drift_request = &following;
    this->reconstruction.optimise(0, false, rounds, true, nullptr);
    this->reconstruction.drift_request = nullptr;
    if (!following.refined) {
        return true;
    }
    this->calibration_drift_.update(following);
    // The frames to come are tracked through the rig, which takes the newest chunk's extrinsics as they take its cameras.
    const mapping::calibration_drift::chunk* const newest = this->calibration_drift_.newest();
    for (size_t camera = 1; (newest != nullptr) && (camera < rig.camera_count()) && (camera < newest->cameras.size()) && (camera < following.cameras.size()); ++camera) {
        if (following.cameras[camera].follow_extrinsic) {
            rig.set_reference_from_sensor(rig.camera(camera), newest->cameras[camera].reference_from_camera);
        }
    }
    // The loop closure's keyframes each hold the camera of their own chunk.
    for (const int keyframe_id : ordered) {
        const std::unordered_map<int, mapping::frame>::const_iterator found = this->reconstruction.frames.find(keyframe_id);
        if ((found != this->reconstruction.frames.end()) && (found->second.calibration_chunk >= 0)) {
            static_cast<void>(this->loop_closure_.set_camera(keyframe_id, found->second.camera));
        }
    }
    return true;
}

#endif // ZEROSLAM_SLAM_CALIBRATION_DRIFT_HPP
