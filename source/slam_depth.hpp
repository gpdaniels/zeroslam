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
#ifndef ZEROSLAM_SLAM_DEPTH_HPP
#define ZEROSLAM_SLAM_DEPTH_HPP

// The part of the system that takes a depth camera registered to the reference camera, declared with the rest of the system
// in slam.hpp, which includes this file at its end. A frame given with its depth image keeps it while it may become a
// keyframe; when the map has metric scale (mapping::map::metric_scale) and depth images are given, the map starts from one
// frame with depth, the keyframes make their landmarks from the depths at their tracks, and the observations' depths enter
// the adjustments as factors (mapping::depth_factor). Without depth none of this is reached.

#include "slam.hpp"

// Takes the depth given for the frame being made, and lets the previous frame's go unless it became a keyframe.
inline void slam::attach_depth(mapping::frame& frame) {
    if ((this->depth_frame_id_ >= 0) && (this->keyframe_ids_.count(this->depth_frame_id_) == 0)) {
        const std::unordered_map<int, mapping::frame>::iterator previous = this->reconstruction.frames.find(this->depth_frame_id_);
        if (previous != this->reconstruction.frames.end()) {
            previous->second.depth.release();
        }
    }
    this->depth_frame_id_ = -1;
    if (!this->next_depth_.empty()) {
        frame.depth = static_cast<sensor::depth&&>(this->next_depth_);
        this->next_depth_ = sensor::depth();
        this->depth_frame_id_ = frame.id;
        this->depth_given_ = true;
    }
}

// Whether a metric map can start from the frame by its depth alone.
inline bool slam::starts_from_depth(const mapping::frame& frame) const {
    return this->reconstruction.metric_scale && !frame.depth.empty();
}

// Starts a metric map from one frame and its depth image: the frame is the first keyframe, at the pose it was given (or
// dead reckoned to, for a submap), and its landmarks are made from the depths at its tracks as the keyframe is processed,
// so no second view and no baseline are needed and the scale is the depth camera's.
inline bool slam::initialise_from_depth(mapping::frame& frame_current) {
    size_t measured = 0;
    for (const feature::tracker::tracker::track* const t : this->active_point_tracks()) {
        double value = 0.0;
        measured += frame_current.depth.sample(static_cast<double>(t->x), static_cast<double>(t->y), value) ? 1u : 0u;
    }
    if (measured < slam::depth_initialisation_tracks_minimum) {
        core::logger::log(core::logger::level::info, "Init REJECTED (frame %d): %zu tracks have a depth, %zu are needed to start a metric map.", frame_current.id, measured, slam::depth_initialisation_tracks_minimum);
        return false;
    }
    if (this->reinitialising_) {
        math::matrix<double, 3, 3> reckoned_rotation;
        math::matrix<double, 3, 1> reckoned_centre;
        const std::unordered_map<int, mapping::frame>::const_iterator anchor = this->reconstruction.frames.find(this->init_anchor_frame_id_);
        if ((this->reckoning_.frame_id == frame_current.id) && this->reckoned_pose(reckoned_rotation, reckoned_centre)) {
            frame_current.rotation = reckoned_rotation;
            frame_current.translation = -(reckoned_rotation * reckoned_centre);
        }
        else if (anchor != this->reconstruction.frames.end()) {
            frame_current.rotation = anchor->second.rotation;
            frame_current.translation = anchor->second.translation;
        }
    }
    else {
        this->reconstruction.gauge_frame_id = frame_current.id;
    }
    // The frame a two-view initialisation would have started from is not needed: this one starts the map alone.
    if ((this->init_anchor_frame_id_ != frame_current.id) && (this->keyframe_ids_.count(this->init_anchor_frame_id_) == 0)) {
        this->reconstruction.frames.erase(this->init_anchor_frame_id_);
    }
    this->init_anchor_frame_id_ = frame_current.id;
    this->state_ = tracking_state::tracking;
    this->blind_frames_ = 0;
    this->reckon_from(frame_current);
    this->keyframe_ids_.insert(frame_current.id);
    this->keyframes_inserted_ += 1;
    this->keyframes_since_global_adjustment_ += 1;
    this->last_keyframe_id_ = frame_current.id;
    this->last_tracked_frame_id_ = frame_current.id;
    this->relocalised_keyframe_id_ = -1;
    if (this->reinitialising_) {
        this->start_submap(frame_current);
    }
    core::logger::log(core::logger::level::note, "Metric map started at frame %d from its depth image, %zu tracks with a depth.", frame_current.id, measured);
    return true;
}

// A landmark for a track from the depth the keyframe measured at it: the ray the camera unprojects the track to, scaled
// to the depth. The track's observations in earlier keyframes join it where it reprojects within the gate a
// triangulated landmark is held to.
inline bool slam::create_landmark_from_depth(const mapping::frame& frame_current, feature::tracker::tracker::track& t, const double measured, const image::image& image_grey, double& depth_in_frame) {
    math::matrix<double, 3, 1> point;
    if (!mapping::depth_factor::location(frame_current, static_cast<double>(t.x), static_cast<double>(t.y), measured, point, depth_in_frame)) {
        return false;
    }
    const size_t colour_col = static_cast<size_t>(math::max(0, math::min(static_cast<int>(image_grey.get_cols()) - 1, static_cast<int>(t.x))));
    const size_t colour_row = static_cast<size_t>(math::max(0, math::min(static_cast<int>(image_grey.get_rows()) - 1, static_cast<int>(t.y))));
    const double colour = static_cast<double>(image_grey.get_data()[colour_row * image_grey.get_cols() + colour_col]) / 255.0;
    mapping::point landmark(this->reconstruction.allocate_landmark_id(), point, math::matrix<double, 3, 1>{ { colour, colour, colour } });
    if (this->frontend.inverse_depth && !landmark.anchor(frame_current.rotation, frame_current.translation)) {
        return false;
    }
    if (t.descriptor_valid) {
        this->add_landmark_descriptors(landmark, frame_current, t.x, t.y, t.octave, t.descriptor);
    }
    this->reconstruction.add_landmark(landmark);
    t.landmark_id = landmark.id;
    const size_t component = this->map_component(frame_current.id);
    for (const feature::tracker::tracker::observation& obs : t.history) {
        if (this->keyframe_ids_.count(obs.frame_id) == 0) {
            continue;
        }
        if (obs.frame_id != frame_current.id) {
            const std::unordered_map<int, mapping::frame>::const_iterator earlier = this->reconstruction.frames.find(obs.frame_id);
            math::matrix<double, 2, 1> reprojected;
            if ((earlier == this->reconstruction.frames.end()) || (this->map_component(obs.frame_id) != component) || !mapping::map::project_landmark(earlier->second, landmark, reprojected)) {
                continue;
            }
            const math::matrix<double, 2, 1> observed = { { static_cast<double>(obs.x), static_cast<double>(obs.y) } };
            if ((observed - reprojected).get_length_squared() > (2 * 2)) {
                continue;
            }
        }
        this->reconstruction.add_observation(obs.frame_id, landmark, static_cast<double>(obs.x), static_cast<double>(obs.y), t.octave, obs.measurement_octave);
    }
    this->reconstruction.estimate_landmark_uncertainty(this->reconstruction.landmarks.at(landmark.id));
    ++this->landmarks_from_depth;
    return true;
}

#endif // ZEROSLAM_SLAM_DEPTH_HPP
