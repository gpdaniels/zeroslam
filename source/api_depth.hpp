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
#ifndef ZEROSLAM_API_DEPTH_HPP
#define ZEROSLAM_API_DEPTH_HPP

#include "sensor/depth.hpp"
#include "zeroslam/zeroslam.h"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// The depth cameras of the api's rig (zeroslam_sensor_depth) and the depth frames given and not yet paired with a frame of
// their camera. A depth camera is registered to one camera of the rig, its frames that camera's size, and a camera frame
// takes the depth frame nearest it in time within the pairing tolerance from those given before it or with it.
class api_depth final {
public:
    struct sensor final {
        int sensor_id;
        zeroslam_sensor_parameters_depth_struct parameters;
    };

    // How many unpaired depth frames are held at most: a camera that stops leaves no more than these.
    constexpr static const size_t pending_maximum = 8;

private:
    struct pending final {
        long long int timestamp;
        int camera_id;
        ::sensor::depth image;
    };

    std::vector<sensor> sensors;
    std::vector<pending> held;
    long long int latest_timestamp = 0;
    bool any_frame = false;

public:
    // Whether the parameters describe a depth camera the system takes: a known encoding and measure, registered to its camera,
    // a positive scale for 16 bit values, a range above zero, a noise model of non-negative coefficients not all zero, and a
    // non-negative pairing tolerance.
    static bool valid(const zeroslam_sensor_parameters_depth_struct& parameters);

    // The depth cameras of a rig being set, each checked with valid; the camera each names is checked by the rig.
    static bool parse(const zeroslam_sensor_rig_struct& entry, sensor& parsed);

    void set_sensors(const std::vector<sensor>& rig_sensors);

    bool empty() const;

    // The depth camera with the sensor id, or null.
    const sensor* find(const int sensor_id) const;

    // The bytes a frame of the depth camera holds for a camera of the given size.
    static size_t frame_length(const sensor& depth_sensor, const int width, const int height);

    // Whether a depth frame's timestamp follows the last one given, as they must increase strictly; next is the newest so far
    // within a batch being validated, seen whether there is one.
    bool follows(const long long int timestamp, bool& seen, long long int& next) const;

    void validation_start(bool& seen, long long int& next) const;

    // Holds a depth frame of the camera's size, converted to metres with the sensor's settings.
    void hold(const sensor& depth_sensor, const long long int timestamp, const int width, const int height, const void* const data);

    // The depth frame for a camera frame: the held frame of its camera nearest in time within the tolerance, the earlier on a
    // tie, empty when there is none. It and the frames before it are used up, those after it may pair with the next camera
    // frame; frames too old to pair with this camera frame, which cannot pair with a later one, are let go.
    ::sensor::depth take(const int camera_id, const long long int timestamp);
};

#endif // ZEROSLAM_API_DEPTH_HPP
