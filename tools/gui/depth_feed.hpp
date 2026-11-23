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
#ifndef ZEROSLAM_TOOLS_GUI_DEPTH_FEED_HPP
#define ZEROSLAM_TOOLS_GUI_DEPTH_FEED_HPP

#include "scene.hpp"
#include "zeroslam/zeroslam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// Feeds a scene's depth camera with the camera's frames (--depth): its frame nearest each camera frame, decoded to metres,
// registered to the camera, a metric RGB-D map. A rig of several cameras takes no depth camera.
namespace gui {
    class depth_feed final {
    public:
        constexpr static const int sensor_id = 9;
        // The depth noise model and pairing the process tool uses on TUM RGB-D (see README "Processing a scene").
        constexpr static const double noise_constant = 0.002;
        constexpr static const double noise_quadratic = 0.005;
        constexpr static const long long pairing_nanoseconds = 16000000;

    private:
        long long last_timestamp = -1;
        scene::decoded_image decoded;

    public:
        // Sets the rig of the camera, the first of the entries (the rest are the other sensors), with the depth camera registered
        // to it when enabled.
        static zeroslam_return_enum set_sensor_rig(const zeroslam::system& system, std::vector<zeroslam_sensor_rig_struct> rig, const bool enabled) {
            zeroslam_sensor_parameters_depth_struct depth = zeroslam::registered_depth(rig.front().sensor_id, zeroslam_depth_encoding_float_32, 1.0);
            depth.noise_constant = depth_feed::noise_constant;
            depth.noise_quadratic = depth_feed::noise_quadratic;
            depth.pairing_tolerance = depth_feed::pairing_nanoseconds;
            if (enabled) {
                zeroslam_sensor_rig_struct entry{};
                entry.type = zeroslam_sensor_depth;
                entry.sensor_id = depth_feed::sensor_id;
                entry.parameters_length = static_cast<int>(sizeof(depth));
                entry.parameters_data = &depth;
                rig.push_back(entry);
            }
            return system.set_sensor_rig(rig.data(), static_cast<int>(rig.size()));
        }

        void reset() {
            this->last_timestamp = -1;
            this->decoded.clear();
        }

        // Adds the scene's depth frame nearest the camera frame within the pairing tolerance to the measurements, when enabled
        // and there is one of the camera's size not given before; it stays held until the next call.
        void append(const scene::scene& loaded, const scene::decoded_image& camera_frame, const bool enabled, std::vector<zeroslam_sensor_data_struct>& measurements) {
            if (!enabled) {
                return;
            }
            for (const scene::image_channel& candidate : loaded.image_channels()) {
                if (!candidate.is_depth() || candidate.log_times.empty()) {
                    continue;
                }
                const long long timestamp = camera_frame.timestamp_nanoseconds;
                const std::vector<long long>::const_iterator after = std::lower_bound(candidate.log_times.begin(), candidate.log_times.end(), timestamp);
                std::size_t best = candidate.log_times.size();
                long long best_gap = 0;
                for (std::vector<long long>::const_iterator it = (after == candidate.log_times.begin()) ? after : (after - 1); (it != candidate.log_times.end()) && (it <= after); ++it) {
                    const long long gap = std::llabs(*it - timestamp);
                    if ((best == candidate.log_times.size()) || (gap < best_gap)) {
                        best = static_cast<std::size_t>(it - candidate.log_times.begin());
                        best_gap = gap;
                    }
                }
                const bool found = (best < candidate.log_times.size()) && (best_gap <= depth_feed::pairing_nanoseconds) && loaded.decode_image(candidate, best, this->decoded) && !this->decoded.metres.empty();
                if (!found || (this->decoded.width != camera_frame.width) || (this->decoded.height != camera_frame.height) || (this->decoded.timestamp_nanoseconds <= this->last_timestamp)) {
                    return;
                }
                zeroslam_sensor_data_struct measurement{};
                measurement.timestamp = this->decoded.timestamp_nanoseconds;
                measurement.sensor_id = depth_feed::sensor_id;
                measurement.measurement_length = static_cast<int>(this->decoded.metres.size() * sizeof(float));
                measurement.measurement_data = this->decoded.metres.data();
                measurements.push_back(measurement);
                this->last_timestamp = this->decoded.timestamp_nanoseconds;
                return;
            }
        }
    };
}

#endif // ZEROSLAM_TOOLS_GUI_DEPTH_FEED_HPP
