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
#ifndef ZEROSLAM_TOOLS_PROCESS_CALIBRATION_DRIFT_HPP
#define ZEROSLAM_TOOLS_PROCESS_CALIBRATION_DRIFT_HPP

#include "zeroslam/zeroslam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

// The calibration a run followed over time (--config calibration_drift=on), written as a line per chunk of time and camera.
namespace process_calibration_drift {
    // Writes the chunks the system holds, when it holds any, and prints over what range each camera's calibration went: a
    // line per chunk and camera of the chunk's start and length in nanoseconds, the camera's sensor id, the observations that
    // held it, its focal lengths, and its extrinsic [x y z qx qy qz qw]. False when the file could not be written.
    inline bool write(const zeroslam::system& slam, const char* const path) {
        int length = 0;
        if (slam.get_calibration_drift(nullptr, &length) != zeroslam_return_failure_insufficient_data_length) {
            return true;
        }
        std::vector<zeroslam_calibration_chunk_struct> chunks(static_cast<size_t>(length));
        if (slam.get_calibration_drift(chunks.data(), &length) != zeroslam_return_success) {
            return false;
        }
        std::FILE* const file = std::fopen(path, "wb");
        if (file == nullptr) {
            return false;
        }
        bool written = std::fprintf(file, "# timestamp duration sensor_id observations focal_x focal_y x y z qx qy qz qw\n") > 0;

        struct range final {
            size_t chunks = 0;
            size_t held = 0;
            double focal[2] = { 0.0, 0.0 };
            double first_pose[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
            double turn = 0.0;
            double shift = 0.0;
        };

        std::map<int, range> ranges;
        for (const zeroslam_calibration_chunk_struct& chunk : chunks) {
            written = written && (std::fprintf(file, "%lld %lld %d %d %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n", chunk.timestamp, chunk.duration, chunk.sensor_id, chunk.observations, chunk.focal_x, chunk.focal_y, chunk.pose[0], chunk.pose[1], chunk.pose[2], chunk.pose[3], chunk.pose[4], chunk.pose[5], chunk.pose[6]) > 0);
            range& camera = ranges[chunk.sensor_id];
            if (camera.chunks == 0) {
                camera.focal[0] = chunk.focal_x;
                camera.focal[1] = chunk.focal_x;
                std::copy(&chunk.pose[0], &chunk.pose[0] + 7, &camera.first_pose[0]);
            }
            ++camera.chunks;
            camera.held += (chunk.observations > 0) ? 1u : 0u;
            camera.focal[0] = std::min(camera.focal[0], chunk.focal_x);
            camera.focal[1] = std::max(camera.focal[1], chunk.focal_x);
            // How far the extrinsic is from the first chunk's: the angle between the two rotations, and the distance.
            double dot = 0.0;
            double distance_squared = 0.0;
            for (int i = 0; i < 4; ++i) {
                dot += chunk.pose[3 + i] * camera.first_pose[3 + i];
            }
            for (int i = 0; i < 3; ++i) {
                distance_squared += (chunk.pose[i] - camera.first_pose[i]) * (chunk.pose[i] - camera.first_pose[i]);
            }
            camera.turn = std::max(camera.turn, 2.0 * std::acos(std::min(1.0, std::abs(dot))) * 180.0 / 3.14159265358979323846);
            camera.shift = std::max(camera.shift, std::sqrt(distance_squared));
        }
        for (const auto& [sensor_id, camera] : ranges) {
            std::printf("Calibration drift of sensor %d: %zu chunks, %zu held by observations, fx %.4f to %.4f, extrinsic up to %.4f degrees and %.5f m from the first chunk's\n", sensor_id, camera.chunks, camera.held, camera.focal[0], camera.focal[1], camera.turn, camera.shift);
        }
        std::printf("Calibration drift written to %s\n", path);
        return (std::fclose(file) == 0) && written;
    }
}

#endif // ZEROSLAM_TOOLS_PROCESS_CALIBRATION_DRIFT_HPP
