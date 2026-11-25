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
#ifndef ZEROSLAM_API_CALIBRATION_DRIFT_HPP
#define ZEROSLAM_API_CALIBRATION_DRIFT_HPP

#include "api.hpp"
#include "mapping/calibration_drift.hpp"
#include "mapping/rig.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

// What the api gives of a calibration followed over time (mapping::calibration_drift): its keys of the configuration, and
// the chunks of time the system holds as zeroslam_get_calibration_drift gives them.
class api_calibration_drift final {
public:
    // The keys: whether the calibration is followed (off or on), the length of a chunk in seconds, and the walks over five
    // seconds of a focal length (as a fraction of it), of an extrinsic's rotation (radians) and of its translation (metres),
    // each of which may be zero.
    constexpr static const char key_enabled[] = "calibration_drift";
    constexpr static const char key_chunk[] = "calibration_drift_chunk";
    constexpr static const char key_focal[] = "calibration_drift_focal";
    constexpr static const char key_rotation[] = "calibration_drift_rotation";
    constexpr static const char key_translation[] = "calibration_drift_translation";

    // The largest walk a key takes: a tenth of a focal length, a tenth of a radian and a tenth of a metre over five seconds.
    constexpr static const double walk_maximum = 0.1;

    enum class parsed {
        // The key is not one of these.
        unknown,
        taken,
        // The key is one of these and its value is not one it takes.
        invalid
    };

    // Takes one key and its value into the settings, which an invalid value leaves as they were.
    static parsed parse(const char* const key, const size_t key_length, const char* const value, const size_t value_length, mapping::calibration_drift::settings& settings);

    // Prints the keys with their values, a line each, as snprintf does: the characters the lines take are returned, and as
    // many of them as fit are written, ended with a zero, when there is a buffer.
    static int print(char* const buffer, const size_t capacity, const mapping::calibration_drift::settings& settings);

    // How many entries the chunks make, one per chunk and camera.
    static int count(const mapping::calibration_drift& drift);

    // Fills the entries, by time and then in the rig's order of cameras: each camera's focal lengths in the chunk, its
    // extrinsic there as zeroslam_sensor_extrinsics_struct gives one, sensor to body on the rig given, and its sensor id, zero
    // for a camera the ids do not reach.
    static void fill(const mapping::calibration_drift& drift, const mapping::rig& rig, const std::vector<int>& sensor_ids, zeroslam_calibration_chunk_struct* const chunks);
};

#endif // ZEROSLAM_API_CALIBRATION_DRIFT_HPP
