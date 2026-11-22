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
#ifndef ZEROSLAM_API_IMU_HPP
#define ZEROSLAM_API_IMU_HPP

#include "api.hpp"
#include "math/lie.hpp"
#include "sensor/imu/measurement.hpp"

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

// The imu of the api's rig: its declaration (a zeroslam_sensor_imu entry of zeroslam_set_sensor_rig), its extrinsic (its entry
// of zeroslam_set_sensor_extrinsics) and its samples (zeroslam_sensor_measurement_imu_struct entries of zeroslam_set_sensor_data),
// checked before they reach the system.
class api_imu final {
private:
    bool declared_ = false;
    int sensor_id_ = 0;
    zeroslam_sensor_parameters_imu_struct parameters_{};
    zeroslam_sensor_extrinsics_struct extrinsic_{};
    bool extrinsic_set_ = false;
    long long latest_timestamp_ = 0;
    bool any_sample_ = false;

public:
    // Whether a rig entry declares a valid imu.
    static bool accepts(const zeroslam_sensor_rig_struct& entry);

    // Declares the imu of a rig entry, which accepts must have passed, or none.
    void declare(const zeroslam_sensor_rig_struct* const entry);

    bool declared() const;
    int sensor_id() const;

    // Takes the imu's entry out of a set of extrinsics, leaving the cameras' in the rest, false when the imu's is not valid.
    bool take_extrinsic(const zeroslam_sensor_extrinsics_struct* const extrinsics, const int length, std::vector<zeroslam_sensor_extrinsics_struct>& rest);

    // The noise model, the imu's pose on the body, and whether it is refined.
    sensor::imu::parameters noise() const;
    math::se3<double> body_from_imu() const;
    bool refined() const;

    // Whether a data entry is the imu's, and whether the imu's entries of a batch are valid and follow in time.
    bool owns(const zeroslam_sensor_data_struct& entry) const;
    bool accepts_samples(const zeroslam_sensor_data_struct* const data, const int length) const;

    // The sample of an entry the imu owns, in the cameras' clock, and the entry's time noted as the latest.
    sensor::imu::sample take_sample(const zeroslam_sensor_data_struct& entry);
};

#endif // ZEROSLAM_API_IMU_HPP
