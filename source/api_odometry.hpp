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
#ifndef ZEROSLAM_API_ODOMETRY_HPP
#define ZEROSLAM_API_ODOMETRY_HPP

#include "api.hpp"
#include "mapping/map.hpp"
#include "mapping/odometry.hpp"
#include "mapping/rig.hpp"
#include "slam.hpp"

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

// What the api holds of a wheel odometer: the rig entry that declared it (zeroslam_sensor_wheel_odometry, at most one), the
// extrinsic zeroslam_set_sensor_extrinsics gave it, and the time of its newest sample. Its samples go to the map's odometry
// (mapping::odometry) in the order they come among the cameras' images, and the time of each frame with them.
class api_odometry final {
public:
    // The doubles of a sample: the two wheels' rotations and the steering reading.
    constexpr static const size_t sample_values = 3;

private:
    bool declared = false;
    int sensor_id = 0;
    zeroslam_sensor_parameters_wheel_odometry_struct parameters = {};
    bool extrinsic_given = false;
    zeroslam_sensor_extrinsics_struct extrinsic = {};
    bool any_sample = false;
    long long latest_sample = 0;
    bool started = false;
    bool dispatching = false;

public:
    static bool declares(const zeroslam_sensor_rig_struct& entry);

    // Takes the odometer's rig entry, false for a second odometer or parameters the model refuses.
    bool take(const zeroslam_sensor_rig_struct& entry);
    bool is_declared() const;
    int get_sensor_id() const;

    // The settings of the map's odometry from the parameters.
    static bool settings_of(const zeroslam_sensor_parameters_wheel_odometry_struct& values, const int id, mapping::odometry::settings& result);

    // Takes the odometer's extrinsic from those given, the others into rest; false for an odometer's extrinsic that is not a
    // rotation or of an unknown refinement.
    bool take_extrinsics(const zeroslam_sensor_extrinsics_struct* const values, const int length, std::vector<zeroslam_sensor_extrinsics_struct>& rest);

    // The rig of the cameras with the odometer, at its extrinsic (the body's origin, held, when it was given none).
    mapping::rig with_odometer(const mapping::rig& cameras) const;

    // Whether every sample among the data is three finite doubles after the samples before it.
    bool accepts(const zeroslam_sensor_data_struct* const data, const int length) const;

    // Data with the odometer's samples among it: the samples to the map's odometry, the runs of images between them through
    // camera_runs in order. The first time, the map's odometry is configured, and a single camera's system given the rig of its
    // camera_rig and the odometer.
    template <typename camera_runs_type, typename camera_rig_type>
    zeroslam_return_enum set_sensor_data(slam& system, const zeroslam_sensor_data_struct* const data, const int length, const bool single_camera, const camera_runs_type& camera_runs, const camera_rig_type& camera_rig);
    bool is_dispatching() const;

    // Records the time of the frame the system processes next.
    void note_frame(slam& system, const long long timestamp) const;

    // The odometer's calibration and extrinsic as the map holds them, or as declared before the first data.
    bool fill_calibration(const mapping::map& map, zeroslam_odometry_calibration_struct* const calibration) const;

private:
    void start(slam& system, const bool single_camera, const mapping::rig& cameras);
};

template <typename camera_runs_type, typename camera_rig_type>
zeroslam_return_enum api_odometry::set_sensor_data(slam& system, const zeroslam_sensor_data_struct* const data, const int length, const bool single_camera, const camera_runs_type& camera_runs, const camera_rig_type& camera_rig) {
    if (!this->accepts(data, length)) {
        return zeroslam_return_failure_invalid_sensor_data;
    }
    int begin = 0;
    while (begin < length) {
        const bool samples = (data[begin].sensor_id == this->sensor_id);
        int end = begin;
        while ((end < length) && ((data[end].sensor_id == this->sensor_id) == samples)) {
            ++end;
        }
        if (!this->started) {
            this->start(system, single_camera, camera_rig());
        }
        if (samples) {
            for (int i = begin; i < end; ++i) {
                const double* const values = static_cast<const double*>(data[i].measurement_data);
                sensor::odometry::sample sample;
                sample.timestamp = data[i].timestamp;
                sample.rotation[0] = values[0];
                sample.rotation[1] = values[1];
                sample.steering = values[2];
                static_cast<void>(system.reconstruction.wheel_odometry.add_sample(sample));
                this->any_sample = true;
                this->latest_sample = sample.timestamp;
            }
        }
        else {
            this->dispatching = true;
            const zeroslam_return_enum result = camera_runs(data + begin, end - begin);
            this->dispatching = false;
            if (result != zeroslam_return_success) {
                return result;
            }
        }
        begin = end;
    }
    return zeroslam_return_success;
}

#endif // ZEROSLAM_API_ODOMETRY_HPP
