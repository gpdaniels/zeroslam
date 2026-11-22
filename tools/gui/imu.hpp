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
#ifndef ZEROSLAM_TOOLS_GUI_IMU_HPP
#define ZEROSLAM_TOOLS_GUI_IMU_HPP

#include "scene.hpp"
#include "zeroslam/zeroslam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstddef>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// Runs a scene's imu with its cameras (--imu imu_01): declared with the rig at its mounting on the ego, with the noise densities of
// the EuRoC MAV's ADIS16448 (a scene carries none), and its samples fed before each frame they precede.
namespace gui_imu {
    constexpr static const int sensor_id = 100;

    inline const scene::imu_channel* find(const scene::scene& loaded, const std::string& name) {
        for (const scene::imu_channel& channel : loaded.imu_channels()) {
            if (!name.empty() && (channel.sensor_name == name)) {
                return &channel;
            }
        }
        return nullptr;
    }

    inline zeroslam_sensor_parameters_imu_struct parameters() {
        zeroslam_sensor_parameters_imu_struct values{};
        values.gyroscope_noise_density = 1.6968e-04;
        values.accelerometer_noise_density = 2.0e-03;
        values.gyroscope_random_walk = 1.9393e-05;
        values.accelerometer_random_walk = 3.0e-03;
        values.rate = 200.0;
        return values;
    }

    // Adds the imu to a rig description, its parameters held by the caller until the rig is set.
    inline bool append(const scene::scene& loaded, const std::string& name, zeroslam_sensor_parameters_imu_struct& values, std::vector<zeroslam_sensor_rig_struct>& rig, std::vector<zeroslam_sensor_extrinsics_struct>& extrinsics) {
        if (name.empty()) {
            return true;
        }
        const scene::imu_channel* const channel = gui_imu::find(loaded, name);
        if (channel == nullptr) {
            return false;
        }
        values = gui_imu::parameters();
        zeroslam_sensor_rig_struct entry{};
        entry.type = zeroslam_sensor_imu;
        entry.sensor_id = gui_imu::sensor_id;
        entry.parameters_length = static_cast<int>(sizeof(values));
        entry.parameters_data = &values;
        rig.push_back(entry);
        zeroslam_sensor_extrinsics_struct extrinsic{};
        extrinsic.sensor_id = gui_imu::sensor_id;
        extrinsic.refinement = zeroslam_extrinsic_fixed;
        for (int axis = 0; axis < 3; ++axis) {
            extrinsic.pose[axis] = channel->mounting.translation[axis];
        }
        for (int component = 0; component < 4; ++component) {
            extrinsic.pose[3 + component] = channel->mounting.rotation[component];
        }
        extrinsics.push_back(extrinsic);
        return true;
    }

    // Feeds the samples up to a frame's time, from the next one not yet fed.
    inline void feed_until(const zeroslam::system& system, const scene::scene& loaded, const std::string& name, std::size_t& next, const long long timestamp) {
        const scene::imu_channel* const channel = gui_imu::find(loaded, name);
        if (channel == nullptr) {
            return;
        }
        std::vector<zeroslam_sensor_measurement_imu_struct> measurements;
        std::vector<zeroslam_sensor_data_struct> data;
        const std::size_t first = next;
        while ((next < channel->samples.size()) && (channel->samples[next].timestamp_nanoseconds <= timestamp)) {
            ++next;
        }
        // Only the last second before the first frame is of use.
        const std::size_t begin = ((first == 0) && (next > 200)) ? (next - 200) : first;
        measurements.resize(next - begin);
        data.resize(next - begin);
        for (std::size_t i = begin; i < next; ++i) {
            const scene::imu_sample& sample = channel->samples[i];
            zeroslam_sensor_measurement_imu_struct& measurement = measurements[i - begin];
            for (int axis = 0; axis < 3; ++axis) {
                measurement.angular_velocity[axis] = sample.angular_velocity[axis];
                measurement.linear_acceleration[axis] = sample.linear_acceleration[axis];
            }
            data[i - begin].timestamp = sample.timestamp_nanoseconds;
            data[i - begin].sensor_id = gui_imu::sensor_id;
            data[i - begin].measurement_length = static_cast<int>(sizeof(measurement));
            data[i - begin].measurement_data = &measurement;
        }
        if (!data.empty()) {
            static_cast<void>(system.set_sensor_data(data.data(), static_cast<int>(data.size())));
        }
    }
}

#endif // ZEROSLAM_TOOLS_GUI_IMU_HPP
