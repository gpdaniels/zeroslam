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
#ifndef ZEROSLAM_TOOLS_PROCESS_IMU_HPP
#define ZEROSLAM_TOOLS_PROCESS_IMU_HPP

#include "cdr.hpp"
#include "mcap.hpp"
#include "zeroslam/zeroslam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// Runs a scene's imu with its cameras (--imu imu_01): the imu is declared with the rig at the extrinsic its first
// 'ego -> sensor/[name]' transform on /tf gives, and its samples are fed to the library between the frames, in time order. A scene
// carries no noise model, so the imu is given the densities of the EuRoC MAV's ADIS16448 unless --imu-noise names others.
namespace process_imu {
    struct settings {
        std::string name;
        double noise[4] = { 1.6968e-04, 2.0e-03, 1.9393e-05, 3.0e-03 };
        double rate = 200.0;
        bool refine = false;

        bool enabled() const {
            return !this->name.empty();
        }
    };

    // The imu's sensor id, after the cameras'.
    constexpr static const int sensor_id = 100;

    // gyroscope noise,accelerometer noise,gyroscope walk,accelerometer walk.
    inline bool parse_noise(const char* const text, double (&noise)[4]) {
        const char* cursor = text;
        for (int i = 0; i < 4; ++i) {
            char* end_pointer = nullptr;
            noise[i] = std::strtod(cursor, &end_pointer);
            if ((end_pointer == cursor) || !(noise[i] > 0.0) || (*end_pointer != ((i < 3) ? ',' : 0))) {
                return false;
            }
            cursor = end_pointer + 1;
        }
        return true;
    }

    struct selected_imu {
        const mcap::channel_type* channel = nullptr;
        bool extrinsic_found = false;
        double extrinsic[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
        zeroslam_sensor_parameters_imu_struct parameters{};
        // The samples that come before the rig is declared, fed once it is.
        std::vector<std::pair<long long, zeroslam_sensor_measurement_imu_struct>> pending;
        size_t samples = 0;
    };

    inline bool find(const mcap& reader, const settings& wanted, selected_imu& selected) {
        if (!wanted.enabled()) {
            return true;
        }
        for (const mcap::channel_type& channel : reader.get_channels()) {
            const mcap::schema_type* schema = reader.find_schema(channel.schema_id);
            if ((schema != nullptr) && (schema->name == "sensor_msgs/msg/Imu") && (channel.topic == "/sensor/" + wanted.name)) {
                selected.channel = &channel;
            }
        }
        if (selected.channel == nullptr) {
            std::fprintf(stderr, "The scene has no imu '%s' (an imu topic '/sensor/%s').\n", wanted.name.c_str(), wanted.name.c_str());
            return false;
        }
        selected.parameters.gyroscope_noise_density = wanted.noise[0];
        selected.parameters.accelerometer_noise_density = wanted.noise[1];
        selected.parameters.gyroscope_random_walk = wanted.noise[2];
        selected.parameters.accelerometer_random_walk = wanted.noise[3];
        selected.parameters.rate = wanted.rate;
        return true;
    }

    // Takes the imu's extrinsic from a /tf message's transforms.
    inline void take_extrinsic(const settings& wanted, const std::vector<cdr::transform_stamped>& transforms, selected_imu& selected) {
        for (const cdr::transform_stamped& transform : transforms) {
            if (!wanted.enabled() || selected.extrinsic_found || (transform.frame_header.frame_id != "ego") || (transform.child_frame_id != ("sensor/" + wanted.name))) {
                continue;
            }
            for (int axis = 0; axis < 3; ++axis) {
                selected.extrinsic[axis] = transform.translation[axis];
            }
            for (int component = 0; component < 4; ++component) {
                selected.extrinsic[3 + component] = transform.rotation[component];
            }
            selected.extrinsic_found = true;
        }
    }

    // Adds the imu to the rig and the extrinsics a rig declaration gives the library.
    inline void declare(const settings& wanted, selected_imu& selected, std::vector<zeroslam_sensor_rig_struct>& rig, std::vector<zeroslam_sensor_extrinsics_struct>& extrinsics, const int verbosity) {
        if (!wanted.enabled()) {
            return;
        }
        zeroslam_sensor_rig_struct entry{};
        entry.type = zeroslam_sensor_imu;
        entry.sensor_id = process_imu::sensor_id;
        entry.parameters_length = static_cast<int>(sizeof(selected.parameters));
        entry.parameters_data = &selected.parameters;
        rig.push_back(entry);
        zeroslam_sensor_extrinsics_struct extrinsic{};
        extrinsic.sensor_id = process_imu::sensor_id;
        extrinsic.refinement = wanted.refine ? zeroslam_extrinsic_refine : zeroslam_extrinsic_fixed;
        for (int i = 0; i < 7; ++i) {
            extrinsic.pose[i] = selected.extrinsic[i];
        }
        extrinsics.push_back(extrinsic);
        if (!selected.extrinsic_found && (verbosity >= 2)) {
            std::printf("The scene does not pose imu %s on the ego: it is taken to sit at the ego's origin.\n", wanted.name.c_str());
        }
        if (verbosity >= 3) {
            std::printf("Imu %s (sensor %d): extrinsic [%.6f %.6f %.6f] [%.6f %.6f %.6f %.6f].\n", wanted.name.c_str(), process_imu::sensor_id, extrinsic.pose[0], extrinsic.pose[1], extrinsic.pose[2], extrinsic.pose[3], extrinsic.pose[4], extrinsic.pose[5], extrinsic.pose[6]);
        }
    }

    inline bool send(const zeroslam::system& slam, const long long timestamp, zeroslam_sensor_measurement_imu_struct& measurement) {
        zeroslam_sensor_data_struct data{};
        data.timestamp = timestamp;
        data.sensor_id = process_imu::sensor_id;
        data.measurement_length = static_cast<int>(sizeof(measurement));
        data.measurement_data = &measurement;
        const zeroslam_return_enum result = slam.set_sensor_data(&data, 1);
        if (result != zeroslam_return_success) {
            std::fprintf(stderr, "The library rejected the imu sample at %lld: %s.\n", timestamp, zeroslam_return_enum_to_string(result));
            return false;
        }
        return true;
    }

    // Feeds a message of the imu's channel, held until the rig is declared.
    inline bool feed(mcap& reader, const zeroslam::system& slam, const size_t message_number, const bool rig_declared, selected_imu& selected) {
        mcap::message_type message;
        cdr::imu sample;
        if (!reader.read_message(message_number, message) || !cdr::read_imu(message.data, message.length, sample)) {
            std::fprintf(stderr, "An imu message does not decode.\n");
            return false;
        }
        zeroslam_sensor_measurement_imu_struct measurement{};
        for (int axis = 0; axis < 3; ++axis) {
            measurement.angular_velocity[axis] = sample.angular_velocity[axis];
            measurement.linear_acceleration[axis] = sample.linear_acceleration[axis];
        }
        const long long timestamp = static_cast<long long>(message.log_time);
        ++selected.samples;
        if (!rig_declared) {
            selected.pending.push_back({ timestamp, measurement });
            return true;
        }
        return process_imu::send(slam, timestamp, measurement);
    }

    // Feeds the samples held until the rig was declared, the last of them only.
    inline bool flush(const zeroslam::system& slam, selected_imu& selected) {
        // Only the last second before the first frame is of use.
        const size_t keep = 400;
        const size_t first = (selected.pending.size() > keep) ? (selected.pending.size() - keep) : 0;
        for (size_t i = first; i < selected.pending.size(); ++i) {
            if (!process_imu::send(slam, selected.pending[i].first, selected.pending[i].second)) {
                return false;
            }
        }
        selected.pending.clear();
        return true;
    }

    // Writes the imu's calibration as the library holds it at the end of the run: the matrix of the gyroscope's scale and
    // misalignment, zero unless the imu_calibration key had it refined.
    inline void report_calibration(const zeroslam::system& slam) {
        zeroslam_imu_calibration_struct calibration{};
        if (slam.get_imu_calibration(&calibration) != zeroslam_return_success) {
            return;
        }
        const double* const m = calibration.gyroscope;
        std::printf("Gyroscope of sensor %d (refined %d times): [%.6f %.6f %.6f; %.6f %.6f %.6f; %.6f %.6f %.6f]\n", calibration.sensor_id, calibration.refinements, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
    }
}

#endif // ZEROSLAM_TOOLS_PROCESS_IMU_HPP
