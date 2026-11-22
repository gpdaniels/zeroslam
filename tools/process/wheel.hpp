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
#ifndef ZEROSLAM_TOOLS_PROCESS_WHEEL_HPP
#define ZEROSLAM_TOOLS_PROCESS_WHEEL_HPP

// Summary: A scene's wheel odometer through the process tool (--wheel wheel_01): declared beside the camera with its scene
// parameters and extrinsic, its samples fed up to each frame's time before the frame, and its refined calibration written
// at the end (odometry.txt).

#include "mcap.hpp"
#include "wheel.hpp"
#include "zeroslam/zeroslam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace process_wheel {
    // Reads up to count numbers separated by spaces.
    inline bool parse_numbers(const std::string& text, double* const values, const std::size_t count) {
        const char* cursor = text.c_str();
        for (std::size_t i = 0; i < count; ++i) {
            char* end = nullptr;
            values[i] = std::strtod(cursor, &end);
            if (end == cursor) {
                return i > 0;
            }
            cursor = end;
        }
        return true;
    }

    // The parameters of a scene's odometer from its key=value text, separated by spaces or new lines: model
    // (ackermann_rear_axle, ackermann_front_wheel or differential_drive), parameters, free_mask, parameter_deviations, noise and
    // extrinsic_deviations, the numbers of a key separated by commas. The noise defaults to the library's.
    inline bool parse_parameters(const std::string& text, zeroslam_sensor_parameters_wheel_odometry_struct& result, std::string& error) {
        result = zeroslam_sensor_parameters_wheel_odometry_struct();
        const double noise[8] = { 0.01, 0.002, 0.005, 0.002, 0.01, 0.005, 0.001, 0.0005 };
        for (std::size_t i = 0; i < 8; ++i) {
            result.noise[i] = noise[i];
        }
        result.model = -1;
        std::size_t position = 0;
        while (position < text.size()) {
            const std::size_t end = text.find_first_of(" \n", position);
            const std::string token = text.substr(position, (end == std::string::npos) ? std::string::npos : (end - position));
            position = (end == std::string::npos) ? text.size() : (end + 1);
            const std::size_t equals = token.find('=');
            if (token.empty() || (equals == std::string::npos)) {
                continue;
            }
            const std::string key = token.substr(0, equals);
            std::string value = token.substr(equals + 1);
            for (char& character : value) {
                character = (character == ',') ? ' ' : character;
            }
            if (key == "model") {
                result.model = (value == "ackermann_rear_axle") ? zeroslam_odometry_ackermann_rear_axle : ((value == "ackermann_front_wheel") ? zeroslam_odometry_ackermann_front_wheel : ((value == "differential_drive") ? zeroslam_odometry_differential_drive : -1));
            }
            else if (key == "parameters") {
                parse_numbers(value, &result.parameters[0], 8);
            }
            else if (key == "free_mask") {
                result.free_mask = std::atoi(value.c_str());
            }
            else if (key == "parameter_deviations") {
                parse_numbers(value, &result.parameter_deviations[0], 8);
            }
            else if (key == "noise") {
                parse_numbers(value, &result.noise[0], 8);
            }
            else if (key == "extrinsic_deviations") {
                parse_numbers(value, &result.extrinsic_deviations[0], 6);
            }
        }
        if (result.model < 0) {
            error = "the odometer's parameters name no model (ackermann_rear_axle, ackermann_front_wheel or differential_drive)";
            return false;
        }
        return true;
    }

    class feeder final {
    private:
        bool active = false;
        int sensor_id = 2;
        wheel::odometer odometer;
        zeroslam_sensor_parameters_wheel_odometry_struct parameters = {};
        std::size_t next = 0;

    public:
        bool is_active() const {
            return this->active;
        }

        // Reads the named odometer of a scene.
        bool open(mcap& reader, const std::string& name, std::string& error) {
            std::vector<wheel::odometer> odometers;
            if (!wheel::read_scene(reader, odometers, error)) {
                return false;
            }
            for (const wheel::odometer& candidate : odometers) {
                if (candidate.name == name) {
                    this->odometer = candidate;
                    this->active = true;
                }
            }
            if (!this->active) {
                error = "the scene has no wheel odometer '" + name + "'";
                return false;
            }
            return parse_parameters(this->odometer.parameters, this->parameters, error);
        }

        // Adds the odometer to the rig's entries.
        void declare(std::vector<zeroslam_sensor_rig_struct>& rig) {
            if (!this->active) {
                return;
            }
            zeroslam_sensor_rig_struct entry{};
            entry.type = zeroslam_sensor_wheel_odometry;
            entry.sensor_id = this->sensor_id;
            entry.parameters_length = static_cast<int>(sizeof(this->parameters));
            entry.parameters_data = &this->parameters;
            rig.push_back(entry);
        }

        // Gives the odometer its scene extrinsic, refined when refine is set.
        bool set_extrinsic(const zeroslam::system& slam, const bool refine) const {
            if (!this->active) {
                return true;
            }
            zeroslam_sensor_extrinsics_struct extrinsic{};
            extrinsic.sensor_id = this->sensor_id;
            extrinsic.refinement = refine ? zeroslam_extrinsic_refine : zeroslam_extrinsic_fixed;
            for (std::size_t i = 0; i < 7; ++i) {
                extrinsic.pose[i] = this->odometer.samples.front().extrinsic[i];
            }
            return slam.set_sensor_extrinsics(&extrinsic, 1) == zeroslam_return_success;
        }

        // Feeds the samples up to a time.
        bool feed_until(const zeroslam::system& slam, const long long time) {
            if (!this->active) {
                return true;
            }
            std::vector<zeroslam_sensor_data_struct> batch;
            const std::size_t first = this->next;
            while ((this->next < this->odometer.samples.size()) && (this->odometer.samples[this->next].timestamp <= time)) {
                ++this->next;
            }
            for (std::size_t i = first; i < this->next; ++i) {
                wheel::sample& sample = this->odometer.samples[i];
                batch.push_back({ sample.timestamp, this->sensor_id, static_cast<int>(sizeof(sample.values)), &sample.values[0] });
            }
            return batch.empty() || (slam.set_sensor_data(batch.data(), static_cast<int>(batch.size())) == zeroslam_return_success);
        }

        // Writes the refined calibration and extrinsic: the parameter names of the model, the values, and the extrinsic.
        bool write_calibration(const zeroslam::system& slam, const char* const path) const {
            if (!this->active) {
                return true;
            }
            zeroslam_odometry_calibration_struct calibration{};
            if (slam.get_odometry_calibration(&calibration) != zeroslam_return_success) {
                return false;
            }
            std::FILE* const file = std::fopen(path, "wb");
            if (file == nullptr) {
                return false;
            }
            bool written = std::fprintf(file, "# %s refined: parameters, then the extrinsic [x y z qx qy qz qw]\n", this->odometer.name.c_str()) > 0;
            written = written && (std::fprintf(file, "parameters=") > 0);
            for (int i = 0; i < calibration.parameters_length; ++i) {
                written = written && (std::fprintf(file, "%s%.9g", (i == 0) ? "" : ",", calibration.parameters[i]) > 0);
            }
            written = written && (std::fprintf(file, "\nextrinsic=") > 0);
            for (int i = 0; i < 7; ++i) {
                written = written && (std::fprintf(file, "%s%.9g", (i == 0) ? "" : ",", calibration.pose[i]) > 0);
            }
            written = written && (std::fprintf(file, "\n") > 0);
            std::printf("Odometry calibration: parameters");
            for (int i = 0; i < calibration.parameters_length; ++i) {
                std::printf(" %.6g", calibration.parameters[i]);
            }
            std::printf(" (written to %s)\n", path);
            return (std::fclose(file) == 0) && written;
        }
    };
}

#endif // ZEROSLAM_TOOLS_PROCESS_WHEEL_HPP
