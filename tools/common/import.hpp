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
#ifndef ZEROSLAM_TOOLS_COMMON_IMPORT_HPP
#define ZEROSLAM_TOOLS_COMMON_IMPORT_HPP

#include "dataset.hpp"
#include "file.hpp"
#include "paths.hpp"
#include "process.hpp"
#include "rotation.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace import {
    inline char* next_line(char*& cursor, char* const end) {
        if ((cursor == nullptr) || (cursor >= end)) {
            return nullptr;
        }
        char* const line = cursor;
        char* const newline = static_cast<char*>(std::memchr(line, '\n', static_cast<std::size_t>(end - line)));
        if (newline != nullptr) {
            *newline = '\0';
            cursor = newline + 1;
        }
        else {
            cursor = end;
        }
        char* const carriage = std::strchr(line, '\r');
        if (carriage != nullptr) {
            *carriage = '\0';
        }
        return line;
    }

    template <typename handler_type>
    inline bool for_each_line(const std::string& path, std::string& error, handler_type&& handle) {
        std::string text;
        if (!dataset::read_text_file(path, text)) {
            error = "cannot read '" + path + "'";
            return false;
        }
        char* cursor = &text[0];
        char* const end = &text[0] + text.size();
        for (char* line = next_line(cursor, end); line != nullptr; line = next_line(cursor, end)) {
            while ((*line == ' ') || (*line == '\t')) {
                ++line;
            }
            if ((*line == '\0') || (*line == '#')) {
                continue;
            }
            handle(line);
        }
        return true;
    }

    struct ground_truth_sample {
        long long timestamp_nanoseconds = 0;
        double position[3] = {};
        double quaternion_xyzw[4] = { 0.0, 0.0, 0.0, 1.0 };
    };

    inline bool ground_truth_covers(const std::vector<ground_truth_sample>& samples, const long long timestamp_nanoseconds) {
        return !samples.empty() && (timestamp_nanoseconds >= samples.front().timestamp_nanoseconds) && (timestamp_nanoseconds <= samples.back().timestamp_nanoseconds);
    }

    inline bool interpolate_ground_truth(const std::vector<ground_truth_sample>& samples, const long long timestamp_nanoseconds, double position[3], double quaternion_xyzw[4]) {
        if (!ground_truth_covers(samples, timestamp_nanoseconds)) {
            return false;
        }
        const auto after = std::lower_bound(samples.begin(), samples.end(), timestamp_nanoseconds, [](const ground_truth_sample& sample, const long long value) {
            return sample.timestamp_nanoseconds < value;
        });
        if (after->timestamp_nanoseconds == timestamp_nanoseconds) {
            std::copy(&after->position[0], &after->position[0] + 3, position);
            std::copy(&after->quaternion_xyzw[0], &after->quaternion_xyzw[0] + 4, quaternion_xyzw);
            return true;
        }
        const ground_truth_sample& sample_1 = *after;
        const ground_truth_sample& sample_0 = *(after - 1);
        const double alpha = static_cast<double>(timestamp_nanoseconds - sample_0.timestamp_nanoseconds) / static_cast<double>(sample_1.timestamp_nanoseconds - sample_0.timestamp_nanoseconds);
        for (int axis = 0; axis < 3; ++axis) {
            position[axis] = sample_0.position[axis] + (alpha * (sample_1.position[axis] - sample_0.position[axis]));
        }
        rotation::slerp(&sample_0.quaternion_xyzw[0], &sample_1.quaternion_xyzw[0], alpha, &quaternion_xyzw[0]);
        return true;
    }

    inline void remove_absent_scene_file(const std::string& path) {
        std::remove(path.c_str());
    }

    inline void pixel_centre_principal_point(double intrinsics[4]) {
        intrinsics[2] += 0.5;
        intrinsics[3] += 0.5;
    }

    inline bool write_calibration_line(gtl::file& file, const long long timestamp_nanoseconds, const double translation[3], const double rotation_xyzw[4], const char* const model, const double intrinsics[4], const double* const distortion, const std::size_t distortion_count) {
        const double extrinsic[7] = { translation[0], translation[1], translation[2], rotation_xyzw[0], rotation_xyzw[1], rotation_xyzw[2], rotation_xyzw[3] };
        std::string line;
        bool formatted = dataset::format_sample_line(line, timestamp_nanoseconds, &extrinsic[0], 7);
        line += std::string(" ") + model;
        formatted = formatted && dataset::append_values(line, intrinsics, 4) && dataset::append_values(line, distortion, distortion_count);
        line += '\n';
        return formatted && dataset::write_line(file, line.c_str());
    }

    inline bool convert_image(const std::string& source_path, const std::string& options, const std::string& destination_path, std::string& error) {
        const std::string command = "convert " + platform::quote_for_shell(source_path) + " " + options + " " + platform::quote_for_shell(destination_path);
        std::string output;
        int exit_code = -1;
        if (!platform::run_command(command, false, output, exit_code) || (exit_code != 0)) {
            error = "Failed to convert: " + source_path;
            return false;
        }
        return true;
    }

    inline bool convert_depth_image(const std::string& source_path, const std::string& destination_path, const double scale, std::string& error) {
        const std::string temporary_path = destination_path + ".pgm";
        if (!convert_image(source_path, "-depth 16", temporary_path, error)) {
            return false;
        }
        dataset::pnm_format format = dataset::pnm_format::pgm8;
        unsigned int width = 0;
        unsigned int height = 0;
        std::vector<unsigned char> samples;
        const bool read = dataset::read_pnm(temporary_path, format, width, height, samples);
        std::remove(temporary_path.c_str());
        if (!read || (format != dataset::pnm_format::pgm16)) {
            error = "Failed to read the converted 16 bit depth of: " + source_path;
            return false;
        }
        std::vector<float> depth(static_cast<std::size_t>(width) * height);
        for (std::size_t index = 0; index < depth.size(); ++index) {
            unsigned short millimetres = 0;
            std::memcpy(&millimetres, &samples[index * 2], sizeof(millimetres));
            depth[index] = static_cast<float>(static_cast<double>(millimetres) / scale);
        }
        if (!dataset::write_pnm(destination_path, dataset::pnm_format::pfm, width, height, reinterpret_cast<const unsigned char*>(depth.data()))) {
            error = "Failed to write the depth: " + destination_path;
            return false;
        }
        return true;
    }

    inline bool locate_tool(const std::string& tools_directory, const std::string& tool, std::string& path, std::string& error) {
        std::string directory = tools_directory;
        if (directory.empty() && !gtl::paths::get_executable_directory(directory)) {
            directory = ".";
        }
#if defined(_WIN32)
        path = directory + "/zeroslam-" + tool + ".exe";
#else
        path = directory + "/zeroslam-" + tool;
#endif
        if (!gtl::paths::is_regular_file(path)) {
            error = "Tool not found: '" + path + "' (build all tools, or provide --tools-dir).";
            return false;
        }
        return true;
    }

    inline bool collapse_scene(const std::string& tools_directory, const std::string& directory, const std::string& output_path, std::string& error) {
        std::string dataset_binary;
        if (!locate_tool(tools_directory, "dataset", dataset_binary, error)) {
            return false;
        }
        std::printf("\nPacking with 'zeroslam-dataset collapse'...\n");
        std::fflush(stdout);
        const std::string command = platform::quote_for_shell(dataset_binary) + " collapse " + platform::quote_for_shell(directory) + " " + platform::quote_for_shell(output_path) + " 2>&1";
        std::string output;
        int exit_code = -1;
        if (!platform::run_command(command, true, output, exit_code) || (exit_code != 0)) {
            error = "Failed to pack the scene (exit code " + std::to_string(exit_code) + ").";
            return false;
        }
        return true;
    }
}

#endif // ZEROSLAM_TOOLS_COMMON_IMPORT_HPP
