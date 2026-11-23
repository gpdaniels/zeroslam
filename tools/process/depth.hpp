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
#ifndef ZEROSLAM_TOOLS_PROCESS_DEPTH_HPP
#define ZEROSLAM_TOOLS_PROCESS_DEPTH_HPP

#include "cdr.hpp"
#include "dataset.hpp"
#include "mcap.hpp"
#include "metrics.hpp"
#include "zeroslam/zeroslam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// Runs a scene with its depth camera (--depth): the lowest numbered /sensor/depth_XX topic, registered to the camera, its
// encoding read from its first frame, each depth frame given to the library with the camera frame it came with, before it.
namespace process_depth {
    struct options {
        bool enabled = false;
        // The depth noise fitted on TUM RGB-D fr1_desk, above the Kinect's random error of 1.425e-3 d^2 (Khoshelham and
        // Elberink, 2012) as it carries the depth's registration to the camera and its timing too, and the range a Kinect
        // measures.
        double noise[3] = { 0.002, 0.0, 0.005 };
        double range[2] = { 0.1, 10.0 };
    };

    // Parses "a,b[,c...]" into exactly count finite numbers.
    inline bool parse_numbers(const char* const text, double* const values, const std::size_t count) {
        const char* cursor = text;
        for (std::size_t index = 0; index < count; ++index) {
            char* end_pointer = nullptr;
            values[index] = std::strtod(cursor, &end_pointer);
            if ((end_pointer == cursor) || !std::isfinite(values[index])) {
                return false;
            }
            cursor = end_pointer;
            if (index + 1 < count) {
                if (*cursor != ',') {
                    return false;
                }
                ++cursor;
            }
        }
        return *cursor == 0;
    }

    // Takes --depth, --depth-noise c0,c1,c2 and --depth-range min,max at argv[index], returning whether it was one of them;
    // failed is set when its value is missing or invalid.
    inline bool parse_option(const int argc, char* argv[], int& index, options& depth, bool& failed) {
        failed = false;
        if (std::strcmp(argv[index], "--depth") == 0) {
            depth.enabled = true;
            return true;
        }
        const bool noise = std::strcmp(argv[index], "--depth-noise") == 0;
        if (!noise && (std::strcmp(argv[index], "--depth-range") != 0)) {
            return false;
        }
        if (index + 1 >= argc) {
            std::fprintf(stderr, "Missing value for option: %s\n", argv[index]);
            failed = true;
            return true;
        }
        const char* const text = argv[++index];
        double values[3] = { 0.0, 0.0, 0.0 };
        const bool valid = noise ? (parse_numbers(text, &values[0], 3) && (values[0] >= 0.0) && (values[1] >= 0.0) && (values[2] >= 0.0) && ((values[0] + values[1] + values[2]) > 0.0)) : (parse_numbers(text, &values[0], 2) && (values[0] >= 0.0) && (values[1] > values[0]));
        if (!valid) {
            std::fprintf(stderr, "Invalid value for %s: '%s' (%s).\n", argv[index - 1], text, noise ? "expected three non-negative coefficients c0,c1,c2, not all zero" : "expected minimum,maximum metres with maximum above minimum");
            failed = true;
            return true;
        }
        for (std::size_t value = 0; value < (noise ? 3u : 2u); ++value) {
            (noise ? depth.noise : depth.range)[value] = values[value];
        }
        depth.enabled = true;
        return true;
    }

    inline void print_usage() {
        std::printf("    --depth   - Give the library the scene's depth camera (/sensor/depth_01) with each frame: a metric RGB-D map.\n");
        std::printf("    --depth-noise - The depth's standard deviation c0 + c1 d + c2 d^2 metres, default 0.002,0,0.005 (implies --depth).\n");
        std::printf("    --depth-range - The depths taken, in metres, default 0.1,10 (implies --depth).\n");
    }

    class feeder final {
    private:
        constexpr static const int sensor_id = 2;
        const mcap::channel_type* channel = nullptr;
        zeroslam_sensor_parameters_depth_struct parameters{};
        // The depth frames read and not yet given to the library, which takes them with the next camera frame.
        std::vector<std::size_t> messages;

    public:
        bool enabled() const {
            return this->channel != nullptr;
        }

        // Finds the depth camera and reads its encoding from its first frame, as the rig is set before any data.
        bool prepare(mcap& reader, const options& depth, const int camera_id, const std::size_t camera_count, std::string& error) {
            if (!depth.enabled) {
                return true;
            }
            if (camera_count > 1) {
                error = "--depth takes a single camera, not a rig of cameras";
                return false;
            }
            const mcap::channel_type* const found = dataset::find_depth_channel(reader);
            if (found == nullptr) {
                error = "The scene has no depth camera (/sensor/depth_01) for --depth";
                return false;
            }
            const std::vector<mcap::message_index_type>& index = reader.get_message_index();
            for (std::size_t message_number = 0; message_number < index.size(); ++message_number) {
                if (index[message_number].channel_id != found->id) {
                    continue;
                }
                mcap::message_type message;
                cdr::image image;
                bool floating_point = false;
                double scale = 1.0;
                if (!reader.read_message(message_number, message) || !cdr::read_image(message.data, message.length, image) || !dataset::depth_encoding_scale(image.encoding, floating_point, scale)) {
                    error = "The depth camera's first frame is not a 32FC1 or mono16 image";
                    return false;
                }
                this->parameters.camera_id = camera_id;
                this->parameters.encoding = floating_point ? zeroslam_depth_encoding_float_32 : zeroslam_depth_encoding_unsigned_16;
                this->parameters.measure = zeroslam_depth_measure_axial;
                this->parameters.registration = zeroslam_depth_registration_camera;
                this->parameters.scale = scale;
                this->parameters.minimum_depth = depth.range[0];
                this->parameters.maximum_depth = depth.range[1];
                this->parameters.noise_constant = depth.noise[0];
                this->parameters.noise_linear = depth.noise[1];
                this->parameters.noise_quadratic = depth.noise[2];
                // TUM RGB-D's depth frames are stored at their rgb frame's time; other scenes pair within half a 30 Hz frame.
                this->parameters.pairing_tolerance = 16000000;
                this->channel = found;
                return true;
            }
            error = "The depth camera has no frames";
            return false;
        }

        const char* topic() const {
            return (this->channel != nullptr) ? this->channel->topic.c_str() : "";
        }

        // Holds the message when it is the depth camera's.
        void note(const unsigned short channel_id, const std::size_t message_number) {
            if ((this->channel != nullptr) && (channel_id == this->channel->id)) {
                this->messages.push_back(message_number);
            }
        }

        // Lets the held messages go, as with a skipped camera frame.
        void clear() {
            this->messages.clear();
        }

        // Sets the rig of the camera, the first of the entries (the rest are the other sensors), with the depth camera registered
        // to it when there is one.
        zeroslam_return_enum set_sensor_rig(const zeroslam::system& slam, std::vector<zeroslam_sensor_rig_struct> rig) {
            if (this->enabled()) {
                zeroslam_sensor_rig_struct depth{};
                depth.type = zeroslam_sensor_depth;
                depth.sensor_id = feeder::sensor_id;
                depth.parameters_length = static_cast<int>(sizeof(this->parameters));
                depth.parameters_data = &this->parameters;
                rig.push_back(depth);
            }
            return slam.set_sensor_rig(rig.data(), static_cast<int>(rig.size()));
        }

        // Gives the camera frame, with the depth frames held since the last one before it.
        zeroslam_return_enum set_sensor_data(const zeroslam::system& slam, mcap& reader, const zeroslam_sensor_data_struct& camera_frame, const unsigned int width, const unsigned int height) {
            std::vector<cdr::image> images;
            std::vector<zeroslam_sensor_data_struct> measurements;
            for (const std::size_t message_number : this->messages) {
                mcap::message_type message;
                cdr::image image;
                if (!reader.read_message(message_number, message) || !cdr::read_image(message.data, message.length, image)) {
                    std::fprintf(stderr, "A depth message does not decode.\n");
                    return zeroslam_return_failure_invalid_sensor_data;
                }
                const std::size_t pixel_size = (this->parameters.encoding == zeroslam_depth_encoding_float_32) ? 4u : 2u;
                const bool encoding_matches = (image.encoding == ((pixel_size == 4u) ? "32FC1" : "mono16")) && (image.is_bigendian == 0);
                if (!encoding_matches || (image.width != width) || (image.height != height) || (image.step != pixel_size * image.width) || (image.data.size() != pixel_size * static_cast<std::size_t>(image.width) * image.height)) {
                    std::fprintf(stderr, "A depth message is not a little endian %s image of the camera's size.\n", (pixel_size == 4u) ? "32FC1" : "mono16");
                    return zeroslam_return_failure_invalid_sensor_data;
                }
                zeroslam_sensor_data_struct measurement{};
                measurement.timestamp = static_cast<long long>(message.log_time);
                measurement.sensor_id = feeder::sensor_id;
                measurement.measurement_length = static_cast<int>(image.data.size());
                images.push_back(static_cast<cdr::image&&>(image));
                measurements.push_back(measurement);
            }
            this->messages.clear();
            for (std::size_t index = 0; index < images.size(); ++index) {
                measurements[index].measurement_data = images[index].data.data();
            }
            measurements.push_back(camera_frame);
            return slam.set_sensor_data(measurements.data(), static_cast<int>(measurements.size()));
        }
    };

    // The error against the ground truth without a fitted scale, which a metric map is judged by.
    template <typename pose_type>
    inline void print_rigid_evaluation(const char* const trajectory_path, const std::vector<pose_type>& poses) {
        std::vector<dataset::trajectory_pose> ground_truth;
        if (!dataset::load_trajectory(trajectory_path, ground_truth)) {
            return;
        }
        std::vector<dataset::trajectory_pose> estimated;
        for (const pose_type& pose : poses) {
            dataset::trajectory_pose converted;
            converted.timestamp_nanoseconds = pose.timestamp;
            converted.x_coordinate = pose.centre[0];
            converted.y_coordinate = pose.centre[1];
            converted.z_coordinate = pose.centre[2];
            converted.quaternion_x = pose.quaternion_xyzw[0];
            converted.quaternion_y = pose.quaternion_xyzw[1];
            converted.quaternion_z = pose.quaternion_xyzw[2];
            converted.quaternion_w = pose.quaternion_xyzw[3];
            estimated.push_back(converted);
        }
        const metrics::result rigid = metrics::evaluate(ground_truth, estimated, true, metrics::default_association_tolerance_nanoseconds, false);
        if (rigid.valid) {
            std::printf("  Rigid:     rmse %.6f m, max %.6f m (first pose anchored alignment, no scale)\n", rigid.ate_rmse, rigid.ate_maximum);
        }
    }
}

#endif // ZEROSLAM_TOOLS_PROCESS_DEPTH_HPP
