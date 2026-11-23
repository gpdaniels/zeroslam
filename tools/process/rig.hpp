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
#ifndef ZEROSLAM_TOOLS_PROCESS_RIG_HPP
#define ZEROSLAM_TOOLS_PROCESS_RIG_HPP

#include "cdr.hpp"
#include "dataset.hpp"
#include "imu.hpp"
#include "mcap.hpp"
#include "wheel_feed.hpp"
#include "zeroslam/zeroslam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// Runs several cameras of a scene together as a rig (--cameras image_01,image_02): the first the primary camera whose poses
// are written, each at the extrinsic its first 'ego -> sensor/[name]' transform on /tf gives, and every moment an image of each
// camera has is fed to the library as one frame.
namespace process_rig {
    struct selected_camera {
        std::string name;
        const mcap::channel_type* image = nullptr;
        const mcap::channel_type* info = nullptr;
        zeroslam_sensor_parameters_camera_model_struct parameters{};
        bool intrinsics_found = false;
        bool extrinsic_found = false;
        double extrinsic[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
    };

    // Comma separated camera names, none empty.
    inline bool parse_names(const char* const text, std::vector<std::string>& names) {
        names.clear();
        const std::string list(text);
        size_t begin = 0;
        while (begin <= list.size()) {
            const size_t end = std::min(list.find(',', begin), list.size());
            if (end == begin) {
                return false;
            }
            names.push_back(list.substr(begin, end - begin));
            begin = end + 1;
        }
        return true;
    }

    // degrees,metres.
    inline bool parse_perturbation(const char* const text, double (&perturbation)[2]) {
        char* end_pointer = nullptr;
        perturbation[0] = std::strtod(text, &end_pointer);
        if ((end_pointer == text) || (*end_pointer != ',')) {
            return false;
        }
        const char* const second = end_pointer + 1;
        perturbation[1] = std::strtod(second, &end_pointer);
        return (end_pointer != second) && (*end_pointer == 0);
    }

    // Moves a pose by a turn of the given degrees about (1, 1, 1) and a shift of the given metres along (1, -1, 1), for runs
    // that start the extrinsic refinement off the calibration.
    inline void perturb_pose(double (&pose)[7], const double degrees, const double metres) {
        const double angle = degrees * 3.14159265358979323846 / 180.0;
        const double axis = 1.0 / std::sqrt(3.0);
        const double half_sine = std::sin(0.5 * angle);
        const double turn[4] = { axis * half_sine, axis * half_sine, axis * half_sine, std::cos(0.5 * angle) };
        const double q[4] = { pose[3], pose[4], pose[5], pose[6] };
        // turn * q, both scalar last.
        pose[3] = (turn[3] * q[0]) + (turn[0] * q[3]) + (turn[1] * q[2]) - (turn[2] * q[1]);
        pose[4] = (turn[3] * q[1]) - (turn[0] * q[2]) + (turn[1] * q[3]) + (turn[2] * q[0]);
        pose[5] = (turn[3] * q[2]) + (turn[0] * q[1]) - (turn[1] * q[0]) + (turn[2] * q[3]);
        pose[6] = (turn[3] * q[3]) - (turn[0] * q[0]) - (turn[1] * q[1]) - (turn[2] * q[2]);
        pose[0] += metres * axis;
        pose[1] -= metres * axis;
        pose[2] += metres * axis;
    }

    inline bool find_cameras(const mcap& reader, const std::vector<std::string>& names, std::vector<selected_camera>& selected) {
        for (const std::string& name : names) {
            selected_camera entry;
            entry.name = name;
            for (const mcap::channel_type& channel : reader.get_channels()) {
                const mcap::schema_type* schema = reader.find_schema(channel.schema_id);
                if (schema == nullptr) {
                    continue;
                }
                if ((schema->name == "sensor_msgs/msg/Image") && (channel.topic == "/sensor/" + name)) {
                    entry.image = &channel;
                }
                if ((schema->name == "sensor_msgs/msg/CameraInfo") && (channel.topic == "/sensor/" + name + "/camera_info")) {
                    entry.info = &channel;
                }
            }
            if ((entry.image == nullptr) || (entry.info == nullptr)) {
                std::fprintf(stderr, "The scene has no camera '%s' (an image topic '/sensor/%s' and its camera info).\n", name.c_str(), name.c_str());
                return false;
            }
            for (const selected_camera& existing : selected) {
                if (existing.name == name) {
                    std::fprintf(stderr, "The camera '%s' is named twice in --cameras.\n", name.c_str());
                    return false;
                }
            }
            selected.push_back(entry);
        }
        return !selected.empty();
    }

    // Gives the library the cameras and their extrinsics, the cameras other than the primary perturbed when asked.
    inline bool declare_rig(const zeroslam::system& slam, std::vector<selected_camera>& selected, const double (&perturbation)[2], const int verbosity, const process_imu::settings& imu, process_imu::selected_imu& imu_selected) {
        std::vector<zeroslam_sensor_rig_struct> rig(selected.size());
        std::vector<zeroslam_sensor_extrinsics_struct> extrinsics(selected.size());
        for (size_t index = 0; index < selected.size(); ++index) {
            selected_camera& entry = selected[index];
            rig[index].type = zeroslam_sensor_camera;
            rig[index].sensor_id = 1 + static_cast<int>(index);
            rig[index].parameters_length = static_cast<int>(sizeof(entry.parameters));
            rig[index].parameters_data = &entry.parameters;
            if (!entry.extrinsic_found && (verbosity >= 2)) {
                std::printf("The scene does not pose camera %s on the ego: it is taken to sit at the ego's origin.\n", entry.name.c_str());
            }
            extrinsics[index].sensor_id = rig[index].sensor_id;
            extrinsics[index].refinement = zeroslam_extrinsic_fixed;
            for (int i = 0; i < 7; ++i) {
                extrinsics[index].pose[i] = entry.extrinsic[i];
            }
            if ((index > 0) && ((perturbation[0] != 0.0) || (perturbation[1] != 0.0))) {
                perturb_pose(extrinsics[index].pose, perturbation[0], perturbation[1]);
            }
            if (verbosity >= 3) {
                std::printf("Camera %s (sensor %d): %dx%d, extrinsic [%.6f %.6f %.6f] [%.6f %.6f %.6f %.6f].\n", entry.name.c_str(), rig[index].sensor_id, entry.parameters.width, entry.parameters.height, extrinsics[index].pose[0], extrinsics[index].pose[1], extrinsics[index].pose[2], extrinsics[index].pose[3], extrinsics[index].pose[4], extrinsics[index].pose[5], extrinsics[index].pose[6]);
            }
        }
        process_imu::declare(imu, imu_selected, rig, extrinsics, verbosity);
        wheel_feed::declare_registered(rig);
        if (slam.set_sensor_rig(rig.data(), static_cast<int>(rig.size())) != zeroslam_return_success) {
            std::fprintf(stderr, "The library rejected the camera rig.\n");
            return false;
        }
        if (slam.set_sensor_extrinsics(extrinsics.data(), static_cast<int>(extrinsics.size())) != zeroslam_return_success) {
            std::fprintf(stderr, "The library rejected the cameras' extrinsics.\n");
            return false;
        }
        if (!wheel_feed::set_registered_extrinsic(slam)) {
            std::fprintf(stderr, "The library rejected the wheel odometer's extrinsic.\n");
            return false;
        }
        return true;
    }

    // Writes the cameras' extrinsics as the library holds them at the end of the run, refined where it was asked to.
    inline void report_extrinsics(const zeroslam::system& slam) {
        int length = 0;
        if (slam.get_sensor_extrinsics(nullptr, &length) != zeroslam_return_failure_insufficient_data_length) {
            return;
        }
        std::vector<zeroslam_sensor_extrinsics_struct> extrinsics(static_cast<size_t>(length));
        if (slam.get_sensor_extrinsics(extrinsics.data(), &length) != zeroslam_return_success) {
            return;
        }
        for (const zeroslam_sensor_extrinsics_struct& extrinsic : extrinsics) {
            std::printf("Extrinsic of camera %d (%s): [%.6f %.6f %.6f] [%.6f %.6f %.6f %.6f]\n", extrinsic.sensor_id, (extrinsic.refinement == zeroslam_extrinsic_refine) ? "refined" : "held", extrinsic.pose[0], extrinsic.pose[1], extrinsic.pose[2], extrinsic.pose[3], extrinsic.pose[4], extrinsic.pose[5], extrinsic.pose[6]);
        }
    }

    // Feeds the scene's cameras to the library, frame by frame: the frames' timestamps, the primary camera's intrinsics and the
    // frame count come back as the single camera run gives them.
    inline bool feed(mcap& reader, const zeroslam::system& slam, const std::vector<std::string>& names, const double (&perturbation)[2], const size_t frame_limit, size_t skip_count, const int verbosity, const std::atomic<bool>& stop, const double focal_scale, std::vector<long long>& timestamps, size_t& frames, zeroslam_sensor_parameters_camera_model_struct& primary, const process_imu::settings& imu = process_imu::settings()) {
        std::vector<selected_camera> selected;
        if (!find_cameras(reader, names, selected)) {
            return false;
        }
        process_imu::selected_imu imu_selected;
        if (!process_imu::find(reader, imu, imu_selected)) {
            return false;
        }
        const mcap::channel_type* tf_channel = nullptr;
        for (const mcap::channel_type& channel : reader.get_channels()) {
            const mcap::schema_type* schema = reader.find_schema(channel.schema_id);
            if ((schema != nullptr) && (schema->name == "tf2_msgs/msg/TFMessage") && (channel.topic == "/tf")) {
                tf_channel = &channel;
            }
        }
        bool rig_set = false;
        long long current_timestamp = -1;
        std::vector<size_t> image_messages;
        const auto flush = [&]() {
            std::vector<cdr::image> images(selected.size());
            std::vector<unsigned char> present(selected.size(), 0);
            for (const size_t image_index : image_messages) {
                mcap::message_type buffered;
                if (!reader.read_message(image_index, buffered)) {
                    std::fprintf(stderr, "Failed to read an image message: %s.\n", reader.get_read_error().c_str());
                    return false;
                }
                for (size_t index = 0; index < selected.size(); ++index) {
                    if (buffered.channel_id != selected[index].image->id) {
                        continue;
                    }
                    if (!cdr::read_image(buffered.data, buffered.length, images[index])) {
                        std::fprintf(stderr, "Image message %zu of %s does not decode.\n", frames, selected[index].name.c_str());
                        return false;
                    }
                    present[index] = 1;
                }
            }
            image_messages.clear();
            size_t present_count = 0;
            for (const unsigned char flag : present) {
                present_count += flag;
            }
            if ((present_count == 0) || (frames >= frame_limit)) {
                return true;
            }
            if (present_count != selected.size()) {
                if (verbosity >= 2) {
                    std::printf("The frame at %lld lacks an image of a camera (the cameras are not synchronised), skipped.\n", current_timestamp);
                }
                return true;
            }
            if (skip_count > 0) {
                --skip_count;
                return true;
            }
            std::vector<std::vector<unsigned char>> greyscale(selected.size());
            std::vector<zeroslam_sensor_data_struct> data(selected.size());
            for (size_t index = 0; index < selected.size(); ++index) {
                selected_camera& entry = selected[index];
                const cdr::image& image = images[index];
                if (!entry.intrinsics_found) {
                    std::fprintf(stderr, "An image of %s arrived before valid camera intrinsics.\n", entry.name.c_str());
                    return false;
                }
                const bool dimensions_valid = (image.width > 0) && (image.height > 0) && (image.width <= 4096) && (image.height <= 4096);
                const bool mono = (image.encoding == "mono8") && (image.step == image.width) && (image.data.size() == static_cast<std::size_t>(image.width) * image.height);
                const bool colour = (image.encoding == "rgb8") && (image.step == 3u * image.width) && (image.data.size() == 3u * static_cast<std::size_t>(image.width) * image.height);
                const bool matching = !rig_set || ((static_cast<int>(image.width) == entry.parameters.width) && (static_cast<int>(image.height) == entry.parameters.height));
                if (!dimensions_valid || !(mono || colour) || !matching) {
                    std::fprintf(stderr, "Image message %zu of %s is not a consistent mono8 or rgb8 image.\n", frames, entry.name.c_str());
                    return false;
                }
                entry.parameters.width = static_cast<int>(image.width);
                entry.parameters.height = static_cast<int>(image.height);
                data[index].timestamp = current_timestamp;
                data[index].sensor_id = 1 + static_cast<int>(index);
                data[index].measurement_length = static_cast<int>(image.width * image.height);
                if (colour) {
                    const std::size_t pixels = static_cast<std::size_t>(image.width) * image.height;
                    greyscale[index].resize(pixels);
                    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                        greyscale[index][pixel] = static_cast<unsigned char>(((77 * static_cast<unsigned int>(image.data[3 * pixel + 0])) + (150 * static_cast<unsigned int>(image.data[3 * pixel + 1])) + (29 * static_cast<unsigned int>(image.data[3 * pixel + 2]))) >> 8);
                    }
                    data[index].measurement_data = greyscale[index].data();
                }
                else {
                    data[index].measurement_data = const_cast<unsigned char*>(image.data.data());
                }
            }
            if (!rig_set) {
                if (!declare_rig(slam, selected, perturbation, verbosity, imu, imu_selected) || !process_imu::flush(slam, imu_selected)) {
                    return false;
                }
                primary = selected[0].parameters;
                rig_set = true;
            }
            timestamps.push_back(current_timestamp);
            const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            if (!wheel_feed::feed_registered(slam, current_timestamp)) {
                std::fprintf(stderr, "The library rejected the wheel samples before frame %zu.\n", frames + 1);
                return false;
            }
            const zeroslam_return_enum result = slam.set_sensor_data(data.data(), static_cast<int>(data.size()));
            if (result != zeroslam_return_success) {
                std::fprintf(stderr, "The library rejected frame %zu: %s.\n", frames + 1, zeroslam_return_enum_to_string(result));
                return false;
            }
            if (verbosity >= 4) {
                std::printf("Finished frame %zu, took: %f seconds\n", frames + 1, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
            }
            ++frames;
            return true;
        };
        const std::vector<mcap::message_index_type>& message_index = reader.get_message_index();
        for (size_t message_number = 0; (message_number < message_index.size()) && !stop; ++message_number) {
            const mcap::message_index_type& entry = message_index[message_number];
            const long long message_time = static_cast<long long>(entry.log_time);
            if ((current_timestamp != -1) && (message_time != current_timestamp)) {
                if (!flush()) {
                    return false;
                }
                if (frames >= frame_limit) {
                    break;
                }
            }
            current_timestamp = message_time;
            if ((imu_selected.channel != nullptr) && (entry.channel_id == imu_selected.channel->id) && !process_imu::feed(reader, slam, message_number, rig_set, imu_selected)) {
                return false;
            }
            for (selected_camera& camera : selected) {
                if (entry.channel_id == camera.image->id) {
                    image_messages.push_back(message_number);
                }
                else if ((entry.channel_id == camera.info->id) && !camera.intrinsics_found) {
                    // Any camera model of the scene, read as the single camera's run reads it (dataset::camera_info_to_model).
                    mcap::message_type message;
                    cdr::camera_info information;
                    int model = 0;
                    double parameters[32] = {};
                    size_t parameters_length = 0;
                    std::string model_error;
                    if (!reader.read_message(message_number, message) || !cdr::read_camera_info(message.data, message.length, information)) {
                        continue;
                    }
                    if (!dataset::camera_info_to_model(information, model, &parameters[0], parameters_length, model_error) || (zeroslam_camera_model_parameter_count(static_cast<zeroslam_camera_model_enum>(model)) != static_cast<int>(parameters_length))) {
                        std::fprintf(stderr, "Unsupported camera info of %s (%s).\n", camera.name.c_str(), model_error.c_str());
                        return false;
                    }
                    // Every model of the C api starts with the focal lengths, which --focal-scale scales.
                    parameters[0] *= focal_scale;
                    parameters[1] *= focal_scale;
                    camera.parameters.model = static_cast<zeroslam_camera_model_enum>(model);
                    camera.parameters.parameters_length = static_cast<int>(parameters_length);
                    for (size_t index = 0; index < parameters_length; ++index) {
                        camera.parameters.parameters[index] = parameters[index];
                    }
                    camera.intrinsics_found = true;
                }
            }
            if ((tf_channel != nullptr) && (entry.channel_id == tf_channel->id) && !rig_set) {
                mcap::message_type message;
                std::vector<cdr::transform_stamped> transforms;
                if (reader.read_message(message_number, message) && cdr::read_tf_message(message.data, message.length, transforms)) {
                    process_imu::take_extrinsic(imu, transforms, imu_selected);
                    for (const cdr::transform_stamped& transform : transforms) {
                        for (selected_camera& camera : selected) {
                            if (camera.extrinsic_found || (transform.frame_header.frame_id != "ego") || (transform.child_frame_id != ("sensor/" + camera.name))) {
                                continue;
                            }
                            for (int axis = 0; axis < 3; ++axis) {
                                camera.extrinsic[axis] = transform.translation[axis];
                            }
                            for (int component = 0; component < 4; ++component) {
                                camera.extrinsic[3 + component] = transform.rotation[component];
                            }
                            camera.extrinsic_found = true;
                        }
                    }
                }
            }
        }
        return flush();
    }
}

#endif // ZEROSLAM_TOOLS_PROCESS_RIG_HPP
