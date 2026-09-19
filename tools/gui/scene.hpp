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
#ifndef ZEROSLAM_TOOLS_GUI_SCENE_HPP
#define ZEROSLAM_TOOLS_GUI_SCENE_HPP

#include "cdr.hpp"
#include "dataset.hpp"
#include "file.hpp"
#include "mcap.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace scene {
    constexpr static const std::size_t camera_parameter_count = 12;

    enum class image_semantics {
        visual,
        depth
    };

    struct decoded_image {
        unsigned int width = 0;
        unsigned int height = 0;
        long long timestamp_nanoseconds = 0;
        std::vector<unsigned char> rgb;
        std::vector<unsigned char> grey;
        std::vector<float> metres;

        void clear() {
            this->width = 0;
            this->height = 0;
            this->timestamp_nanoseconds = 0;
            this->rgb.clear();
            this->grey.clear();
            this->metres.clear();
        }

        bool is_valid() const {
            return (this->width > 0) && (this->height > 0);
        }
    };

    struct extrinsic {
        bool valid = false;
        double translation[3] = {};
        double rotation[4] = { 0.0, 0.0, 0.0, 1.0 };
    };

    struct image_channel {
        std::string topic;
        std::string sensor_name;
        std::string encoding;
        unsigned int width = 0;
        unsigned int height = 0;
        image_semantics semantics = image_semantics::visual;
        bool has_intrinsics = false;
        double fx = 0.0;
        double fy = 0.0;
        double cx = 0.0;
        double cy = 0.0;
        double camera_parameters[camera_parameter_count] = {};
        bool distortion_model_recognised = true;
        double depth_scale = 1.0;
        bool selected_for_slam = false;
        extrinsic mounting;
        std::vector<std::size_t> messages;
        std::vector<long long> log_times;

        bool is_depth() const {
            return this->semantics == image_semantics::depth;
        }
    };

    struct imu_sample {
        long long timestamp_nanoseconds = 0;
        double angular_velocity[3] = {};
        double linear_acceleration[3] = {};
    };

    struct imu_channel {
        std::string topic;
        std::string sensor_name;
        bool selected_for_slam = false;
        extrinsic mounting;
        std::vector<imu_sample> samples;
    };

    class scene final {
    private:
        mutable mcap reader;
        mutable std::mutex reader_mutex;
        bool scene_loaded = false;

        std::vector<image_channel> image_channels_data;
        std::vector<imu_channel> imu_channels_data;
        std::vector<dataset::trajectory_pose> ground_truth_data;
        long long time_first = 0;
        long long time_last = 0;

    public:
        scene() = default;
        scene(const scene&) = delete;
        scene& operator=(const scene&) = delete;
        scene(scene&&) = delete;
        scene& operator=(scene&&) = delete;

        ~scene() = default;

        void unload() {
            this->image_channels_data.clear();
            this->imu_channels_data.clear();
            this->ground_truth_data.clear();
            this->reader.close();
            this->scene_loaded = false;
            this->time_first = 0;
            this->time_last = 0;
        }

        bool is_loaded() const {
            return this->scene_loaded;
        }

        const std::vector<image_channel>& image_channels() const {
            return this->image_channels_data;
        }

        std::vector<image_channel>& image_channels() {
            return this->image_channels_data;
        }

        const std::vector<imu_channel>& imu_channels() const {
            return this->imu_channels_data;
        }

        std::vector<imu_channel>& imu_channels() {
            return this->imu_channels_data;
        }

        const std::vector<dataset::trajectory_pose>& ground_truth() const {
            return this->ground_truth_data;
        }

        long long begin_time() const {
            return this->time_first;
        }

        long long end_time() const {
            return this->time_last;
        }

        long long duration_nanoseconds() const {
            return (this->time_last > this->time_first) ? (this->time_last - this->time_first) : 0;
        }

        bool load(const std::string& path, std::string& error) {
            this->unload();

            if (!this->reader.open(path, error)) {
                this->unload();
                return false;
            }

            dataset::mcap_scene_information information;
            if (!dataset::inspect_mcap_scene(this->reader, information, error, false)) {
                this->unload();
                return false;
            }

            if (!this->index_image_channels(information, error)) {
                this->unload();
                return false;
            }
            this->index_imu_channels(information);
            this->index_extrinsics(information);
            this->index_ground_truth(information);
            this->compute_timeline();

            if (this->image_channels_data.empty() && this->imu_channels_data.empty()) {
                error = "the scene holds no camera or inertial channels";
                this->unload();
                return false;
            }

            this->scene_loaded = true;
            return true;
        }

        static long long message_index_at(const image_channel& channel, const long long timestamp_nanoseconds) {
            long long low = 0;
            long long high = static_cast<long long>(channel.log_times.size()) - 1;
            long long result = -1;
            while (low <= high) {
                const long long middle = low + ((high - low) / 2);
                if (channel.log_times[static_cast<std::size_t>(middle)] <= timestamp_nanoseconds) {
                    result = middle;
                    low = middle + 1;
                }
                else {
                    high = middle - 1;
                }
            }
            return result;
        }

        static long long sample_index_at(const imu_channel& channel, const long long timestamp_nanoseconds) {
            long long low = 0;
            long long high = static_cast<long long>(channel.samples.size()) - 1;
            long long result = -1;
            while (low <= high) {
                const long long middle = low + ((high - low) / 2);
                if (channel.samples[static_cast<std::size_t>(middle)].timestamp_nanoseconds <= timestamp_nanoseconds) {
                    result = middle;
                    low = middle + 1;
                }
                else {
                    high = middle - 1;
                }
            }
            return result;
        }

        bool decode_image(const image_channel& channel, const std::size_t index, decoded_image& output) const {
            output.clear();
            if (index >= channel.messages.size()) {
                return false;
            }
            cdr::image image;
            {
                const std::lock_guard<std::mutex> lock(this->reader_mutex);
                mcap::message_type message;
                if (!this->reader.read_message(channel.messages[index], message) || !cdr::read_image(message.data, message.length, image)) {
                    return false;
                }
            }
            const std::size_t pixels = static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height);
            if (pixels == 0) {
                return false;
            }
            const std::size_t bytes_per_pixel = (image.encoding == "mono16") ? 2 : ((image.encoding == "32FC1") ? 4 : ((image.encoding == "rgb8") ? 3 : 1));
            if ((static_cast<std::size_t>(image.step) != static_cast<std::size_t>(image.width) * bytes_per_pixel) || (image.data.size() < pixels * bytes_per_pixel)) {
                return false;
            }

            output.width = image.width;
            output.height = image.height;
            output.timestamp_nanoseconds = channel.log_times[index];
            output.rgb.resize(pixels * 3);
            output.grey.resize(pixels);

            std::vector<double> scalars;
            bool have_scalars = false;
            if (image.encoding == "mono8") {
                scalars.resize(pixels);
                for (std::size_t i = 0; i < pixels; ++i) {
                    scalars[i] = static_cast<double>(image.data[i]);
                }
                have_scalars = true;
            }
            else if (image.encoding == "mono16") {
                scalars.resize(pixels);
                for (std::size_t i = 0; i < pixels; ++i) {
                    const unsigned int low = image.data[(i * 2) + 0];
                    const unsigned int high = image.data[(i * 2) + 1];
                    scalars[i] = static_cast<double>((high << 8u) | low);
                }
                have_scalars = true;
            }
            else if (image.encoding == "32FC1") {
                scalars.resize(pixels);
                for (std::size_t i = 0; i < pixels; ++i) {
                    float value = 0.0f;
                    const unsigned char bytes[4] = { image.data[(i * 4) + 0], image.data[(i * 4) + 1], image.data[(i * 4) + 2], image.data[(i * 4) + 3] };
                    std::memcpy(&value, &bytes[0], sizeof(value));
                    scalars[i] = static_cast<double>(value);
                }
                have_scalars = true;
            }
            else if (image.encoding != "rgb8") {
                return false;
            }

            if (channel.is_depth()) {
                if (!have_scalars) {
                    return false;
                }
                output.metres.resize(pixels);
                for (std::size_t i = 0; i < pixels; ++i) {
                    double range = scalars[i] * channel.depth_scale;
                    if (!(range > 0.0) || (range != range) || (range > 1.0e6)) {
                        range = 0.0;
                    }
                    output.metres[i] = static_cast<float>(range);
                }
                constexpr static const float display_near_metres = 0.2f;
                constexpr static const float display_far_metres = 10.0f;
                for (std::size_t i = 0; i < pixels; ++i) {
                    const float range = output.metres[i];
                    if (!(range > 0.0f)) {
                        output.grey[i] = 0;
                        output.rgb[(i * 3) + 0] = 0;
                        output.rgb[(i * 3) + 1] = 0;
                        output.rgb[(i * 3) + 2] = 0;
                        continue;
                    }
                    float normalised = (range - display_near_metres) / (display_far_metres - display_near_metres);
                    normalised = (normalised < 0.0f) ? 0.0f : ((normalised > 1.0f) ? 1.0f : normalised);
                    const unsigned char value = static_cast<unsigned char>(normalised * 255.0f);
                    output.grey[i] = value;
                    output.rgb[(i * 3) + 0] = static_cast<unsigned char>(255 - value);
                    output.rgb[(i * 3) + 1] = static_cast<unsigned char>((value < 128) ? (value * 2) : ((255 - value) * 2));
                    output.rgb[(i * 3) + 2] = value;
                }
                return true;
            }

            if (image.encoding == "rgb8") {
                for (std::size_t i = 0; i < pixels; ++i) {
                    const unsigned char red = image.data[(i * 3) + 0];
                    const unsigned char green = image.data[(i * 3) + 1];
                    const unsigned char blue = image.data[(i * 3) + 2];
                    output.rgb[(i * 3) + 0] = red;
                    output.rgb[(i * 3) + 1] = green;
                    output.rgb[(i * 3) + 2] = blue;
                    output.grey[i] = static_cast<unsigned char>(((77u * static_cast<unsigned int>(red)) + (150u * static_cast<unsigned int>(green)) + (29u * static_cast<unsigned int>(blue))) >> 8);
                }
                return true;
            }

            double maximum = 255.0;
            if (image.encoding != "mono8") {
                maximum = 0.0;
                for (std::size_t i = 0; i < pixels; ++i) {
                    maximum = (scalars[i] > maximum) ? scalars[i] : maximum;
                }
                if (!(maximum > 0.0)) {
                    maximum = 1.0;
                }
            }
            for (std::size_t i = 0; i < pixels; ++i) {
                double normalised = (scalars[i] / maximum) * 255.0;
                normalised = (normalised < 0.0) ? 0.0 : ((normalised > 255.0) ? 255.0 : normalised);
                const unsigned char value = static_cast<unsigned char>(normalised);
                output.grey[i] = value;
                output.rgb[(i * 3) + 0] = value;
                output.rgb[(i * 3) + 1] = value;
                output.rgb[(i * 3) + 2] = value;
            }
            return true;
        }

    private:
        bool index_image_channels(const dataset::mcap_scene_information& information, std::string& error) {
            for (const dataset::camera_information& camera : information.cameras) {
                if (camera.frames == 0) {
                    continue;
                }
                image_channel channel;
                channel.topic = camera.image_topic;
                channel.sensor_name = camera.camera_name;
                channel.encoding = camera.encoding;
                channel.width = camera.width;
                channel.height = camera.height;
                channel.has_intrinsics = camera.camera_info;
                channel.fx = camera.fx;
                channel.fy = camera.fy;
                channel.cx = camera.cx;
                channel.cy = camera.cy;
                if (!camera.camera_infos_data.empty()) {
                    channel.has_intrinsics = dataset::camera_info_to_parameters(
                        camera.camera_infos_data.front(),
                        &channel.camera_parameters[0],
                        camera_parameter_count,
                        channel.distortion_model_recognised
                    );
                }
                const bool named_depth = dataset::is_valid_sensor_name(camera.camera_name, "depth");
                const bool encoded_depth = (camera.encoding == "mono16") || (camera.encoding == "32FC1");
                if (named_depth || encoded_depth) {
                    channel.semantics = image_semantics::depth;
                    if (camera.encoding == "32FC1") {
                        channel.depth_scale = 1.0;
                    }
                    else if (camera.encoding == "mono16") {
                        channel.depth_scale = 1.0 / 1000.0;
                    }
                    else {
                        channel.depth_scale = 1.0 / 25.0;
                    }
                }
                const mcap::channel_type* const record = this->reader.find_channel(camera.image_topic);
                if (record == nullptr) {
                    error = "the camera topic '" + camera.image_topic + "' has no channel record";
                    return false;
                }
                const std::vector<mcap::message_index_type>& message_index = this->reader.get_message_index();
                for (std::size_t index = 0; index < message_index.size(); ++index) {
                    if (message_index[index].channel_id == record->id) {
                        channel.messages.push_back(index);
                        channel.log_times.push_back(static_cast<long long>(message_index[index].log_time));
                    }
                }
                if (channel.messages.empty()) {
                    continue;
                }
                channel.selected_for_slam = (channel.semantics == image_semantics::visual) && !this->has_selected_visual_channel();
                this->image_channels_data.push_back(channel);
            }
            return true;
        }

        bool has_selected_visual_channel() const {
            for (const image_channel& channel : this->image_channels_data) {
                if (channel.selected_for_slam) {
                    return true;
                }
            }
            return false;
        }

        void index_imu_channels(const dataset::mcap_scene_information& information) {
            for (const dataset::imu_information& imu : information.imus) {
                if (imu.imu_data.empty()) {
                    continue;
                }
                imu_channel channel;
                channel.topic = imu.imu_topic;
                channel.sensor_name = imu.imu_name;
                channel.samples.reserve(imu.imu_data.size());
                for (std::size_t i = 0; i < imu.imu_data.size(); ++i) {
                    const cdr::imu& message = imu.imu_data[i];
                    imu_sample sample;
                    sample.timestamp_nanoseconds = (i < imu.imu_log_times.size()) ? static_cast<long long>(imu.imu_log_times[i]) : 0;
                    for (int axis = 0; axis < 3; ++axis) {
                        sample.angular_velocity[axis] = message.angular_velocity[axis];
                        sample.linear_acceleration[axis] = message.linear_acceleration[axis];
                    }
                    channel.samples.push_back(sample);
                }
                channel.selected_for_slam = this->imu_channels_data.empty();
                this->imu_channels_data.push_back(channel);
            }
        }

        void index_extrinsics(const dataset::mcap_scene_information& information) {
            const auto find_mounting = [&](const std::string& sensor_name, extrinsic& mounting) {
                const std::string child = "sensor/" + sensor_name;
                for (const cdr::transform_stamped& transform : information.dynamics) {
                    if ((transform.frame_header.frame_id != "ego") || (transform.child_frame_id != child)) {
                        continue;
                    }
                    mounting.valid = true;
                    for (int i = 0; i < 3; ++i) {
                        mounting.translation[i] = transform.translation[i];
                    }
                    for (int i = 0; i < 4; ++i) {
                        mounting.rotation[i] = transform.rotation[i];
                    }
                    return;
                }
            };
            for (image_channel& channel : this->image_channels_data) {
                find_mounting(channel.sensor_name, channel.mounting);
            }
            for (imu_channel& channel : this->imu_channels_data) {
                find_mounting(channel.sensor_name, channel.mounting);
            }
        }

        void index_ground_truth(const dataset::mcap_scene_information& information) {
            for (const cdr::transform_stamped& transform : information.dynamics) {
                if ((transform.frame_header.frame_id != "root") || (transform.child_frame_id != "ego")) {
                    continue;
                }
                dataset::trajectory_pose pose;
                pose.timestamp_nanoseconds = (static_cast<long long>(transform.frame_header.stamp.sec) * 1000000000LL) + static_cast<long long>(transform.frame_header.stamp.nanosec);
                pose.x_coordinate = transform.translation[0];
                pose.y_coordinate = transform.translation[1];
                pose.z_coordinate = transform.translation[2];
                pose.quaternion_x = transform.rotation[0];
                pose.quaternion_y = transform.rotation[1];
                pose.quaternion_z = transform.rotation[2];
                pose.quaternion_w = transform.rotation[3];
                this->ground_truth_data.push_back(pose);
            }
        }

        void compute_timeline() {
            bool first = true;
            const auto extend = [&](const long long timestamp) {
                if (first) {
                    this->time_first = timestamp;
                    this->time_last = timestamp;
                    first = false;
                    return;
                }
                if (timestamp < this->time_first) {
                    this->time_first = timestamp;
                }
                if (timestamp > this->time_last) {
                    this->time_last = timestamp;
                }
            };
            for (const image_channel& channel : this->image_channels_data) {
                for (const long long log_time : channel.log_times) {
                    extend(log_time);
                }
            }
            for (const imu_channel& channel : this->imu_channels_data) {
                for (const imu_sample& sample : channel.samples) {
                    extend(sample.timestamp_nanoseconds);
                }
            }
        }
    };
}

#endif // ZEROSLAM_TOOLS_GUI_SCENE_HPP
