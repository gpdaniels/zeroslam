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
#ifndef ZEROSLAM_TOOLS_COMMON_DATASET_HPP
#define ZEROSLAM_TOOLS_COMMON_DATASET_HPP

#include "cdr.hpp"
#include "directory.hpp"
#include "file.hpp"
#include "mcap.hpp"
#include "paths.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// Inspection of the pieces of a dataset: frame data, timestamp files, and trajectories.
// These helpers only measure and check, the policy of what counts as valid enough belongs
// to each tool.
namespace dataset {
    inline bool read_file(const std::string& path, std::vector<unsigned char>& data) {
        data.clear();
        const gtl::file input(path.c_str());
        if (!input.is_open()) {
            return false;
        }
        gtl::file::size_type size = 0;
        if (!input.get_size(size)) {
            return false;
        }
        if (size == 0) {
            return true;
        }
        data.resize(static_cast<std::size_t>(size));
        gtl::file::size_type index = 0;
        while (index < size) {
            gtl::file::size_type remaining = size - index;
            if ((!input.read(reinterpret_cast<char*>(&data[static_cast<std::size_t>(index)]), remaining)) || (remaining == 0)) {
                data.clear();
                return false;
            }
            index += remaining;
        }
        return true;
    }

    inline bool write_all(gtl::file& file, const char* const data, const std::size_t size) {
        gtl::file::size_type index = 0;
        while (index < static_cast<gtl::file::size_type>(size)) {
            gtl::file::size_type remaining = static_cast<gtl::file::size_type>(size) - index;
            if ((!file.write(data + static_cast<std::size_t>(index), remaining)) || (remaining == 0)) {
                return false;
            }
            index += remaining;
        }
        return true;
    }

    inline bool read_text_file(const std::string& path, std::string& text) {
        std::vector<unsigned char> data;
        if (!read_file(path, data)) {
            text.clear();
            return false;
        }
        text.assign(reinterpret_cast<const char*>(data.data()), data.size());
        return true;
    }

    inline bool write_line(gtl::file& file, const char* const line) {
        return write_all(file, line, std::strlen(line));
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    inline bool write_formatted_line(gtl::file& file, const char* const format, ...) {
        char line[1024];
        va_list arguments;
        va_start(arguments, format);
        const int length = std::vsnprintf(&line[0], sizeof(line), format, arguments);
        va_end(arguments);
        if ((length < 0) || (static_cast<std::size_t>(length) >= sizeof(line))) {
            return false;
        }
        return write_line(file, &line[0]);
    }

    // Whether 'name' is a valid sensor instance name of the given type: "[type]_01" to
    // "[type]_99" ("image_01", "imu_01", ...).
    inline bool is_valid_sensor_name(const std::string& name, const std::string& type) {
        const std::string prefix = type + "_";
        if (name.size() != prefix.size() + 2) {
            return false;
        }
        if (name.compare(0, prefix.size(), prefix) != 0) {
            return false;
        }
        const char tens = name[prefix.size() + 0];
        const char ones = name[prefix.size() + 1];
        if ((tens < '0') || (tens > '9') || (ones < '0') || (ones > '9')) {
            return false;
        }
        return (tens != '0') || (ones != '0');
    }

    enum class inertial_type {
        imu,
        accelerometer,
        gyroscope,
        unknown
    };

    inline inertial_type inertial_type_of(const std::string& name) {
        if (is_valid_sensor_name(name, "imu")) {
            return inertial_type::imu;
        }
        if (is_valid_sensor_name(name, "accelerometer")) {
            return inertial_type::accelerometer;
        }
        if (is_valid_sensor_name(name, "gyroscope")) {
            return inertial_type::gyroscope;
        }
        return inertial_type::unknown;
    }

    inline bool measures_angular_velocity(const inertial_type type) {
        return (type == inertial_type::imu) || (type == inertial_type::gyroscope);
    }

    inline bool measures_linear_acceleration(const inertial_type type) {
        return (type == inertial_type::imu) || (type == inertial_type::accelerometer);
    }

    // One camera sensor's messages: the frames and their per-frame calibration.
    struct camera_information {
        std::string camera_name;
        std::string image_topic;
        std::string camera_info_topic;
        std::size_t frames = 0;
        unsigned int width = 0;
        unsigned int height = 0;
        std::string encoding;
        bool camera_info = false;
        double fx = 0.0;
        double fy = 0.0;
        double cx = 0.0;
        double cy = 0.0;
        std::size_t camera_infos = 0;
        std::vector<cdr::camera_info> camera_infos_data;
        bool frame_ids_consistent = true;
        bool log_times_consistent = true;
        std::vector<unsigned long long> image_log_times;
        std::vector<unsigned long long> camera_info_log_times;
    };

    // camera's frame count.
    struct imu_information {
        std::string imu_name;
        std::string imu_topic;
        std::size_t samples = 0;
        bool frame_ids_consistent = true;
        bool log_times_consistent = true;
        std::vector<unsigned long long> imu_log_times;
        std::vector<cdr::imu> imu_data;
    };

    // A description of a raw image scene mcap.
    struct mcap_scene_information {
        std::vector<camera_information> cameras;
        std::vector<imu_information> imus;
        std::size_t primary_camera = 0;
        std::size_t poses = 0;
        // The measurements below let a tool decide whether the file fully round trips
        // through the directory form.
        std::vector<std::string> attachments;
        std::vector<unsigned long long> dynamic_log_times;
        std::vector<cdr::transform_stamped> dynamics;
        bool dynamics_log_times_consistent = true;
        std::vector<std::string> extra_topics;

        const camera_information& primary() const {
            return cameras[primary_camera];
        }
    };

    // Whether a message's mcap log time (nanoseconds since the mcap's own epoch) is exactly
    // the same instant as its own header stamp: every text form in the project stores
    // timestamps as a plain signed 64 bit nanosecond count, so this is exact integer
    // arithmetic, not an approximation.
    inline bool log_time_matches_stamp(const unsigned long long log_time, const cdr::time& stamp) {
        return stamp.nanoseconds() == static_cast<long long>(log_time);
    }

    inline bool find_camera_channels(
        const mcap& reader,
        const mcap::channel_type*& image_channel,
        const mcap::channel_type*& info_channel
    ) {
        image_channel = nullptr;
        info_channel = nullptr;
        for (const mcap::channel_type& channel : reader.get_channels()) {
            const mcap::schema_type* schema = reader.find_schema(channel.schema_id);
            if ((schema == nullptr) || (schema->name != "sensor_msgs/msg/Image")) {
                continue;
            }
            const bool preferred = (channel.topic.rfind("/sensor/image_", 0) == 0);
            const bool current_preferred = (image_channel != nullptr) && (image_channel->topic.rfind("/sensor/image_", 0) == 0);
            if ((image_channel == nullptr) || (preferred && !current_preferred)) {
                image_channel = &channel;
            }
        }
        if (image_channel == nullptr) {
            return false;
        }
        for (const mcap::channel_type& channel : reader.get_channels()) {
            const mcap::schema_type* schema = reader.find_schema(channel.schema_id);
            if ((schema != nullptr) && (schema->name == "sensor_msgs/msg/CameraInfo") && (channel.topic == image_channel->topic + "/camera_info")) {
                info_channel = &channel;
                return true;
            }
        }
        return false;
    }

    inline bool camera_info_to_parameters(
        const cdr::camera_info& information,
        double* const parameters,
        const std::size_t count,
        bool& distortion_model_recognised
    ) {
        constexpr static const std::size_t required = 12;
        distortion_model_recognised = true;
        if ((parameters == nullptr) || (count < required)) {
            return false;
        }
        for (std::size_t index = 0; index < count; ++index) {
            parameters[index] = 0.0;
        }
        parameters[0] = information.k[0];
        parameters[1] = information.k[4];
        parameters[2] = information.k[2];
        parameters[3] = information.k[5];
        if ((information.distortion_model.empty()) || (information.distortion_model == "plumb_bob") || (information.distortion_model == "rational_polynomial")) {
            for (std::size_t index = 0; (index < information.d.size()) && (index < 8); ++index) {
                parameters[4 + index] = information.d[index];
            }
        }
        else {
            distortion_model_recognised = false;
        }
        return (parameters[0] > 0.0) && (parameters[1] > 0.0);
    }

    enum class pnm_format {
        pgm8,
        pgm16,
        ppm,
        pfm
    };

    inline const char* pnm_encoding(const pnm_format format) {
        switch (format) {
            case pnm_format::pgm8:
                return "mono8";
            case pnm_format::pgm16:
                return "mono16";
            case pnm_format::ppm:
                return "rgb8";
            case pnm_format::pfm:
                return "32FC1";
        }
        return "";
    }

    inline std::size_t pnm_bytes_per_pixel(const pnm_format format) {
        switch (format) {
            case pnm_format::pgm8:
                return 1;
            case pnm_format::pgm16:
                return 2;
            case pnm_format::ppm:
                return 3;
            case pnm_format::pfm:
                return 4;
        }
        return 0;
    }

    inline bool pnm_format_of_encoding(const std::string& encoding, pnm_format& format) {
        for (const pnm_format candidate : { pnm_format::pgm8, pnm_format::pgm16, pnm_format::ppm, pnm_format::pfm }) {
            if (encoding == pnm_encoding(candidate)) {
                format = candidate;
                return true;
            }
        }
        return false;
    }

    // Every Image/CameraInfo channel becomes a camera (grouped by their shared
    // "/sensor/[name]" topic prefix, in channel-add order) and every Imu channel becomes an
    inline bool inspect_mcap_scene(mcap& reader, mcap_scene_information& information, std::string& error, const bool decode_every_image = true) {
        information = mcap_scene_information();
        const mcap::channel_type* dynamic_channel = nullptr;
        std::vector<const mcap::channel_type*> image_channels;
        std::vector<const mcap::channel_type*> imu_channels;
        // camera_of_image[id]/camera_of_info[id]/imu_of_channel[id] resolve a channel's
        // (unsigned short) id to an index into information.cameras/imus, so the message loop
        // below is a simple lookup; sized to the full id range since a channel id need not be
        // dense (the mcap reader accepts any spec-compliant file, not only our own writer's).
        std::vector<int> camera_of_image(65536, -1);
        std::vector<int> camera_of_info(65536, -1);
        std::vector<int> imu_of_channel(65536, -1);
        for (const mcap::channel_type& channel : reader.get_channels()) {
            const mcap::schema_type* schema = reader.find_schema(channel.schema_id);
            if (schema == nullptr) {
                continue;
            }
            if (schema->name == "sensor_msgs/msg/Image") {
                image_channels.push_back(&channel);
            }
            else if (schema->name == "sensor_msgs/msg/Imu") {
                imu_channels.push_back(&channel);
            }
            else if ((schema->name == "tf2_msgs/msg/TFMessage") && (channel.topic == "/tf") && (dynamic_channel == nullptr)) {
                dynamic_channel = &channel;
            }
            else if (schema->name == "sensor_msgs/msg/CompressedImage") {
                error = "the scene holds compressed images, only raw images are supported";
                return false;
            }
        }
        if (image_channels.empty()) {
            error = "no raw image channel found";
            return false;
        }
        // Build the camera list, one entry per Image channel, matched to its CameraInfo
        // channel by the "/sensor/[name]/camera_info" topic convention.
        for (const mcap::channel_type* image_channel : image_channels) {
            camera_information camera;
            camera.image_topic = image_channel->topic;
            const std::size_t separator = camera.image_topic.rfind('/');
            if (separator != std::string::npos) {
                camera.camera_name = camera.image_topic.substr(separator + 1);
            }
            camera_of_image[image_channel->id] = static_cast<int>(information.cameras.size());
            for (const mcap::channel_type& channel : reader.get_channels()) {
                const mcap::schema_type* schema = reader.find_schema(channel.schema_id);
                if ((schema != nullptr) && (schema->name == "sensor_msgs/msg/CameraInfo") && (channel.topic == (camera.image_topic + "/camera_info"))) {
                    camera.camera_info_topic = channel.topic;
                    camera_of_info[channel.id] = static_cast<int>(information.cameras.size());
                    break;
                }
            }
            information.cameras.push_back(camera);
        }
        for (const mcap::channel_type* imu_channel : imu_channels) {
            imu_information imu;
            imu.imu_topic = imu_channel->topic;
            const std::size_t separator = imu.imu_topic.rfind('/');
            if (separator != std::string::npos) {
                imu.imu_name = imu.imu_topic.substr(separator + 1);
            }
            imu_of_channel[imu_channel->id] = static_cast<int>(information.imus.size());
            information.imus.push_back(imu);
        }
        // Any channel beyond those recognised above cannot be represented by the directory
        // form, so a round trip would lose it.
        for (const mcap::channel_type& channel : reader.get_channels()) {
            const bool recognised = (camera_of_image[channel.id] >= 0) || (camera_of_info[channel.id] >= 0) || (imu_of_channel[channel.id] >= 0) || (&channel == dynamic_channel);
            if (!recognised) {
                information.extra_topics.push_back(channel.topic);
            }
        }
        const std::vector<mcap::message_index_type>& message_index = reader.get_message_index();
        for (std::size_t index = 0; index < message_index.size(); ++index) {
            const mcap::message_index_type& entry = message_index[index];
            const bool image_channel = camera_of_image[entry.channel_id] >= 0;
            const bool wanted = image_channel || (camera_of_info[entry.channel_id] >= 0) || (imu_of_channel[entry.channel_id] >= 0) || ((dynamic_channel != nullptr) && (entry.channel_id == dynamic_channel->id));
            if (!wanted) {
                continue;
            }
            if (image_channel && !decode_every_image && (information.cameras[static_cast<std::size_t>(camera_of_image[entry.channel_id])].frames > 0)) {
                camera_information& camera = information.cameras[static_cast<std::size_t>(camera_of_image[entry.channel_id])];
                camera.image_log_times.push_back(entry.log_time);
                ++camera.frames;
                continue;
            }
            mcap::message_type message;
            if (!reader.read_message(index, message)) {
                error = "a message does not read: " + reader.get_read_error();
                return false;
            }
            if (image_channel) {
                camera_information& camera = information.cameras[static_cast<std::size_t>(camera_of_image[message.channel_id])];
                cdr::image image;
                if (!cdr::read_image(message.data, message.length, image)) {
                    error = "an image message does not decode";
                    return false;
                }
                pnm_format format = pnm_format::pgm8;
                if (!pnm_format_of_encoding(image.encoding, format)) {
                    error = "image encoding '" + image.encoding + "' is unsupported";
                    return false;
                }
                const std::size_t bytes_per_pixel = pnm_bytes_per_pixel(format);
                if ((image.width == 0) || (image.height == 0) || (image.step != bytes_per_pixel * image.width) || (image.data.size() != bytes_per_pixel * static_cast<std::size_t>(image.width) * image.height)) {
                    error = "an image message has inconsistent dimensions";
                    return false;
                }
                if (camera.frames == 0) {
                    camera.width = image.width;
                    camera.height = image.height;
                    camera.encoding = image.encoding;
                }
                else if ((image.width != camera.width) || (image.height != camera.height)) {
                    error = "the image dimensions change mid scene";
                    return false;
                }
                else if (image.encoding != camera.encoding) {
                    error = "the image encoding changes mid scene";
                    return false;
                }
                if (image.frame_header.frame_id != ("sensor/" + camera.camera_name)) {
                    camera.frame_ids_consistent = false;
                }
                if (!log_time_matches_stamp(message.log_time, image.frame_header.stamp)) {
                    camera.log_times_consistent = false;
                }
                camera.image_log_times.push_back(message.log_time);
                ++camera.frames;
            }
            else if (camera_of_info[message.channel_id] >= 0) {
                camera_information& camera = information.cameras[static_cast<std::size_t>(camera_of_info[message.channel_id])];
                cdr::camera_info camera_info;
                if (!cdr::read_camera_info(message.data, message.length, camera_info)) {
                    error = "a camera info message does not decode";
                    return false;
                }
                camera.camera_info = true;
                camera.fx = camera_info.k[0];
                camera.fy = camera_info.k[4];
                camera.cx = camera_info.k[2];
                camera.cy = camera_info.k[5];
                if (camera_info.frame_header.frame_id != ("sensor/" + camera.camera_name)) {
                    camera.frame_ids_consistent = false;
                }
                if (!log_time_matches_stamp(message.log_time, camera_info.frame_header.stamp)) {
                    camera.log_times_consistent = false;
                }
                camera.camera_info_log_times.push_back(message.log_time);
                ++camera.camera_infos;
                camera.camera_infos_data.push_back(camera_info);
            }
            else if (imu_of_channel[message.channel_id] >= 0) {
                imu_information& imu = information.imus[static_cast<std::size_t>(imu_of_channel[message.channel_id])];
                cdr::imu sample;
                if (!cdr::read_imu(message.data, message.length, sample)) {
                    error = "an imu message does not decode";
                    return false;
                }
                if (sample.frame_header.frame_id != ("sensor/" + imu.imu_name)) {
                    imu.frame_ids_consistent = false;
                }
                if (!log_time_matches_stamp(message.log_time, sample.frame_header.stamp)) {
                    imu.log_times_consistent = false;
                }
                imu.imu_log_times.push_back(message.log_time);
                imu.imu_data.push_back(sample);
                ++imu.samples;
            }
            else if ((dynamic_channel != nullptr) && (message.channel_id == dynamic_channel->id)) {
                const std::size_t before = information.dynamics.size();
                if (!cdr::read_tf_message(message.data, message.length, information.dynamics)) {
                    error = "a transform message does not decode";
                    return false;
                }
                for (std::size_t transform = before; transform < information.dynamics.size(); ++transform) {
                    if (!log_time_matches_stamp(message.log_time, information.dynamics[transform].frame_header.stamp)) {
                        information.dynamics_log_times_consistent = false;
                    }
                    if ((information.dynamics[transform].frame_header.frame_id == "root") && (information.dynamics[transform].child_frame_id == "ego")) {
                        ++information.poses;
                    }
                    information.dynamic_log_times.push_back(message.log_time);
                }
            }
        }
        for (const mcap::attachment_type& attachment : reader.get_attachments()) {
            information.attachments.push_back(attachment.name);
        }
        for (std::size_t camera_index = 0; camera_index < information.cameras.size(); ++camera_index) {
            if (information.cameras[camera_index].image_topic.rfind("/sensor/image_", 0) == 0) {
                information.primary_camera = camera_index;
                break;
            }
        }
        return true;
    }

    inline void swap_16_bit_samples(unsigned char* const data, const std::size_t samples) {
        for (std::size_t sample = 0; sample < samples; ++sample) {
            const unsigned char high = data[(sample * 2) + 0];
            data[(sample * 2) + 0] = data[(sample * 2) + 1];
            data[(sample * 2) + 1] = high;
        }
    }

    inline bool write_pnm(const std::string& path, const pnm_format format, const unsigned int width, const unsigned int height, const unsigned char* data) {
        char header[64];
        int header_length = 0;
        switch (format) {
            case pnm_format::pgm8:
                header_length = std::snprintf(&header[0], sizeof(header), "P5\n%u %u\n255\n", width, height);
                break;
            case pnm_format::pgm16:
                header_length = std::snprintf(&header[0], sizeof(header), "P5\n%u %u\n65535\n", width, height);
                break;
            case pnm_format::ppm:
                header_length = std::snprintf(&header[0], sizeof(header), "P6\n%u %u\n255\n", width, height);
                break;
            case pnm_format::pfm:
                header_length = std::snprintf(&header[0], sizeof(header), "Pf\n%u %u\n-1.0\n", width, height);
                break;
        }
        if ((header_length <= 0) || (static_cast<std::size_t>(header_length) >= sizeof(header))) {
            return false;
        }
        gtl::file file(path.c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
        if (!file.is_open()) {
            return false;
        }
        if (!write_all(file, &header[0], static_cast<std::size_t>(header_length))) {
            return false;
        }
        const std::size_t size = static_cast<std::size_t>(width) * height * pnm_bytes_per_pixel(format);
        std::vector<unsigned char> swapped;
        if (format == pnm_format::pgm16) {
            swapped.assign(data, data + size);
            swap_16_bit_samples(swapped.data(), static_cast<std::size_t>(width) * height);
            data = swapped.data();
        }
        return write_all(file, reinterpret_cast<const char*>(data), size);
    }

    inline bool read_pnm_header(gtl::file& file, pnm_format& format, unsigned int& width, unsigned int& height) {
        width = 0;
        height = 0;
        format = pnm_format::pgm8;
        char byte = 0;
        const auto read_byte = [&byte, &file]() -> bool {
            gtl::file::size_type length = 1;
            return file.read(&byte, length) && (length == 1);
        };
        const auto is_whitespace = [](const char character) -> bool {
            return (character == ' ') || (character == '\t') || (character == '\r') || (character == '\n') || (character == '\v') || (character == '\f');
        };
        const auto read_number = [&](unsigned long long& value) -> bool {
            value = 0;
            do {
                if (!read_byte()) {
                    return false;
                }
            } while (is_whitespace(byte));
            if ((byte < '0') || (byte > '9')) {
                return false;
            }
            while ((byte >= '0') && (byte <= '9')) {
                value = (value * 10ull) + static_cast<unsigned long long>(byte - '0');
                if (value > 1000000000ull) {
                    return false;
                }
                if (!read_byte()) {
                    return false;
                }
            }
            return true;
        };
        if (!read_byte() || (byte != 'P')) {
            return false;
        }
        if (!read_byte()) {
            return false;
        }
        const char magic = byte;
        if ((magic != '5') && (magic != '6') && (magic != 'f')) {
            return false;
        }
        unsigned long long values[2] = {};
        if (!read_number(values[0]) || !read_number(values[1])) {
            return false;
        }
        if (magic == 'f') {
            do {
                if (!read_byte()) {
                    return false;
                }
            } while (is_whitespace(byte));
            while (!is_whitespace(byte)) {
                if (!read_byte()) {
                    return false;
                }
            }
            format = pnm_format::pfm;
        }
        else {
            unsigned long long maxval = 0;
            if (!read_number(maxval) || (maxval == 0) || (maxval > 65535)) {
                return false;
            }
            format = (magic == '6') ? pnm_format::ppm : ((maxval > 255) ? pnm_format::pgm16 : pnm_format::pgm8);
        }
        width = static_cast<unsigned int>(values[0]);
        height = static_cast<unsigned int>(values[1]);
        return (width != 0) && (height != 0) && (width <= 4096) && (height <= 4096);
    }

    inline bool read_pnm(const std::string& path, pnm_format& format, unsigned int& width, unsigned int& height, std::vector<unsigned char>& data) {
        data.clear();
        gtl::file file(path.c_str(), gtl::file::access_type::read_only);
        if (!file.is_open() || !read_pnm_header(file, format, width, height)) {
            return false;
        }
        data.resize(static_cast<std::size_t>(width) * height * pnm_bytes_per_pixel(format));
        gtl::file::size_type length = data.size();
        if (!file.read(reinterpret_cast<char*>(data.data()), length) || (length != data.size())) {
            data.clear();
            return false;
        }
        if (format == pnm_format::pgm16) {
            swap_16_bit_samples(data.data(), static_cast<std::size_t>(width) * height);
        }
        return true;
    }

    // The name of a frame file: its timestamp in nanoseconds, zero padded to the width of
    // a 64 bit value so that lexicographic and numeric order agree.
    inline std::string frame_filename(const unsigned long long timestamp_nanoseconds) {
        char filename[32];
        std::snprintf(&filename[0], sizeof(filename), "%020llu.pnm", timestamp_nanoseconds);
        return &filename[0];
    }

    inline std::size_t count_frame_directory(const std::string& directory) {
        std::size_t count = 0;
        std::vector<std::string> entries;
        if (!gtl::directory::list_directory(directory, entries)) {
            return 0;
        }
        for (const std::string& entry : entries) {
            if (gtl::paths::is_regular_file(directory + "/" + entry) && (gtl::paths::path_extension(entry) == ".pnm")) {
                ++count;
            }
        }
        return count;
    }

    // Parse a 'seconds.subseconds' timestamp into total nanoseconds. The subsecond digits
    // are scaled by their count, so '1.5' is half a second and '1.1234567890123' truncates
    // to nanosecond precision, and any other trailing characters are rejected.
    inline bool parse_timestamp_nanoseconds(const char* const text, long long& timestamp_nanoseconds) {
        timestamp_nanoseconds = 0;
        if ((text == nullptr) || (*text == '\0')) {
            return false;
        }
        const char* position = text;
        if ((*position < '0') || (*position > '9')) {
            return false;
        }
        long long seconds = 0;
        while ((*position >= '0') && (*position <= '9')) {
            const int digit = *position - '0';
            if (seconds > ((std::numeric_limits<long long>::max() - digit) / 10)) {
                return false;
            }
            seconds = (seconds * 10) + digit;
            ++position;
        }
        long long nanoseconds = 0;
        if (*position == '.') {
            ++position;
            unsigned int fractional_digits = 0;
            while ((*position >= '0') && (*position <= '9')) {
                if (fractional_digits < 9) {
                    nanoseconds = (nanoseconds * 10) + (*position - '0');
                    ++fractional_digits;
                }
                ++position;
            }
            while (fractional_digits < 9) {
                nanoseconds *= 10;
                ++fractional_digits;
            }
        }
        if (*position != '\0') {
            return false;
        }
        if (seconds > ((std::numeric_limits<long long>::max() - nanoseconds) / 1000000000LL)) {
            return false;
        }
        timestamp_nanoseconds = (seconds * 1000000000LL) + nanoseconds;
        return true;
    }

    inline bool parse_sample_line(char* const line, long long& timestamp_nanoseconds, std::vector<double>& values) {
        values.clear();
        char* cursor = line;
        while ((*cursor != '\0') && (*cursor != ' ') && (*cursor != '\t')) {
            ++cursor;
        }
        const char saved_character = *cursor;
        *cursor = '\0';
        const bool timestamp_ok = parse_timestamp_nanoseconds(line, timestamp_nanoseconds);
        *cursor = saved_character;
        if (!timestamp_ok) {
            return false;
        }
        while (*cursor != '\0') {
            char* value_end = nullptr;
            const double value = std::strtod(cursor, &value_end);
            if (value_end == cursor) {
                break;
            }
            values.push_back(value);
            cursor = value_end;
        }
        return true;
    }

    inline bool append_values(std::string& line, const double* const values, const std::size_t count) {
        char token[32];
        for (std::size_t index = 0; index < count; ++index) {
            const int length = std::snprintf(&token[0], sizeof(token), " %.17g", values[index]);
            if ((length <= 0) || (static_cast<std::size_t>(length) >= sizeof(token))) {
                return false;
            }
            line += &token[0];
        }
        return true;
    }

    inline bool format_sample_line(std::string& line, const long long timestamp_nanoseconds, const double* const values, const std::size_t count) {
        const cdr::time stamp = cdr::time::from_nanoseconds(timestamp_nanoseconds);
        char token[32];
        const int length = std::snprintf(&token[0], sizeof(token), "%d.%09u", stamp.sec, stamp.nanosec);
        if ((length <= 0) || (static_cast<std::size_t>(length) >= sizeof(token))) {
            return false;
        }
        line = &token[0];
        return append_values(line, values, count);
    }

    inline bool write_sample_line(gtl::file& file, const long long timestamp_nanoseconds, const double* const values, const std::size_t count) {
        std::string line;
        if (!format_sample_line(line, timestamp_nanoseconds, values, count)) {
            return false;
        }
        line += '\n';
        return write_line(file, line.c_str());
    }

    inline bool write_pose_line(gtl::file& file, const long long timestamp_nanoseconds, const double translation[3], const double rotation_xyzw[4]) {
        const double values[7] = { translation[0], translation[1], translation[2], rotation_xyzw[0], rotation_xyzw[1], rotation_xyzw[2], rotation_xyzw[3] };
        return write_sample_line(file, timestamp_nanoseconds, &values[0], 7);
    }

    inline bool write_trajectory(const mcap_scene_information& scene, const std::string& path, std::size_t& poses) {
        poses = 0;
        gtl::file file(path.c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
        if (!file.is_open()) {
            return false;
        }
        for (const cdr::transform_stamped& transform : scene.dynamics) {
            if ((transform.frame_header.frame_id != "root") || (transform.child_frame_id != "ego")) {
                continue;
            }
            if (!write_pose_line(file, transform.frame_header.stamp.nanoseconds(), &transform.translation[0], &transform.rotation[0])) {
                return false;
            }
            ++poses;
        }
        return true;
    }

    // One parsed line of a TUM format trajectory file: a seconds.subseconds timestamp,
    // then a position, then an orientation quaternion.
    struct trajectory_pose {
        long long timestamp_nanoseconds = 0;
        double x_coordinate = 0.0;
        double y_coordinate = 0.0;
        double z_coordinate = 0.0;
        double quaternion_x = 0.0;
        double quaternion_y = 0.0;
        double quaternion_z = 0.0;
        double quaternion_w = 0.0;
    };

    // Load every parseable TUM format line ('timestamp x y z qx qy qz qw') from a file,
    // skipping lines that do not parse.
    inline bool load_trajectory(const std::string& path, std::vector<trajectory_pose>& poses) {
        poses.clear();
        std::string text;
        if (!read_text_file(path, text)) {
            return false;
        }
        std::vector<double> values;
        for (char* line = &text[0]; line != nullptr;) {
            char* end = std::strchr(line, '\n');
            if (end != nullptr) {
                *end = '\0';
            }
            trajectory_pose pose;
            if (parse_sample_line(line, pose.timestamp_nanoseconds, values) && (values.size() == 7)) {
                pose.x_coordinate = values[0];
                pose.y_coordinate = values[1];
                pose.z_coordinate = values[2];
                pose.quaternion_x = values[3];
                pose.quaternion_y = values[4];
                pose.quaternion_z = values[5];
                pose.quaternion_w = values[6];
                poses.push_back(pose);
            }
            line = (end != nullptr) ? (end + 1) : nullptr;
        }
        return true;
    }

    // Count the lines parseable as TUM format poses: "timestamp x y z qx qy qz qw".
    inline std::size_t count_trajectory_poses(const std::string& path) {
        std::vector<trajectory_pose> poses;
        return load_trajectory(path, poses) ? poses.size() : 0;
    }

    inline void associate_trajectories(
        const std::vector<trajectory_pose>& reference,
        const std::vector<trajectory_pose>& estimate,
        const long long tolerance_nanoseconds,
        std::vector<std::pair<std::size_t, std::size_t>>& pairs
    ) {
        pairs.clear();
        std::vector<std::size_t> order(reference.size());
        for (std::size_t index = 0; index < order.size(); ++index) {
            order[index] = index;
        }
        std::sort(order.begin(), order.end(), [&reference](const std::size_t left, const std::size_t right) {
            return reference[left].timestamp_nanoseconds < reference[right].timestamp_nanoseconds;
        });
        std::vector<bool> taken(reference.size(), false);
        for (std::size_t estimate_index = 0; estimate_index < estimate.size(); ++estimate_index) {
            const long long timestamp = estimate[estimate_index].timestamp_nanoseconds;
            const auto after = std::lower_bound(order.begin(), order.end(), timestamp, [&reference](const std::size_t index, const long long value) {
                return reference[index].timestamp_nanoseconds < value;
            });
            std::size_t best = reference.size();
            long long best_difference = tolerance_nanoseconds + 1;
            for (const auto candidate : { after, (after == order.begin()) ? order.end() : (after - 1) }) {
                if (candidate == order.end()) {
                    continue;
                }
                const long long difference = reference[*candidate].timestamp_nanoseconds - timestamp;
                const long long magnitude = (difference < 0) ? -difference : difference;
                if ((magnitude < best_difference) && !taken[*candidate]) {
                    best = *candidate;
                    best_difference = magnitude;
                }
            }
            if (best < reference.size()) {
                taken[best] = true;
                pairs.push_back({ best, estimate_index });
            }
        }
    }
}

#endif // ZEROSLAM_TOOLS_COMMON_DATASET_HPP
