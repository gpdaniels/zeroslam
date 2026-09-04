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
#include "file.hpp"
#include "filesystem.hpp"
#include "mcap.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// Inspection of the pieces of a dataset: frame data, timestamp files, and trajectories.
// These helpers only measure and check, the policy of what counts as valid enough belongs
// to each tool.
namespace dataset {
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

    // One camera sensor's messages: the frames and their per-frame calibration.
    struct camera_information {
        std::string camera_name;
        std::string image_topic;
        std::string camera_info_topic;
        std::size_t frames = 0;
        unsigned int width = 0;
        unsigned int height = 0;
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

    // One IMU sensor's messages: an independently time-stamped stream, not tied to any
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
        // The first camera, kept as its own fields for tools that only ever look at "the"
        // camera of a scene; 'cameras[0]' describes the same camera.
        std::string camera_name;
        std::string image_topic;
        std::string camera_info_topic;
        std::size_t frames = 0;
        unsigned int width = 0;
        unsigned int height = 0;
        std::size_t poses = 0;
        bool camera_info = false;
        double fx = 0.0;
        double fy = 0.0;
        double cx = 0.0;
        double cy = 0.0;
        // The measurements below let a tool decide whether the file fully round trips
        // through the directory form.
        std::vector<std::string> attachments;
        std::size_t camera_infos = 0;
        std::vector<cdr::camera_info> camera_infos_data;
        bool frame_ids_consistent = true;
        bool log_times_consistent = true;
        std::vector<unsigned long long> image_log_times;
        std::vector<unsigned long long> camera_info_log_times;
        std::vector<unsigned long long> dynamic_log_times;
        std::vector<cdr::transform_stamped> dynamics;
        std::vector<std::string> extra_topics;
        // Every camera (cameras[0] is the primary camera above) and every IMU in the scene.
        std::vector<camera_information> cameras;
        std::vector<imu_information> imus;
    };

    // Whether a message's mcap log time (nanoseconds since the mcap's own epoch) is exactly
    // the same instant as its own header stamp: every text form in the project stores
    // timestamps as a plain signed 64 bit nanosecond count, so this is exact integer
    // arithmetic, not an approximation.
    inline bool log_time_matches_stamp(const unsigned long long log_time, const cdr::time& stamp) {
        return stamp.nanoseconds() == static_cast<long long>(log_time);
    }

    // Inspect a parsed scene mcap: the image messages must be raw mono8 with consistent
    // dimensions; the pose count, camera intrinsics, and attachment names are collected.
    // Every Image/CameraInfo channel becomes a camera (grouped by their shared
    // "/sensor/[name]" topic prefix, in channel-add order) and every Imu channel becomes an
    // IMU; the first camera found also fills the scene's singular fields, for tools that only
    // ever look at "the" camera of a scene.
    inline bool inspect_mcap_scene(const mcap& reader, mcap_scene_information& information, std::string& error) {
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
                error = "the scene holds compressed images, only raw mono8 images are supported";
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
        for (const mcap::message_type& message : reader.get_messages()) {
            if (camera_of_image[message.channel_id] >= 0) {
                camera_information& camera = information.cameras[static_cast<std::size_t>(camera_of_image[message.channel_id])];
                cdr::image image;
                if (!cdr::read_image(message.data, message.length, image)) {
                    error = "an image message does not decode";
                    return false;
                }
                if (image.encoding != "mono8") {
                    error = "image encoding '" + image.encoding + "' is unsupported, only mono8 is";
                    return false;
                }
                if ((image.width == 0) || (image.height == 0) || (image.step != image.width) || (image.data.size() != static_cast<std::size_t>(image.width) * image.height)) {
                    error = "an image message has inconsistent dimensions";
                    return false;
                }
                if (camera.frames == 0) {
                    camera.width = image.width;
                    camera.height = image.height;
                }
                else if ((image.width != camera.width) || (image.height != camera.height)) {
                    error = "the image dimensions change mid scene";
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
                        information.log_times_consistent = false;
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
        // Mirror the first camera onto the scene's singular fields and fold the per-camera
        // 'log_times_consistent'/'frame_ids_consistent' flags (and every IMU's) into the
        // scene-wide ones, so existing single-camera consumers see no behavioural change.
        const camera_information& primary = information.cameras.front();
        information.camera_name = primary.camera_name;
        information.image_topic = primary.image_topic;
        information.camera_info_topic = primary.camera_info_topic;
        information.frames = primary.frames;
        information.width = primary.width;
        information.height = primary.height;
        information.camera_info = primary.camera_info;
        information.fx = primary.fx;
        information.fy = primary.fy;
        information.cx = primary.cx;
        information.cy = primary.cy;
        information.camera_infos = primary.camera_infos;
        information.camera_infos_data = primary.camera_infos_data;
        information.image_log_times = primary.image_log_times;
        information.camera_info_log_times = primary.camera_info_log_times;
        for (const camera_information& camera : information.cameras) {
            information.frame_ids_consistent = information.frame_ids_consistent && camera.frame_ids_consistent;
            information.log_times_consistent = information.log_times_consistent && camera.log_times_consistent;
        }
        for (const imu_information& imu : information.imus) {
            information.frame_ids_consistent = information.frame_ids_consistent && imu.frame_ids_consistent;
            information.log_times_consistent = information.log_times_consistent && imu.log_times_consistent;
        }
        return true;
    }

    // Write one 8 bit binary pgm image.
    inline bool write_pgm(const std::string& path, const unsigned int width, const unsigned int height, const unsigned char* data) {
        char header[32];
        const int header_length = std::snprintf(&header[0], sizeof(header), "P5\n%u %u\n255\n", width, height);
        if ((header_length <= 0) || (static_cast<std::size_t>(header_length) >= sizeof(header))) {
            return false;
        }
        gtl::file file(path.c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
        if (!file.is_open()) {
            return false;
        }
        gtl::file::size_type length = static_cast<gtl::file::size_type>(header_length);
        if (!file.write(&header[0], length) || (length != static_cast<gtl::file::size_type>(header_length))) {
            return false;
        }
        length = static_cast<gtl::file::size_type>(width) * height;
        return file.write(reinterpret_cast<const char*>(data), length) && (length == (static_cast<gtl::file::size_type>(width) * height));
    }

    // Read one 8 bit binary pgm image.
    inline bool read_pgm(const std::string& path, unsigned int& width, unsigned int& height, std::vector<unsigned char>& data) {
        width = 0;
        height = 0;
        data.clear();
        gtl::file file(path.c_str(), gtl::file::access_type::read_only);
        if (!file.is_open()) {
            return false;
        }
        // The header is 'P5' then the width, height, and maximum value as decimal tokens
        // separated by runs of whitespace; exactly one whitespace after the maximum value
        // separates it from the binary pixel data.
        char byte = 0;
        const auto read_byte = [&byte, &file]() -> bool {
            gtl::file::size_type length = 1;
            return file.read(&byte, length) && (length == 1);
        };
        const auto is_whitespace = [](const char character) -> bool {
            return (character == ' ') || (character == '\t') || (character == '\r') || (character == '\n') || (character == '\v') || (character == '\f');
        };
        if (!read_byte() || (byte != 'P')) {
            return false;
        }
        if (!read_byte() || (byte != '5')) {
            return false;
        }
        unsigned long long values[3] = {};
        for (unsigned long long& value : values) {
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
        }
        // Here 'byte' holds the single whitespace that ended the maximum value.
        width = static_cast<unsigned int>(values[0]);
        height = static_cast<unsigned int>(values[1]);
        const unsigned int maximum = static_cast<unsigned int>(values[2]);
        if ((width == 0) || (height == 0) || (width > 4096) || (height > 4096) || (maximum == 0) || (maximum > 255)) {
            return false;
        }
        data.resize(static_cast<std::size_t>(width) * height);
        gtl::file::size_type length = data.size();
        return file.read(reinterpret_cast<char*>(&data[0]), length) && (length == data.size());
    }

    // The name of a frame file: its timestamp in nanoseconds, zero padded to the width of
    // a 64 bit value so that lexicographic and numeric order agree.
    inline std::string frame_filename(const unsigned long long timestamp_nanoseconds) {
        char filename[32];
        std::snprintf(&filename[0], sizeof(filename), "%020llu.pgm", timestamp_nanoseconds);
        return &filename[0];
    }

    // Count the '*.pgm' frames in a directory (named by their nanosecond timestamp).
    inline std::size_t count_frame_directory(const std::string& directory) {
        std::size_t count = 0;
        std::vector<std::string> entries;
        if (!platform::list_directory(directory, entries)) {
            return 0;
        }
        for (const std::string& entry : entries) {
            if (platform::is_regular_file(directory + "/" + entry) && (platform::path_extension(entry) == ".pgm")) {
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
        gtl::file file(path.c_str(), gtl::file::access_type::read_only);
        if (!file.is_open()) {
            return false;
        }
        gtl::file::size_type size = 0;
        if (!file.get_size(size) || (size == 0)) {
            return false;
        }
        const gtl::file::size_type expected_size = size;
        std::vector<char> buffer(static_cast<std::size_t>(size) + 1);
        if (!file.read(&buffer[0], size) || (size != expected_size)) {
            return false;
        }
        buffer[static_cast<std::size_t>(expected_size)] = '\0';
        for (char* line = &buffer[0]; line != nullptr;) {
            char* end = std::strchr(line, '\n');
            if (end != nullptr) {
                *end = '\0';
            }
            char* timestamp_end = line;
            while ((*timestamp_end != '\0') && (*timestamp_end != ' ') && (*timestamp_end != '\t')) {
                ++timestamp_end;
            }
            const char saved_character = *timestamp_end;
            *timestamp_end = '\0';
            trajectory_pose pose;
            const bool timestamp_ok = parse_timestamp_nanoseconds(line, pose.timestamp_nanoseconds);
            *timestamp_end = saved_character;
            if (timestamp_ok && (std::sscanf(timestamp_end, "%lf %lf %lf %lf %lf %lf %lf", &pose.x_coordinate, &pose.y_coordinate, &pose.z_coordinate, &pose.quaternion_x, &pose.quaternion_y, &pose.quaternion_z, &pose.quaternion_w) == 7)) {
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
}

#endif // ZEROSLAM_TOOLS_COMMON_DATASET_HPP
