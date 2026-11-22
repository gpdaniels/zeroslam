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
#ifndef ZEROSLAM_TOOLS_COMMON_WHEEL_HPP
#define ZEROSLAM_TOOLS_COMMON_WHEEL_HPP

// Summary: A wheel odometer in a scene. Its samples are sensor_msgs/msg/JointState messages on /sensor/wheel_[01-99], the
// positions the cumulative rotations in radians of the measured (or left) wheel and the right wheel and the steering reading;
// its model, calibration and noise are one std_msgs/msg/String of key=value lines on /sensor/wheel_[01-99]/parameters; its
// extrinsic is the ego -> sensor/wheel_[01-99] transform on /tf. In the directory form it is sensor/wheel_[01-99].txt, the
// parameters on its first line after '# ' and one line per sample:
// `[timestamp] [x] [y] [z] [qx] [qy] [qz] [qw] [rotation_0] [rotation_1] [steering]`.

#include "cdr.hpp"
#include "mcap.hpp"

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

namespace wheel {
    struct sample {
        long long timestamp = 0;
        double extrinsic[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
        double values[3] = { 0.0, 0.0, 0.0 };
    };

    struct odometer {
        std::string name;
        std::string parameters;
        std::vector<sample> samples;
    };

    inline std::string joint_state_schema() {
        return std::string(
            "std_msgs/Header header\n"
            "string[] name\n"
            "float64[] position\n"
            "float64[] velocity\n"
            "float64[] effort\n"
            "================================================================================\n"
            "MSG: std_msgs/Header\n"
            "builtin_interfaces/Time stamp\n"
            "string frame_id\n"
            "================================================================================\n"
            "MSG: builtin_interfaces/Time\n"
            "int32 sec\n"
            "uint32 nanosec\n"
        );
    }

    inline std::string string_schema() {
        return std::string("string data\n");
    }

    // Whether a name is 'wheel_01' to 'wheel_99'.
    inline bool is_wheel_name(const std::string& name) {
        return (name.size() == 8) && (name.compare(0, 6, "wheel_") == 0) && (name[6] >= '0') && (name[6] <= '9') && (name[7] >= '0') && (name[7] <= '9') && ((name[6] != '0') || (name[7] != '0'));
    }

    // Whether a topic is a wheel odometer's samples or parameters.
    inline bool is_wheel_topic(const std::string& topic) {
        if (topic.compare(0, 8, "/sensor/") != 0) {
            return false;
        }
        const std::string rest = topic.substr(8);
        return is_wheel_name(rest) || ((rest.size() == 19) && is_wheel_name(rest.substr(0, 8)) && (rest.compare(8, 11, "/parameters") == 0));
    }

    inline std::vector<unsigned char> write_joint_state(const long long timestamp, const std::string& frame_id, const double (&values)[3]) {
        cdr::writer stream;
        cdr::header header;
        header.stamp = cdr::time::from_nanoseconds(timestamp);
        header.frame_id = frame_id;
        cdr::write_header(stream, header);
        const char* const names[3] = { "wheel_0", "wheel_1", "steering" };
        stream.write_u32(3);
        for (const char* const name : names) {
            stream.write_string(name);
        }
        stream.write_u32(3);
        for (const double value : values) {
            stream.write_f64(value);
        }
        stream.write_u32(0);
        stream.write_u32(0);
        return stream.finish();
    }

    inline bool read_joint_state(const unsigned char* payload, const unsigned long long length, long long& timestamp, double (&values)[3]) {
        cdr::reader stream;
        cdr::header header;
        if (!stream.open(payload, length) || !cdr::read_header(stream, header)) {
            return false;
        }
        timestamp = header.stamp.nanoseconds();
        const unsigned int names = stream.read_u32();
        std::string name;
        for (unsigned int i = 0; (i < names) && stream.is_valid(); ++i) {
            stream.read_string(name);
        }
        const unsigned int positions = stream.read_u32();
        if (!stream.is_valid() || (positions != 3)) {
            return false;
        }
        for (double& value : values) {
            value = stream.read_f64();
        }
        return stream.is_valid();
    }

    inline std::vector<unsigned char> write_string(const std::string& text) {
        cdr::writer stream;
        stream.write_string(text);
        return stream.finish();
    }

    inline bool read_string(const unsigned char* payload, const unsigned long long length, std::string& text) {
        cdr::reader stream;
        return stream.open(payload, length) && stream.read_string(text) && stream.is_valid();
    }

    // The odometers of a scene with their parameters, samples and extrinsics, in topic order.
    inline bool read_scene(mcap& reader, std::vector<odometer>& odometers, std::string& error) {
        odometers.clear();
        std::vector<int> sample_of(65536, -1);
        std::vector<int> parameters_of(65536, -1);
        for (const mcap::channel_type& channel : reader.get_channels()) {
            if (!is_wheel_topic(channel.topic)) {
                continue;
            }
            const std::string name = channel.topic.substr(8, 8);
            int index = -1;
            for (std::size_t i = 0; i < odometers.size(); ++i) {
                index = (odometers[i].name == name) ? static_cast<int>(i) : index;
            }
            if (index < 0) {
                index = static_cast<int>(odometers.size());
                odometers.push_back(odometer());
                odometers.back().name = name;
            }
            ((channel.topic.size() == 16) ? sample_of : parameters_of)[channel.id] = index;
        }
        int transform_channel = -1;
        for (const mcap::channel_type& channel : reader.get_channels()) {
            transform_channel = (channel.topic == "/tf") ? static_cast<int>(channel.id) : transform_channel;
        }
        std::vector<std::vector<cdr::transform_stamped>> extrinsics(odometers.size());
        const std::vector<mcap::message_index_type>& index = reader.get_message_index();
        for (std::size_t number = 0; number < index.size(); ++number) {
            const std::size_t channel = static_cast<std::size_t>(index[number].channel_id);
            const bool transform = (static_cast<int>(channel) == transform_channel) && !odometers.empty();
            if ((sample_of[channel] < 0) && (parameters_of[channel] < 0) && !transform) {
                continue;
            }
            mcap::message_type message;
            if (!reader.read_message(number, message)) {
                error = "a wheel odometer's message cannot be read";
                return false;
            }
            if (sample_of[channel] >= 0) {
                sample value;
                if (!read_joint_state(message.data, message.length, value.timestamp, value.values)) {
                    error = "a sample of '" + odometers[static_cast<std::size_t>(sample_of[channel])].name + "' is not a joint state of three positions";
                    return false;
                }
                odometers[static_cast<std::size_t>(sample_of[channel])].samples.push_back(value);
            }
            else if (parameters_of[channel] >= 0) {
                if (!read_string(message.data, message.length, odometers[static_cast<std::size_t>(parameters_of[channel])].parameters)) {
                    error = "the parameters of '" + odometers[static_cast<std::size_t>(parameters_of[channel])].name + "' are not a string";
                    return false;
                }
            }
            else {
                std::vector<cdr::transform_stamped> transforms;
                if (!cdr::read_tf_message(message.data, message.length, transforms)) {
                    continue;
                }
                for (const cdr::transform_stamped& value : transforms) {
                    for (std::size_t i = 0; i < odometers.size(); ++i) {
                        if ((value.frame_header.frame_id == "ego") && (value.child_frame_id == ("sensor/" + odometers[i].name))) {
                            extrinsics[i].push_back(value);
                        }
                    }
                }
            }
        }
        for (std::size_t i = 0; i < odometers.size(); ++i) {
            if (odometers[i].samples.empty()) {
                error = "the wheel odometer '" + odometers[i].name + "' has no samples";
                return false;
            }
            for (std::size_t s = 1; s < odometers[i].samples.size(); ++s) {
                if (odometers[i].samples[s].timestamp <= odometers[i].samples[s - 1].timestamp) {
                    error = "the samples of '" + odometers[i].name + "' are not in time order";
                    return false;
                }
            }
            // Each sample takes the extrinsic logged at its time, or the latest before it, or the first.
            std::size_t cursor = 0;
            for (sample& value : odometers[i].samples) {
                while (((cursor + 1) < extrinsics[i].size()) && (extrinsics[i][cursor + 1].frame_header.stamp.nanoseconds() <= value.timestamp)) {
                    ++cursor;
                }
                if (!extrinsics[i].empty()) {
                    for (std::size_t k = 0; k < 3; ++k) {
                        value.extrinsic[k] = extrinsics[i][cursor].translation[k];
                    }
                    for (std::size_t k = 0; k < 4; ++k) {
                        value.extrinsic[3 + k] = extrinsics[i][cursor].rotation[k];
                    }
                }
            }
        }
        return true;
    }

    // Writes an odometer in the directory form.
    inline bool write_directory(const std::string& path, const odometer& value) {
        std::FILE* const file = std::fopen(path.c_str(), "wb");
        if (file == nullptr) {
            return false;
        }
        bool written = std::fprintf(file, "# %s\n", value.parameters.c_str()) > 0;
        for (const sample& entry : value.samples) {
            written = written && (std::fprintf(file, "%lld.%09lld", entry.timestamp / 1000000000ll, entry.timestamp % 1000000000ll) > 0);
            for (const double number : entry.extrinsic) {
                written = written && (std::fprintf(file, " %.17g", number) > 0);
            }
            for (const double number : entry.values) {
                written = written && (std::fprintf(file, " %.17g", number) > 0);
            }
            written = written && (std::fprintf(file, "\n") > 0);
        }
        return (std::fclose(file) == 0) && written;
    }

    // Reads an odometer in the directory form.
    inline bool read_directory(const std::string& path, const std::string& name, odometer& value, std::string& error) {
        value = odometer();
        value.name = name;
        std::FILE* const file = std::fopen(path.c_str(), "rb");
        if (file == nullptr) {
            error = "cannot read '" + path + "'";
            return false;
        }
        char line[4096];
        bool first = true;
        while (std::fgets(line, sizeof(line), file) != nullptr) {
            std::size_t length = std::strlen(line);
            while ((length > 0) && ((line[length - 1] == '\n') || (line[length - 1] == '\r'))) {
                line[--length] = 0;
            }
            if (first && (line[0] == '#')) {
                value.parameters = (length > 2) ? std::string(line + 2) : std::string();
                first = false;
                continue;
            }
            first = false;
            if (length == 0) {
                continue;
            }
            char* cursor = line;
            char* end = nullptr;
            const long long seconds = std::strtoll(cursor, &end, 10);
            long long nanoseconds = 0;
            if ((end != cursor) && (*end == '.')) {
                cursor = end + 1;
                nanoseconds = std::strtoll(cursor, &end, 10);
            }
            sample entry;
            entry.timestamp = (seconds * 1000000000ll) + nanoseconds;
            bool parsed = (end != line);
            for (std::size_t k = 0; (k < 10) && parsed; ++k) {
                cursor = end;
                const double number = std::strtod(cursor, &end);
                parsed = (end != cursor);
                (k < 7) ? (entry.extrinsic[k] = number) : (entry.values[k - 7] = number);
            }
            if (!parsed || (!value.samples.empty() && (entry.timestamp <= value.samples.back().timestamp))) {
                std::fclose(file);
                error = "'" + path + "' has a sample line that is not a timestamp, a pose and three values after the one before";
                return false;
            }
            value.samples.push_back(entry);
        }
        std::fclose(file);
        if (value.samples.empty()) {
            error = "'" + path + "' holds no samples";
            return false;
        }
        return true;
    }

    // Adds an odometer's channels and messages to a scene being written, its extrinsic on the /tf channel.
    inline void write_scene(mcap::writer& writer, const unsigned short tf_channel, const odometer& value) {
        const unsigned short joint_schema = writer.add_schema("sensor_msgs/msg/JointState", "ros2msg", joint_state_schema());
        const unsigned short text_schema = writer.add_schema("std_msgs/msg/String", "ros2msg", string_schema());
        const unsigned short samples_channel = writer.add_channel(joint_schema, "/sensor/" + value.name, "cdr");
        const unsigned short parameters_channel = writer.add_channel(text_schema, "/sensor/" + value.name + "/parameters", "cdr");
        const unsigned long long first_time = static_cast<unsigned long long>(value.samples.front().timestamp);
        const std::vector<unsigned char> parameters = write_string(value.parameters);
        writer.add_message(parameters_channel, 0, first_time, first_time, parameters.data(), parameters.size());
        for (std::size_t i = 0; i < value.samples.size(); ++i) {
            const sample& entry = value.samples[i];
            const unsigned long long log_time = static_cast<unsigned long long>(entry.timestamp);
            const std::vector<unsigned char> payload = write_joint_state(entry.timestamp, "sensor/" + value.name, entry.values);
            writer.add_message(samples_channel, static_cast<unsigned int>(i), log_time, log_time, payload.data(), payload.size());
            const std::vector<unsigned char> transform = cdr::write_tf_message(cdr::time::from_nanoseconds(entry.timestamp), "ego", "sensor/" + value.name, &entry.extrinsic[0], &entry.extrinsic[3]);
            writer.add_message(tf_channel, static_cast<unsigned int>(i), log_time, log_time, transform.data(), transform.size());
        }
    }
}

#endif // ZEROSLAM_TOOLS_COMMON_WHEEL_HPP
