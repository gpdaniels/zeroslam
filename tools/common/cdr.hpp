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
#ifndef ZEROSLAM_TOOLS_COMMON_CDR_HPP
#define ZEROSLAM_TOOLS_COMMON_CDR_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstring>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// Decoding and encoding of the little-endian cdr (xcdr1) serialization that ROS2 uses for
// its messages, for a fixed set of message types. Alignment is relative to the start of
// the serialized data, after the four byte encapsulation header. The schema texts carry
// the full concatenated definitions (every referenced type included), which external
// viewers parse to decode the messages.
namespace cdr {
    constexpr const char* const header_schema_suffix =
        "================================================================================\n"
        "MSG: std_msgs/Header\n"
        "builtin_interfaces/Time stamp\n"
        "string frame_id\n"
        "================================================================================\n"
        "MSG: builtin_interfaces/Time\n"
        "int32 sec\n"
        "uint32 nanosec\n";

    inline std::string image_schema() {
        return std::string(
                   "std_msgs/Header header\n"
                   "uint32 height\n"
                   "uint32 width\n"
                   "string encoding\n"
                   "uint8 is_bigendian\n"
                   "uint32 step\n"
                   "uint8[] data\n"
               ) +
               header_schema_suffix;
    }

    inline std::string camera_info_schema() {
        return std::string(
                   "std_msgs/Header header\n"
                   "uint32 height\n"
                   "uint32 width\n"
                   "string distortion_model\n"
                   "float64[] d\n"
                   "float64[9] k\n"
                   "float64[9] r\n"
                   "float64[12] p\n"
                   "uint32 binning_x\n"
                   "uint32 binning_y\n"
                   "sensor_msgs/RegionOfInterest roi\n"
                   "================================================================================\n"
                   "MSG: sensor_msgs/RegionOfInterest\n"
                   "uint32 x_offset\n"
                   "uint32 y_offset\n"
                   "uint32 height\n"
                   "uint32 width\n"
                   "bool do_rectify\n"
               ) +
               header_schema_suffix;
    }

    inline std::string tf_message_schema() {
        return std::string(
                   "geometry_msgs/TransformStamped[] transforms\n"
                   "================================================================================\n"
                   "MSG: geometry_msgs/TransformStamped\n"
                   "std_msgs/Header header\n"
                   "string child_frame_id\n"
                   "geometry_msgs/Transform transform\n"
                   "================================================================================\n"
                   "MSG: geometry_msgs/Transform\n"
                   "geometry_msgs/Vector3 translation\n"
                   "geometry_msgs/Quaternion rotation\n"
                   "================================================================================\n"
                   "MSG: geometry_msgs/Vector3\n"
                   "float64 x\n"
                   "float64 y\n"
                   "float64 z\n"
                   "================================================================================\n"
                   "MSG: geometry_msgs/Quaternion\n"
                   "float64 x\n"
                   "float64 y\n"
                   "float64 z\n"
                   "float64 w\n"
               ) +
               header_schema_suffix;
    }

    inline std::string imu_schema() {
        return std::string(
                   "std_msgs/Header header\n"
                   "geometry_msgs/Quaternion orientation\n"
                   "float64[9] orientation_covariance\n"
                   "geometry_msgs/Vector3 angular_velocity\n"
                   "float64[9] angular_velocity_covariance\n"
                   "geometry_msgs/Vector3 linear_acceleration\n"
                   "float64[9] linear_acceleration_covariance\n"
                   "================================================================================\n"
                   "MSG: geometry_msgs/Quaternion\n"
                   "float64 x\n"
                   "float64 y\n"
                   "float64 z\n"
                   "float64 w\n"
                   "================================================================================\n"
                   "MSG: geometry_msgs/Vector3\n"
                   "float64 x\n"
                   "float64 y\n"
                   "float64 z\n"
               ) +
               header_schema_suffix;
    }

    class reader final {
    private:
        const unsigned char* data = nullptr;
        unsigned long long length = 0;
        unsigned long long position = 0;
        bool valid = false;

    public:
        bool open(const unsigned char* payload, const unsigned long long payload_length) {
            data = payload;
            length = payload_length;
            position = 4;
            // The encapsulation header: 0x00 0x01 = plain cdr, little-endian.
            valid = (payload != nullptr) && (payload_length >= 4) && (payload[0] == 0x00) && (payload[1] == 0x01);
            return valid;
        }

        bool is_valid() const {
            return valid;
        }

        // All of the payload was consumed, allowing for alignment padding at the end.
        bool is_complete() const {
            return valid && (length - position < 4);
        }

    private:
        bool align_and_check(const unsigned long long alignment, const unsigned long long size) {
            if (!valid) {
                return false;
            }
            const unsigned long long relative = position - 4;
            position = 4 + ((relative + alignment - 1) / alignment) * alignment;
            if ((position > length) || (size > length - position)) {
                valid = false;
                return false;
            }
            return true;
        }

    public:
        unsigned char read_u8() {
            if (!align_and_check(1, 1)) {
                return 0;
            }
            return data[position++];
        }

        unsigned int read_u32() {
            if (!align_and_check(4, 4)) {
                return 0;
            }
            const unsigned int value = static_cast<unsigned int>(data[position]) | (static_cast<unsigned int>(data[position + 1]) << 8) | (static_cast<unsigned int>(data[position + 2]) << 16) | (static_cast<unsigned int>(data[position + 3]) << 24);
            position += 4;
            return value;
        }

        int read_i32() {
            return static_cast<int>(read_u32());
        }

        unsigned long long read_u64() {
            if (!align_and_check(8, 8)) {
                return 0;
            }
            unsigned long long value = 0;
            for (int i = 7; i >= 0; --i) {
                value = (value << 8) | data[position + static_cast<unsigned long long>(i)];
            }
            position += 8;
            return value;
        }

        double read_f64() {
            const unsigned long long bits = read_u64();
            double value = 0.0;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }

        // A cdr string: a length including the null terminator, then the characters.
        bool read_string(std::string& value) {
            const unsigned int string_length = read_u32();
            if (!valid || (string_length > length - position)) {
                valid = false;
                return false;
            }
            if (string_length == 0) {
                value.clear();
            }
            else {
                value.assign(reinterpret_cast<const char*>(&data[position]), string_length - 1);
            }
            position += string_length;
            return true;
        }

        // A sequence of bytes: a length, then the bytes.
        bool read_bytes(std::vector<unsigned char>& value) {
            const unsigned int byte_length = read_u32();
            if (!valid || (byte_length > length - position)) {
                valid = false;
                return false;
            }
            value.assign(&data[position], &data[position] + byte_length);
            position += byte_length;
            return true;
        }
    };

    class writer final {
    private:
        std::vector<unsigned char> output;

    public:
        writer() {
            // The encapsulation header: plain cdr, little-endian.
            output = { 0x00, 0x01, 0x00, 0x00 };
        }

    private:
        void align(const unsigned long long alignment) {
            while (((output.size() - 4) % alignment) != 0) {
                output.push_back(0);
            }
        }

    public:
        void write_u8(const unsigned char value) {
            output.push_back(value);
        }

        void write_u32(const unsigned int value) {
            align(4);
            for (int i = 0; i < 4; ++i) {
                output.push_back(static_cast<unsigned char>((value >> (8 * i)) & 0xFF));
            }
        }

        void write_i32(const int value) {
            write_u32(static_cast<unsigned int>(value));
        }

        void write_f64(const double value) {
            align(8);
            unsigned long long bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            for (int i = 0; i < 8; ++i) {
                output.push_back(static_cast<unsigned char>((bits >> (8 * i)) & 0xFF));
            }
        }

        void write_string(const std::string& value) {
            write_u32(static_cast<unsigned int>(value.size() + 1));
            output.insert(output.end(), value.begin(), value.end());
            output.push_back(0);
        }

        void write_bytes(const unsigned char* data, const unsigned long long length) {
            write_u32(static_cast<unsigned int>(length));
            output.insert(output.end(), data, data + length);
        }

        const std::vector<unsigned char>& finish() {
            return output;
        }
    };

    // builtin_interfaces/msg/Time
    struct time {
        int sec = 0;
        unsigned int nanosec = 0;

        // The exact (integer, no floating point) nanosecond count since the epoch this stamp
        // is relative to; this is what every text form in the project uses for timestamps.
        long long nanoseconds() const {
            return static_cast<long long>(sec) * 1000000000LL + static_cast<long long>(nanosec);
        }

        // The inverse of nanoseconds(): a floored sec/nanosec split (nanosec always in
        // [0, 1e9), sec carrying the sign), so this round trips exactly for any value.
        static time from_nanoseconds(const long long value) {
            long long whole_seconds = value / 1000000000LL;
            long long remainder_nanoseconds = value % 1000000000LL;
            if (remainder_nanoseconds < 0) {
                remainder_nanoseconds += 1000000000LL;
                whole_seconds -= 1;
            }
            time result;
            result.sec = static_cast<int>(whole_seconds);
            result.nanosec = static_cast<unsigned int>(remainder_nanoseconds);
            return result;
        }
    };

    // std_msgs/msg/Header
    struct header {
        time stamp;
        std::string frame_id;
    };

    // sensor_msgs/msg/Image
    struct image {
        header frame_header;
        unsigned int height = 0;
        unsigned int width = 0;
        std::string encoding;
        unsigned char is_bigendian = 0;
        unsigned int step = 0;
        std::vector<unsigned char> data;
    };

    // sensor_msgs/msg/CompressedImage
    struct compressed_image {
        header frame_header;
        std::string format;
        std::vector<unsigned char> data;
    };

    // sensor_msgs/msg/CameraInfo
    struct camera_info {
        header frame_header;
        unsigned int height = 0;
        unsigned int width = 0;
        std::string distortion_model;
        std::vector<double> d;
        double k[9] = {};
        double r[9] = {};
        double p[12] = {};
        unsigned int binning_x = 0;
        unsigned int binning_y = 0;
        unsigned int roi_x_offset = 0;
        unsigned int roi_y_offset = 0;
        unsigned int roi_height = 0;
        unsigned int roi_width = 0;
        unsigned char roi_do_rectify = 0;
    };

    struct transform_stamped {
        // The header frame id is the parent frame.
        header frame_header;
        std::string child_frame_id;
        double translation[3] = {};
        // The ros quaternion field order: x, y, z, w.
        double rotation[4] = {};
    };

    // sensor_msgs/msg/Imu. No orientation estimate is carried (only raw gyro/accel), so
    // 'orientation' is zero and 'orientation_covariance[0]' is -1, the ros convention for
    // "orientation not provided"; the other two covariances are left zero (unknown).
    struct imu {
        header frame_header;
        double orientation[4] = { 0.0, 0.0, 0.0, 0.0 };
        double orientation_covariance[9] = {};
        double angular_velocity[3] = {};
        double angular_velocity_covariance[9] = {};
        double linear_acceleration[3] = {};
        double linear_acceleration_covariance[9] = {};

        imu() {
            orientation_covariance[0] = -1.0;
        }
    };

    inline bool read_header(reader& stream, header& value) {
        value.stamp.sec = stream.read_i32();
        value.stamp.nanosec = stream.read_u32();
        return stream.read_string(value.frame_id);
    }

    inline bool read_image(const unsigned char* payload, const unsigned long long length, image& value) {
        reader stream;
        if (!stream.open(payload, length)) {
            return false;
        }
        if (!read_header(stream, value.frame_header)) {
            return false;
        }
        value.height = stream.read_u32();
        value.width = stream.read_u32();
        if (!stream.read_string(value.encoding)) {
            return false;
        }
        value.is_bigendian = stream.read_u8();
        value.step = stream.read_u32();
        return stream.read_bytes(value.data) && stream.is_valid();
    }

    inline bool read_compressed_image(const unsigned char* payload, const unsigned long long length, compressed_image& value) {
        reader stream;
        if (!stream.open(payload, length)) {
            return false;
        }
        if (!read_header(stream, value.frame_header)) {
            return false;
        }
        if (!stream.read_string(value.format)) {
            return false;
        }
        return stream.read_bytes(value.data) && stream.is_valid();
    }

    inline bool read_camera_info(const unsigned char* payload, const unsigned long long length, camera_info& value) {
        reader stream;
        if (!stream.open(payload, length)) {
            return false;
        }
        if (!read_header(stream, value.frame_header)) {
            return false;
        }
        value.height = stream.read_u32();
        value.width = stream.read_u32();
        if (!stream.read_string(value.distortion_model)) {
            return false;
        }
        const unsigned int distortion_count = stream.read_u32();
        value.d.clear();
        for (unsigned int i = 0; stream.is_valid() && (i < distortion_count); ++i) {
            value.d.push_back(stream.read_f64());
        }
        for (int i = 0; i < 9; ++i) {
            value.k[i] = stream.read_f64();
        }
        for (int i = 0; i < 9; ++i) {
            value.r[i] = stream.read_f64();
        }
        for (int i = 0; i < 12; ++i) {
            value.p[i] = stream.read_f64();
        }
        value.binning_x = stream.read_u32();
        value.binning_y = stream.read_u32();
        value.roi_x_offset = stream.read_u32();
        value.roi_y_offset = stream.read_u32();
        value.roi_height = stream.read_u32();
        value.roi_width = stream.read_u32();
        value.roi_do_rectify = stream.read_u8();
        return stream.is_valid();
    }

    inline bool read_tf_message(const unsigned char* payload, const unsigned long long length, std::vector<transform_stamped>& value) {
        reader stream;
        if (!stream.open(payload, length)) {
            return false;
        }
        const unsigned int count = stream.read_u32();
        for (unsigned int i = 0; stream.is_valid() && (i < count); ++i) {
            transform_stamped transform;
            if (!read_header(stream, transform.frame_header)) {
                return false;
            }
            if (!stream.read_string(transform.child_frame_id)) {
                return false;
            }
            for (int axis = 0; axis < 3; ++axis) {
                transform.translation[axis] = stream.read_f64();
            }
            for (int axis = 0; axis < 4; ++axis) {
                transform.rotation[axis] = stream.read_f64();
            }
            value.push_back(transform);
        }
        return stream.is_valid();
    }

    inline bool read_imu(const unsigned char* payload, const unsigned long long length, imu& value) {
        reader stream;
        if (!stream.open(payload, length)) {
            return false;
        }
        if (!read_header(stream, value.frame_header)) {
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            value.orientation[i] = stream.read_f64();
        }
        for (int i = 0; i < 9; ++i) {
            value.orientation_covariance[i] = stream.read_f64();
        }
        for (int i = 0; i < 3; ++i) {
            value.angular_velocity[i] = stream.read_f64();
        }
        for (int i = 0; i < 9; ++i) {
            value.angular_velocity_covariance[i] = stream.read_f64();
        }
        for (int i = 0; i < 3; ++i) {
            value.linear_acceleration[i] = stream.read_f64();
        }
        for (int i = 0; i < 9; ++i) {
            value.linear_acceleration_covariance[i] = stream.read_f64();
        }
        return stream.is_valid();
    }

    inline void write_header(writer& stream, const header& value) {
        stream.write_i32(value.stamp.sec);
        stream.write_u32(value.stamp.nanosec);
        stream.write_string(value.frame_id);
    }

    inline std::vector<unsigned char> write_image(const image& value) {
        writer stream;
        write_header(stream, value.frame_header);
        stream.write_u32(value.height);
        stream.write_u32(value.width);
        stream.write_string(value.encoding);
        stream.write_u8(value.is_bigendian);
        stream.write_u32(value.step);
        stream.write_bytes(value.data.data(), value.data.size());
        return stream.finish();
    }

    inline std::vector<unsigned char> write_compressed_image(const compressed_image& value) {
        writer stream;
        write_header(stream, value.frame_header);
        stream.write_string(value.format);
        stream.write_bytes(value.data.data(), value.data.size());
        return stream.finish();
    }

    inline std::vector<unsigned char> write_camera_info(const camera_info& value) {
        writer stream;
        write_header(stream, value.frame_header);
        stream.write_u32(value.height);
        stream.write_u32(value.width);
        stream.write_string(value.distortion_model);
        stream.write_u32(static_cast<unsigned int>(value.d.size()));
        for (const double element : value.d) {
            stream.write_f64(element);
        }
        for (int i = 0; i < 9; ++i) {
            stream.write_f64(value.k[i]);
        }
        for (int i = 0; i < 9; ++i) {
            stream.write_f64(value.r[i]);
        }
        for (int i = 0; i < 12; ++i) {
            stream.write_f64(value.p[i]);
        }
        stream.write_u32(value.binning_x);
        stream.write_u32(value.binning_y);
        stream.write_u32(value.roi_x_offset);
        stream.write_u32(value.roi_y_offset);
        stream.write_u32(value.roi_height);
        stream.write_u32(value.roi_width);
        stream.write_u8(value.roi_do_rectify);
        return stream.finish();
    }

    inline std::vector<unsigned char> write_imu(const imu& value) {
        writer stream;
        write_header(stream, value.frame_header);
        for (int i = 0; i < 4; ++i) {
            stream.write_f64(value.orientation[i]);
        }
        for (int i = 0; i < 9; ++i) {
            stream.write_f64(value.orientation_covariance[i]);
        }
        for (int i = 0; i < 3; ++i) {
            stream.write_f64(value.angular_velocity[i]);
        }
        for (int i = 0; i < 9; ++i) {
            stream.write_f64(value.angular_velocity_covariance[i]);
        }
        for (int i = 0; i < 3; ++i) {
            stream.write_f64(value.linear_acceleration[i]);
        }
        for (int i = 0; i < 9; ++i) {
            stream.write_f64(value.linear_acceleration_covariance[i]);
        }
        return stream.finish();
    }

    // A tf2_msgs/msg/TFMessage holding one transform: the pose of the child frame in the
    // parent frame, with the quaternion in the ros field order x, y, z, w.
    inline std::vector<unsigned char> write_tf_message(const time& stamp, const std::string& parent_frame, const std::string& child_frame, const double translation[3], const double rotation_xyzw[4]) {
        writer stream;
        stream.write_u32(1); // One transform.
        stream.write_i32(stamp.sec);
        stream.write_u32(stamp.nanosec);
        stream.write_string(parent_frame);
        stream.write_string(child_frame);
        for (int i = 0; i < 3; ++i) {
            stream.write_f64(translation[i]);
        }
        for (int i = 0; i < 4; ++i) {
            stream.write_f64(rotation_xyzw[i]);
        }
        return stream.finish();
    }

}

#endif // ZEROSLAM_TOOLS_COMMON_CDR_HPP
