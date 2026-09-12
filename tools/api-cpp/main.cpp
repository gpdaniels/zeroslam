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

#include "zeroslam/zeroslam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#define REQUIRE(EXPRESSION) require((EXPRESSION), #EXPRESSION, __LINE__)

namespace {
    void require(bool passed, const char* expression, int line) {
        if (passed) {
            return;
        }
        std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", line, expression);
        std::exit(EXIT_FAILURE);
    }

    class random_generator final {
    private:
        unsigned int state;

    public:
        explicit random_generator(unsigned int seed)
            : state(seed) {
        }

        unsigned int next() {
            this->state = (this->state * 1664525u) + 1013904223u;
            return this->state >> 8;
        }
    };

    void generate_frame(std::vector<unsigned char>& pixels, int width, int height, int frame_index) {
        random_generator random(2026u + static_cast<unsigned int>(frame_index));
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                const int shifted_column = column + (2 * frame_index);
                const int blob = ((shifted_column / 16) + (row / 16)) % 2;
                const int noise = static_cast<int>(random.next() % 40u);
                const int value = 100 + (60 * blob) + noise;
                pixels[static_cast<size_t>((row * width) + column)] = static_cast<unsigned char>((value > 255) ? 255 : value);
            }
        }
    }

    void print_pose(const char* label, const zeroslam_pose_struct& pose) {
        std::printf("%s: timestamp %lld, centre (%.3f, %.3f, %.3f), rotation (%.3f, %.3f, %.3f, %.3f)\n", label, pose.timestamp, static_cast<double>(pose.pose[0]), static_cast<double>(pose.pose[1]), static_cast<double>(pose.pose[2]), static_cast<double>(pose.pose[3]), static_cast<double>(pose.pose[4]), static_cast<double>(pose.pose[5]), static_cast<double>(pose.pose[6]));
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    zeroslam::system system;
    std::printf("create: %s\n", system.is_valid() ? "success" : "failure");
    REQUIRE(system.is_valid());

    {
        int length = 0;
        zeroslam_return_enum result = system.get_configuration(nullptr, &length);
        REQUIRE(result == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(length > 0);
        std::vector<char> configuration(static_cast<size_t>(length));
        result = system.get_configuration(configuration.data(), &length);
        std::printf("get_configuration: %s (%d characters)\n", zeroslam_return_enum_to_string(result), length);
        REQUIRE(result == zeroslam_return_success);
        std::printf("%s", configuration.data());

        const char* const delta = "verbosity=1\n";
        result = system.set_configuration(delta, static_cast<int>(std::strlen(delta)));
        std::printf("set_configuration: %s\n", zeroslam_return_enum_to_string(result));
        REQUIRE(result == zeroslam_return_success);

        const char* const unknown = "warp_speed=9\n";
        result = system.set_configuration(unknown, static_cast<int>(std::strlen(unknown)));
        std::printf("set_configuration (unknown key): %s\n", zeroslam_return_enum_to_string(result));
        REQUIRE(result == zeroslam_return_failure_invalid_configuration);
    }

    constexpr static const int width = 320;
    constexpr static const int height = 240;
    constexpr static const int sensor_id = 1;
    zeroslam_sensor_parameters_camera_struct camera_parameters{};
    camera_parameters.width = width;
    camera_parameters.height = height;
    camera_parameters.focal_x = 262.5;
    camera_parameters.focal_y = 262.5;
    camera_parameters.centre_x = 160.0;
    camera_parameters.centre_y = 120.0;
    zeroslam_sensor_rig_struct rig{};
    rig.type = zeroslam_sensor_camera;
    rig.sensor_id = sensor_id;
    rig.parameters_length = static_cast<int>(sizeof(camera_parameters));
    rig.parameters_data = &camera_parameters;
    zeroslam_return_enum result = system.set_sensor_rig(&rig, 1);
    std::printf("set_sensor_rig: %s (%s, id %d, %dx%d)\n", zeroslam_return_enum_to_string(result), zeroslam_sensor_enum_to_string(rig.type), rig.sensor_id, width, height);
    REQUIRE(result == zeroslam_return_success);

    constexpr static const int frame_count = 5;
    constexpr static const long long int frame_interval = 33333333ll;
    std::vector<unsigned char> pixels(static_cast<size_t>(width * height));
    for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
        generate_frame(pixels, width, height, frame_index);
        zeroslam_sensor_data_struct data{};
        data.timestamp = 1000000000ll + (frame_interval * frame_index);
        data.sensor_id = sensor_id;
        data.measurement_length = width * height;
        data.measurement_data = pixels.data();
        result = system.set_sensor_data(&data, 1);
        std::printf("set_sensor_data: %s (timestamp %lld)\n", zeroslam_return_enum_to_string(result), data.timestamp);
        REQUIRE(result == zeroslam_return_success);
    }

    long long int timestamp = 0;
    result = system.get_timestamp(&timestamp);
    std::printf("get_timestamp: %s (%lld)\n", zeroslam_return_enum_to_string(result), timestamp);
    REQUIRE(result == zeroslam_return_success);
    REQUIRE(timestamp == 1000000000ll + (frame_interval * (frame_count - 1)));

    {
        zeroslam_pose_struct pose{};
        result = system.get_pose(&pose);
        std::printf("get_pose: %s\n", zeroslam_return_enum_to_string(result));
        REQUIRE(result == zeroslam_return_success);
        print_pose("newest pose", pose);

        pose = zeroslam_pose_struct{};
        result = system.get_pose_at_timestamp(&pose, 1000000000ll);
        std::printf("get_pose_at_timestamp: %s\n", zeroslam_return_enum_to_string(result));
        REQUIRE(result == zeroslam_return_success);
        print_pose("first pose", pose);
        REQUIRE(pose.pose[0] == 0.0);
        REQUIRE(pose.pose[6] == 1.0);

        result = system.get_pose_at_timestamp(&pose, 12345ll);
        std::printf("get_pose_at_timestamp (unknown): %s\n", zeroslam_return_enum_to_string(result));
        REQUIRE(result == zeroslam_return_failure_invalid_argument);
    }

    {
        zeroslam_map_chunk_struct chunk{};
        result = system.get_map_chunk(0.0f, 0.0f, 0.0f, &chunk);
        std::printf("get_map_chunk: %s (%d points)\n", zeroslam_return_enum_to_string(result), chunk.points_length);
        if (result == zeroslam_return_failure_insufficient_data_length) {
            REQUIRE(chunk.points_length > 0);
            std::vector<zeroslam_point_struct> points(static_cast<size_t>(chunk.points_length));
            chunk.points = points.data();
            result = system.get_map_chunk(0.0f, 0.0f, 0.0f, &chunk);
            std::printf("get_map_chunk (sized): %s (%d points)\n", zeroslam_return_enum_to_string(result), chunk.points_length);
            REQUIRE(result == zeroslam_return_success);
            for (int index = 0; (index < chunk.points_length) && (index < 5); ++index) {
                const zeroslam_point_struct& point = points[static_cast<size_t>(index)];
                std::printf("  point %d: (%.3f, %.3f, %.3f) confidence %.3f\n", index, static_cast<double>(point.x), static_cast<double>(point.y), static_cast<double>(point.z), static_cast<double>(point.confidence));
            }
        }
        else {
            REQUIRE(result == zeroslam_return_success);
            REQUIRE(chunk.points_length == 0);
        }
    }

    {
        zeroslam_map_lines_struct lines{};
        result = system.get_map_lines(&lines);
        std::printf("get_map_lines: %s (%d lines)\n", zeroslam_return_enum_to_string(result), lines.lines_length);
        std::vector<zeroslam_line_struct> line_buffer;
        if (result == zeroslam_return_failure_insufficient_data_length) {
            line_buffer.resize(static_cast<size_t>(lines.lines_length));
            lines.lines = line_buffer.data();
            REQUIRE(system.get_map_lines(&lines) == zeroslam_return_success);
        }
        else {
            REQUIRE(result == zeroslam_return_success);
        }
        zeroslam_map_edges_struct edges{};
        result = system.get_map_edges(&edges);
        std::printf("get_map_edges: %s (%d edges)\n", zeroslam_return_enum_to_string(result), edges.edges_length);
        std::vector<zeroslam_edge_struct> edge_buffer;
        if (result == zeroslam_return_failure_insufficient_data_length) {
            edge_buffer.resize(static_cast<size_t>(edges.edges_length));
            edges.edges = edge_buffer.data();
            REQUIRE(system.get_map_edges(&edges) == zeroslam_return_success);
            for (int index = 0; (index < edges.edges_length) && (index < 5); ++index) {
                const zeroslam_edge_struct& edge = edge_buffer[static_cast<size_t>(index)];
                std::printf("  edge %d: %lld - %lld type %d weight %d\n", index, edge.timestamp_a, edge.timestamp_b, edge.type, edge.weight);
            }
        }
        else {
            REQUIRE(result == zeroslam_return_success);
        }
    }

    zeroslam::system moved(static_cast<zeroslam::system&&>(system));
    REQUIRE(!system.is_valid());
    REQUIRE(moved.is_valid());
    REQUIRE(system.get_timestamp(&timestamp) == zeroslam_return_failure_invalid_system);
    REQUIRE(moved.get_timestamp(&timestamp) == zeroslam_return_success);
    std::printf("destroy: on scope exit\n");

    return EXIT_SUCCESS;
}
