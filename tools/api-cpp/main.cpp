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

#include <cmath>
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

    void render_corridor(std::vector<unsigned char>& pixels, int width, int height, const zeroslam_sensor_parameters_camera_struct& camera, double camera_x) {
        constexpr static const double wall_depth = 6.0;
        for (int row = 0; row < height; ++row) {
            const double ray_y = ((static_cast<double>(row) + 0.5) - camera.centre_y) / camera.focal_y;
            const bool wall = (std::abs(ray_y) * wall_depth) <= 1.0;
            const double depth = wall ? wall_depth : (1.0 / std::abs(ray_y));
            const unsigned long long int surface = wall ? 1ull : ((ray_y > 0.0) ? 2ull : 3ull);
            const long long int cell_v = static_cast<long long int>(std::floor(4.0 * (wall ? (ray_y * depth) : depth)));
            for (int column = 0; column < width; ++column) {
                const double ray_x = ((static_cast<double>(column) + 0.5) - camera.centre_x) / camera.focal_x;
                const long long int cell_u = static_cast<long long int>(std::floor(4.0 * (camera_x + (ray_x * depth))));
                unsigned long long int hash = (static_cast<unsigned long long int>(cell_u) * 73856093ull) ^ (static_cast<unsigned long long int>(cell_v) * 19349663ull) ^ (surface * 83492791ull);
                hash ^= hash >> 13;
                hash *= 0x5bd1e995ull;
                hash ^= hash >> 15;
                pixels[static_cast<size_t>((row * width) + column)] = static_cast<unsigned char>(40ull + (hash % 176ull));
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

        const char* const delta = "verbosity=1\nlines=on\n";
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

    // Where each sensor sits on the rig's body, sensor to body, and whether it is refined: a single camera is the body.
    zeroslam_sensor_extrinsics_struct extrinsic{};
    extrinsic.sensor_id = sensor_id;
    extrinsic.refinement = zeroslam_extrinsic_fixed;
    extrinsic.pose[6] = 1.0;
    result = system.set_sensor_extrinsics(&extrinsic, 1);
    std::printf("set_sensor_extrinsics: %s (id %d at the body's origin, fixed)\n", zeroslam_return_enum_to_string(result), extrinsic.sensor_id);
    REQUIRE(result == zeroslam_return_success);

    constexpr static const int frame_count = 20;
    constexpr static const long long int frame_interval = 33333333ll;
    std::vector<unsigned char> pixels(static_cast<size_t>(width * height));
    for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
        render_corridor(pixels, width, height, camera_parameters, 0.05 * frame_index);
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
        // The camera's calibration as the system holds it: the rig's, as the calibration key is off.
        zeroslam_sensor_parameters_camera_model_struct calibration{};
        result = system.get_camera_calibration(sensor_id, &calibration);
        std::printf("get_camera_calibration: %s (%s, fx %.3f, fy %.3f, cx %.3f, cy %.3f)\n", zeroslam_return_enum_to_string(result), zeroslam_camera_model_enum_to_string(calibration.model), calibration.parameters[0], calibration.parameters[1], calibration.parameters[2], calibration.parameters[3]);
        REQUIRE(result == zeroslam_return_success);
        REQUIRE(calibration.parameters_length == 12);
        REQUIRE(calibration.parameters[0] == camera_parameters.focal_x);
        REQUIRE(calibration.parameters[3] == camera_parameters.centre_y);
        REQUIRE(system.get_camera_calibration(sensor_id + 1, &calibration) == zeroslam_return_failure_invalid_argument);
    }

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
        REQUIRE(result == zeroslam_return_failure_insufficient_data_length);
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

    {
        zeroslam_map_lines_struct lines{};
        result = system.get_map_lines(&lines);
        std::printf("get_map_lines: %s (%d lines)\n", zeroslam_return_enum_to_string(result), lines.lines_length);
        REQUIRE(result == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(lines.lines_length > 0);
        std::vector<zeroslam_line_struct> line_buffer(static_cast<size_t>(lines.lines_length));
        lines.lines = line_buffer.data();
        REQUIRE(system.get_map_lines(&lines) == zeroslam_return_success);
        zeroslam_map_edges_struct edges{};
        result = system.get_map_edges(&edges);
        std::printf("get_map_edges: %s (%d edges)\n", zeroslam_return_enum_to_string(result), edges.edges_length);
        REQUIRE(result == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(edges.edges_length > 0);
        std::vector<zeroslam_edge_struct> edge_buffer(static_cast<size_t>(edges.edges_length));
        edges.edges = edge_buffer.data();
        REQUIRE(system.get_map_edges(&edges) == zeroslam_return_success);
        for (int index = 0; (index < edges.edges_length) && (index < 5); ++index) {
            const zeroslam_edge_struct& edge = edge_buffer[static_cast<size_t>(index)];
            std::printf("  edge %d: %lld - %lld type %d weight %d\n", index, edge.timestamp_a, edge.timestamp_b, edge.type, edge.weight);
        }
        zeroslam_map_voxels_struct voxels{};
        result = system.get_map_voxels(&voxels);
        std::printf("get_map_voxels: %s (%d voxels)\n", zeroslam_return_enum_to_string(result), voxels.voxels_length);
        REQUIRE(result == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(voxels.voxels_length > 0);
        std::vector<zeroslam_voxel_struct> voxel_buffer(static_cast<size_t>(voxels.voxels_length));
        voxels.voxels = voxel_buffer.data();
        REQUIRE(system.get_map_voxels(&voxels) == zeroslam_return_success);
        for (int index = 0; (index < voxels.voxels_length) && (index < 5); ++index) {
            const zeroslam_voxel_struct& voxel = voxel_buffer[static_cast<size_t>(index)];
            std::printf("  voxel %d: corner (%.3f, %.3f, %.3f) size %.3f, %d points\n", index, static_cast<double>(voxel.x), static_cast<double>(voxel.y), static_cast<double>(voxel.z), static_cast<double>(voxels.voxel_size), voxel.points);
        }
    }

    zeroslam::system moved(static_cast<zeroslam::system&&>(system));
    REQUIRE(!system.is_valid());
    REQUIRE(moved.is_valid());
    REQUIRE(system.get_timestamp(&timestamp) == zeroslam_return_failure_invalid_system);
    REQUIRE(moved.get_timestamp(&timestamp) == zeroslam_return_success);

    {
        // A depth camera registered to the camera makes the map metric: the depth frames go with the camera frames.
        zeroslam::system rgbd;
        REQUIRE(rgbd.is_valid());
        constexpr static const int depth_sensor_id = 2;
        zeroslam_sensor_parameters_depth_struct depth_parameters = zeroslam::registered_depth(sensor_id, zeroslam_depth_encoding_float_32, 1.0);
        zeroslam_sensor_rig_struct sensors[2] = { rig, {} };
        sensors[1].type = zeroslam_sensor_depth;
        sensors[1].sensor_id = depth_sensor_id;
        sensors[1].parameters_length = static_cast<int>(sizeof(depth_parameters));
        sensors[1].parameters_data = &depth_parameters;
        zeroslam_return_enum rgbd_result = rgbd.set_sensor_rig(&sensors[0], 2);
        std::printf("set_sensor_rig (rgb-d): %s\n", zeroslam_return_enum_to_string(rgbd_result));
        REQUIRE(rgbd_result == zeroslam_return_success);
        std::vector<unsigned char> frame_pixels(static_cast<size_t>(width * height));
        std::vector<float> metres(static_cast<size_t>(width * height));
        for (int row = 0; row < height; ++row) {
            const double ray_y = ((static_cast<double>(row) + 0.5) - camera_parameters.centre_y) / camera_parameters.focal_y;
            const double depth = ((std::abs(ray_y) * 6.0) <= 1.0) ? 6.0 : (1.0 / std::abs(ray_y));
            for (int column = 0; column < width; ++column) {
                metres[static_cast<size_t>((row * width) + column)] = static_cast<float>(depth);
            }
        }
        for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
            render_corridor(frame_pixels, width, height, camera_parameters, 0.05 * frame_index);
            zeroslam_sensor_data_struct data[2]{};
            data[0].timestamp = 1000000000ll + (frame_interval * frame_index);
            data[0].sensor_id = depth_sensor_id;
            data[0].measurement_length = width * height * static_cast<int>(sizeof(float));
            data[0].measurement_data = metres.data();
            data[1].timestamp = data[0].timestamp;
            data[1].sensor_id = sensor_id;
            data[1].measurement_length = width * height;
            data[1].measurement_data = frame_pixels.data();
            REQUIRE(rgbd.set_sensor_data(&data[0], 2) == zeroslam_return_success);
        }
        zeroslam_pose_struct pose{};
        rgbd_result = rgbd.get_pose(&pose);
        std::printf("get_pose (rgb-d): %s\n", zeroslam_return_enum_to_string(rgbd_result));
        REQUIRE(rgbd_result == zeroslam_return_success);
        print_pose("newest pose, metres", pose);
        REQUIRE((pose.pose[0] > 0.85) && (pose.pose[0] < 1.05));
    }
    std::printf("destroy: on scope exit\n");

    return EXIT_SUCCESS;
}
