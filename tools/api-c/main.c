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

#include "zeroslam/zeroslam.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(EXPRESSION) require((EXPRESSION) ? 1 : 0, #EXPRESSION, __LINE__)

static void require(int passed, const char* expression, int line) {
    if (passed) {
        return;
    }
    fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", line, expression);
    exit(EXIT_FAILURE);
}

static long long int floor_to_integer(double value) {
    const long long int truncated = (long long int)value;
    return ((double)truncated > value) ? (truncated - 1) : truncated;
}

static void render_corridor(unsigned char* pixels, int width, int height, const zeroslam_sensor_parameters_camera_struct* camera, double camera_x) {
    const double wall_depth = 6.0;
    for (int row = 0; row < height; ++row) {
        const double ray_y = (((double)row + 0.5) - camera->centre_y) / camera->focal_y;
        const double ray_y_magnitude = (ray_y < 0.0) ? -ray_y : ray_y;
        const int wall = (ray_y_magnitude * wall_depth) <= 1.0;
        const double depth = wall ? wall_depth : (1.0 / ray_y_magnitude);
        const unsigned long long int surface = wall ? 1ull : ((ray_y > 0.0) ? 2ull : 3ull);
        const long long int cell_v = floor_to_integer(4.0 * (wall ? (ray_y * depth) : depth));
        for (int column = 0; column < width; ++column) {
            const double ray_x = (((double)column + 0.5) - camera->centre_x) / camera->focal_x;
            const long long int cell_u = floor_to_integer(4.0 * (camera_x + (ray_x * depth)));
            unsigned long long int hash = ((unsigned long long int)cell_u * 73856093ull) ^ ((unsigned long long int)cell_v * 19349663ull) ^ (surface * 83492791ull);
            hash ^= hash >> 13;
            hash *= 0x5bd1e995ull;
            hash ^= hash >> 15;
            pixels[(row * width) + column] = (unsigned char)(40ull + (hash % 176ull));
        }
    }
}

// The square root by Newton's method, so the demonstration needs no maths library.
static double square_root(double value) {
    if (value <= 0.0) {
        return 0.0;
    }
    double root = (value > 1.0) ? value : 1.0;
    for (int iteration = 0; iteration < 64; ++iteration) {
        const double next = 0.5 * (root + (value / root));
        if (next >= root) {
            break;
        }
        root = next;
    }
    return root;
}

static double absolute(double value) {
    return (value < 0.0) ? -value : value;
}

// The ray of a pixel through a double sphere camera [fx fy cx cy xi alpha] (Usenko, Demmel and Cremers), 0 outside its image.
static int double_sphere_ray(const double* parameters, double u, double v, double* ray) {
    const double mx = (u - parameters[2]) / parameters[0];
    const double my = (v - parameters[3]) / parameters[1];
    const double r2 = (mx * mx) + (my * my);
    const double xi = parameters[4];
    const double alpha = parameters[5];
    const double discriminant = 1.0 - (((2.0 * alpha) - 1.0) * r2);
    if (discriminant <= 0.0) {
        return 0;
    }
    const double mz = (1.0 - (alpha * alpha * r2)) / ((alpha * square_root(discriminant)) + 1.0 - alpha);
    const double sphere = (mz * mz) + ((1.0 - (xi * xi)) * r2);
    if (sphere <= 0.0) {
        return 0;
    }
    const double k = ((mz * xi) + square_root(sphere)) / ((mz * mz) + r2);
    ray[0] = k * mx;
    ray[1] = k * my;
    ray[2] = (k * mz) - xi;
    return 1;
}

// The corridor seen along any ray: the floor and the ceiling a unit below and above the camera, the end walls six units
// ahead and behind, so a camera that sees past ninety degrees from its axis sees something there too.
static void render_corridor_wide(unsigned char* pixels, int width, int height, const double* parameters, double camera_x) {
    const double wall_depth = 6.0;
    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            double ray[3];
            unsigned char* pixel = &pixels[(row * width) + column];
            if (!double_sphere_ray(parameters, (double)column + 0.5, (double)row + 0.5, ray)) {
                *pixel = 0;
                continue;
            }
            const double ray_y_magnitude = (ray[1] < 0.0) ? -ray[1] : ray[1];
            const double ray_z_magnitude = (ray[2] < 0.0) ? -ray[2] : ray[2];
            const double to_plane = (ray_y_magnitude > 0.0) ? (1.0 / ray_y_magnitude) : 1.0e9;
            const double to_wall = (ray_z_magnitude > 0.0) ? (wall_depth / ray_z_magnitude) : 1.0e9;
            const int wall = to_wall <= to_plane;
            const double distance = wall ? to_wall : to_plane;
            const unsigned long long int surface = wall ? ((ray[2] > 0.0) ? 1ull : 4ull) : ((ray[1] > 0.0) ? 2ull : 3ull);
            const long long int cell_u = floor_to_integer(4.0 * (camera_x + (ray[0] * distance)));
            const long long int cell_v = floor_to_integer(4.0 * (wall ? (ray[1] * distance) : (ray[2] * distance)));
            unsigned long long int hash = ((unsigned long long int)cell_u * 73856093ull) ^ ((unsigned long long int)cell_v * 19349663ull) ^ (surface * 83492791ull);
            hash ^= hash >> 13;
            hash *= 0x5bd1e995ull;
            hash ^= hash >> 15;
            *pixel = (unsigned char)(40ull + (hash % 176ull));
        }
    }
}

static void print_pose(const char* label, const zeroslam_pose_struct* pose) {
    printf("%s: timestamp %lld, centre (%.3f, %.3f, %.3f), rotation (%.3f, %.3f, %.3f, %.3f)\n", label, pose->timestamp, (double)pose->pose[0], (double)pose->pose[1], (double)pose->pose[2], (double)pose->pose[3], (double)pose->pose[4], (double)pose->pose[5], (double)pose->pose[6]);
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    zeroslam_system* system = NULL;
    zeroslam_return_enum result = zeroslam_create(&system);
    printf("create: %s\n", zeroslam_return_enum_to_string(result));
    REQUIRE(result == zeroslam_return_success);
    REQUIRE(system != NULL);

    {
        int length = 0;
        result = zeroslam_get_configuration(system, NULL, &length);
        REQUIRE(result == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(length > 0);
        char* configuration = (char*)malloc((size_t)length);
        REQUIRE(configuration != NULL);
        result = zeroslam_get_configuration(system, configuration, &length);
        printf("get_configuration: %s (%d characters)\n", zeroslam_return_enum_to_string(result), length);
        REQUIRE(result == zeroslam_return_success);
        printf("%s", configuration);
        free(configuration);

        const char* const delta = "verbosity=1\nlines=on\n";
        result = zeroslam_set_configuration(system, delta, (int)strlen(delta));
        printf("set_configuration: %s\n", zeroslam_return_enum_to_string(result));
        REQUIRE(result == zeroslam_return_success);

        const char* const unknown = "warp_speed=9\n";
        result = zeroslam_set_configuration(system, unknown, (int)strlen(unknown));
        printf("set_configuration (unknown key): %s\n", zeroslam_return_enum_to_string(result));
        REQUIRE(result == zeroslam_return_failure_invalid_configuration);
    }

    const int width = 320;
    const int height = 240;
    const int sensor_id = 1;
    zeroslam_sensor_parameters_camera_struct camera_parameters;
    memset(&camera_parameters, 0, sizeof(camera_parameters));
    camera_parameters.width = width;
    camera_parameters.height = height;
    camera_parameters.focal_x = 262.5;
    camera_parameters.focal_y = 262.5;
    camera_parameters.centre_x = 160.0;
    camera_parameters.centre_y = 120.0;
    zeroslam_sensor_rig_struct rig;
    memset(&rig, 0, sizeof(rig));
    rig.type = zeroslam_sensor_camera;
    rig.sensor_id = sensor_id;
    rig.parameters_length = (int)sizeof(camera_parameters);
    rig.parameters_data = &camera_parameters;
    result = zeroslam_set_sensor_rig(system, &rig, 1);
    printf("set_sensor_rig: %s (%s, id %d, %dx%d)\n", zeroslam_return_enum_to_string(result), zeroslam_sensor_enum_to_string(rig.type), rig.sensor_id, width, height);
    REQUIRE(result == zeroslam_return_success);

    /* Where each sensor sits on the rig's body, sensor to body, and whether it is refined: a single camera is the body. */
    zeroslam_sensor_extrinsics_struct extrinsic;
    memset(&extrinsic, 0, sizeof(extrinsic));
    extrinsic.sensor_id = sensor_id;
    extrinsic.refinement = zeroslam_extrinsic_fixed;
    extrinsic.pose[6] = 1.0;
    result = zeroslam_set_sensor_extrinsics(system, &extrinsic, 1);
    printf("set_sensor_extrinsics: %s (id %d at the body's origin, fixed)\n", zeroslam_return_enum_to_string(result), extrinsic.sensor_id);
    REQUIRE(result == zeroslam_return_success);

    const int frame_count = 20;
    const long long int frame_interval = 33333333ll;
    unsigned char* pixels = (unsigned char*)malloc((size_t)(width * height));
    REQUIRE(pixels != NULL);
    for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
        render_corridor(pixels, width, height, &camera_parameters, 0.05 * frame_index);
        zeroslam_sensor_data_struct data;
        memset(&data, 0, sizeof(data));
        data.timestamp = 1000000000ll + (frame_interval * frame_index);
        data.sensor_id = sensor_id;
        data.measurement_length = width * height;
        data.measurement_data = pixels;
        result = zeroslam_set_sensor_data(system, &data, 1);
        printf("set_sensor_data: %s (timestamp %lld)\n", zeroslam_return_enum_to_string(result), data.timestamp);
        REQUIRE(result == zeroslam_return_success);
    }
    free(pixels);

    long long int timestamp = 0;
    result = zeroslam_get_timestamp(system, &timestamp);
    printf("get_timestamp: %s (%lld)\n", zeroslam_return_enum_to_string(result), timestamp);
    REQUIRE(result == zeroslam_return_success);
    REQUIRE(timestamp == 1000000000ll + (frame_interval * (frame_count - 1)));

    {
        /* The camera's calibration as the system holds it: the rig's, as the calibration key is off. */
        zeroslam_sensor_parameters_camera_model_struct calibration;
        memset(&calibration, 0, sizeof(calibration));
        result = zeroslam_get_camera_calibration(system, sensor_id, &calibration);
        printf("get_camera_calibration: %s (%s, fx %.3f, fy %.3f, cx %.3f, cy %.3f)\n", zeroslam_return_enum_to_string(result), zeroslam_camera_model_enum_to_string(calibration.model), calibration.parameters[0], calibration.parameters[1], calibration.parameters[2], calibration.parameters[3]);
        REQUIRE(result == zeroslam_return_success);
        REQUIRE(calibration.model == zeroslam_camera_model_radial_tangential);
        REQUIRE(calibration.parameters_length == 12);
        REQUIRE(calibration.parameters[0] == camera_parameters.focal_x);
        REQUIRE(calibration.parameters[3] == camera_parameters.centre_y);
        REQUIRE(zeroslam_get_camera_calibration(system, sensor_id + 1, &calibration) == zeroslam_return_failure_invalid_argument);
    }

    {
        /* The calibration over time: no chunks, as the calibration_drift key is off. */
        int chunks_length = -1;
        result = zeroslam_get_calibration_drift(system, NULL, &chunks_length);
        printf("get_calibration_drift: %s (%d chunks)\n", zeroslam_return_enum_to_string(result), chunks_length);
        REQUIRE(result == zeroslam_return_success);
        REQUIRE(chunks_length == 0);
    }

    {
        zeroslam_pose_struct pose;
        memset(&pose, 0, sizeof(pose));
        result = zeroslam_get_pose(system, &pose);
        printf("get_pose: %s\n", zeroslam_return_enum_to_string(result));
        REQUIRE(result == zeroslam_return_success);
        print_pose("newest pose", &pose);

        memset(&pose, 0, sizeof(pose));
        result = zeroslam_get_pose_at_timestamp(system, &pose, 1000000000ll);
        printf("get_pose_at_timestamp: %s\n", zeroslam_return_enum_to_string(result));
        REQUIRE(result == zeroslam_return_success);
        print_pose("first pose", &pose);
        REQUIRE(pose.pose[0] == 0.0);
        REQUIRE(pose.pose[6] == 1.0);

        result = zeroslam_get_pose_at_timestamp(system, &pose, 12345ll);
        printf("get_pose_at_timestamp (unknown): %s\n", zeroslam_return_enum_to_string(result));
        REQUIRE(result == zeroslam_return_failure_invalid_argument);
    }

    {
        zeroslam_map_chunk_struct chunk;
        memset(&chunk, 0, sizeof(chunk));
        result = zeroslam_get_map_chunk(system, 0.0f, 0.0f, 0.0f, &chunk);
        printf("get_map_chunk: %s (%d points)\n", zeroslam_return_enum_to_string(result), chunk.points_length);
        REQUIRE(result == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(chunk.points_length > 0);
        chunk.points = (zeroslam_point_struct*)malloc(sizeof(zeroslam_point_struct) * (size_t)chunk.points_length);
        REQUIRE(chunk.points != NULL);
        result = zeroslam_get_map_chunk(system, 0.0f, 0.0f, 0.0f, &chunk);
        printf("get_map_chunk (sized): %s (%d points)\n", zeroslam_return_enum_to_string(result), chunk.points_length);
        REQUIRE(result == zeroslam_return_success);
        for (int index = 0; (index < chunk.points_length) && (index < 5); ++index) {
            printf("  point %d: (%.3f, %.3f, %.3f) confidence %.3f\n", index, (double)chunk.points[index].x, (double)chunk.points[index].y, (double)chunk.points[index].z, (double)chunk.points[index].confidence);
        }
        free(chunk.points);
    }

    {
        zeroslam_map_lines_struct lines;
        memset(&lines, 0, sizeof(lines));
        result = zeroslam_get_map_lines(system, &lines);
        printf("get_map_lines: %s (%d lines)\n", zeroslam_return_enum_to_string(result), lines.lines_length);
        REQUIRE(result == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(lines.lines_length > 0);
        lines.lines = (zeroslam_line_struct*)malloc(sizeof(zeroslam_line_struct) * (size_t)lines.lines_length);
        REQUIRE(lines.lines != NULL);
        REQUIRE(zeroslam_get_map_lines(system, &lines) == zeroslam_return_success);
        free(lines.lines);
        zeroslam_map_edges_struct edges;
        memset(&edges, 0, sizeof(edges));
        result = zeroslam_get_map_edges(system, &edges);
        printf("get_map_edges: %s (%d edges)\n", zeroslam_return_enum_to_string(result), edges.edges_length);
        REQUIRE(result == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(edges.edges_length > 0);
        edges.edges = (zeroslam_edge_struct*)malloc(sizeof(zeroslam_edge_struct) * (size_t)edges.edges_length);
        REQUIRE(edges.edges != NULL);
        REQUIRE(zeroslam_get_map_edges(system, &edges) == zeroslam_return_success);
        for (int index = 0; (index < edges.edges_length) && (index < 5); ++index) {
            printf("  edge %d: %lld - %lld type %d weight %d\n", index, edges.edges[index].timestamp_a, edges.edges[index].timestamp_b, edges.edges[index].type, edges.edges[index].weight);
        }
        free(edges.edges);
        zeroslam_map_voxels_struct voxels;
        memset(&voxels, 0, sizeof(voxels));
        result = zeroslam_get_map_voxels(system, &voxels);
        printf("get_map_voxels: %s (%d voxels)\n", zeroslam_return_enum_to_string(result), voxels.voxels_length);
        REQUIRE(result == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(voxels.voxels_length > 0);
        voxels.voxels = (zeroslam_voxel_struct*)malloc(sizeof(zeroslam_voxel_struct) * (size_t)voxels.voxels_length);
        REQUIRE(voxels.voxels != NULL);
        REQUIRE(zeroslam_get_map_voxels(system, &voxels) == zeroslam_return_success);
        for (int index = 0; (index < voxels.voxels_length) && (index < 5); ++index) {
            printf("  voxel %d: corner (%.3f, %.3f, %.3f) size %.3f, %d points\n", index, (double)voxels.voxels[index].x, (double)voxels.voxels[index].y, (double)voxels.voxels[index].z, (double)voxels.voxel_size, voxels.voxels[index].points);
        }
        free(voxels.voxels);
    }

    REQUIRE(zeroslam_get_timestamp(NULL, &timestamp) == zeroslam_return_failure_invalid_system);

    result = zeroslam_destroy(&system);
    printf("destroy: %s\n", zeroslam_return_enum_to_string(result));
    REQUIRE(result == zeroslam_return_success);
    REQUIRE(system == NULL);

    {
        // Any camera model through zeroslam_sensor_parameters_camera_model_struct: a double sphere lens seeing past 90 degrees
        // from its axis, moving along the same corridor.
        result = zeroslam_create(&system);
        REQUIRE(result == zeroslam_return_success);
        zeroslam_sensor_parameters_camera_model_struct wide_camera;
        memset(&wide_camera, 0, sizeof(wide_camera));
        wide_camera.width = width;
        wide_camera.height = height;
        wide_camera.model = zeroslam_camera_model_double_sphere;
        wide_camera.parameters_length = zeroslam_camera_model_parameter_count(zeroslam_camera_model_double_sphere);
        const double wide_parameters[6] = { 110.0, 110.0, 160.0, 120.0, -0.2, 0.6 };
        memcpy(wide_camera.parameters, wide_parameters, sizeof(wide_parameters));
        zeroslam_sensor_rig_struct wide_rig;
        memset(&wide_rig, 0, sizeof(wide_rig));
        wide_rig.type = zeroslam_sensor_camera;
        wide_rig.sensor_id = sensor_id;
        wide_rig.parameters_length = (int)sizeof(wide_camera);
        wide_rig.parameters_data = &wide_camera;
        wide_camera.parameters_length = 5;
        REQUIRE(zeroslam_set_sensor_rig(system, &wide_rig, 1) == zeroslam_return_failure_invalid_rig_sensor);
        wide_camera.parameters_length = 6;
        wide_camera.parameters[5] = 1.5;
        REQUIRE(zeroslam_set_sensor_rig(system, &wide_rig, 1) == zeroslam_return_failure_invalid_rig_sensor);
        wide_camera.parameters[5] = wide_parameters[5];
        result = zeroslam_set_sensor_rig(system, &wide_rig, 1);
        printf("set_sensor_rig: %s (%s camera, %d parameters)\n", zeroslam_return_enum_to_string(result), zeroslam_camera_model_enum_to_string(wide_camera.model), wide_camera.parameters_length);
        REQUIRE(result == zeroslam_return_success);
        unsigned char* wide_pixels = (unsigned char*)malloc((size_t)(width * height));
        REQUIRE(wide_pixels != NULL);
        for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
            render_corridor_wide(wide_pixels, width, height, wide_parameters, 0.05 * frame_index);
            zeroslam_sensor_data_struct data;
            memset(&data, 0, sizeof(data));
            data.timestamp = 1000000000ll + (frame_interval * frame_index);
            data.sensor_id = sensor_id;
            data.measurement_length = width * height;
            data.measurement_data = wide_pixels;
            REQUIRE(zeroslam_set_sensor_data(system, &data, 1) == zeroslam_return_success);
        }
        free(wide_pixels);
        zeroslam_pose_struct first;
        zeroslam_pose_struct newest;
        memset(&first, 0, sizeof(first));
        memset(&newest, 0, sizeof(newest));
        REQUIRE(zeroslam_get_pose_at_timestamp(system, &first, 1000000000ll) == zeroslam_return_success);
        REQUIRE(zeroslam_get_pose(system, &newest) == zeroslam_return_success);
        print_pose("double sphere newest pose", &newest);
        // The camera slid along x without turning: the map's unit is unknown, but the motion is along its x axis.
        const double moved = square_root((newest.pose[0] * newest.pose[0]) + (newest.pose[1] * newest.pose[1]) + (newest.pose[2] * newest.pose[2]));
        REQUIRE(moved > 0.0);
        REQUIRE(absolute(newest.pose[0]) > 0.95 * moved);
        REQUIRE(absolute(newest.pose[6]) > 0.999);
        result = zeroslam_destroy(&system);
        REQUIRE(result == zeroslam_return_success);
    }

    {
        /* A depth camera registered to the camera makes the map metric: the depth frames go with the camera frames. */
        result = zeroslam_create(&system);
        REQUIRE(result == zeroslam_return_success);
        const int depth_sensor_id = 2;
        zeroslam_sensor_parameters_depth_struct depth_parameters;
        memset(&depth_parameters, 0, sizeof(depth_parameters));
        depth_parameters.camera_id = sensor_id;
        depth_parameters.encoding = zeroslam_depth_encoding_unsigned_16;
        depth_parameters.measure = zeroslam_depth_measure_axial;
        depth_parameters.registration = zeroslam_depth_registration_camera;
        depth_parameters.scale = 0.001;
        depth_parameters.minimum_depth = 0.1;
        depth_parameters.maximum_depth = 10.0;
        depth_parameters.noise_quadratic = 1.425e-3;
        depth_parameters.pairing_tolerance = frame_interval / 2;
        zeroslam_sensor_rig_struct rgbd[2];
        memset(&rgbd[0], 0, sizeof(rgbd));
        rgbd[0] = rig;
        rgbd[1].type = zeroslam_sensor_depth;
        rgbd[1].sensor_id = depth_sensor_id;
        rgbd[1].parameters_length = (int)sizeof(depth_parameters);
        rgbd[1].parameters_data = &depth_parameters;
        result = zeroslam_set_sensor_rig(system, &rgbd[0], 2);
        printf("set_sensor_rig (rgb-d): %s (%s, id %d, of camera %d)\n", zeroslam_return_enum_to_string(result), zeroslam_sensor_enum_to_string(rgbd[1].type), depth_sensor_id, sensor_id);
        REQUIRE(result == zeroslam_return_success);

        unsigned char* frame_pixels = (unsigned char*)malloc((size_t)(width * height));
        unsigned short* millimetres = (unsigned short*)malloc(sizeof(unsigned short) * (size_t)(width * height));
        REQUIRE((frame_pixels != NULL) && (millimetres != NULL));
        for (int row = 0; row < height; ++row) {
            const double ray_y = (((double)row + 0.5) - camera_parameters.centre_y) / camera_parameters.focal_y;
            const double ray_y_magnitude = (ray_y < 0.0) ? -ray_y : ray_y;
            const double depth = ((ray_y_magnitude * 6.0) <= 1.0) ? 6.0 : (1.0 / ray_y_magnitude);
            for (int column = 0; column < width; ++column) {
                millimetres[(row * width) + column] = (unsigned short)((depth * 1000.0) + 0.5);
            }
        }
        for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
            render_corridor(frame_pixels, width, height, &camera_parameters, 0.05 * frame_index);
            zeroslam_sensor_data_struct data[2];
            memset(&data[0], 0, sizeof(data));
            data[0].timestamp = 1000000000ll + (frame_interval * frame_index);
            data[0].sensor_id = depth_sensor_id;
            data[0].measurement_length = width * height * (int)sizeof(unsigned short);
            data[0].measurement_data = millimetres;
            data[1].timestamp = data[0].timestamp;
            data[1].sensor_id = sensor_id;
            data[1].measurement_length = width * height;
            data[1].measurement_data = frame_pixels;
            result = zeroslam_set_sensor_data(system, &data[0], 2);
            REQUIRE(result == zeroslam_return_success);
        }
        free(millimetres);
        free(frame_pixels);
        zeroslam_pose_struct pose;
        memset(&pose, 0, sizeof(pose));
        result = zeroslam_get_pose(system, &pose);
        printf("get_pose (rgb-d): %s\n", zeroslam_return_enum_to_string(result));
        REQUIRE(result == zeroslam_return_success);
        print_pose("newest pose, metres", &pose);
        REQUIRE((pose.pose[0] > 0.85) && (pose.pose[0] < 1.05));
        result = zeroslam_destroy(&system);
        REQUIRE(result == zeroslam_return_success);
    }

    {
        /* A camera with an imu: the imu is a sensor of the rig with its noise model, posed on the body like a camera, and its samples go
           in with the frames in time order. This camera glides at a steady speed, so the imu measures no turn and only gravity's pull
           (the camera's y axis points down): it is taken in, and the system initialises it only once the motion is excited enough. */
        result = zeroslam_create(&system);
        REQUIRE(result == zeroslam_return_success);
        zeroslam_sensor_parameters_imu_struct imu_parameters;
        memset(&imu_parameters, 0, sizeof(imu_parameters));
        imu_parameters.gyroscope_noise_density = 1.6968e-04;
        imu_parameters.accelerometer_noise_density = 2.0e-03;
        imu_parameters.gyroscope_random_walk = 1.9393e-05;
        imu_parameters.accelerometer_random_walk = 3.0e-03;
        imu_parameters.rate = 200.0;
        const int imu_id = 2;
        zeroslam_sensor_rig_struct sensors[2];
        memset(&sensors[0], 0, sizeof(sensors));
        sensors[0] = rig;
        sensors[1].type = zeroslam_sensor_imu;
        sensors[1].sensor_id = imu_id;
        sensors[1].parameters_length = (int)sizeof(imu_parameters);
        sensors[1].parameters_data = &imu_parameters;
        result = zeroslam_set_sensor_rig(system, &sensors[0], 2);
        printf("set_sensor_rig (camera and imu): %s (%s, id %d)\n", zeroslam_return_enum_to_string(result), zeroslam_sensor_enum_to_string(sensors[1].type), imu_id);
        REQUIRE(result == zeroslam_return_success);
        zeroslam_sensor_extrinsics_struct placements[2];
        memset(&placements[0], 0, sizeof(placements));
        placements[0] = extrinsic;
        placements[1].sensor_id = imu_id;
        placements[1].refinement = zeroslam_extrinsic_fixed;
        placements[1].pose[0] = 0.05;
        placements[1].pose[6] = 1.0;
        result = zeroslam_set_sensor_extrinsics(system, &placements[0], 2);
        printf("set_sensor_extrinsics (camera and imu): %s (imu 5 cm along the camera's x axis)\n", zeroslam_return_enum_to_string(result));
        REQUIRE(result == zeroslam_return_success);
        unsigned char* frame_pixels = (unsigned char*)malloc((size_t)(width * height));
        REQUIRE(frame_pixels != NULL);
        zeroslam_sensor_measurement_imu_struct still;
        memset(&still, 0, sizeof(still));
        still.linear_acceleration[1] = -9.81;
        long long int imu_time = 1000000000ll - 5000000ll;
        int samples = 0;
        for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
            const long long int frame_time = 1000000000ll + (frame_interval * frame_index);
            zeroslam_sensor_data_struct batch[16];
            int batch_length = 0;
            while ((imu_time + 5000000ll <= frame_time) && (batch_length < 15)) {
                imu_time += 5000000ll;
                memset(&batch[batch_length], 0, sizeof(batch[batch_length]));
                batch[batch_length].timestamp = imu_time;
                batch[batch_length].sensor_id = imu_id;
                batch[batch_length].measurement_length = (int)sizeof(still);
                batch[batch_length].measurement_data = &still;
                ++batch_length;
                ++samples;
            }
            render_corridor(frame_pixels, width, height, &camera_parameters, 0.05 * frame_index);
            memset(&batch[batch_length], 0, sizeof(batch[batch_length]));
            batch[batch_length].timestamp = frame_time;
            batch[batch_length].sensor_id = sensor_id;
            batch[batch_length].measurement_length = width * height;
            batch[batch_length].measurement_data = frame_pixels;
            ++batch_length;
            result = zeroslam_set_sensor_data(system, &batch[0], batch_length);
            REQUIRE(result == zeroslam_return_success);
        }
        free(frame_pixels);
        printf("set_sensor_data (camera and imu): %s (%d frames, %d imu samples between them)\n", zeroslam_return_enum_to_string(result), frame_count, samples);
        zeroslam_pose_struct pose;
        memset(&pose, 0, sizeof(pose));
        result = zeroslam_get_pose(system, &pose);
        REQUIRE(result == zeroslam_return_success);
        print_pose("newest pose (camera and imu)", &pose);
        /* The imu's calibration: the matrix of its gyroscope's scale and misalignment, zero until an adjustment refines it, which
           the configuration's imu_calibration key asks for. */
        zeroslam_imu_calibration_struct imu_calibration;
        memset(&imu_calibration, 0, sizeof(imu_calibration));
        result = zeroslam_get_imu_calibration(system, &imu_calibration);
        printf("get_imu_calibration: %s (sensor %d, gyroscope refined %d times, diagonal %.4f %.4f %.4f)\n", zeroslam_return_enum_to_string(result), imu_calibration.sensor_id, imu_calibration.refinements, imu_calibration.gyroscope[0], imu_calibration.gyroscope[4], imu_calibration.gyroscope[8]);
        REQUIRE(result == zeroslam_return_success);
        result = zeroslam_destroy(&system);
        REQUIRE(result == zeroslam_return_success);
    }

    return EXIT_SUCCESS;
}
