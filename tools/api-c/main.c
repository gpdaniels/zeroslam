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

static unsigned int random_next(unsigned int* state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state >> 8;
}

static void generate_frame(unsigned char* pixels, int width, int height, int frame_index) {
    unsigned int state = 2026u + (unsigned int)frame_index;
    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            const int shifted_column = column + (2 * frame_index);
            const int blob = ((shifted_column / 16) + (row / 16)) % 2;
            const int noise = (int)(random_next(&state) % 40u);
            const int value = 100 + (60 * blob) + noise;
            pixels[(row * width) + column] = (unsigned char)((value > 255) ? 255 : value);
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

        const char* const delta = "verbosity=1\n";
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

    const int frame_count = 5;
    const long long int frame_interval = 33333333ll;
    unsigned char* pixels = (unsigned char*)malloc((size_t)(width * height));
    REQUIRE(pixels != NULL);
    for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
        generate_frame(pixels, width, height, frame_index);
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
        if (result == zeroslam_return_failure_insufficient_data_length) {
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
        else {
            REQUIRE(result == zeroslam_return_success);
            REQUIRE(chunk.points_length == 0);
        }
    }

    {
        zeroslam_map_lines_struct lines;
        memset(&lines, 0, sizeof(lines));
        result = zeroslam_get_map_lines(system, &lines);
        printf("get_map_lines: %s (%d lines)\n", zeroslam_return_enum_to_string(result), lines.lines_length);
        if (result == zeroslam_return_failure_insufficient_data_length) {
            lines.lines = (zeroslam_line_struct*)malloc(sizeof(zeroslam_line_struct) * (size_t)lines.lines_length);
            REQUIRE(lines.lines != NULL);
            REQUIRE(zeroslam_get_map_lines(system, &lines) == zeroslam_return_success);
            free(lines.lines);
        }
        else {
            REQUIRE(result == zeroslam_return_success);
        }
        zeroslam_map_edges_struct edges;
        memset(&edges, 0, sizeof(edges));
        result = zeroslam_get_map_edges(system, &edges);
        printf("get_map_edges: %s (%d edges)\n", zeroslam_return_enum_to_string(result), edges.edges_length);
        if (result == zeroslam_return_failure_insufficient_data_length) {
            edges.edges = (zeroslam_edge_struct*)malloc(sizeof(zeroslam_edge_struct) * (size_t)edges.edges_length);
            REQUIRE(edges.edges != NULL);
            REQUIRE(zeroslam_get_map_edges(system, &edges) == zeroslam_return_success);
            for (int index = 0; (index < edges.edges_length) && (index < 5); ++index) {
                printf("  edge %d: %lld - %lld type %d weight %d\n", index, edges.edges[index].timestamp_a, edges.edges[index].timestamp_b, edges.edges[index].type, edges.edges[index].weight);
            }
            free(edges.edges);
        }
        else {
            REQUIRE(result == zeroslam_return_success);
        }
    }

    REQUIRE(zeroslam_get_timestamp(NULL, &timestamp) == zeroslam_return_failure_invalid_system);

    result = zeroslam_destroy(&system);
    printf("destroy: %s\n", zeroslam_return_enum_to_string(result));
    REQUIRE(result == zeroslam_return_success);
    REQUIRE(system == NULL);

    return EXIT_SUCCESS;
}
