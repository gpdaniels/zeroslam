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
#ifndef ZEROSLAM_ZEROSLAM_H
#define ZEROSLAM_ZEROSLAM_H

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(_MSC_VER)
#if defined(ZEROSLAM_API_EXPORT)
#define ZEROSLAM_API_VISIBILITY __declspec(dllexport)
#else
#define ZEROSLAM_API_VISIBILITY __declspec(dllimport)
#endif
#define ZEROSLAM_API_CALL __stdcall
#else
#define ZEROSLAM_API_VISIBILITY __attribute__((visibility("default")))
#define ZEROSLAM_API_CALL
#endif

#if defined(__cplusplus)
#define ZEROSLAM_API_STATIC_ASSERT(ASSERTION, MESSAGE) static_assert(ASSERTION, MESSAGE);
#else
#define ZEROSLAM_API_STATIC_ASSERT(ASSERTION, MESSAGE) _Static_assert(ASSERTION, MESSAGE);
#endif

ZEROSLAM_API_STATIC_ASSERT(sizeof(char) == 1, "For ABI compatibility the sizeof(char) must be 1 byte.")
ZEROSLAM_API_STATIC_ASSERT(sizeof(short int) == 2, "For ABI compatibility the sizeof(short int) must be 2 bytes.")
ZEROSLAM_API_STATIC_ASSERT(sizeof(int) == 4, "For ABI compatibility the sizeof(int) must be 4 bytes.")
ZEROSLAM_API_STATIC_ASSERT(sizeof(long long int) == 8, "For ABI compatibility the sizeof(long long int) must be 8 bytes.")
ZEROSLAM_API_STATIC_ASSERT(sizeof(float) == 4, "For ABI compatibility the sizeof(float) must be 4 bytes.")

#pragma pack(push, 1)

typedef struct zeroslam_system zeroslam_system;

typedef enum zeroslam_return_enum {
    zeroslam_return_success = 0x00000000,
    zeroslam_return_failure_insufficient_data_length = 0x00000001,
    zeroslam_return_failure_invalid_system = 0x00000010,
    zeroslam_return_failure_invalid_argument = 0x00000100,
    zeroslam_return_failure_invalid_configuration = 0x00001000,
    zeroslam_return_failure_invalid_rig_sensor = 0x00010000,
    zeroslam_return_failure_invalid_sensor_data = 0x00100000,
    zeroslam_return_invalid = -1
} zeroslam_return_enum;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_return_enum) == 4, "For ABI compatibility the sizeof(zeroslam_return_enum) must be 4 bytes.")

static inline const char* zeroslam_return_enum_to_string(zeroslam_return_enum return_enum) {
    switch (return_enum) {
        case zeroslam_return_success:
            return "success";
        case zeroslam_return_failure_insufficient_data_length:
            return "failure: insufficient data length";
        case zeroslam_return_failure_invalid_system:
            return "failure: invalid system";
        case zeroslam_return_failure_invalid_argument:
            return "failure: invalid argument";
        case zeroslam_return_failure_invalid_configuration:
            return "failure: invalid configuration";
        case zeroslam_return_failure_invalid_rig_sensor:
            return "failure: invalid rig sensor";
        case zeroslam_return_failure_invalid_sensor_data:
            return "failure: invalid sensor data";
        case zeroslam_return_invalid:
            return "invalid";
    }
    return "unknown";
}

typedef enum zeroslam_sensor_enum {
    zeroslam_sensor_empty = 0x00000000,
    zeroslam_sensor_map_chunk = 0x00000001,
    zeroslam_sensor_local_scale = 0x00000010,
    zeroslam_sensor_local_linear = 0x00000020,
    zeroslam_sensor_local_angular = 0x00000040,
    zeroslam_sensor_remote_range = 0x00000100,
    zeroslam_sensor_remote_bearing = 0x00000200,
    zeroslam_sensor_remote_description = 0x00000400,
    zeroslam_sensor_camera = 0x00000800,
    zeroslam_sensor_invalid = -1
} zeroslam_sensor_enum;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_sensor_enum) == 4, "For ABI compatibility the sizeof(zeroslam_sensor_enum) must be 4 bytes.")

static inline const char* zeroslam_sensor_enum_to_string(zeroslam_sensor_enum sensor_enum) {
    switch (sensor_enum) {
        case zeroslam_sensor_empty:
            return "empty";
        case zeroslam_sensor_map_chunk:
            return "map chunk";
        case zeroslam_sensor_local_scale:
            return "local scale";
        case zeroslam_sensor_local_linear:
            return "local linear";
        case zeroslam_sensor_local_angular:
            return "local angular";
        case zeroslam_sensor_remote_range:
            return "remote range";
        case zeroslam_sensor_remote_bearing:
            return "remote bearing";
        case zeroslam_sensor_remote_description:
            return "remote description";
        case zeroslam_sensor_camera:
            return "camera";
        case zeroslam_sensor_invalid:
            return "invalid";
    }
    return "unknown";
}

// Intrinsics in pixels with the centre of pixel i at i + 0.5, distortion is [k1 k2 p1 p2 k3 k4 k5 k6].
typedef struct zeroslam_sensor_parameters_camera_struct {
    int width;
    int height;
    double focal_x;
    double focal_y;
    double centre_x;
    double centre_y;
    double distortion[8];
} zeroslam_sensor_parameters_camera_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_sensor_parameters_camera_struct) == 104, "For ABI compatibility the sizeof(zeroslam_sensor_parameters_camera_struct) must be 104 bytes.")

typedef struct zeroslam_sensor_rig_struct {
    zeroslam_sensor_enum type;
    int sensor_id;
    int parameters_length;
    void* parameters_data;
} zeroslam_sensor_rig_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_sensor_rig_struct) == 12 + sizeof(void*), "For ABI compatibility the sizeof(zeroslam_sensor_rig_struct) must be 12+sizeof(void*) bytes.")

typedef struct zeroslam_sensor_data_struct {
    long long int timestamp;
    int sensor_id;
    int measurement_length;
    void* measurement_data;
} zeroslam_sensor_data_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_sensor_data_struct) == 16 + sizeof(void*), "For ABI compatibility the sizeof(zeroslam_sensor_data_struct) must be 16+sizeof(void*) bytes.")

// The camera-to-world pose { x, y, z, qx, qy, qz, qw }, see README.md "Coordinate conventions".
typedef struct zeroslam_pose_struct {
    long long int timestamp;
    double pose[7];
    double covariance[7 * 7];
} zeroslam_pose_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_pose_struct) == 456, "For ABI compatibility the sizeof(zeroslam_pose_struct) must be 456 bytes.")

typedef struct zeroslam_point_struct {
    float x, y, z;
    float confidence;
    float r, g, b;
    float a;
} zeroslam_point_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_point_struct) == 32, "For ABI compatibility the sizeof(zeroslam_point_struct) must be 32 bytes.")

typedef struct zeroslam_map_chunk_struct {
    long long int timestamp;
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    int points_length;
    zeroslam_point_struct* points;
} zeroslam_map_chunk_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_map_chunk_struct) == 36 + sizeof(void*), "For ABI compatibility the sizeof(zeroslam_map_chunk_struct) must be 36+sizeof(void*) bytes.")

typedef struct zeroslam_line_struct {
    float x1, y1, z1;
    float x2, y2, z2;
    float confidence;
    float r, g, b;
    float a;
} zeroslam_line_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_line_struct) == 44, "For ABI compatibility the sizeof(zeroslam_line_struct) must be 44 bytes.")

typedef struct zeroslam_map_lines_struct {
    long long int timestamp;
    int lines_length;
    int reserved;
    zeroslam_line_struct* lines;
} zeroslam_map_lines_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_map_lines_struct) == 16 + sizeof(void*), "For ABI compatibility the sizeof(zeroslam_map_lines_struct) must be 16+sizeof(void*) bytes.")

typedef enum zeroslam_edge_enum {
    zeroslam_edge_covisibility = 0,
    zeroslam_edge_loop = 1
} zeroslam_edge_enum;

typedef struct zeroslam_edge_struct {
    long long int timestamp_a;
    long long int timestamp_b;
    int type;
    int weight;
} zeroslam_edge_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_edge_struct) == 24, "For ABI compatibility the sizeof(zeroslam_edge_struct) must be 24 bytes.")

typedef struct zeroslam_map_keyframes_struct {
    long long int timestamp;
    int keyframes_length;
    int reserved;
    long long int* keyframes;
} zeroslam_map_keyframes_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_map_keyframes_struct) == 16 + sizeof(void*), "For ABI compatibility the sizeof(zeroslam_map_keyframes_struct) must be 16+sizeof(void*) bytes.")

typedef struct zeroslam_map_edges_struct {
    long long int timestamp;
    int edges_length;
    int reserved;
    zeroslam_edge_struct* edges;
} zeroslam_map_edges_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_map_edges_struct) == 16 + sizeof(void*), "For ABI compatibility the sizeof(zeroslam_map_edges_struct) must be 16+sizeof(void*) bytes.")

#pragma pack(pop)

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_create(zeroslam_system** system);
ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_destroy(zeroslam_system** system);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_timestamp(zeroslam_system* system, long long int* timestamp);

// Buffers: a null or short buffer fails with insufficient_data_length and reports the required length.
ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_configuration(zeroslam_system* system, char* configuration, int* length);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_set_configuration(zeroslam_system* system, const char* configuration, int length);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_set_sensor_rig(zeroslam_system* system, const zeroslam_sensor_rig_struct* rig, int length);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_set_sensor_data(zeroslam_system* system, const zeroslam_sensor_data_struct* data, int length);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_finalise(zeroslam_system* system);
ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_pose(zeroslam_system* system, zeroslam_pose_struct* pose);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_pose_at_timestamp(zeroslam_system* system, zeroslam_pose_struct* pose, long long int timestamp);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_map_chunk(zeroslam_system* system, float x, float y, float z, zeroslam_map_chunk_struct* chunk);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_map_lines(zeroslam_system* system, zeroslam_map_lines_struct* lines);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_map_edges(zeroslam_system* system, zeroslam_map_edges_struct* edges);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_map_keyframes(zeroslam_system* system, zeroslam_map_keyframes_struct* keyframes);

#undef ZEROSLAM_API_STATIC_ASSERT

#if defined(__cplusplus)
}
#endif

#endif // ZEROSLAM_ZEROSLAM_H
