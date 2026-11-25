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
    zeroslam_sensor_depth = 0x00002000,
    zeroslam_sensor_imu = 0x00004000,
    zeroslam_sensor_wheel_odometry = 0x00008000,
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
        case zeroslam_sensor_depth:
            return "depth";
        case zeroslam_sensor_imu:
            return "imu";
        case zeroslam_sensor_wheel_odometry:
            return "wheel odometry";
        case zeroslam_sensor_invalid:
            return "invalid";
    }
    return "unknown";
}

// Intrinsics in pixels with the centre of pixel i at i + 0.5, distortion is [k1 k2 p1 p2 k3 k4 k5 k6] (the radial
// tangential camera model; zeroslam_sensor_parameters_camera_model_struct takes any model).
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

// The camera models, and the parameters each takes. Every model starts with the focal lengths and the principal point
// [fx fy cx cy], in pixels with the centre of pixel i at i + 0.5, see README.md "Camera models".
typedef enum zeroslam_camera_model_enum {
    zeroslam_camera_model_pinhole = 0,           // [fx fy cx cy]
    zeroslam_camera_model_radial_tangential = 1, // [fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6], OpenCV's rational model.
    zeroslam_camera_model_opencv = 2,            // [fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4 tau_x tau_y], with thin prism and tilt.
    zeroslam_camera_model_kannala_brandt = 3,    // [fx fy cx cy k1 k2 k3 k4], OpenCV's fisheye.
    zeroslam_camera_model_field_of_view = 4,     // [fx fy cx cy w], PTAM's arctangent model.
    zeroslam_camera_model_unified = 5,           // [fx fy cx cy xi]
    zeroslam_camera_model_extended_unified = 6,  // [fx fy cx cy alpha beta]
    zeroslam_camera_model_double_sphere = 7,     // [fx fy cx cy xi alpha]
    zeroslam_camera_model_fisheye_624 = 8,       // [fx fy cx cy k0 k1 k2 k3 k4 k5 p0 p1 s0 s1 s2 s3], Project Aria's.
    zeroslam_camera_model_invalid = -1
} zeroslam_camera_model_enum;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_camera_model_enum) == 4, "For ABI compatibility the sizeof(zeroslam_camera_model_enum) must be 4 bytes.")

static inline const char* zeroslam_camera_model_enum_to_string(zeroslam_camera_model_enum camera_model_enum) {
    switch (camera_model_enum) {
        case zeroslam_camera_model_pinhole:
            return "pinhole";
        case zeroslam_camera_model_radial_tangential:
            return "radial tangential";
        case zeroslam_camera_model_opencv:
            return "opencv";
        case zeroslam_camera_model_kannala_brandt:
            return "kannala brandt";
        case zeroslam_camera_model_field_of_view:
            return "field of view";
        case zeroslam_camera_model_unified:
            return "unified";
        case zeroslam_camera_model_extended_unified:
            return "extended unified";
        case zeroslam_camera_model_double_sphere:
            return "double sphere";
        case zeroslam_camera_model_fisheye_624:
            return "fisheye 624";
        case zeroslam_camera_model_invalid:
            return "invalid";
    }
    return "unknown";
}

// How many parameters a camera model takes, zero for one that is not a model.
static inline int zeroslam_camera_model_parameter_count(zeroslam_camera_model_enum camera_model_enum) {
    switch (camera_model_enum) {
        case zeroslam_camera_model_pinhole:
            return 4;
        case zeroslam_camera_model_radial_tangential:
            return 12;
        case zeroslam_camera_model_opencv:
            return 18;
        case zeroslam_camera_model_kannala_brandt:
            return 8;
        case zeroslam_camera_model_field_of_view:
            return 5;
        case zeroslam_camera_model_unified:
            return 5;
        case zeroslam_camera_model_extended_unified:
            return 6;
        case zeroslam_camera_model_double_sphere:
            return 6;
        case zeroslam_camera_model_fisheye_624:
            return 16;
        case zeroslam_camera_model_invalid:
            return 0;
    }
    return 0;
}

// A camera of any model, taken in place of zeroslam_sensor_parameters_camera_struct when a rig entry's parameters_length
// is this struct's size: the image size, the model, and parameters_length of its parameters (the rest unused).
typedef struct zeroslam_sensor_parameters_camera_model_struct {
    int width;
    int height;
    zeroslam_camera_model_enum model;
    int parameters_length;
    double parameters[32];
} zeroslam_sensor_parameters_camera_model_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_sensor_parameters_camera_model_struct) == 272, "For ABI compatibility the sizeof(zeroslam_sensor_parameters_camera_model_struct) must be 272 bytes.")

// The calibration of one camera of the rig over one chunk of time, where the configuration's calibration_drift key has the
// calibration followed over time (see README.md "Calibration drift"): when the chunk starts and how long it is, in
// nanoseconds, the camera, its focal lengths in the chunk (the rest of its parameters are the rig's), its extrinsic there as
// zeroslam_sensor_extrinsics_struct gives one, and how many observations held the chunk in the last adjustment that followed
// the calibration: with none it is where the drift between the chunks around it leaves it, or still as it started.
typedef struct zeroslam_calibration_chunk_struct {
    long long int timestamp;
    long long int duration;
    int sensor_id;
    int observations;
    double focal_x;
    double focal_y;
    double pose[7];
} zeroslam_calibration_chunk_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_calibration_chunk_struct) == 96, "For ABI compatibility the sizeof(zeroslam_calibration_chunk_struct) must be 96 bytes.")

// How a depth image's pixels are stored: 16 bit unsigned values of scale metres each (zero is no measurement), or 32 bit floats
// of metres (zero or not finite is no measurement).
typedef enum zeroslam_depth_encoding_enum {
    zeroslam_depth_encoding_unsigned_16 = 0,
    zeroslam_depth_encoding_float_32 = 1
} zeroslam_depth_encoding_enum;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_depth_encoding_enum) == 4, "For ABI compatibility the sizeof(zeroslam_depth_encoding_enum) must be 4 bytes.")

// What a depth is: the distance along the optical axis, the z of the point in the camera frame (structured light and stereo
// cameras, TUM RGB-D, the OAK-D), or the distance along the pixel's ray from the camera centre (time of flight cameras).
typedef enum zeroslam_depth_measure_enum {
    zeroslam_depth_measure_axial = 0,
    zeroslam_depth_measure_radial = 1
} zeroslam_depth_measure_enum;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_depth_measure_enum) == 4, "For ABI compatibility the sizeof(zeroslam_depth_measure_enum) must be 4 bytes.")

// How a depth image is related to its camera: registered to it, on the camera's own pixel grid as aligned depth comes, so
// pixel (u, v) of the depth image is the depth of what pixel (u, v) of the camera sees. The only relation taken so far.
typedef enum zeroslam_depth_registration_enum {
    zeroslam_depth_registration_camera = 0
} zeroslam_depth_registration_enum;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_depth_registration_enum) == 4, "For ABI compatibility the sizeof(zeroslam_depth_registration_enum) must be 4 bytes.")

// A depth camera paired with a camera of the rig, whose images are the camera's size. Depths outside [minimum_depth,
// maximum_depth] metres are taken as missing. The standard deviation of a depth d is noise_constant + noise_linear d +
// noise_quadratic d^2 metres, a structured light camera's error growing with the square of the depth (the Kinect's is about
// 1.425e-3 d^2). A camera frame takes the depth frame nearest it in time within pairing_tolerance nanoseconds, from those
// given before it or with it.
typedef struct zeroslam_sensor_parameters_depth_struct {
    int camera_id;
    int encoding;
    int measure;
    int registration;
    double scale;
    double minimum_depth;
    double maximum_depth;
    double noise_constant;
    double noise_linear;
    double noise_quadratic;
    long long int pairing_tolerance;
} zeroslam_sensor_parameters_depth_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_sensor_parameters_depth_struct) == 72, "For ABI compatibility the sizeof(zeroslam_sensor_parameters_depth_struct) must be 72 bytes.")

// An imu's noise model, the continuous time densities its calibration publishes (rad/s/sqrt(Hz), m/s^2/sqrt(Hz), rad/s^2/sqrt(Hz)
// and m/s^3/sqrt(Hz)), its sample rate (Hz), the magnitude of gravity (m/s^2, 0 for 9.81) and the offset of its clock from the
// cameras' (seconds, the imu's time of a moment less the cameras'). See README.md "Inertial sensors".
typedef struct zeroslam_sensor_parameters_imu_struct {
    double gyroscope_noise_density;
    double accelerometer_noise_density;
    double gyroscope_random_walk;
    double accelerometer_random_walk;
    double rate;
    double gravity;
    double time_offset;
} zeroslam_sensor_parameters_imu_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_sensor_parameters_imu_struct) == 56, "For ABI compatibility the sizeof(zeroslam_sensor_parameters_imu_struct) must be 56 bytes.")

// One imu sample in the imu's own frame, as measured: the angular velocity (rad/s) and the specific force (m/s^2).
typedef struct zeroslam_sensor_measurement_imu_struct {
    double angular_velocity[3];
    double linear_acceleration[3];
} zeroslam_sensor_measurement_imu_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_sensor_measurement_imu_struct) == 48, "For ABI compatibility the sizeof(zeroslam_sensor_measurement_imu_struct) must be 48 bytes.")

typedef struct zeroslam_sensor_rig_struct {
    zeroslam_sensor_enum type;
    int sensor_id;
    int parameters_length;
    void* parameters_data;
} zeroslam_sensor_rig_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_sensor_rig_struct) == 12 + sizeof(void*), "For ABI compatibility the sizeof(zeroslam_sensor_rig_struct) must be 12+sizeof(void*) bytes.")

typedef enum zeroslam_extrinsic_enum {
    zeroslam_extrinsic_fixed = 0,
    zeroslam_extrinsic_refine = 1
} zeroslam_extrinsic_enum;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_extrinsic_enum) == 4, "For ABI compatibility the sizeof(zeroslam_extrinsic_enum) must be 4 bytes.")

// Where a sensor of the rig sits on the rig's body: the sensor-to-body pose { x, y, z, qx, qy, qz, qw }, p_body = R * p_sensor + t
// (the scene's ego -> sensor/[name] on /tf), and whether the system refines it (a zeroslam_extrinsic_enum in refinement).
typedef struct zeroslam_sensor_extrinsics_struct {
    int sensor_id;
    int refinement;
    double pose[7];
} zeroslam_sensor_extrinsics_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_sensor_extrinsics_struct) == 64, "For ABI compatibility the sizeof(zeroslam_sensor_extrinsics_struct) must be 64 bytes.")

// Wheel odometry, a rig entry of type zeroslam_sensor_wheel_odometry beside the cameras with these parameters, its extrinsic the
// odometry frame's pose on the body (x forward, z up, the origin the rear axle's centre or the drive axle's middle) given with
// zeroslam_set_sensor_extrinsics. Its samples are zeroslam_set_sensor_data entries of three doubles: the cumulative rotation in
// radians of the measured wheel (Ackermann) or the left wheel (differential drive), of the right wheel, and the steering reading.
typedef enum zeroslam_odometry_model_enum {
    zeroslam_odometry_ackermann_rear_axle = 0,
    zeroslam_odometry_ackermann_front_wheel = 1,
    zeroslam_odometry_differential_drive = 2
} zeroslam_odometry_model_enum;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_odometry_model_enum) == 4, "For ABI compatibility the sizeof(zeroslam_odometry_model_enum) must be 4 bytes.")

// The model (a zeroslam_odometry_model_enum) and its calibration: an Ackermann vehicle's wheel radius, wheelbase, steering gain
// and steering offset (steering angle = gain * reading + offset), or a differential drive's left and right wheel radii and track
// width, in metres and radians; bit i of free_mask refines parameter i, with a prior of parameter_deviations[i] (zero for
// none). The noise is the wheel slip, the steering reading, the lateral slip, the heading, the height and the tilt, as standard
// deviations per square root metre but the steering's per reading, then the least translation and rotation of any interval.
// The extrinsic's prior deviations are its rotation then its translation about the odometry frame's axes, zero for none: planar
// motion leaves the height above the plane, the last, unobservable, so refining the extrinsic wants a tight one there.
typedef struct zeroslam_sensor_parameters_wheel_odometry_struct {
    int model;
    int free_mask;
    double parameters[8];
    double parameter_deviations[8];
    double noise[8];
    double extrinsic_deviations[6];
} zeroslam_sensor_parameters_wheel_odometry_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_sensor_parameters_wheel_odometry_struct) == 248, "For ABI compatibility the sizeof(zeroslam_sensor_parameters_wheel_odometry_struct) must be 248 bytes.")

// The odometer's calibration as the system holds it, refined where asked for, and its extrinsic as zeroslam_sensor_extrinsics_struct.
typedef struct zeroslam_odometry_calibration_struct {
    int sensor_id;
    int parameters_length;
    double parameters[8];
    double pose[7];
} zeroslam_odometry_calibration_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_odometry_calibration_struct) == 128, "For ABI compatibility the sizeof(zeroslam_odometry_calibration_struct) must be 128 bytes.")

// The imu's own calibration as the system holds it: the matrix M that corrects its gyroscope's scale and the alignment of its axes,
// row by row, the angular velocity being (I + M) * measured - bias. It is zero until an adjustment has refined it, which the
// configuration's imu_calibration key asks for, and refinements counts the adjustments that have. See README.md "Inertial sensors".
typedef struct zeroslam_imu_calibration_struct {
    int sensor_id;
    int refinements;
    double gyroscope[9];
} zeroslam_imu_calibration_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_imu_calibration_struct) == 80, "For ABI compatibility the sizeof(zeroslam_imu_calibration_struct) must be 80 bytes.")

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

typedef struct zeroslam_voxel_struct {
    float x, y, z;
    int points;
} zeroslam_voxel_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_voxel_struct) == 16, "For ABI compatibility the sizeof(zeroslam_voxel_struct) must be 16 bytes.")

typedef struct zeroslam_map_voxels_struct {
    long long int timestamp;
    float voxel_size;
    int voxels_length;
    zeroslam_voxel_struct* voxels;
} zeroslam_map_voxels_struct;
ZEROSLAM_API_STATIC_ASSERT(sizeof(zeroslam_map_voxels_struct) == 16 + sizeof(void*), "For ABI compatibility the sizeof(zeroslam_map_voxels_struct) must be 16+sizeof(void*) bytes.")

#pragma pack(pop)

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_create(zeroslam_system** system);
ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_destroy(zeroslam_system** system);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_timestamp(zeroslam_system* system, long long int* timestamp);

// Buffers: a null or short buffer fails with insufficient_data_length and reports the required length.
ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_configuration(zeroslam_system* system, char* configuration, int* length);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_set_configuration(zeroslam_system* system, const char* configuration, int length);

// The rig's cameras, the first of them the primary camera whose poses the system reports. Several cameras are synchronised:
// each frame is one image of every camera with the same timestamp, in one batch or across calls.
ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_set_sensor_rig(zeroslam_system* system, const zeroslam_sensor_rig_struct* rig, int length);

// The extrinsics of sensors of the rig, after the rig and before any data; a sensor given none sits at the body's origin, fixed.
ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_set_sensor_extrinsics(zeroslam_system* system, const zeroslam_sensor_extrinsics_struct* extrinsics, int length);

// The extrinsics of the rig's cameras as the system holds them, one per camera in the rig's order, refined where asked for.
ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_sensor_extrinsics(zeroslam_system* system, zeroslam_sensor_extrinsics_struct* extrinsics, int* length);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_set_sensor_data(zeroslam_system* system, const zeroslam_sensor_data_struct* data, int length);

// The wheel odometer's calibration and extrinsic as the system holds them; fails with invalid_rig_sensor without an odometer.
ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_odometry_calibration(zeroslam_system* system, zeroslam_odometry_calibration_struct* calibration);

// The imu's calibration as the system holds it; fails with invalid_rig_sensor without an imu.
ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_imu_calibration(zeroslam_system* system, zeroslam_imu_calibration_struct* calibration);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_finalise(zeroslam_system* system);

// The calibration of a camera of the rig as the system holds it now, of whichever model the rig gave: the rig's, refined by
// the adjustments when the configuration's calibration key asks for it (see README.md "Calibrating the camera").
ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_camera_calibration(zeroslam_system* system, int sensor_id, zeroslam_sensor_parameters_camera_model_struct* calibration);

// The calibration over time as the system holds it, one entry per chunk of time and camera, by time and then in the rig's
// order of cameras; none while the calibration is not followed (the calibration_drift key, and frames with timestamps).
ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_calibration_drift(zeroslam_system* system, zeroslam_calibration_chunk_struct* chunks, int* length);
ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_pose(zeroslam_system* system, zeroslam_pose_struct* pose);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_pose_at_timestamp(zeroslam_system* system, zeroslam_pose_struct* pose, long long int timestamp);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_map_chunk(zeroslam_system* system, float x, float y, float z, zeroslam_map_chunk_struct* chunk);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_map_lines(zeroslam_system* system, zeroslam_map_lines_struct* lines);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_map_edges(zeroslam_system* system, zeroslam_map_edges_struct* edges);

ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_map_keyframes(zeroslam_system* system, zeroslam_map_keyframes_struct* keyframes);

// The occupied voxels of the voxel map the system keeps of its landmarks, each by its minimum corner in map coordinates (a
// cube of voxel_size) and the landmarks in it, sorted by corner.
ZEROSLAM_API_VISIBILITY zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_map_voxels(zeroslam_system* system, zeroslam_map_voxels_struct* voxels);

#undef ZEROSLAM_API_STATIC_ASSERT

#if defined(__cplusplus)
}
#endif

#endif // ZEROSLAM_ZEROSLAM_H
