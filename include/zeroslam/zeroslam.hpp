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
#ifndef ZEROSLAM_ZEROSLAM_HPP
#define ZEROSLAM_ZEROSLAM_HPP

#include "zeroslam/zeroslam.h"

namespace zeroslam {
    // The parameters of a depth camera registered to the rig's camera camera_id (on its pixel grid, as aligned depth comes),
    // measuring the depth along the optical axis between 0.1 and 10 m with the Kinect's noise model, 1.425e-3 d^2 metres,
    // its frames paired with the camera's within half a 30 Hz frame. Any field can be changed after.
    inline zeroslam_sensor_parameters_depth_struct registered_depth(const int camera_id, const zeroslam_depth_encoding_enum encoding, const double scale) {
        zeroslam_sensor_parameters_depth_struct parameters{};
        parameters.camera_id = camera_id;
        parameters.encoding = encoding;
        parameters.measure = zeroslam_depth_measure_axial;
        parameters.registration = zeroslam_depth_registration_camera;
        parameters.scale = scale;
        parameters.minimum_depth = 0.1;
        parameters.maximum_depth = 10.0;
        parameters.noise_constant = 0.0;
        parameters.noise_linear = 0.0;
        parameters.noise_quadratic = 1.425e-3;
        parameters.pairing_tolerance = 16000000;
        return parameters;
    }

    class system final {
    private:
        zeroslam_system* handle = nullptr;

    public:
        system() {
            if (zeroslam_create(&this->handle) != zeroslam_return_success) {
                this->handle = nullptr;
            }
        }

        ~system() {
            if (this->handle != nullptr) {
                zeroslam_destroy(&this->handle);
            }
        }

        system(const system& other) = delete;

        system(system&& other)
            : handle(other.handle) {
            other.handle = nullptr;
        }

        system& operator=(const system& other) = delete;

        system& operator=(system&& other) {
            if (this != &other) {
                if (this->handle != nullptr) {
                    zeroslam_destroy(&this->handle);
                }
                this->handle = other.handle;
                other.handle = nullptr;
            }
            return *this;
        }

    public:
        bool is_valid() const {
            return (this->handle != nullptr);
        }

    public:
        zeroslam_return_enum get_timestamp(long long int* timestamp) const {
            return zeroslam_get_timestamp(this->handle, timestamp);
        }

        zeroslam_return_enum get_configuration(char* configuration, int* length) const {
            return zeroslam_get_configuration(this->handle, configuration, length);
        }

        zeroslam_return_enum set_configuration(const char* configuration, int length) const {
            return zeroslam_set_configuration(this->handle, configuration, length);
        }

        zeroslam_return_enum set_sensor_rig(const zeroslam_sensor_rig_struct* rig, int length) const {
            return zeroslam_set_sensor_rig(this->handle, rig, length);
        }

        zeroslam_return_enum set_sensor_extrinsics(const zeroslam_sensor_extrinsics_struct* extrinsics, int length) const {
            return zeroslam_set_sensor_extrinsics(this->handle, extrinsics, length);
        }

        zeroslam_return_enum get_sensor_extrinsics(zeroslam_sensor_extrinsics_struct* extrinsics, int* length) const {
            return zeroslam_get_sensor_extrinsics(this->handle, extrinsics, length);
        }

        zeroslam_return_enum get_odometry_calibration(zeroslam_odometry_calibration_struct* calibration) const {
            return zeroslam_get_odometry_calibration(this->handle, calibration);
        }

        zeroslam_return_enum get_imu_calibration(zeroslam_imu_calibration_struct* calibration) const {
            return zeroslam_get_imu_calibration(this->handle, calibration);
        }

        zeroslam_return_enum set_sensor_data(const zeroslam_sensor_data_struct* data, int length) const {
            return zeroslam_set_sensor_data(this->handle, data, length);
        }

        zeroslam_return_enum finalise() const {
            return zeroslam_finalise(this->handle);
        }

        zeroslam_return_enum get_camera_calibration(int sensor_id, zeroslam_sensor_parameters_camera_model_struct* calibration) const {
            return zeroslam_get_camera_calibration(this->handle, sensor_id, calibration);
        }

        zeroslam_return_enum get_pose(zeroslam_pose_struct* pose) const {
            return zeroslam_get_pose(this->handle, pose);
        }

        zeroslam_return_enum get_pose_at_timestamp(zeroslam_pose_struct* pose, long long int timestamp) const {
            return zeroslam_get_pose_at_timestamp(this->handle, pose, timestamp);
        }

        zeroslam_return_enum get_map_chunk(float x, float y, float z, zeroslam_map_chunk_struct* chunk) const {
            return zeroslam_get_map_chunk(this->handle, x, y, z, chunk);
        }

        zeroslam_return_enum get_map_lines(zeroslam_map_lines_struct* lines) const {
            return zeroslam_get_map_lines(this->handle, lines);
        }

        zeroslam_return_enum get_map_edges(zeroslam_map_edges_struct* edges) const {
            return zeroslam_get_map_edges(this->handle, edges);
        }

        zeroslam_return_enum get_map_keyframes(zeroslam_map_keyframes_struct* keyframes) const {
            return zeroslam_get_map_keyframes(this->handle, keyframes);
        }

        zeroslam_return_enum get_map_voxels(zeroslam_map_voxels_struct* voxels) const {
            return zeroslam_get_map_voxels(this->handle, voxels);
        }
    };
}

#endif // ZEROSLAM_ZEROSLAM_HPP
