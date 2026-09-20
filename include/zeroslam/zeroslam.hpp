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

        zeroslam_return_enum set_sensor_data(const zeroslam_sensor_data_struct* data, int length) const {
            return zeroslam_set_sensor_data(this->handle, data, length);
        }

        zeroslam_return_enum finalise() const {
            return zeroslam_finalise(this->handle);
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
    };
}

#endif // ZEROSLAM_ZEROSLAM_HPP
