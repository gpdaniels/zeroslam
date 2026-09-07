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

#include "feature/point.hpp"

namespace feature::detector {
    size_t detect_cpu(const unsigned char* __restrict const data, const int width, const int height, const int stride, const int threshold, const size_t feature_point_buffer_size, point* __restrict feature_point_buffer);

    size_t detect_cpu(
        const unsigned char* __restrict const data,
        const int width,
        const int height,
        const int stride,
        const int threshold,
        const size_t feature_point_buffer_size,
        point* __restrict feature_point_buffer
    ) {
        constexpr static const int patch_size = 7;
        constexpr static const int patch_radius = patch_size / 2;
        constexpr static const int pattern_size = 16;
        constexpr static const int pattern_size_half = pattern_size / 2;
        const int pattern_offset[24] = {
            +patch_radius * stride + 0,
            +patch_radius * stride + 1,
            +2 * stride + 2,
            +patch_radius + stride + 0,
            +patch_radius,
            +patch_radius - stride + 0,
            -2 * stride + 2,
            -patch_radius * stride + 1,
            -patch_radius * stride + 0,
            -patch_radius * stride - 1,
            -2 * stride - 2,
            -patch_radius - stride + 0,
            -patch_radius,
            -patch_radius + stride + 0,
            +2 * stride - 2,
            +patch_radius * stride - 1,
            +patch_radius * stride + 0,
            +patch_radius * stride + 1,
            +2 * stride + 2,
            +patch_radius + stride + 0,
            +patch_radius,
            +patch_radius - stride + 0,
            -2 * stride + 2,
            -patch_radius * stride + 1
        };
        unsigned char threshold_map[255 + 1 + 255];
        for (int value = -255; value <= 255; ++value) {
            threshold_map[value + 255] = (value < -threshold) + 2 * (value > threshold);
        }
        size_t feature_count = 0;
        for (int y = patch_radius; y < height - patch_radius; ++y) {
            const unsigned char* data_pointer = data + y * stride + patch_radius;
            for (int x = patch_radius; x < width - patch_radius; ++x, ++data_pointer) {
                const unsigned char* threshold_pointer = &threshold_map[255 - data_pointer[0]];
                unsigned char threshold_accumulator = 1 | 2;
                threshold_accumulator &= threshold_pointer[data_pointer[pattern_offset[0]]] | threshold_pointer[data_pointer[pattern_offset[8]]];
                threshold_accumulator &= threshold_pointer[data_pointer[pattern_offset[1]]] | threshold_pointer[data_pointer[pattern_offset[9]]];
                threshold_accumulator &= threshold_pointer[data_pointer[pattern_offset[2]]] | threshold_pointer[data_pointer[pattern_offset[10]]];
                threshold_accumulator &= threshold_pointer[data_pointer[pattern_offset[3]]] | threshold_pointer[data_pointer[pattern_offset[11]]];
                threshold_accumulator &= threshold_pointer[data_pointer[pattern_offset[4]]] | threshold_pointer[data_pointer[pattern_offset[12]]];
                threshold_accumulator &= threshold_pointer[data_pointer[pattern_offset[5]]] | threshold_pointer[data_pointer[pattern_offset[13]]];
                threshold_accumulator &= threshold_pointer[data_pointer[pattern_offset[6]]] | threshold_pointer[data_pointer[pattern_offset[14]]];
                threshold_accumulator &= threshold_pointer[data_pointer[pattern_offset[7]]] | threshold_pointer[data_pointer[pattern_offset[15]]];
                if (threshold_accumulator & 1) {
                    unsigned int valid_pixels = 0;
                    const int pixel_threshold = static_cast<int>(data_pointer[0]) - threshold;
                    for (unsigned int index = 0; index < pattern_size + pattern_size_half; ++index) {
                        const int pixel = data_pointer[pattern_offset[index]];
                        if (pixel < pixel_threshold) {
                            ++valid_pixels;
                            if (valid_pixels > pattern_size_half) {
                                feature_point_buffer[feature_count++] = { static_cast<float>(x), static_cast<float>(y), 0, 0, 0 };
                                if (feature_count == feature_point_buffer_size) {
                                    return feature_count;
                                }
                                break;
                            }
                        }
                        else {
                            if (index > pattern_size) {
                                break;
                            }
                            valid_pixels = 0;
                        }
                    }
                }
                if (threshold_accumulator & 2) {
                    unsigned int valid_pixels = 0;
                    const int pixel_threshold = static_cast<int>(data_pointer[0]) + threshold;
                    for (unsigned int index = 0; index < pattern_size + pattern_size_half; ++index) {
                        const int pixel = data_pointer[pattern_offset[index]];
                        if (pixel > pixel_threshold) {
                            ++valid_pixels;
                            if (valid_pixels > pattern_size_half) {
                                feature_point_buffer[feature_count++] = { static_cast<float>(x), static_cast<float>(y), 0, 0, 0 };
                                if (feature_count == feature_point_buffer_size) {
                                    return feature_count;
                                }
                                break;
                            }
                        }
                        else {
                            if (index > pattern_size) {
                                break;
                            }
                            valid_pixels = 0;
                        }
                    }
                }
            }
        }
        return feature_count;
    }
}
