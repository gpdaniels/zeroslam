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
#ifndef ZEROSLAM_FEATURE_FEATURE_HPP
#define ZEROSLAM_FEATURE_FEATURE_HPP

#include "core/assert.hpp"
#include "feature/descriptor/binary.hpp"
#include "feature/point.hpp"
#include "match/pair.hpp"
#include "math/math.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace feature {
    static inline size_t detect(
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

    static inline float score(
        const unsigned char* __restrict const data,
        const int stride
    ) {
        constexpr static const int patch_size = 7;
        constexpr static const int patch_radius = patch_size / 2;
        constexpr static const int pattern_size = 16;
        constexpr static const int pattern_size_half = pattern_size / 2;
        const int pattern_offset[24] = {
            patch_radius * stride + 0,
            patch_radius * stride + 1,
            +2 * stride + 2,
            patch_radius + stride + 0,
            patch_radius,
            patch_radius - stride + 0,
            -2 * stride + 2,
            -patch_radius * stride + 1,
            -patch_radius * stride + 0,
            -patch_radius * stride - 1,
            -2 * stride - 2,
            -patch_radius - stride + 0,
            -patch_radius,
            -patch_radius + stride + 0,
            +2 * stride - 2,
            patch_radius * stride - 1,
            patch_radius * stride + 0,
            patch_radius * stride + 1,
            +2 * stride + 2,
            patch_radius + stride + 0,
            patch_radius,
            patch_radius - stride + 0,
            -2 * stride + 2,
            -patch_radius * stride + 1
        };
        int pattern_differences[pattern_size + pattern_size_half];
        for (unsigned int i = 0; i < pattern_size + pattern_size_half; ++i) {
            pattern_differences[i] = static_cast<int>(data[0]) - static_cast<int>(data[pattern_offset[i]]);
        }
        int threshold_min = 0;
        for (int i = 0; i < pattern_size; i += 2) {
            int pixel_min = pattern_differences[i + 1];
            pixel_min = math::min(pixel_min, pattern_differences[i + 2]);
            pixel_min = math::min(pixel_min, pattern_differences[i + 3]);
            pixel_min = math::min(pixel_min, pattern_differences[i + 4]);
            pixel_min = math::min(pixel_min, pattern_differences[i + 5]);
            pixel_min = math::min(pixel_min, pattern_differences[i + 6]);
            pixel_min = math::min(pixel_min, pattern_differences[i + 7]);
            pixel_min = math::min(pixel_min, pattern_differences[i + 8]);
            threshold_min = math::max(threshold_min, math::min(pixel_min, pattern_differences[i + 0]));
            threshold_min = math::max(threshold_min, math::min(pixel_min, pattern_differences[i + 9]));
        }
        int threshold_max = -threshold_min;
        for (int i = 0; i < pattern_size; i += 2) {
            int pixel_max = pattern_differences[i + 1];
            pixel_max = math::max(pixel_max, pattern_differences[i + 2]);
            pixel_max = math::max(pixel_max, pattern_differences[i + 3]);
            pixel_max = math::max(pixel_max, pattern_differences[i + 4]);
            pixel_max = math::max(pixel_max, pattern_differences[i + 5]);
            pixel_max = math::max(pixel_max, pattern_differences[i + 6]);
            pixel_max = math::max(pixel_max, pattern_differences[i + 7]);
            pixel_max = math::max(pixel_max, pattern_differences[i + 8]);
            threshold_max = math::min(threshold_max, math::max(pixel_max, pattern_differences[i + 0]));
            threshold_max = math::min(threshold_max, math::max(pixel_max, pattern_differences[i + 9]));
        }
        return static_cast<float>(-threshold_max - 1);
    }

    static inline size_t suppress(
        const point* __restrict const features,
        const size_t features_count,
        const size_t max_row,
        point* __restrict const features_suppressed
    ) {
        if (features_count == 0) {
            return 0;
        }
        size_t* index_rows = new size_t[max_row + 1];
        for (size_t i = 0; i < max_row + 1; ++i) {
            index_rows[i] = static_cast<size_t>(-1);
        }
        size_t index_row = static_cast<size_t>(-1);
        for (size_t i = 0; i < features_count; ++i) {
            if (static_cast<size_t>(features[i].y) != index_row) {
                if (static_cast<size_t>(features[i].y) > max_row) {
                    delete[] index_rows;
                    return 0;
                }
                index_rows[static_cast<size_t>(features[i].y)] = i;
                index_row = static_cast<size_t>(features[i].y);
            }
        }
        size_t index_row_above = 0;
        size_t index_row_below = 0;
        size_t suppressed_count = 0;
        for (size_t i = 0; i < features_count; ++i) {
            if (i > 0) {
                if ((features[i - 1].x == features[i].x - 1) && (features[i - 1].y == features[i].y) && (features[i - 1].response > features[i].response)) {
                    continue;
                }
            }
            if (i < (features_count - 1)) {
                if ((features[i + 1].x == features[i].x + 1) && (features[i + 1].y == features[i].y) && (features[i + 1].response > features[i].response)) {
                    continue;
                }
            }
            if ((features[i].y != 0) && (index_rows[static_cast<size_t>(features[i].y - 1)] != static_cast<size_t>(-1))) {
                if (features[index_row_above].y < (features[i].y - 1)) {
                    index_row_above = index_rows[static_cast<size_t>(features[i].y - 1)];
                }
                while ((features[index_row_above].y < features[i].y) && (features[index_row_above].x < (features[i].x - 1))) {
                    ++index_row_above;
                }
                bool skip = false;
                for (size_t j = index_row_above; (features[j].y < features[i].y) && (features[j].x <= (features[i].x + 1)); ++j) {
                    if (((features[j].x == (features[i].x - 1)) || (features[j].x == (features[i].x + 0)) || (features[j].x == (features[i].x + 1))) && (features[j].response > features[i].response)) {
                        skip = true;
                        break;
                    }
                }
                if (skip) {
                    continue;
                }
            }
            if ((static_cast<size_t>(features[i].y) != max_row) && (index_rows[static_cast<size_t>(features[i].y + 1)] != static_cast<size_t>(-1)) && (index_row_below < features_count)) {
                if (features[index_row_below].y < (features[i].y + 1)) {
                    index_row_below = index_rows[static_cast<size_t>(features[i].y + 1)];
                }
                while ((index_row_below < features_count) && (features[index_row_below].y == (features[i].y + 1)) && (features[index_row_below].x < (features[i].x - 1))) {
                    ++index_row_below;
                }
                bool skip = false;
                for (size_t j = index_row_below; (j < features_count) && (features[j].y == (features[i].y + 1)) && (features[j].x <= (features[i].x + 1)); ++j) {
                    if (((features[j].x == (features[i].x - 1)) || (features[j].x == (features[i].x + 0)) || (features[j].x == (features[i].x + 1))) && (features[j].response > features[i].response)) {
                        skip = true;
                        break;
                    }
                }
                if (skip) {
                    continue;
                }
            }
            features_suppressed[suppressed_count++] = features[i];
        }
        delete[] index_rows;
        return suppressed_count;
    }

    static inline float dominant_angle(
        const unsigned char* __restrict const data,
        const int stride
    ) {
        constexpr static const int patch_size = 31;
        constexpr static const int patch_radius = patch_size / 2;
        constexpr static const int patch_width[patch_radius + 1] = { 15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3 };
        int sum_x = 0;
        for (int x = -patch_radius; x <= patch_radius; ++x) {
            sum_x += x * data[x];
        }
        int sum_y = 0;
        for (int y = 1; y <= patch_radius; ++y) {
            int sum = 0;
            for (int x = -patch_width[y]; x <= patch_width[y]; ++x) {
                const int above = data[x - y * stride];
                const int below = data[x + y * stride];
                sum += (below - above);
                sum_x += x * (below + above);
            }
            sum_y += y * sum;
        }
        return static_cast<float>(math::atan2(static_cast<double>(sum_y), static_cast<double>(sum_x)));
    }

    static inline void describe(
        const unsigned char* __restrict const data,
        const int stride,
        const float angle_radians,
        descriptor::binary<256>& descriptor
    ) {
        constexpr static const int pattern_size = 256;
        constexpr static const int pattern[pattern_size][2][2] = {
            { { 8, -3 }, { 9, 5 } },
            { { 4, 2 }, { 7, -12 } },
            { { -11, 9 }, { -8, 2 } },
            { { 7, -12 }, { 12, -13 } },
            { { 2, -13 }, { 2, 12 } },
            { { 1, -7 }, { 1, 6 } },
            { { -2, -10 }, { -2, -4 } },
            { { -13, -13 }, { -11, -8 } },
            { { -13, -3 }, { -12, -9 } },
            { { 10, 4 }, { 11, 9 } },
            { { -13, -8 }, { -8, -9 } },
            { { -11, 7 }, { -9, 12 } },
            { { 7, 7 }, { 12, 6 } },
            { { -4, -5 }, { -3, 0 } },
            { { -13, 2 }, { -12, -3 } },
            { { -9, 0 }, { -7, 5 } },
            { { 12, -6 }, { 12, -1 } },
            { { -3, 6 }, { -2, 12 } },
            { { -6, -13 }, { -4, -8 } },
            { { 11, -13 }, { 12, -8 } },
            { { 4, 7 }, { 5, 1 } },
            { { 5, -3 }, { 10, -3 } },
            { { 3, -7 }, { 6, 12 } },
            { { -8, -7 }, { -6, -2 } },
            { { -2, 11 }, { -1, -10 } },
            { { -13, 12 }, { -8, 10 } },
            { { -7, 3 }, { -5, -3 } },
            { { -4, 2 }, { -3, 7 } },
            { { -10, -12 }, { -6, 11 } },
            { { 5, -12 }, { 6, -7 } },
            { { 5, -6 }, { 7, -1 } },
            { { 1, 0 }, { 4, -5 } },
            { { 9, 11 }, { 11, -13 } },
            { { 4, 7 }, { 4, 12 } },
            { { 2, -1 }, { 4, 4 } },
            { { -4, -12 }, { -2, 7 } },
            { { -8, -5 }, { -7, -10 } },
            { { 4, 11 }, { 9, 12 } },
            { { 0, -8 }, { 1, -13 } },
            { { -13, -2 }, { -8, 2 } },
            { { -3, -2 }, { -2, 3 } },
            { { -6, 9 }, { -4, -9 } },
            { { 8, 12 }, { 10, 7 } },
            { { 0, 9 }, { 1, 3 } },
            { { 7, -5 }, { 11, -10 } },
            { { -13, -6 }, { -11, 0 } },
            { { 10, 7 }, { 12, 1 } },
            { { -6, -3 }, { -6, 12 } },
            { { 10, -9 }, { 12, -4 } },
            { { -13, 8 }, { -8, -12 } },
            { { -13, 0 }, { -8, -4 } },
            { { 3, 3 }, { 7, 8 } },
            { { 5, 7 }, { 10, -7 } },
            { { -1, 7 }, { 1, -12 } },
            { { 3, -10 }, { 5, 6 } },
            { { 2, -4 }, { 3, -10 } },
            { { -13, 0 }, { -13, 5 } },
            { { -13, -7 }, { -12, 12 } },
            { { -13, 3 }, { -11, 8 } },
            { { -7, 12 }, { -4, 7 } },
            { { 6, -10 }, { 12, 8 } },
            { { -9, -1 }, { -7, -6 } },
            { { -2, -5 }, { 0, 12 } },
            { { -12, 5 }, { -7, 5 } },
            { { 3, -10 }, { 8, -13 } },
            { { -7, -7 }, { -4, 5 } },
            { { -3, -2 }, { -1, -7 } },
            { { 2, 9 }, { 5, -11 } },
            { { -11, -13 }, { -5, -13 } },
            { { -1, 6 }, { 0, -1 } },
            { { 5, -3 }, { 5, 2 } },
            { { -4, -13 }, { -4, 12 } },
            { { -9, -6 }, { -9, 6 } },
            { { -12, -10 }, { -8, -4 } },
            { { 10, 2 }, { 12, -3 } },
            { { 7, 12 }, { 12, 12 } },
            { { -7, -13 }, { -6, 5 } },
            { { -4, 9 }, { -3, 4 } },
            { { 7, -1 }, { 12, 2 } },
            { { -7, 6 }, { -5, 1 } },
            { { -13, 11 }, { -12, 5 } },
            { { -3, 7 }, { -2, -6 } },
            { { 7, -8 }, { 12, -7 } },
            { { -13, -7 }, { -11, -12 } },
            { { 1, -3 }, { 12, 12 } },
            { { 2, -6 }, { 3, 0 } },
            { { -4, 3 }, { -2, -13 } },
            { { -1, -13 }, { 1, 9 } },
            { { 7, 1 }, { 8, -6 } },
            { { 1, -1 }, { 3, 12 } },
            { { 9, 1 }, { 12, 6 } },
            { { -1, -9 }, { -1, 3 } },
            { { -13, -13 }, { -10, 5 } },
            { { 7, 7 }, { 10, 12 } },
            { { 12, -5 }, { 12, 9 } },
            { { 6, 3 }, { 7, 11 } },
            { { 5, -13 }, { 6, 10 } },
            { { 2, -12 }, { 2, 3 } },
            { { 3, 8 }, { 4, -6 } },
            { { 2, 6 }, { 12, -13 } },
            { { 9, -12 }, { 10, 3 } },
            { { -8, 4 }, { -7, 9 } },
            { { -11, 12 }, { -4, -6 } },
            { { 1, 12 }, { 2, -8 } },
            { { 6, -9 }, { 7, -4 } },
            { { 2, 3 }, { 3, -2 } },
            { { 6, 3 }, { 11, 0 } },
            { { 3, -3 }, { 8, -8 } },
            { { 7, 8 }, { 9, 3 } },
            { { -11, -5 }, { -6, -4 } },
            { { -10, 11 }, { -5, 10 } },
            { { -5, -8 }, { -3, 12 } },
            { { -10, 5 }, { -9, 0 } },
            { { 8, -1 }, { 12, -6 } },
            { { 4, -6 }, { 6, -11 } },
            { { -10, 12 }, { -8, 7 } },
            { { 4, -2 }, { 6, 7 } },
            { { -2, 0 }, { -2, 12 } },
            { { -5, -8 }, { -5, 2 } },
            { { 7, -6 }, { 10, 12 } },
            { { -9, -13 }, { -8, -8 } },
            { { -5, -13 }, { -5, -2 } },
            { { 8, -8 }, { 9, -13 } },
            { { -9, -11 }, { -9, 0 } },
            { { 1, -8 }, { 1, -2 } },
            { { 7, -4 }, { 9, 1 } },
            { { -2, 1 }, { -1, -4 } },
            { { 11, -6 }, { 12, -11 } },
            { { -12, -9 }, { -6, 4 } },
            { { 3, 7 }, { 7, 12 } },
            { { 5, 5 }, { 10, 8 } },
            { { 0, -4 }, { 2, 8 } },
            { { -9, 12 }, { -5, -13 } },
            { { 0, 7 }, { 2, 12 } },
            { { -1, 2 }, { 1, 7 } },
            { { 5, 11 }, { 7, -9 } },
            { { 3, 5 }, { 6, -8 } },
            { { -13, -4 }, { -8, 9 } },
            { { -5, 9 }, { -3, -3 } },
            { { -4, -7 }, { -3, -12 } },
            { { 6, 5 }, { 8, 0 } },
            { { -7, 6 }, { -6, 12 } },
            { { -13, 6 }, { -5, -2 } },
            { { 1, -10 }, { 3, 10 } },
            { { 4, 1 }, { 8, -4 } },
            { { -2, -2 }, { 2, -13 } },
            { { 2, -12 }, { 12, 12 } },
            { { -2, -13 }, { 0, -6 } },
            { { 4, 1 }, { 9, 3 } },
            { { -6, -10 }, { -3, -5 } },
            { { -3, -13 }, { -1, 1 } },
            { { 7, 5 }, { 12, -11 } },
            { { 4, -2 }, { 5, -7 } },
            { { -13, 9 }, { -9, -5 } },
            { { 7, 1 }, { 8, 6 } },
            { { 7, -8 }, { 7, 6 } },
            { { -7, -4 }, { -7, 1 } },
            { { -8, 11 }, { -7, -8 } },
            { { -13, 6 }, { -12, -8 } },
            { { 2, 4 }, { 3, 9 } },
            { { 10, -5 }, { 12, 3 } },
            { { -6, -5 }, { -6, 7 } },
            { { 8, -3 }, { 9, -8 } },
            { { 2, -12 }, { 2, 8 } },
            { { -11, -2 }, { -10, 3 } },
            { { -12, -13 }, { -7, -9 } },
            { { -11, 0 }, { -10, -5 } },
            { { 5, -3 }, { 11, 8 } },
            { { -2, -13 }, { -1, 12 } },
            { { -1, -8 }, { 0, 9 } },
            { { -13, -11 }, { -12, -5 } },
            { { -10, -2 }, { -10, 11 } },
            { { -3, 9 }, { -2, -13 } },
            { { 2, -3 }, { 3, 2 } },
            { { -9, -13 }, { -4, 0 } },
            { { -4, 6 }, { -3, -10 } },
            { { -4, 12 }, { -2, -7 } },
            { { -6, -11 }, { -4, 9 } },
            { { 6, -3 }, { 6, 11 } },
            { { -13, 11 }, { -5, 5 } },
            { { 11, 11 }, { 12, 6 } },
            { { 7, -5 }, { 12, -2 } },
            { { -1, 12 }, { 0, 7 } },
            { { -4, -8 }, { -3, -2 } },
            { { -7, 1 }, { -6, 7 } },
            { { -13, -12 }, { -8, -13 } },
            { { -7, -2 }, { -6, -8 } },
            { { -8, 5 }, { -6, -9 } },
            { { -5, -1 }, { -4, 5 } },
            { { -13, 7 }, { -8, 10 } },
            { { 1, 5 }, { 5, -13 } },
            { { 1, 0 }, { 10, -13 } },
            { { 9, 12 }, { 10, -1 } },
            { { 5, -8 }, { 10, -9 } },
            { { -1, 11 }, { 1, -13 } },
            { { -9, -3 }, { -6, 2 } },
            { { -1, -10 }, { 1, 12 } },
            { { -13, 1 }, { -8, -10 } },
            { { 8, -11 }, { 10, -6 } },
            { { 2, -13 }, { 3, -6 } },
            { { 7, -13 }, { 12, -9 } },
            { { -10, -10 }, { -5, -7 } },
            { { -10, -8 }, { -8, -13 } },
            { { 4, -6 }, { 8, 5 } },
            { { 3, 12 }, { 8, -13 } },
            { { -4, 2 }, { -3, -3 } },
            { { 5, -13 }, { 10, -12 } },
            { { 4, -13 }, { 5, -1 } },
            { { -9, 9 }, { -4, 3 } },
            { { 0, 3 }, { 3, -9 } },
            { { -12, 1 }, { -6, 1 } },
            { { 3, 2 }, { 4, -8 } },
            { { -10, -10 }, { -10, 9 } },
            { { 8, -13 }, { 12, 12 } },
            { { -8, -12 }, { -6, -5 } },
            { { 2, 2 }, { 3, 7 } },
            { { 10, 6 }, { 11, -8 } },
            { { 6, 8 }, { 8, -12 } },
            { { -7, 10 }, { -6, 5 } },
            { { -3, -9 }, { -3, 9 } },
            { { -1, -13 }, { -1, 5 } },
            { { -3, -7 }, { -3, 4 } },
            { { -8, -2 }, { -8, 3 } },
            { { 4, 2 }, { 12, 12 } },
            { { 2, -5 }, { 3, 11 } },
            { { 6, -9 }, { 11, -13 } },
            { { 3, -1 }, { 7, 12 } },
            { { 11, -1 }, { 12, 4 } },
            { { -3, 0 }, { -3, 6 } },
            { { 4, -11 }, { 4, 12 } },
            { { 2, -4 }, { 2, 1 } },
            { { -10, -6 }, { -8, 1 } },
            { { -13, 7 }, { -11, 1 } },
            { { -13, 12 }, { -11, -13 } },
            { { 6, 0 }, { 11, -13 } },
            { { 0, -1 }, { 1, 4 } },
            { { -13, 3 }, { -9, -2 } },
            { { -9, 8 }, { -6, -3 } },
            { { -13, -6 }, { -8, -2 } },
            { { 5, -9 }, { 8, 10 } },
            { { 2, 7 }, { 3, -9 } },
            { { -1, -6 }, { -1, -1 } },
            { { 9, 5 }, { 11, -2 } },
            { { 11, -3 }, { 12, -8 } },
            { { 3, 0 }, { 3, 5 } },
            { { -1, 4 }, { 0, 10 } },
            { { 3, -6 }, { 4, 5 } },
            { { -13, 0 }, { -10, 5 } },
            { { 5, 8 }, { 12, 11 } },
            { { 8, 9 }, { 9, -6 } },
            { { 7, -4 }, { 8, -12 } },
            { { -10, 4 }, { -10, 9 } },
            { { 7, 3 }, { 12, 4 } },
            { { 9, -7 }, { 10, -2 } },
            { { 7, 0 }, { 12, -2 } },
            { { -1, -6 }, { 0, -11 } }
        };
        float angle_sin = 0;
        float angle_cos = 0;
        math::sincos(angle_radians, angle_sin, angle_cos);
        for (int index = 0; index < pattern_size; index += 8) {
            const unsigned char pixels_lhs[8] = {
                data[(math::round(pattern[index + 0][0][0] * angle_sin + pattern[index + 0][0][1] * angle_cos)) * stride + (math::round(pattern[index + 0][0][0] * angle_cos - pattern[index + 0][0][1] * angle_sin))],
                data[(math::round(pattern[index + 1][0][0] * angle_sin + pattern[index + 1][0][1] * angle_cos)) * stride + (math::round(pattern[index + 1][0][0] * angle_cos - pattern[index + 1][0][1] * angle_sin))],
                data[(math::round(pattern[index + 2][0][0] * angle_sin + pattern[index + 2][0][1] * angle_cos)) * stride + (math::round(pattern[index + 2][0][0] * angle_cos - pattern[index + 2][0][1] * angle_sin))],
                data[(math::round(pattern[index + 3][0][0] * angle_sin + pattern[index + 3][0][1] * angle_cos)) * stride + (math::round(pattern[index + 3][0][0] * angle_cos - pattern[index + 3][0][1] * angle_sin))],
                data[(math::round(pattern[index + 4][0][0] * angle_sin + pattern[index + 4][0][1] * angle_cos)) * stride + (math::round(pattern[index + 4][0][0] * angle_cos - pattern[index + 4][0][1] * angle_sin))],
                data[(math::round(pattern[index + 5][0][0] * angle_sin + pattern[index + 5][0][1] * angle_cos)) * stride + (math::round(pattern[index + 5][0][0] * angle_cos - pattern[index + 5][0][1] * angle_sin))],
                data[(math::round(pattern[index + 6][0][0] * angle_sin + pattern[index + 6][0][1] * angle_cos)) * stride + (math::round(pattern[index + 6][0][0] * angle_cos - pattern[index + 6][0][1] * angle_sin))],
                data[(math::round(pattern[index + 7][0][0] * angle_sin + pattern[index + 7][0][1] * angle_cos)) * stride + (math::round(pattern[index + 7][0][0] * angle_cos - pattern[index + 7][0][1] * angle_sin))],
            };
            const unsigned char pixels_rhs[8] = {
                data[(math::round(pattern[index + 0][1][0] * angle_sin + pattern[index + 0][1][1] * angle_cos)) * stride + (math::round(pattern[index + 0][1][0] * angle_cos - pattern[index + 0][1][1] * angle_sin))],
                data[(math::round(pattern[index + 1][1][0] * angle_sin + pattern[index + 1][1][1] * angle_cos)) * stride + (math::round(pattern[index + 1][1][0] * angle_cos - pattern[index + 1][1][1] * angle_sin))],
                data[(math::round(pattern[index + 2][1][0] * angle_sin + pattern[index + 2][1][1] * angle_cos)) * stride + (math::round(pattern[index + 2][1][0] * angle_cos - pattern[index + 2][1][1] * angle_sin))],
                data[(math::round(pattern[index + 3][1][0] * angle_sin + pattern[index + 3][1][1] * angle_cos)) * stride + (math::round(pattern[index + 3][1][0] * angle_cos - pattern[index + 3][1][1] * angle_sin))],
                data[(math::round(pattern[index + 4][1][0] * angle_sin + pattern[index + 4][1][1] * angle_cos)) * stride + (math::round(pattern[index + 4][1][0] * angle_cos - pattern[index + 4][1][1] * angle_sin))],
                data[(math::round(pattern[index + 5][1][0] * angle_sin + pattern[index + 5][1][1] * angle_cos)) * stride + (math::round(pattern[index + 5][1][0] * angle_cos - pattern[index + 5][1][1] * angle_sin))],
                data[(math::round(pattern[index + 6][1][0] * angle_sin + pattern[index + 6][1][1] * angle_cos)) * stride + (math::round(pattern[index + 6][1][0] * angle_cos - pattern[index + 6][1][1] * angle_sin))],
                data[(math::round(pattern[index + 7][1][0] * angle_sin + pattern[index + 7][1][1] * angle_cos)) * stride + (math::round(pattern[index + 7][1][0] * angle_cos - pattern[index + 7][1][1] * angle_sin))],
            };
            descriptor[static_cast<size_t>(index / 8)] = static_cast<unsigned char>(
                ((pixels_lhs[0] < pixels_rhs[0]) << 0) |
                ((pixels_lhs[1] < pixels_rhs[1]) << 1) |
                ((pixels_lhs[2] < pixels_rhs[2]) << 2) |
                ((pixels_lhs[3] < pixels_rhs[3]) << 3) |
                ((pixels_lhs[4] < pixels_rhs[4]) << 4) |
                ((pixels_lhs[5] < pixels_rhs[5]) << 5) |
                ((pixels_lhs[6] < pixels_rhs[6]) << 6) |
                ((pixels_lhs[7] < pixels_rhs[7]) << 7)
            );
        }
    }

    template <unsigned int bits>
    static inline unsigned int distance(
        const unsigned char* __restrict const data_lhs,
        const unsigned char* __restrict const data_rhs
    );

    template <>
    inline unsigned int distance<64>(
        const unsigned char* __restrict const data_lhs,
        const unsigned char* __restrict const data_rhs
    ) {
#if __has_builtin(__builtin_popcountll)
        unsigned long long int internal_lhs;
        unsigned long long int internal_rhs;
        unsigned char* __restrict const data_internal_lhs = reinterpret_cast<unsigned char*>(&internal_lhs);
        unsigned char* __restrict const data_internal_rhs = reinterpret_cast<unsigned char*>(&internal_rhs);
        for (decltype(sizeof(unsigned long long int)) i = 0; i < sizeof(unsigned long long int); ++i) {
            data_internal_lhs[i] = data_lhs[i];
            data_internal_rhs[i] = data_rhs[i];
        }
        return static_cast<unsigned int>(__builtin_popcountll(internal_lhs ^ internal_rhs));
#else
        constexpr static const unsigned long long int Ox01 = 0x0101010101010101ull;
        constexpr static const unsigned long long int Ox0F = 0x0F0F0F0F0F0F0F0Full;
        constexpr static const unsigned long long int Ox33 = 0x3333333333333333ull;
        constexpr static const unsigned long long int Ox55 = 0x5555555555555555ull;
        unsigned long long int internal_lhs;
        unsigned long long int internal_rhs;
        unsigned char* __restrict const data_internal_lhs = reinterpret_cast<unsigned char*>(&internal_lhs);
        unsigned char* __restrict const data_internal_rhs = reinterpret_cast<unsigned char*>(&internal_rhs);
        for (decltype(sizeof(unsigned long long int)) i = 0; i < sizeof(unsigned long long int); ++i) {
            data_internal_lhs[i] = data_lhs[i];
            data_internal_rhs[i] = data_rhs[i];
        }
        unsigned long long int difference = internal_lhs ^ internal_rhs;
        difference = difference - ((difference >> 1u) & Ox55);
        difference = (difference & Ox33) + ((difference >> 2u) & Ox33);
        return static_cast<unsigned long long int>(((difference + (difference >> 4u)) & Ox0F) * Ox01) >> 56u;
#endif
    }

    template <>
    inline unsigned int distance<128>(
        const unsigned char* __restrict const data_lhs,
        const unsigned char* __restrict const data_rhs
    ) {
        return distance<64>(&data_lhs[0], &data_rhs[0]) + distance<64>(&data_lhs[8], &data_rhs[8]);
    }

    template <>
    inline unsigned int distance<256>(
        const unsigned char* __restrict const data_lhs,
        const unsigned char* __restrict const data_rhs
    ) {
        return distance<128>(&data_lhs[0], &data_rhs[0]) + distance<128>(&data_lhs[16], &data_rhs[16]);
    }

    static inline size_t find_matches(
        const descriptor::binary<256>* lhs_descriptors,
        const size_t lhs_descriptors_size,
        const descriptor::binary<256>* rhs_descriptors,
        const size_t rhs_descriptors_size,
        const float threshold,
        const size_t matches_count,
        match::pair* matches,
        const size_t matches_size
    ) {
        if ((matches_count == 0) || (matches_size == 0)) {
            return 0;
        }
        size_t count = 0;
        for (size_t lhs_index = 0; lhs_index < lhs_descriptors_size; ++lhs_index) {
            if (count + matches_count > matches_size) {
                break;
            }
            for (size_t matches_index = 0; matches_index < matches_count; ++matches_index) {
                matches[count + matches_index].lhs_index = lhs_index;
                matches[count + matches_index].score = threshold;
            }
            for (size_t rhs_index = 0; rhs_index < rhs_descriptors_size; ++rhs_index) {
                const float score = static_cast<float>(distance<256>(lhs_descriptors[lhs_index].data, rhs_descriptors[rhs_index].data));
                for (size_t matches_index = 0; matches_index < matches_count; ++matches_index) {
                    if (score < matches[count + matches_index].score) {
                        for (size_t shift_index = matches_count - 1; shift_index > matches_index; --shift_index) {
                            matches[count + shift_index].score = matches[count + shift_index - 1].score;
                            matches[count + shift_index].rhs_index = matches[count + shift_index - 1].rhs_index;
                        }
                        matches[count + matches_index].score = score;
                        matches[count + matches_index].rhs_index = rhs_index;
                        break;
                    }
                }
            }
            const size_t save_index = count;
            for (size_t matches_index = 0; matches_index < matches_count; ++matches_index) {
                if (matches[save_index + matches_index].score < threshold) {
                    ++count;
                }
            }
        }
        return count;
    }
}

#endif // ZEROSLAM_FEATURE_FEATURE_HPP
