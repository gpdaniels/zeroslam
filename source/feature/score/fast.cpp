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

#include "feature/score/fast.hpp"

#include "math/math.hpp"

namespace feature::score {
    float fast::score(
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
}
