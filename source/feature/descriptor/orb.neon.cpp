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

#include "feature/descriptor/orb.hpp"

#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <arm_neon.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::descriptor {
    void describe_neon(const unsigned char* __restrict const data, const int stride, const float angle_radians, binary<256>& descriptor);

    void describe_neon(
        const unsigned char* __restrict const data,
        const int stride,
        const float angle_radians,
        binary<256>& descriptor
    ) {
        float angle_sin = 0;
        float angle_cos = 0;
        math::sincos(angle_radians, angle_sin, angle_cos);

        const float32x4_t sinv = vdupq_n_f32(angle_sin);
        const float32x4_t cosv = vdupq_n_f32(angle_cos);
        const int32x4_t step = vdupq_n_s32(stride);

        // Rotate four {x, y} sample offsets and return them as pixel offsets, rounding half away from
        // zero exactly as math::round does.
        //   row = x * sin + y * cos
        //   col = x * cos - y * sin
        const auto rotate = [sinv, cosv, step](const int32x4_t x, const int32x4_t y) -> int32x4_t {
            const float32x4_t a = vcvtq_f32_s32(x);
            const float32x4_t b = vcvtq_f32_s32(y);
            const float32x4_t row = vaddq_f32(vmulq_f32(a, sinv), vmulq_f32(b, cosv));
            const float32x4_t col = vsubq_f32(vmulq_f32(a, cosv), vmulq_f32(b, sinv));
            return vmlaq_s32(vcvtaq_s32_f32(col), vcvtaq_s32_f32(row), step);
        };

        alignas(16) int lhs_offsets[8];
        alignas(16) int rhs_offsets[8];

        for (int index = 0; index < orb::pattern_size; index += 8) {
            // A structure load of sixteen ints de-interleaves four pattern entries into the four vectors
            // {lhs x, lhs y, rhs x, rhs y}.
            const int32x4x4_t first = vld4q_s32(&orb::pattern[index + 0][0][0]);
            const int32x4x4_t second = vld4q_s32(&orb::pattern[index + 4][0][0]);

            vst1q_s32(lhs_offsets + 0, rotate(first.val[0], first.val[1]));
            vst1q_s32(lhs_offsets + 4, rotate(second.val[0], second.val[1]));
            vst1q_s32(rhs_offsets + 0, rotate(first.val[2], first.val[3]));
            vst1q_s32(rhs_offsets + 4, rotate(second.val[2], second.val[3]));

            descriptor[static_cast<size_t>(index / 8)] =
                static_cast<unsigned char>(
                    ((data[lhs_offsets[0]] < data[rhs_offsets[0]]) << 0) |
                    ((data[lhs_offsets[1]] < data[rhs_offsets[1]]) << 1) |
                    ((data[lhs_offsets[2]] < data[rhs_offsets[2]]) << 2) |
                    ((data[lhs_offsets[3]] < data[rhs_offsets[3]]) << 3) |
                    ((data[lhs_offsets[4]] < data[rhs_offsets[4]]) << 4) |
                    ((data[lhs_offsets[5]] < data[rhs_offsets[5]]) << 5) |
                    ((data[lhs_offsets[6]] < data[rhs_offsets[6]]) << 6) |
                    ((data[lhs_offsets[7]] < data[rhs_offsets[7]]) << 7)
                );
        }
    }
}
