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

namespace feature::descriptor {
    void describe_cpu(const unsigned char* __restrict const data, const int stride, const float angle_radians, binary<256>& descriptor);

    void describe_cpu(
        const unsigned char* __restrict const data,
        const int stride,
        const float angle_radians,
        binary<256>& descriptor
    ) {
        float angle_sin = 0;
        float angle_cos = 0;
        math::sincos(angle_radians, angle_sin, angle_cos);
        for (int index = 0; index < orb::pattern_size; index += 8) {
            const unsigned char pixels_lhs[8] = {
                data[(math::round(orb::pattern[index + 0][0][0] * angle_sin + orb::pattern[index + 0][0][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 0][0][0] * angle_cos - orb::pattern[index + 0][0][1] * angle_sin))],
                data[(math::round(orb::pattern[index + 1][0][0] * angle_sin + orb::pattern[index + 1][0][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 1][0][0] * angle_cos - orb::pattern[index + 1][0][1] * angle_sin))],
                data[(math::round(orb::pattern[index + 2][0][0] * angle_sin + orb::pattern[index + 2][0][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 2][0][0] * angle_cos - orb::pattern[index + 2][0][1] * angle_sin))],
                data[(math::round(orb::pattern[index + 3][0][0] * angle_sin + orb::pattern[index + 3][0][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 3][0][0] * angle_cos - orb::pattern[index + 3][0][1] * angle_sin))],
                data[(math::round(orb::pattern[index + 4][0][0] * angle_sin + orb::pattern[index + 4][0][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 4][0][0] * angle_cos - orb::pattern[index + 4][0][1] * angle_sin))],
                data[(math::round(orb::pattern[index + 5][0][0] * angle_sin + orb::pattern[index + 5][0][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 5][0][0] * angle_cos - orb::pattern[index + 5][0][1] * angle_sin))],
                data[(math::round(orb::pattern[index + 6][0][0] * angle_sin + orb::pattern[index + 6][0][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 6][0][0] * angle_cos - orb::pattern[index + 6][0][1] * angle_sin))],
                data[(math::round(orb::pattern[index + 7][0][0] * angle_sin + orb::pattern[index + 7][0][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 7][0][0] * angle_cos - orb::pattern[index + 7][0][1] * angle_sin))],
            };
            const unsigned char pixels_rhs[8] = {
                data[(math::round(orb::pattern[index + 0][1][0] * angle_sin + orb::pattern[index + 0][1][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 0][1][0] * angle_cos - orb::pattern[index + 0][1][1] * angle_sin))],
                data[(math::round(orb::pattern[index + 1][1][0] * angle_sin + orb::pattern[index + 1][1][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 1][1][0] * angle_cos - orb::pattern[index + 1][1][1] * angle_sin))],
                data[(math::round(orb::pattern[index + 2][1][0] * angle_sin + orb::pattern[index + 2][1][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 2][1][0] * angle_cos - orb::pattern[index + 2][1][1] * angle_sin))],
                data[(math::round(orb::pattern[index + 3][1][0] * angle_sin + orb::pattern[index + 3][1][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 3][1][0] * angle_cos - orb::pattern[index + 3][1][1] * angle_sin))],
                data[(math::round(orb::pattern[index + 4][1][0] * angle_sin + orb::pattern[index + 4][1][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 4][1][0] * angle_cos - orb::pattern[index + 4][1][1] * angle_sin))],
                data[(math::round(orb::pattern[index + 5][1][0] * angle_sin + orb::pattern[index + 5][1][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 5][1][0] * angle_cos - orb::pattern[index + 5][1][1] * angle_sin))],
                data[(math::round(orb::pattern[index + 6][1][0] * angle_sin + orb::pattern[index + 6][1][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 6][1][0] * angle_cos - orb::pattern[index + 6][1][1] * angle_sin))],
                data[(math::round(orb::pattern[index + 7][1][0] * angle_sin + orb::pattern[index + 7][1][1] * angle_cos)) * stride + (math::round(orb::pattern[index + 7][1][0] * angle_cos - orb::pattern[index + 7][1][1] * angle_sin))],
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
}
