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

#include "feature/angle/orb.hpp"

#include "math/math.hpp"

namespace feature::angle {
    float orb::dominant_angle(
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
}
