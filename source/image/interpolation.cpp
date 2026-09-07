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

#include "image/interpolation.hpp"

#include "math/math.hpp"

namespace image {
    unsigned char interpolation::nearest(
        const unsigned char* __restrict data,
        size_t source_width,
        size_t source_height,
        float offset_x,
        float offset_y
    ) {
        const int nearest_x = math::max(0, math::min(math::round(offset_x), static_cast<int>(source_width) - 1));
        const int nearest_y = math::max(0, math::min(math::round(offset_y), static_cast<int>(source_height) - 1));
        return data[(static_cast<size_t>(nearest_y) * source_width) + static_cast<size_t>(nearest_x)];
    }

    unsigned char interpolation::linear(
        const unsigned char* __restrict data,
        size_t source_width,
        size_t source_height,
        float offset_x,
        float offset_y
    ) {
        const int floor_x = math::max(0, math::min(static_cast<int>(math::floor(offset_x)), static_cast<int>(source_width) - 1));
        const int floor_y = math::max(0, math::min(static_cast<int>(math::floor(offset_y)), static_cast<int>(source_height) - 1));
        const int ceil_x = math::min(floor_x + 1, static_cast<int>(source_width) - 1);
        const int ceil_y = math::min(floor_y + 1, static_cast<int>(source_height) - 1);
        const size_t row_floor = static_cast<size_t>(floor_y) * source_width;
        const size_t row_ceil = static_cast<size_t>(ceil_y) * source_width;
        const size_t column_floor = static_cast<size_t>(floor_x);
        const size_t column_ceil = static_cast<size_t>(ceil_x);
        const float weight_x = math::max(0.0f, math::min(offset_x - static_cast<float>(floor_x), 1.0f));
        const float weight_y = math::max(0.0f, math::min(offset_y - static_cast<float>(floor_y), 1.0f));
        return static_cast<unsigned char>(
            math::round(
                static_cast<float>(data[row_floor + column_floor]) * (1.0f - weight_x) * (1.0f - weight_y) +
                static_cast<float>(data[row_floor + column_ceil]) * (weight_x) * (1.0f - weight_y) +
                static_cast<float>(data[row_ceil + column_floor]) * (1.0f - weight_x) * (weight_y) +
                static_cast<float>(data[row_ceil + column_ceil]) * (weight_x) * (weight_y)
            )
        );
    }
}
