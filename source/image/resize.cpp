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

#include "image/resize.hpp"

#include "image/interpolation.hpp"

namespace image {
    namespace {
        void sample(
            const unsigned char* __restrict const source_data,
            const size_t source_width,
            const size_t source_height,
            const size_t target_width,
            const size_t target_height,
            unsigned char* __restrict const target_data,
            const interpolation::function_type interpolate_function
        ) {
            for (size_t y = 0; y < target_height; ++y) {
                const float offset_y = ((static_cast<float>(y) + 0.5f) * (static_cast<float>(source_height) / static_cast<float>(target_height))) - 0.5f;
                for (size_t x = 0; x < target_width; ++x) {
                    const float offset_x = ((static_cast<float>(x) + 0.5f) * (static_cast<float>(source_width) / static_cast<float>(target_width))) - 0.5f;
                    target_data[y * target_width + x] = interpolate_function(source_data, source_width, source_height, offset_x, offset_y);
                }
            }
        }
    }

    void resize::decimate(
        const unsigned char* __restrict const source_data,
        const size_t source_width,
        const size_t source_height,
        const size_t target_width,
        const size_t target_height,
        unsigned char* __restrict const target_data
    ) {
        if ((target_width == 0) || (target_height == 0)) {
            return;
        }
        const size_t step_x = source_width / target_width;
        const size_t step_y = source_height / target_height;
        for (size_t y = 0; y < target_height; ++y) {
            const unsigned char* const source_row = source_data + (y * step_y) * source_width;
            unsigned char* const target_row = target_data + y * target_width;
            for (size_t x = 0; x < target_width; ++x) {
                target_row[x] = source_row[x * step_x];
            }
        }
    }

    void resize::nearest(
        const unsigned char* __restrict const source_data,
        const size_t source_width,
        const size_t source_height,
        const size_t target_width,
        const size_t target_height,
        unsigned char* __restrict const target_data
    ) {
        sample(source_data, source_width, source_height, target_width, target_height, target_data, &interpolation::nearest);
    }

    void resize::linear(
        const unsigned char* __restrict const source_data,
        const size_t source_width,
        const size_t source_height,
        const size_t target_width,
        const size_t target_height,
        unsigned char* __restrict const target_data
    ) {
        sample(source_data, source_width, source_height, target_width, target_height, target_data, &interpolation::linear);
    }
}
