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

#include "feature/detector/structure_tensor.hpp"

#include "core/assert.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::detector {
    size_t structure_tensor::detect(
        const unsigned char* __restrict const data,
        const int width,
        const int height,
        const int stride,
        const score::structure_tensor::measure kind,
        const float sigma,
        const float threshold,
        const size_t feature_point_buffer_size,
        point* __restrict feature_point_buffer
    ) {
        ASSERT(threshold >= 0.0f, "The threshold is a corner strength.");
        if ((width < 3) || (height < 3)) {
            return 0;
        }
        std::vector<float> response(static_cast<size_t>(width) * static_cast<size_t>(height));
        score::structure_tensor::respond(data, width, height, stride, kind, sigma, response.data());

        size_t feature_count = 0;
        for (int y = 1; (y < height - 1) && (feature_count < feature_point_buffer_size); ++y) {
            const float* __restrict const row = response.data() + static_cast<size_t>(y) * static_cast<size_t>(width);
            for (int x = 1; (x < width - 1) && (feature_count < feature_point_buffer_size); ++x) {
                const float value = row[x];
                if ((value <= 0.0f) || (score::structure_tensor::strength(kind, value) < threshold)) {
                    continue;
                }
                const float* __restrict const above = row - width;
                const float* __restrict const below = row + width;
                if ((above[x - 1] >= value) || (above[x] >= value) || (above[x + 1] >= value) || (row[x - 1] >= value) || (row[x + 1] >= value) || (below[x - 1] >= value) || (below[x] >= value) || (below[x + 1] >= value)) {
                    continue;
                }
                feature_point_buffer[feature_count].x = static_cast<float>(x);
                feature_point_buffer[feature_count].y = static_cast<float>(y);
                feature_point_buffer[feature_count].response = value;
                feature_point_buffer[feature_count].angle = 0.0f;
                feature_point_buffer[feature_count].octave = 0;
                ++feature_count;
            }
        }
        return feature_count;
    }
}
