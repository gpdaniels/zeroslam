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

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <arm_neon.h>
#include <limits>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::suppressor {
    size_t suppress_neon(const point* __restrict const features, const size_t features_count, const size_t max_row, point* __restrict const features_suppressed);

    size_t suppress_cpu(const point* __restrict const features, const size_t features_count, const size_t max_row, point* __restrict const features_suppressed);

    size_t suppress_neon(
        const point* __restrict const features,
        const size_t features_count,
        const size_t max_row,
        point* __restrict const features_suppressed
    ) {
        if (features_count == 0) {
            return 0;
        }

        size_t max_x = 0;
        for (size_t i = 0; i < features_count; ++i) {
            if (!((features[i].y >= 0.0f) && (features[i].x >= 0.0f))) {
                return suppress_cpu(features, features_count, max_row, features_suppressed);
            }
            const size_t fy = static_cast<size_t>(features[i].y);
            const size_t fx = static_cast<size_t>(features[i].x);
            if (fy > max_row) {
                return 0;
            }
            if (fx > max_x) {
                max_x = fx;
            }
            if ((features[i].y != static_cast<float>(fy)) || (features[i].x != static_cast<float>(fx))) {
                return suppress_cpu(features, features_count, max_row, features_suppressed);
            }
        }

        const size_t row_stride = max_x + 3;
        const size_t map_limit = static_cast<size_t>(1) << 26;
        if ((max_row + 1) > (map_limit / row_stride)) {
            return suppress_cpu(features, features_count, max_row, features_suppressed);
        }
        const size_t map_size = (max_row + 1) * row_stride;
        if (map_size >= (features_count * static_cast<size_t>(32))) {
            return suppress_cpu(features, features_count, max_row, features_suppressed);
        }

        float* const response_map = new float[map_size];
        for (size_t i = 0; i < map_size; ++i) {
            response_map[i] = -std::numeric_limits<float>::max();
        }
        for (size_t i = 0; i < features_count; ++i) {
            const size_t fy = static_cast<size_t>(features[i].y);
            const size_t fx = static_cast<size_t>(features[i].x);
            float& response = response_map[fy * row_stride + fx + 1];
            if (features[i].response > response) {
                response = features[i].response;
            }
        }

        size_t suppressed_count = 0;
        float prev_x = -999.0f;
        float prev_y = -999.0f;
        float prev_resp = -999.0f;
        for (size_t i = 0; i < features_count; ++i) {
            const float rx = features[i].x;
            const float ry = features[i].y;
            const float resp = features[i].response;
            const size_t siy = static_cast<size_t>(ry);
            const size_t six = static_cast<size_t>(rx);

            if (ry == prev_y && rx == prev_x + 1.0f && prev_resp > resp) {
                prev_x = rx;
                prev_y = ry;
                prev_resp = resp;
                continue;
            }
            if (i < (features_count - 1)) {
                if ((features[i + 1].x == rx + 1.0f) && (features[i + 1].y == ry) && (features[i + 1].response > resp)) {
                    prev_x = rx;
                    prev_y = ry;
                    prev_resp = resp;
                    continue;
                }
            }
            if (siy > 0) {
                const float* const above = response_map + (siy - 1) * row_stride + six + 1;
                if ((above[-1] > resp) || (above[0] > resp) || (above[1] > resp)) {
                    prev_x = rx;
                    prev_y = ry;
                    prev_resp = resp;
                    continue;
                }
            }
            if (siy < max_row) {
                const float* const below = response_map + (siy + 1) * row_stride + six + 1;
                if ((below[-1] > resp) || (below[0] > resp) || (below[1] > resp)) {
                    prev_x = rx;
                    prev_y = ry;
                    prev_resp = resp;
                    continue;
                }
            }
            features_suppressed[suppressed_count++] = features[i];
            prev_x = rx;
            prev_y = ry;
            prev_resp = resp;
        }

        delete[] response_map;
        return suppressed_count;
    }
}
