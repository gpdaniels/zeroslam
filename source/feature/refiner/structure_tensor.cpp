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

#include "feature/refiner/structure_tensor.hpp"

#include "core/assert.hpp"
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::refiner {
    int structure_tensor::footprint(const float sigma) {
        return structure_tensor::search_radius + 1 + score::structure_tensor::margin(sigma);
    }

    bool structure_tensor::refine(
        const unsigned char* __restrict const data,
        const int stride,
        const score::structure_tensor::measure kind,
        const float sigma,
        float& offset_x,
        float& offset_y
    ) {
        ASSERT(sigma <= structure_tensor::sigma_maximum, "The smoothing sigma exceeds the refiner's maximum.");
        offset_x = 0.0f;
        offset_y = 0.0f;

        const int half = structure_tensor::footprint(sigma);
        const int size = 2 * half + 1;
        std::vector<float> response(static_cast<size_t>(size) * static_cast<size_t>(size));
        score::structure_tensor::respond(data - half * stride - half, size, size, stride, kind, sigma, response.data());

        int peak_x = half;
        int peak_y = half;
        float peak = response[static_cast<size_t>(half) * static_cast<size_t>(size) + static_cast<size_t>(half)];
        for (int dy = -structure_tensor::search_radius; dy <= structure_tensor::search_radius; ++dy) {
            for (int dx = -structure_tensor::search_radius; dx <= structure_tensor::search_radius; ++dx) {
                const float value = response[static_cast<size_t>(half + dy) * static_cast<size_t>(size) + static_cast<size_t>(half + dx)];
                if (value > peak) {
                    peak = value;
                    peak_x = half + dx;
                    peak_y = half + dy;
                }
            }
        }
        if (peak <= 0.0f) {
            return false;
        }

        const float* __restrict const centre = response.data() + static_cast<size_t>(peak_y) * static_cast<size_t>(size) + static_cast<size_t>(peak_x);
        const float left = centre[-1];
        const float right = centre[1];
        const float above = centre[-size];
        const float below = centre[size];
        if ((left > peak) || (right > peak) || (above > peak) || (below > peak) || (centre[-size - 1] > peak) || (centre[-size + 1] > peak) || (centre[size - 1] > peak) || (centre[size + 1] > peak)) {
            return false;
        }

        const float curvature_x = left - 2.0f * peak + right;
        const float curvature_y = above - 2.0f * peak + below;
        const float subpixel_x = (curvature_x < 0.0f) ? math::min(0.5f, math::max(-0.5f, (0.5f * (left - right)) / curvature_x)) : 0.0f;
        const float subpixel_y = (curvature_y < 0.0f) ? math::min(0.5f, math::max(-0.5f, (0.5f * (above - below)) / curvature_y)) : 0.0f;

        offset_x = static_cast<float>(peak_x - half) + subpixel_x;
        offset_y = static_cast<float>(peak_y - half) + subpixel_y;
        return true;
    }
}
