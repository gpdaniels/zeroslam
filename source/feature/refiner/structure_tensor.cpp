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
        if ((centre[-1] > peak) || (centre[1] > peak) || (centre[-size] > peak) || (centre[size] > peak) || (centre[-size - 1] > peak) || (centre[-size + 1] > peak) || (centre[size - 1] > peak) || (centre[size + 1] > peak)) {
            return false;
        }

        // Note: The response of a corner peaks about sigma inside its wedge, so move to the point nearest every gradient line in the smoothing window (Forstner).
        const int radius = score::structure_tensor::smoothing_radius(sigma);
        float weights[2 * score::structure_tensor::smoothing_radius_maximum + 1];
        score::structure_tensor::smoothing_weights(sigma, weights);
        const unsigned char* __restrict const window = data + (peak_y - half) * stride + (peak_x - half);
        float a11 = 0.0f;
        float a12 = 0.0f;
        float a22 = 0.0f;
        float b1 = 0.0f;
        float b2 = 0.0f;
        for (int v = -radius; v <= radius; ++v) {
            const unsigned char* __restrict const row = window + v * stride;
            for (int u = -radius; u <= radius; ++u) {
                const float gx = 0.5f * (static_cast<float>(row[u + 1]) - static_cast<float>(row[u - 1]));
                const float gy = 0.5f * (static_cast<float>(row[u + stride]) - static_cast<float>(row[u - stride]));
                const float weight = weights[v + radius] * weights[u + radius];
                const float wxx = weight * gx * gx;
                const float wxy = weight * gx * gy;
                const float wyy = weight * gy * gy;
                a11 += wxx;
                a12 += wxy;
                a22 += wyy;
                b1 += wxx * static_cast<float>(u) + wxy * static_cast<float>(v);
                b2 += wxy * static_cast<float>(u) + wyy * static_cast<float>(v);
            }
        }
        const float trace = a11 + a22;
        const float determinant = a11 * a22 - a12 * a12;
        if (!(determinant > 1.0e-6f * trace * trace)) {
            return false;
        }
        const float refined_x = static_cast<float>(peak_x - half) + (a22 * b1 - a12 * b2) / determinant;
        const float refined_y = static_cast<float>(peak_y - half) + (a11 * b2 - a12 * b1) / determinant;
        // Note: The intersection must lie inside the pixels read, beyond them it is an extrapolation.
        if ((math::abs(refined_x) >= static_cast<float>(half)) || (math::abs(refined_y) >= static_cast<float>(half))) {
            return false;
        }

        offset_x = refined_x;
        offset_y = refined_y;
        return true;
    }
}
