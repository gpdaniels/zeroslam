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

#include "feature/score/structure_tensor.hpp"

#include "core/assert.hpp"
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::score {
    float structure_tensor::score(const measure kind, const float ixx, const float ixy, const float iyy) {
        const float trace = ixx + iyy;
        const float determinant = ixx * iyy - ixy * ixy;
        float result = 0.0f;
        switch (kind) {
            case measure::rohr: {
                result = determinant;
            } break;
            case measure::harris: {
                result = determinant - structure_tensor::harris_k * trace * trace;
            } break;
            case measure::forstner: {
                result = (trace > 0.0f) ? (determinant / trace) : 0.0f;
            } break;
            case measure::klt:
            case measure::kenney: {
                const float discriminant = math::sqrt(math::max(0.0f, trace * trace - 4.0f * determinant));
                const float large = 0.5f * (trace + discriminant);
                const float small = 0.5f * (trace - discriminant);
                if (kind == measure::klt) {
                    result = small;
                }
                else if (small > 0.0f) {
                    result = (large * small) / math::sqrt(large * large + small * small);
                }
            } break;
        }
        return result;
    }

    float structure_tensor::strength(const measure kind, const float response) {
        if ((kind == measure::harris) || (kind == measure::rohr)) {
            return (response > 0.0f) ? math::sqrt(response) : 0.0f;
        }
        return response;
    }

    int structure_tensor::smoothing_radius(const float sigma) {
        ASSERT(sigma > 0.0f, "The smoothing sigma must be positive.");
        ASSERT(sigma <= structure_tensor::sigma_maximum, "The smoothing sigma exceeds the maximum.");
        return math::min(structure_tensor::smoothing_radius_maximum, math::max(1, static_cast<int>(2.0f * sigma * sigma + 0.5f)));
    }

    void structure_tensor::smoothing_weights(const float sigma, float* __restrict const weights) {
        const int radius = structure_tensor::smoothing_radius(sigma);
        const int order = 2 * radius;
        const double scale = 1.0 / static_cast<double>(1ull << order);
        double coefficient = 1.0;
        for (int k = 0; k <= order; ++k) {
            weights[k] = static_cast<float>(coefficient * scale);
            coefficient = (coefficient * static_cast<double>(order - k)) / static_cast<double>(k + 1);
        }
    }

    int structure_tensor::margin(const float sigma) {
        return structure_tensor::smoothing_radius(sigma) + 1;
    }

    void structure_tensor::respond(
        const unsigned char* __restrict const data,
        const int width,
        const int height,
        const int stride,
        const measure kind,
        const float sigma,
        float* __restrict const response
    ) {
        ASSERT((width > 0) && (height > 0), "The region must not be empty.");
        ASSERT(stride >= width, "The stride must cover the width.");
        const int radius = structure_tensor::smoothing_radius(sigma);
        const int taps = 2 * radius + 1;
        const int border = structure_tensor::margin(sigma);
        float weights[2 * structure_tensor::smoothing_radius_maximum + 1];
        structure_tensor::smoothing_weights(sigma, weights);

        const size_t area = static_cast<size_t>(width) * static_cast<size_t>(height);
        for (size_t i = 0; i < area; ++i) {
            response[i] = 0.0f;
        }
        if ((width <= 2 * border) || (height <= 2 * border)) {
            return;
        }

        std::vector<float> products(3 * area, 0.0f);
        std::vector<float> scratch(area, 0.0f);
        float* __restrict const ixx = products.data();
        float* __restrict const ixy = products.data() + area;
        float* __restrict const iyy = products.data() + 2 * area;
        for (int y = 1; y < height - 1; ++y) {
            const unsigned char* __restrict const row = data + static_cast<size_t>(y) * static_cast<size_t>(stride);
            for (int x = 1; x < width - 1; ++x) {
                const float gx = 0.5f * (static_cast<float>(row[x + 1]) - static_cast<float>(row[x - 1]));
                const float gy = 0.5f * (static_cast<float>(row[x + stride]) - static_cast<float>(row[x - stride]));
                const size_t i = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
                ixx[i] = gx * gx;
                ixy[i] = gx * gy;
                iyy[i] = gy * gy;
            }
        }

        float* const planes[3] = { ixx, ixy, iyy };
        for (float* __restrict const plane : planes) {
            for (int y = 1; y < height - 1; ++y) {
                const float* __restrict const source = plane + static_cast<size_t>(y) * static_cast<size_t>(width);
                float* __restrict const target = scratch.data() + static_cast<size_t>(y) * static_cast<size_t>(width);
                for (int x = border; x < width - border; ++x) {
                    float sum = 0.0f;
                    for (int k = 0; k < taps; ++k) {
                        sum += weights[k] * source[x - radius + k];
                    }
                    target[x] = sum;
                }
            }
            for (int y = border; y < height - border; ++y) {
                float* __restrict const target = plane + static_cast<size_t>(y) * static_cast<size_t>(width);
                for (int x = border; x < width - border; ++x) {
                    float sum = 0.0f;
                    for (int k = 0; k < taps; ++k) {
                        sum += weights[k] * scratch[static_cast<size_t>(y - radius + k) * static_cast<size_t>(width) + static_cast<size_t>(x)];
                    }
                    target[x] = sum;
                }
            }
        }

        for (int y = border; y < height - border; ++y) {
            for (int x = border; x < width - border; ++x) {
                const size_t i = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
                response[i] = structure_tensor::score(kind, ixx[i], ixy[i], iyy[i]);
            }
        }
    }
}
