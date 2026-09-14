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

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline bool is_value_approx(float lhs, float rhs, float epsilon = 1e-5f) {
    return std::abs(lhs - rhs) <= epsilon;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    using tensor = feature::score::structure_tensor;

    {
        REQUIRE(is_value_approx(tensor::score(tensor::measure::klt, 4.0f, 0.0f, 1.0f), 1.0f));
        REQUIRE(is_value_approx(tensor::score(tensor::measure::forstner, 4.0f, 0.0f, 1.0f), 0.8f));
        REQUIRE(is_value_approx(tensor::score(tensor::measure::harris, 4.0f, 0.0f, 1.0f), 4.0f - 0.04f * 25.0f));
        REQUIRE(is_value_approx(tensor::score(tensor::measure::rohr, 4.0f, 0.0f, 1.0f), 4.0f));
        REQUIRE(is_value_approx(tensor::score(tensor::measure::kenney, 4.0f, 0.0f, 1.0f), 4.0f / std::sqrt(17.0f)));
        REQUIRE(is_value_approx(tensor::score(tensor::measure::klt, 2.5f, 1.5f, 2.5f), 1.0f));
        REQUIRE(is_value_approx(tensor::score(tensor::measure::rohr, 2.5f, 1.5f, 2.5f), 4.0f));
    }
    {
        REQUIRE(is_value_approx(tensor::score(tensor::measure::klt, 4.0f, 0.0f, 0.0f), 0.0f));
        REQUIRE(is_value_approx(tensor::score(tensor::measure::forstner, 4.0f, 0.0f, 0.0f), 0.0f));
        REQUIRE(tensor::score(tensor::measure::harris, 4.0f, 0.0f, 0.0f) < 0.0f);
        REQUIRE(is_value_approx(tensor::score(tensor::measure::rohr, 4.0f, 0.0f, 0.0f), 0.0f));
        REQUIRE(is_value_approx(tensor::score(tensor::measure::kenney, 4.0f, 0.0f, 0.0f), 0.0f));
        REQUIRE(is_value_approx(tensor::score(tensor::measure::forstner, 0.0f, 0.0f, 0.0f), 0.0f));
        REQUIRE(is_value_approx(tensor::score(tensor::measure::kenney, 0.0f, 0.0f, 0.0f), 0.0f));
    }
    {
        REQUIRE(is_value_approx(tensor::strength(tensor::measure::rohr, 4.0f), 2.0f));
        REQUIRE(is_value_approx(tensor::strength(tensor::measure::harris, 9.0f), 3.0f));
        REQUIRE(is_value_approx(tensor::strength(tensor::measure::harris, -1.0f), 0.0f));
        REQUIRE(is_value_approx(tensor::strength(tensor::measure::klt, 4.0f), 4.0f));
        REQUIRE(is_value_approx(tensor::strength(tensor::measure::forstner, 4.0f), 4.0f));
        REQUIRE(is_value_approx(tensor::strength(tensor::measure::kenney, 4.0f), 4.0f));
    }
    {
        REQUIRE(tensor::smoothing_radius(1.0f) == 2);
        REQUIRE(tensor::margin(1.0f) == 3);
        float weights[51];
        tensor::smoothing_weights(1.0f, weights);
        REQUIRE(is_value_approx(weights[0], 1.0f / 16.0f));
        REQUIRE(is_value_approx(weights[1], 4.0f / 16.0f));
        REQUIRE(is_value_approx(weights[2], 6.0f / 16.0f));
        REQUIRE(is_value_approx(weights[3], 4.0f / 16.0f));
        REQUIRE(is_value_approx(weights[4], 1.0f / 16.0f));
        for (float sigma : { 0.5f, 1.5f, 2.5f, tensor::sigma_maximum }) {
            const int radius = tensor::smoothing_radius(sigma);
            REQUIRE((radius >= 1) && (radius <= 25));
            tensor::smoothing_weights(sigma, weights);
            float sum = 0.0f;
            for (int k = 0; k <= 2 * radius; ++k) {
                REQUIRE(weights[k] > 0.0f);
                REQUIRE(is_value_approx(weights[k], weights[2 * radius - k], 1e-7f));
                sum += weights[k];
            }
            REQUIRE(is_value_approx(sum, 1.0f));
        }
    }
    {
        constexpr static const int size = 64;
        std::vector<unsigned char> flat(static_cast<size_t>(size * size), 100);
        std::vector<float> response(static_cast<size_t>(size * size));
        tensor::respond(flat.data(), size, size, size, tensor::measure::harris, 1.5f, response.data());
        for (const float value : response) {
            REQUIRE(value == 0.0f);
        }
        std::vector<unsigned char> saddle(static_cast<size_t>(size * size));
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                saddle[static_cast<size_t>(y * size + x)] = static_cast<unsigned char>(((x < 32) == (y < 32)) ? 255 : 0);
            }
        }
        for (tensor::measure kind : { tensor::measure::klt, tensor::measure::forstner, tensor::measure::harris, tensor::measure::rohr, tensor::measure::kenney }) {
            tensor::respond(saddle.data(), size, size, size, kind, 1.5f, response.data());
            const int border = tensor::margin(1.5f);
            int best_x = 0;
            int best_y = 0;
            float best = 0.0f;
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    const float value = response[static_cast<size_t>(y * size + x)];
                    if ((x < border) || (y < border) || (x >= size - border) || (y >= size - border)) {
                        REQUIRE(value == 0.0f);
                    }
                    if (value > best) {
                        best = value;
                        best_x = x;
                        best_y = y;
                    }
                }
            }
            REQUIRE(best > 0.0f);
            REQUIRE((best_x == 31) || (best_x == 32));
            REQUIRE((best_y == 31) || (best_y == 32));
        }
    }

    return EXIT_SUCCESS;
}
