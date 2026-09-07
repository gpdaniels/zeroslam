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

#include "feature/refiner/subpixel.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-8) {
    if (std::isnan(lhs) && std::isnan(rhs))
        return true;
    if (std::isnan(lhs) != std::isnan(rhs))
        return false;
    if (std::isinf(lhs) != std::isinf(rhs))
        return false;
    if (std::signbit(lhs + epsilon) != std::signbit(rhs + epsilon))
        return false;
    if (std::isinf(lhs) && std::isinf(rhs))
        return true;
    return (std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon);
}

static inline bool is_value_approx(float lhs, float rhs, double epsilon = 1e-8) {
    return is_value_approx(static_cast<double>(lhs), static_cast<double>(rhs), epsilon);
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        constexpr static const size_t data_width = 128;
        constexpr static const size_t data_height = 64;
        unsigned char data[data_height][data_width] = {};
        for (size_t y = 0; y < data_height; ++y) {
            for (size_t x = 0; x < data_width; ++x) {
                if ((x >= data_width / 2) || (y >= data_height / 2)) {
                    if ((x == data_width / 2) || (y == data_height / 2)) {
                        data[y][x] = 128;
                    }
                    else {
                        data[y][x] = 255;
                    }
                }
                else {
                    data[y][x] = 0;
                }
            }
        }
        for (int y = -2; y <= 2; ++y) {
            for (int x = -2; x <= 2; ++x) {
                float offset_x = 0;
                float offset_y = 0;
                REQUIRE(feature::refiner::subpixel::refine(&data[static_cast<size_t>(static_cast<int>(data_height) / 2 + y)][static_cast<size_t>(static_cast<int>(data_width) / 2 + x)], data_width, offset_x, offset_y));
                REQUIRE(is_value_approx(offset_x, static_cast<float>(-x), 1e-2));
                REQUIRE(is_value_approx(offset_y, static_cast<float>(-y), 1e-2));
            }
        }
        {
            // Refinements that converge more than four pixels from the detected feature must be rejected by the acceptance gate.
            float offset_x = 0;
            float offset_y = 0;
            REQUIRE(!feature::refiner::subpixel::refine(&data[data_height / 2 + 4][data_width / 2 + 4], data_width, offset_x, offset_y));
            REQUIRE(!feature::refiner::subpixel::refine(&data[data_height / 2 + 6][data_width / 2 + 6], data_width, offset_x, offset_y));
            REQUIRE(feature::refiner::subpixel::refine(&data[data_height / 2 - 4][data_width / 2 - 4], data_width, offset_x, offset_y));
            REQUIRE(is_value_approx(offset_x, +4.0f, 1e-2));
            REQUIRE(is_value_approx(offset_y, +4.0f, 1e-2));
        }
    }

    return EXIT_SUCCESS;
}
