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
                data[y][x] = static_cast<unsigned char>(x + y);
                if ((x > data_width / 4) && (x < (data_width * 3) / 4) && (y > data_height / 4) && (y < (data_height * 3) / 4)) {
                    data[y][x] = 0;
                }
            }
        }
        constexpr static const size_t features_count = 12;
        const size_t features[features_count][2] = {
            { 93, 17 },
            { 94, 17 },
            { 95, 17 },
            { 94, 18 },
            { 95, 18 },
            { 95, 19 },
            { 95, 45 },
            { 94, 46 },
            { 95, 46 },
            { 93, 47 },
            { 94, 47 },
            { 95, 47 }
        };
        const float angles_radians[features_count] = {
            -0.640070f,
            -0.576437f,
            -0.514927f,
            -0.543202f,
            -0.483588f,
            -0.448683f,
            +0.698690f,
            +0.785398f,
            +0.743421f,
            +0.872107f,
            +0.827376f,
            +0.785398f
        };
        for (size_t i = 0; i < features_count; ++i) {
            REQUIRE(is_value_approx(feature::angle::orb::dominant_angle(&data[features[i][1]][features[i][0]], data_width), angles_radians[i], 1e-5));
        }
    }

    return EXIT_SUCCESS;
}
