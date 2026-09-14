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

#include "estimation/minimal/similarity_3_point.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"

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
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

static void map_points(const math::matrix<double, 3, 3>& rotation, const double* translation, double scale, const double* source, size_t count, double* target) {
    for (size_t i = 0; i < count; ++i) {
        for (size_t row = 0; row < 3; ++row) {
            target[(i * 3) + row] = (scale * ((rotation[row][0] * source[(i * 3) + 0]) + (rotation[row][1] * source[(i * 3) + 1]) + (rotation[row][2] * source[(i * 3) + 2]))) + translation[row];
        }
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const double source[7][3] = {
        { 0.0, 0.0, 0.0 },
        { 1.0, 0.0, 0.0 },
        { 0.0, 2.0, 0.0 },
        { 0.0, 0.0, 3.0 },
        { -1.5, 2.5, -0.5 },
        { 4.0, -1.0, 2.0 },
        { -2.0, -3.0, 1.5 }
    };
    const double truth_scale = 2.5;
    const math::matrix<double, 3, 3> truth_rotation = math::so3<double>::exp({ { 0.4, -0.9, 0.3 } }).get_matrix();
    const double truth_translation[3] = { 1.5, -3.0, 4.2 };
    double target[7 * 3];
    map_points(truth_rotation, truth_translation, truth_scale, &source[0][0], 7, target);

    const size_t counts[2] = { 3, 7 };
    for (size_t c = 0; c < 2; ++c) {
        double rotation[9];
        double translation[3];
        double scale = 0.0;
        REQUIRE(estimation::minimal::similarity_3_point<double>::solve(&source[0][0], target, counts[c], rotation, translation, scale));
        REQUIRE(is_value_approx(scale, truth_scale, 1e-9));
        for (size_t k = 0; k < 9; ++k) {
            REQUIRE(is_value_approx(rotation[k], truth_rotation[k / 3][k % 3], 1e-9));
        }
        for (size_t k = 0; k < 3; ++k) {
            REQUIRE(is_value_approx(translation[k], truth_translation[k], 1e-9));
        }
    }

    {
        const double mirrored[4][3] = {
            { 0.0, 0.0, 0.0 },
            { 1.0, 0.0, 0.0 },
            { 0.0, 1.0, 0.0 },
            { 0.0, 0.0, 1.0 }
        };
        const double mirrored_target[4][3] = {
            { 0.0, 0.0, 0.0 },
            { 1.0, 0.0, 0.0 },
            { 0.0, 1.0, 0.0 },
            { 0.0, 0.0, -1.0 }
        };
        double rotation[9];
        double translation[3];
        double scale = 0.0;
        REQUIRE(estimation::minimal::similarity_3_point<double>::solve(&mirrored[0][0], &mirrored_target[0][0], 4, rotation, translation, scale));
        const double determinant = rotation[0] * ((rotation[4] * rotation[8]) - (rotation[5] * rotation[7])) - rotation[1] * ((rotation[3] * rotation[8]) - (rotation[5] * rotation[6])) + rotation[2] * ((rotation[3] * rotation[7]) - (rotation[4] * rotation[6]));
        REQUIRE(is_value_approx(determinant, 1.0, 1e-9));
        REQUIRE(scale > 0.0);
    }

    {
        double noisy[7 * 3];
        for (size_t k = 0; k < 7 * 3; ++k) {
            noisy[k] = target[k] + ((static_cast<double>(k % 5) - 2.0) * 0.002);
        }
        double rotation[9];
        double translation[3];
        double scale = 0.0;
        REQUIRE(estimation::minimal::similarity_3_point<double>::solve(&source[0][0], noisy, 7, rotation, translation, scale));
        REQUIRE(is_value_approx(scale, truth_scale, 1e-2));
        for (size_t k = 0; k < 9; ++k) {
            REQUIRE(is_value_approx(rotation[k], truth_rotation[k / 3][k % 3], 1e-2));
        }
        for (size_t k = 0; k < 3; ++k) {
            REQUIRE(is_value_approx(translation[k], truth_translation[k], 5e-2));
        }
    }

    {
        double rotation[9];
        double translation[3];
        double scale = 0.0;
        REQUIRE(!estimation::minimal::similarity_3_point<double>::solve(&source[0][0], target, 2, rotation, translation, scale));
        const double coincident[4][3] = {
            { 5.0, -2.0, 7.0 },
            { 5.0, -2.0, 7.0 },
            { 5.0, -2.0, 7.0 },
            { 5.0, -2.0, 7.0 }
        };
        REQUIRE(!estimation::minimal::similarity_3_point<double>::solve(&coincident[0][0], target, 4, rotation, translation, scale));
        const double collinear[5][3] = {
            { 0.0, 0.0, 0.0 },
            { 1.0, 1.0, 1.0 },
            { 2.0, 2.0, 2.0 },
            { 3.0, 3.0, 3.0 },
            { 4.0, 4.0, 4.0 }
        };
        double collinear_target[5 * 3];
        map_points(truth_rotation, truth_translation, truth_scale, &collinear[0][0], 5, collinear_target);
        REQUIRE(!estimation::minimal::similarity_3_point<double>::solve(&collinear[0][0], collinear_target, 5, rotation, translation, scale));
    }

    {
        float source_f[7 * 3];
        float target_f[7 * 3];
        for (size_t k = 0; k < 7 * 3; ++k) {
            source_f[k] = static_cast<float>(source[k / 3][k % 3]);
            target_f[k] = static_cast<float>(target[k]);
        }
        float rotation[9];
        float translation[3];
        float scale = 0.0f;
        REQUIRE(estimation::minimal::similarity_3_point<float>::solve(source_f, target_f, 7, rotation, translation, scale));
        REQUIRE(is_value_approx(static_cast<double>(scale), truth_scale, 1e-4));
        for (size_t k = 0; k < 3; ++k) {
            REQUIRE(is_value_approx(static_cast<double>(translation[k]), truth_translation[k], 1e-3));
        }
    }

    return EXIT_SUCCESS;
}
