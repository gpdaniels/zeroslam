/*
Copyright (C) 2025 Geoffrey Daniels. https://gpdaniels.com/

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

#include "matrix_decomposition_lower_upper.hpp"

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

static inline bool is_value_equal(double lhs, double rhs) {
    return lhs == rhs;
}

void matrix_multiply(const double* lhs, int lhs_width, int lhs_height, const double* rhs, int rhs_width, int rhs_height, double* result);

void matrix_multiply(const double* lhs, int lhs_width, int lhs_height, const double* rhs, int rhs_width, int rhs_height, double* result) {
    REQUIRE(lhs_width == rhs_height);
    for (int lhs_y = 0; lhs_y < lhs_height; ++lhs_y) {
        for (int rhs_x = 0; rhs_x < rhs_width; ++rhs_x) {
            double sum = 0;
            for (int lhs_x_rhs_y = 0; lhs_x_rhs_y < lhs_width; ++lhs_x_rhs_y) {
                sum += lhs[lhs_y * lhs_width + lhs_x_rhs_y] * rhs[lhs_x_rhs_y * rhs_width + rhs_x];
            }
            result[lhs_y * rhs_width + rhs_x] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        using test_type = double;
        constexpr static const int width = 2;
        constexpr static const int height = 2;

        const test_type matrix[height][width] = {
            { 4.0, 3.0 },
            { 6.0, 3.0 }
        };

        test_type L1[height][height];
        test_type U1[height][width];
        test_type P1[height][height];
        int swaps;
        REQUIRE(matrix::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps));

        test_type PA[height][width];
        matrix_multiply(&P1[0][0], height, height, &matrix[0][0], width, height, &PA[0][0]);

        test_type LU[height][width];
        matrix_multiply(&L1[0][0], height, height, &U1[0][0], width, height, &LU[0][0]);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_equal(PA[i][j], LU[i][j]));
            }
        }
    }

    {
        using test_type = double;
        constexpr static const int width = 2;
        constexpr static const int height = 3;

        const test_type matrix[height][width] = {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        };

        test_type L1[height][height];
        test_type U1[height][width];
        test_type P1[height][height];
        int swaps;
        REQUIRE(matrix::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps));

        test_type PA[height][width];
        matrix_multiply(&P1[0][0], height, height, &matrix[0][0], width, height, &PA[0][0]);

        test_type LU[height][width];
        matrix_multiply(&L1[0][0], height, height, &U1[0][0], width, height, &LU[0][0]);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_equal(PA[i][j], LU[i][j]));
            }
        }
    }

    {
        using test_type = double;
        constexpr static const int width = 4;
        constexpr static const int height = 2;

        const test_type matrix[height][width] = {
            { 1.0, 2.0, 3.0, 4.0 },
            { 4.0, 5.0, 6.0, 7.0 }
        };

        test_type L1[height][height];
        test_type U1[height][width];
        test_type P1[height][height];
        int swaps;
        REQUIRE(matrix::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps));

        test_type PA[height][width];
        matrix_multiply(&P1[0][0], height, height, &matrix[0][0], width, height, &PA[0][0]);

        test_type LU[height][width];
        matrix_multiply(&L1[0][0], height, height, &U1[0][0], width, height, &LU[0][0]);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_equal(PA[i][j], LU[i][j]));
            }
        }
    }

    {
        const float A[4][4] = {
            { -0.0f, 1.0f, 1.0f, -0.0f },
            { 2.0f, 2.0f, -2.0f, 3.0f },
            { 1.0f, 2.0f, -3.0f, 4.0f },
            { 1.0f, -1.0f, 1.0f, -2.0f }
        };

        const float B[4] = {
            -1.0f,
            10.0f,
            12.0f,
            -4.0f
        };

        float result[4];
        REQUIRE(matrix::solve_lower_upper<float>(&A[0][0], &B[0], 4, 4, &result[0]));

        REQUIRE(is_value_approx(result[0], 1.0f, 1e-6f));
        REQUIRE(is_value_approx(result[1], 0.0f, 1e-6f));
        REQUIRE(is_value_approx(result[2], -1.0f, 1e-6f));
        REQUIRE(is_value_approx(result[3], 2.0f, 1e-6f));

        float A2[2][2] = {
            { 0.0f, 2.0f },
            { 0.0f, 1.0f },
        };

        float B2[2] = {
            8.0f,
            4.0f,
        };

        float result2[2];
        REQUIRE(matrix::solve_lower_upper<float>(&A2[0][0], &B2[0], 2, 2, &result2[0]) == false);

        float A3[2][2] = {
            { 1.0f, 2.0f },
            { 3.0f, 1.0f },
        };

        float B3[2] = {
            8.0f,
            4.0f,
        };

        float result3[2];
        REQUIRE(matrix::solve_lower_upper<float>(&A3[0][0], &B3[0], 2, 2, &result3[0]));

        REQUIRE(is_value_approx(result3[0], 0.0f, 1e-6f));
        REQUIRE(is_value_approx(result3[1], 4.0f, 1e-6f));
    }

    return EXIT_SUCCESS;
}