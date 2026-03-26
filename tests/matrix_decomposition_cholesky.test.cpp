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

#include "matrix_decomposition_cholesky.hpp"

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

template <typename array_type>
static inline bool are_values_approx(const array_type& lhs, const array_type& rhs, size_t length, double epsilon = 1e-8) {
    for (size_t index = 0; index < length; ++index) {
        if (!is_value_approx(lhs[index], rhs[index], epsilon)) {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // ...

    /*TEST(cholesky, function, solve_cholesky)*/ {
        const double matrix[6][6] = {
            { 0.001616, -0.000103, -0.001936, -0.000328, 0.004693, -0.000626 },
            { -0.000103, 1.496701, -0.513660, -2.005226, 0.523680, 2.010268 },
            { -0.001936, -0.513660, 0.512908, 1.027922, -0.522163, -0.523352 },
            { -0.000328, -2.005226, 1.027922, 3.034133, -1.043544, -2.523810 },
            { 0.004693, 0.523680, -0.522163, -1.043544, 0.542792, 0.537967 },
            { -0.000626, 2.010268, -0.523352, -2.523810, 0.537967, 3.034290 }
        };
        const double rhs[6] = {
            0.000004,
            -0.000003,
            -0.000005,
            0.000002,
            0.000013,
            -0.000006
        };
        const double expected[6] = {
            0.002200,
            0.000019,
            0.000168,
            0.000018,
            0.000187,
            -0.000003
        };
        double result[6] = { 0, 0, 0, 0, 0, 0 };
        double decomposed[6][6] = {
            { 0, 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0, 0 }
        };
        matrix::decompose_cholesky(&matrix[0][0], 6, 6, &decomposed[0][0]);
        matrix::solve_cholesky(&decomposed[0][0], &rhs[0], 6, 6, &result[0]);
        REQUIRE(are_values_approx(result, expected, 6, 1e-4));
    }
}