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

#include "estimation/robust/evaluate/least_median_of_squares.hpp"

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

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-6) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const estimation::robust::evaluate::least_median_of_squares evaluator(2);

    // The cost is the median squared residual, for odd and even counts.
    {
        const float odd[5] = { 3.0f, 1.0f, 2.0f, 10.0f, 0.5f };
        size_t odd_inliers[5] = {};
        size_t odd_inliers_size = 0;
        REQUIRE(is_value_approx(static_cast<double>(evaluator.evaluate(odd, 5, odd_inliers, odd_inliers_size)), 4.0));
        const float even[6] = { 3.0f, 1.0f, 2.0f, 10.0f, 0.5f, 4.0f };
        size_t even_inliers[6] = {};
        size_t even_inliers_size = 0;
        REQUIRE(is_value_approx(static_cast<double>(evaluator.evaluate(even, 6, even_inliers, even_inliers_size)), 0.5 * (4.0 + 9.0)));
    }

    // Gross outliers fall outside the robust band without any threshold.
    {
        float residuals[20];
        for (size_t i = 0; i < 20; ++i) {
            residuals[i] = 0.01f * static_cast<float>(i % 5);
        }
        residuals[3] = 50.0f;
        residuals[11] = -40.0f;
        size_t inliers[20] = {};
        size_t inliers_size = 0;
        const float cost = evaluator.evaluate(residuals, 20, inliers, inliers_size);
        REQUIRE(cost < 0.01f);
        REQUIRE(inliers_size == 18);
        for (size_t i = 0; i < inliers_size; ++i) {
            REQUIRE(inliers[i] != 3);
            REQUIRE(inliers[i] != 11);
        }
    }

    // The input residuals are left untouched.
    {
        const float residuals[4] = { 4.0f, 3.0f, 2.0f, 1.0f };
        size_t inliers[4] = {};
        size_t inliers_size = 0;
        static_cast<void>(evaluator.evaluate(residuals, 4, inliers, inliers_size));
        REQUIRE(residuals[0] == 4.0f);
        REQUIRE(residuals[3] == 1.0f);
    }

    return EXIT_SUCCESS;
}
