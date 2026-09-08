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

#include "estimation/robust/evaluate/inlier_support.hpp"

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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const estimation::robust::evaluate::inlier_support support(1.0f);

    // Inliers are the residuals strictly below the threshold, in order, and the cost counts the outliers.
    {
        const float residuals[6] = { 0.5f, 1.0f, 0.25f, 3.0f, 0.0f, 0.999f };
        size_t inliers[6] = {};
        size_t inliers_size = 0;
        REQUIRE(support.evaluate(residuals, 6, inliers, inliers_size) == 2.0f);
        REQUIRE(inliers_size == 4);
        REQUIRE(inliers[0] == 0);
        REQUIRE(inliers[1] == 2);
        REQUIRE(inliers[2] == 4);
        REQUIRE(inliers[3] == 5);
    }

    // No residuals gives no inliers and no cost.
    {
        size_t inliers[1] = {};
        size_t inliers_size = 7;
        REQUIRE(support.evaluate(nullptr, 0, inliers, inliers_size) == 0.0f);
        REQUIRE(inliers_size == 0);
    }

    // The cost does not depend on how far inside the threshold the inliers are.
    {
        const float close[3] = { 0.9f, 0.9f, 5.0f };
        const float far[3] = { 0.0f, 0.0f, 5.0f };
        size_t inliers[3] = {};
        size_t inliers_size = 0;
        REQUIRE(support.evaluate(close, 3, inliers, inliers_size) == support.evaluate(far, 3, inliers, inliers_size));
        REQUIRE(inliers_size == 2);
    }

    return EXIT_SUCCESS;
}
