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

#include "estimation/robust/evaluator.hpp"

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

class all_inliers final
    : public estimation::robust::evaluator {
public:
    virtual float evaluate(const float* const __restrict residuals, const size_t residuals_size, size_t* const __restrict inliers, size_t& inliers_size) const override final {
        float cost = 0.0f;
        for (size_t i = 0; i < residuals_size; ++i) {
            inliers[i] = i;
            cost += residuals[i];
        }
        inliers_size = residuals_size;
        return cost;
    }
};

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        const all_inliers evaluator;
        const float residuals[3] = { 0.5f, 1.5f, 2.0f };
        size_t inliers[3] = {};
        size_t inliers_size = 0;
        REQUIRE(evaluator.evaluate(residuals, 3, inliers, inliers_size) == 4.0f);
        REQUIRE(inliers_size == 3);
        REQUIRE(inliers[2] == 2);
    }

    return EXIT_SUCCESS;
}
