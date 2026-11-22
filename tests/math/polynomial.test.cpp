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

#include "math/polynomial.hpp"

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

    {
        // 2 - 3 x + x^2 = (x - 1) (x - 2).
        const double coefficients[3] = { 2.0, -3.0, 1.0 };
        REQUIRE(math::polynomial::evaluate(&coefficients[0], 2, 0.0) == 2.0);
        REQUIRE(math::polynomial::evaluate(&coefficients[0], 2, 1.0) == 0.0);
        REQUIRE(math::polynomial::evaluate(&coefficients[0], 2, 3.0) == 2.0);
        double roots[8];
        REQUIRE(math::polynomial::roots(&coefficients[0], 2, 0.0, 10.0, &roots[0]) == 2);
        REQUIRE(std::abs(roots[0] - 1.0) < 1e-15);
        REQUIRE(std::abs(roots[1] - 2.0) < 1e-15);
        REQUIRE(math::polynomial::roots(&coefficients[0], 2, 1.5, 10.0, &roots[0]) == 1);
        REQUIRE(std::abs(roots[0] - 2.0) < 1e-15);
        REQUIRE(std::abs(math::polynomial::first_non_positive(&coefficients[0], 2, 10.0) - 1.0) < 1e-15);
        REQUIRE(std::isinf(math::polynomial::first_non_positive(&coefficients[0], 2, 0.5)));
    }

    {
        // Roots found by bisection are within a unit of rounding: 1 + x - 7 x^3 + x^6 has a root near 0.6.
        const double coefficients[7] = { 1.0, 1.0, 0.0, -7.0, 0.0, 0.0, 1.0 };
        double roots[8];
        const size_t count = math::polynomial::roots(&coefficients[0], 6, 0.0, 10.0, &roots[0]);
        REQUIRE(count == 2);
        for (size_t index = 0; index < count; ++index) {
            const double below = math::polynomial::evaluate(&coefficients[0], 6, std::nextafter(roots[index], 0.0));
            const double above = math::polynomial::evaluate(&coefficients[0], 6, roots[index]);
            REQUIRE((below > 0.0) != (above > 0.0) || (above == 0.0));
        }
        REQUIRE(roots[0] < roots[1]);
    }

    {
        // A polynomial that never reaches zero, a constant, trailing zero coefficients, and a degree beyond the maximum.
        const double positive[5] = { 1.0, 0.0, 2.0, 0.0, 0.0 };
        double roots[8];
        REQUIRE(math::polynomial::roots(&positive[0], 4, 0.0, 100.0, &roots[0]) == 0);
        REQUIRE(std::isinf(math::polynomial::first_non_positive(&positive[0], 4, 100.0)));
        const double linear[4] = { 1.0, -0.5, 0.0, 0.0 };
        REQUIRE(std::abs(math::polynomial::first_non_positive(&linear[0], 3, 100.0) - 2.0) < 1e-14);
        const double high[10] = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0 };
        REQUIRE(math::polynomial::roots(&high[0], 9, 0.0, 2.0, &roots[0]) == 0);
        const double eighth[9] = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0 };
        REQUIRE(math::polynomial::roots(&eighth[0], 8, 0.0, 2.0, &roots[0]) == 1);
        REQUIRE(std::abs(roots[0] - 1.0) < 1e-15);
    }

    return EXIT_SUCCESS;
}
