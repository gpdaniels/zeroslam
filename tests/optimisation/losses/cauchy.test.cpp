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

#include "optimisation/losses/cauchy.hpp"

#include "optimisation/loss.hpp"

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

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-9) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

static inline double numerical_derivative(const optimisation::loss& loss, double error_squared) {
    math::matrix<double, 3, 1> forward;
    math::matrix<double, 3, 1> backward;
    const double step = 1e-6;
    loss.compute(error_squared + step, forward);
    loss.compute(error_squared - step, backward);
    return (forward[0] - backward[0]) / (2.0 * step);
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const double delta = 2.0;
    const optimisation::loss loss{ optimisation::losses::cauchy(delta) };

    {
        math::matrix<double, 3, 1> rho;
        loss.compute(0.0, rho);
        REQUIRE(rho[0] == 0.0);
        REQUIRE(rho[1] == 1.0);
        REQUIRE(rho[2] < 0.0);
    }

    for (int i = 1; i < 10; ++i) {
        const double error_squared = 0.7 * static_cast<double>(i);
        math::matrix<double, 3, 1> rho;
        loss.compute(error_squared, rho);
        REQUIRE(is_value_approx(rho[0], (delta * delta) * std::log(1.0 + (error_squared / (delta * delta)))));
        REQUIRE(is_value_approx(rho[1], numerical_derivative(loss, error_squared), 1e-5));
        REQUIRE(rho[1] > 0.0);
        REQUIRE(rho[1] < 1.0);
        REQUIRE(rho[2] < 0.0);
    }

    return EXIT_SUCCESS;
}
