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

#include "optimisation/vertices/point.hpp"

#include "optimisation/vertex.hpp"

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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    optimisation::vertex vertex{ optimisation::vertices::point() };
    REQUIRE(vertex.get_parameter_dimensions() == 3);
    REQUIRE(vertex.get_local_dimensions() == 3);
    REQUIRE(vertex.get_local_dimensions() == 3);
    const double initial[3] = { 1.0, 2.0, 3.0 };
    REQUIRE(vertex.set_parameters(&initial[0], 3));
    const double delta[3] = { -1.0, 0.5, 0.0 };
    vertex.plus(&delta[0]);
    REQUIRE(is_value_approx(vertex.get_parameters()[0], 0.0));
    REQUIRE(is_value_approx(vertex.get_parameters()[1], 2.5));
    REQUIRE(is_value_approx(vertex.get_parameters()[2], 3.0));

    return EXIT_SUCCESS;
}
