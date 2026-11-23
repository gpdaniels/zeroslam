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

#include "optimisation/vertices/imu_bias.hpp"

#include "optimisation/vertex.hpp"
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-12) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    optimisation::vertex vertex{ optimisation::vertices::imu_bias() };
    REQUIRE(std::string_view(vertex.name()) == "imu_bias");
    REQUIRE(vertex.get_parameter_dimensions() == 6);
    REQUIRE(vertex.get_local_dimensions() == 6);
    const double initial[6] = { 0.01, -0.02, 0.03, 0.1, -0.2, 0.3 };
    REQUIRE(vertex.set_parameters(&initial[0], 6));
    const double delta[6] = { 0.001, 0.002, 0.003, -0.01, -0.02, -0.03 };
    vertex.plus(&delta[0]);
    for (size_t i = 0; i < 6; ++i) {
        REQUIRE(is_value_approx(vertex.get_parameters()[i], initial[i] + delta[i]));
    }
    vertex.backup();
    vertex.plus(&delta[0]);
    vertex.restore();
    for (size_t i = 0; i < 6; ++i) {
        REQUIRE(is_value_approx(vertex.get_parameters()[i], initial[i] + delta[i]));
    }

    return EXIT_SUCCESS;
}
