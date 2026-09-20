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

#include "optimisation/vertices/line.hpp"

#include "geometry/plucker.hpp"
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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    optimisation::vertex vertex{ optimisation::vertices::line() };
    REQUIRE(vertex.get_parameter_dimensions() == 6);
    REQUIRE(vertex.get_local_dimensions() == 4);
    geometry::plucker initial;
    REQUIRE(geometry::plucker::from_points({ { 1.0, 2.0, 3.0 } }, { { -2.0, 0.5, 4.0 } }, initial));
    const double parameters[6] = { initial.moment[0], initial.moment[1], initial.moment[2], initial.direction[0], initial.direction[1], initial.direction[2] };
    REQUIRE(vertex.set_parameters(&parameters[0], 6));

    const double zero[4] = { 0.0, 0.0, 0.0, 0.0 };
    vertex.plus(&zero[0]);
    for (size_t i = 0; i < 6; ++i) {
        REQUIRE(std::abs(vertex.get_parameters()[i] - parameters[i]) < 1.0e-12);
    }

    const double delta[4] = { 0.01, -0.02, 0.03, -0.01 };
    geometry::plucker expected = initial;
    REQUIRE(expected.oplus(delta[0], delta[1], delta[2], delta[3]));
    vertex.plus(&delta[0]);
    const double* const updated = vertex.get_parameters();
    for (size_t i = 0; i < 3; ++i) {
        REQUIRE(std::abs(updated[i] - expected.moment[i]) < 1.0e-12);
        REQUIRE(std::abs(updated[3 + i] - expected.direction[i]) < 1.0e-12);
    }
    const double direction_length_squared = (updated[3] * updated[3]) + (updated[4] * updated[4]) + (updated[5] * updated[5]);
    REQUIRE(std::abs(direction_length_squared - 1.0) < 1.0e-12);
    const double klein = (updated[0] * updated[3]) + (updated[1] * updated[4]) + (updated[2] * updated[5]);
    REQUIRE(std::abs(klein) < 1.0e-12);
    double moved = 0.0;
    for (size_t i = 0; i < 6; ++i) {
        moved += (updated[i] - parameters[i]) * (updated[i] - parameters[i]);
    }
    REQUIRE(moved > 1.0e-8);

    return EXIT_SUCCESS;
}
