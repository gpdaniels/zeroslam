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

#include "optimisation/vertices/gyroscope_intrinsics.hpp"

#include "optimisation/vertex.hpp"
#include "sensor/imu/measurement.hpp"

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

    optimisation::vertex vertex{ optimisation::vertices::gyroscope_intrinsics() };
    REQUIRE(vertex.is_valid());
    REQUIRE(std::string_view(vertex.name()) == "gyroscope_intrinsics");
    REQUIRE(vertex.get_parameter_dimensions() == 9);
    REQUIRE(vertex.get_local_dimensions() == 9);
    // A new vertex holds no correction.
    for (size_t i = 0; i < 9; ++i) {
        REQUIRE(vertex.get_parameters()[i] == 0.0);
    }

    // The parameters are the matrix row by row, and round trip.
    sensor::imu::intrinsics intrinsics;
    intrinsics.gyroscope = { { { 0.011, -0.006, 0.004 }, { 0.007, -0.009, -0.005 }, { -0.003, 0.008, 0.012 } } };
    double parameters[optimisation::vertices::gyroscope_intrinsics::parameter_count];
    optimisation::vertices::gyroscope_intrinsics::to_parameters(intrinsics, parameters);
    REQUIRE(parameters[0] == 0.011);
    REQUIRE(parameters[1] == -0.006);
    REQUIRE(parameters[3] == 0.007);
    REQUIRE(parameters[8] == 0.012);
    REQUIRE(optimisation::vertices::gyroscope_intrinsics::from_parameters(&parameters[0]).gyroscope == intrinsics.gyroscope);
    REQUIRE(vertex.set_parameters(&parameters[0], 9));
    REQUIRE(!vertex.set_parameters(&parameters[0], 8));

    // The update adds to each entry.
    const double delta[9] = { 0.001, 0.002, 0.003, -0.001, -0.002, -0.003, 0.0005, -0.0005, 0.0 };
    vertex.plus(&delta[0]);
    for (size_t i = 0; i < 9; ++i) {
        REQUIRE(is_value_approx(vertex.get_parameters()[i], parameters[i] + delta[i]));
    }
    vertex.backup();
    vertex.plus(&delta[0]);
    vertex.restore();
    for (size_t i = 0; i < 9; ++i) {
        REQUIRE(is_value_approx(vertex.get_parameters()[i], parameters[i] + delta[i]));
    }
    const sensor::imu::intrinsics moved = optimisation::vertices::gyroscope_intrinsics::from_parameters(vertex.get_parameters());
    REQUIRE(is_value_approx(moved.gyroscope[1][2], intrinsics.gyroscope[1][2] + delta[5]));
    REQUIRE(is_value_approx(moved.gyroscope[2][0], intrinsics.gyroscope[2][0] + delta[6]));

    return EXIT_SUCCESS;
}
