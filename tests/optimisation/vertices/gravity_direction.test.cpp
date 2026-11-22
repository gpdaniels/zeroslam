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

#include "optimisation/vertices/gravity_direction.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
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

    optimisation::vertex vertex{ optimisation::vertices::gravity_direction() };
    REQUIRE(std::string_view(vertex.name()) == "gravity_direction");
    REQUIRE(vertex.get_parameter_dimensions() == 4);
    REQUIRE(vertex.get_local_dimensions() == 2);

    const math::so3<double> initial = math::so3<double>::exp({ { 0.3, -0.2, 1.1 } });
    const double parameters[4] = { initial.get_quaternion()[1], initial.get_quaternion()[2], initial.get_quaternion()[3], initial.get_quaternion()[0] };
    REQUIRE(vertex.set_parameters(&parameters[0], 4));
    const double delta[2] = { 0.05, -0.08 };
    vertex.plus(&delta[0]);
    const math::so3<double> expected = initial * math::so3<double>::exp({ { 0.05, -0.08, 0.0 } });
    const double* const updated = vertex.get_parameters();
    REQUIRE(is_value_approx(updated[0], expected.get_quaternion()[1]));
    REQUIRE(is_value_approx(updated[1], expected.get_quaternion()[2]));
    REQUIRE(is_value_approx(updated[2], expected.get_quaternion()[3]));
    REQUIRE(is_value_approx(updated[3], expected.get_quaternion()[0]));
    REQUIRE(is_value_approx((updated[0] * updated[0]) + (updated[1] * updated[1]) + (updated[2] * updated[2]) + (updated[3] * updated[3]), 1.0));

    // A turn about the gravity aligned frame's z axis does not move gravity, and the update cannot make one: the gravity of the
    // updated rotation is moved by the step, at first order by R_wg * (delta x (0, 0, -1)).
    const math::matrix<double, 3, 1> down = { { 0.0, 0.0, -1.0 } };
    const math::so3<double> turned = initial * math::so3<double>::exp({ { 0.0, 0.0, 0.7 } });
    const math::matrix<double, 3, 1> gravity = initial * down;
    const math::matrix<double, 3, 1> turned_gravity = turned * down;
    for (size_t i = 0; i < 3; ++i) {
        REQUIRE(is_value_approx(gravity[i], turned_gravity[i]));
    }
    const double small[2] = { 1e-7, -2e-7 };
    REQUIRE(vertex.set_parameters(&parameters[0], 4));
    vertex.plus(&small[0]);
    const math::so3<double> moved(vertex.get_parameters()[3], vertex.get_parameters()[0], vertex.get_parameters()[1], vertex.get_parameters()[2]);
    const math::matrix<double, 3, 1> moved_gravity = moved * down;
    const math::matrix<double, 3, 1> step = { { small[0], small[1], 0.0 } };
    const math::matrix<double, 3, 1> cross = { { step[1] * down[2] - step[2] * down[1], step[2] * down[0] - step[0] * down[2], step[0] * down[1] - step[1] * down[0] } };
    const math::matrix<double, 3, 1> predicted = gravity + initial * cross;
    for (size_t i = 0; i < 3; ++i) {
        REQUIRE(std::abs(moved_gravity[i] - predicted[i]) < 1e-12);
    }

    return EXIT_SUCCESS;
}
