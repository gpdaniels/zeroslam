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

#include "optimisation/vertices/similarity.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
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

    optimisation::vertex vertex{ optimisation::vertices::similarity() };
    REQUIRE(vertex.get_parameter_dimensions() == 8);
    REQUIRE(vertex.get_local_dimensions() == 7);
    const double initial[8] = { 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
    REQUIRE(vertex.set_parameters(&initial[0], 8));

    const double translation_delta[7] = { 0.0, 0.0, 0.0, 0.5, -1.0, 0.25, 0.0 };
    vertex.plus(&translation_delta[0]);
    REQUIRE(is_value_approx(vertex.get_parameters()[0], 1.5));
    REQUIRE(is_value_approx(vertex.get_parameters()[1], 1.0));
    REQUIRE(is_value_approx(vertex.get_parameters()[2], 3.25));
    REQUIRE(is_value_approx(vertex.get_parameters()[6], 1.0));
    REQUIRE(is_value_approx(vertex.get_parameters()[7], 1.0));

    const double scale_delta[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::log(2.0) };
    vertex.plus(&scale_delta[0]);
    REQUIRE(is_value_approx(vertex.get_parameters()[0], 3.0));
    REQUIRE(is_value_approx(vertex.get_parameters()[1], 2.0));
    REQUIRE(is_value_approx(vertex.get_parameters()[2], 6.5));
    REQUIRE(is_value_approx(vertex.get_parameters()[6], 1.0));
    REQUIRE(is_value_approx(vertex.get_parameters()[7], 2.0));

    const math::matrix<double, 7, 1> delta{ { 0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.1 } };
    const math::so3<double> current_rotation(vertex.get_parameters()[6], vertex.get_parameters()[3], vertex.get_parameters()[4], vertex.get_parameters()[5]);
    const math::matrix<double, 3, 1> current_translation{ { vertex.get_parameters()[0], vertex.get_parameters()[1], vertex.get_parameters()[2] } };
    const math::sim3<double> expected = math::sim3<double>::exp(delta) * math::sim3<double>(math::se3<double>(current_rotation, current_translation), vertex.get_parameters()[7]);
    vertex.plus(delta.data());
    for (size_t i = 0; i < 3; ++i) {
        REQUIRE(is_value_approx(vertex.get_parameters()[i], expected.transformation().translation()[i]));
    }
    REQUIRE(is_value_approx(vertex.get_parameters()[3], expected.transformation().rotation().get_quaternion()[1]));
    REQUIRE(is_value_approx(vertex.get_parameters()[4], expected.transformation().rotation().get_quaternion()[2]));
    REQUIRE(is_value_approx(vertex.get_parameters()[5], expected.transformation().rotation().get_quaternion()[3]));
    REQUIRE(is_value_approx(vertex.get_parameters()[6], expected.transformation().rotation().get_quaternion()[0]));
    REQUIRE(is_value_approx(vertex.get_parameters()[7], expected.scale()));
    const double quaternion_norm = std::sqrt((vertex.get_parameters()[3] * vertex.get_parameters()[3]) + (vertex.get_parameters()[4] * vertex.get_parameters()[4]) + (vertex.get_parameters()[5] * vertex.get_parameters()[5]) + (vertex.get_parameters()[6] * vertex.get_parameters()[6]));
    REQUIRE(is_value_approx(quaternion_norm, 1.0));
    REQUIRE(vertex.get_parameters()[7] > 0.0);

    return EXIT_SUCCESS;
}
