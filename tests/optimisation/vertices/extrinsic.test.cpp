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

#include "optimisation/vertices/extrinsic.hpp"

#include "math/lie.hpp"
#include "optimisation/vertex.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

    optimisation::vertex vertex{ optimisation::vertices::extrinsic() };
    REQUIRE(vertex.get_parameter_dimensions() == 7);
    REQUIRE(vertex.get_local_dimensions() == 6);
    REQUIRE(std::strcmp(vertex.name(), "extrinsic") == 0);

    {
        // The parameters round trip a transform, the quaternion scalar last.
        const math::se3<double> body_from_sensor(math::so3<double>::exp({ { 0.1, -0.3, 0.2 } }), { { 0.11, -0.02, 0.005 } });
        double parameters[7];
        optimisation::vertices::extrinsic::to_parameters(body_from_sensor, parameters);
        REQUIRE(parameters[0] == 0.11);
        REQUIRE(parameters[6] == body_from_sensor.rotation().get_quaternion()[0]);
        const math::se3<double> recovered = optimisation::vertices::extrinsic::from_parameters(&parameters[0]);
        const math::matrix<double, 3, 3> rotation = recovered.rotation().get_matrix();
        const math::matrix<double, 3, 3> expected = body_from_sensor.rotation().get_matrix();
        for (size_t row = 0; row < 3; ++row) {
            REQUIRE(is_value_approx(recovered.translation()[row], body_from_sensor.translation()[row]));
            for (size_t column = 0; column < 3; ++column) {
                REQUIRE(is_value_approx(rotation[row][column], expected[row][column]));
            }
        }
        REQUIRE(vertex.set_parameters(&parameters[0], 7));
    }

    {
        // A step is taken on the left, in the body's frame.
        const math::matrix<double, 6, 1> delta{ { 0.05, 0.02, -0.04, 0.01, -0.03, 0.02 } };
        const math::se3<double> expected = math::se3<double>::exp(delta) * optimisation::vertices::extrinsic::from_parameters(vertex.get_parameters());
        vertex.plus(delta.data());
        const double* const parameters = vertex.get_parameters();
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(is_value_approx(parameters[i], expected.translation()[i]));
        }
        REQUIRE(is_value_approx(parameters[3], expected.rotation().get_quaternion()[1]));
        REQUIRE(is_value_approx(parameters[4], expected.rotation().get_quaternion()[2]));
        REQUIRE(is_value_approx(parameters[5], expected.rotation().get_quaternion()[3]));
        REQUIRE(is_value_approx(parameters[6], expected.rotation().get_quaternion()[0]));
        REQUIRE(is_value_approx((parameters[3] * parameters[3]) + (parameters[4] * parameters[4]) + (parameters[5] * parameters[5]) + (parameters[6] * parameters[6]), 1.0));
    }

    {
        // A pure translation step adds to the translation, and a zero step renormalises the quaternion.
        const double identity[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 + 1e-7 };
        REQUIRE(vertex.set_parameters(&identity[0], 7));
        const double translation_step[6] = { 0.0, 0.0, 0.0, 0.5, -1.0, 0.25 };
        vertex.plus(&translation_step[0]);
        const double* const parameters = vertex.get_parameters();
        REQUIRE(is_value_approx(parameters[0], 0.5));
        REQUIRE(is_value_approx(parameters[1], -1.0));
        REQUIRE(is_value_approx(parameters[2], 0.25));
        REQUIRE(std::abs(((parameters[3] * parameters[3]) + (parameters[4] * parameters[4]) + (parameters[5] * parameters[5]) + (parameters[6] * parameters[6])) - 1.0) < 1e-12);
    }

    return EXIT_SUCCESS;
}
