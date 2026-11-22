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

#include "optimisation/edges/bias_random_walk.hpp"

#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/imu_bias.hpp"
#include "sensor/imu/measurement.hpp"

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

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-12) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::bias_random_walk) <= optimisation::edge::maximum_size, "The bias random walk edge must fit.");

    // The information is the inverse of the walk's variance over the duration.
    sensor::imu::parameters parameters;
    const math::matrix<double, 0, 0> information = optimisation::edges::bias_random_walk::information(parameters, 0.25);
    REQUIRE(information.rows() == 6);
    REQUIRE(information.cols() == 6);
    for (size_t r = 0; r < 6; ++r) {
        for (size_t c = 0; c < 6; ++c) {
            if (r != c) {
                REQUIRE(information[r][c] == 0.0);
            }
        }
        REQUIRE(is_value_approx(information[r][r] * ((r < 3) ? parameters.gyroscope_walk_variance(0.25) : parameters.accelerometer_walk_variance(0.25)), 1.0));
    }

    // The residual is the change of the bias, and its jacobians are minus and plus the identity.
    optimisation::vertex first{ optimisation::vertices::imu_bias() };
    optimisation::vertex second{ optimisation::vertices::imu_bias() };
    const double first_values[6] = { 0.01, 0.02, 0.03, 0.1, 0.2, 0.3 };
    const double second_values[6] = { 0.015, 0.01, 0.03, 0.05, 0.25, 0.4 };
    REQUIRE(first.set_parameters(&first_values[0], 6));
    REQUIRE(second.set_parameters(&second_values[0], 6));
    optimisation::edge edge{ optimisation::edges::bias_random_walk() };
    REQUIRE(edge.add_vertex(&first));
    REQUIRE(edge.add_vertex(&second));
    edge.set_information(information);
    edge.compute_residual();
    edge.compute_jacobians();
    REQUIRE(edge.has_analytic_jacobians());
    for (size_t i = 0; i < 6; ++i) {
        REQUIRE(is_value_approx(edge.get_residual()[i][0], second_values[i] - first_values[i]));
        for (size_t j = 0; j < 6; ++j) {
            REQUIRE(edge.get_jacobians()[0][i][j] == ((i == j) ? -1.0 : 0.0));
            REQUIRE(edge.get_jacobians()[1][i][j] == ((i == j) ? 1.0 : 0.0));
        }
    }

    // Between a fixed bias and a free one, the free one is drawn onto the fixed one.
    {
        optimisation::factor_graph graph;
        optimisation::vertex fixed_bias = first;
        fixed_bias.set_fixed(true);
        optimisation::vertex* const anchor = graph.add_vertex(static_cast<optimisation::vertex&&>(fixed_bias));
        optimisation::vertex* const free_bias = graph.add_vertex(static_cast<optimisation::vertex&&>(second));
        optimisation::edge walk{ optimisation::edges::bias_random_walk() };
        walk.set_information(information);
        REQUIRE(walk.add_vertex(anchor));
        REQUIRE(walk.add_vertex(free_bias));
        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(walk)) != nullptr);
        REQUIRE(graph.solve(10) > 0);
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(std::abs(free_bias->get_parameters()[i] - first_values[i]) < 1e-9);
        }
    }

    return EXIT_SUCCESS;
}
