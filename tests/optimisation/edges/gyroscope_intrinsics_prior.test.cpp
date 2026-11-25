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

#include "optimisation/edges/gyroscope_intrinsics_prior.hpp"

#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/gyroscope_intrinsics.hpp"

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

    static_assert(sizeof(optimisation::edges::gyroscope_intrinsics_prior) <= optimisation::edge::maximum_size, "The gyroscope intrinsics prior must fit.");

    // The information is the inverse of the prior's variance on each entry, the paper's 0.002 unless another is given.
    REQUIRE(optimisation::edges::gyroscope_intrinsics_prior::default_deviation == 0.002);
    const math::matrix<double, 0, 0> information = optimisation::edges::gyroscope_intrinsics_prior::information();
    REQUIRE(information.rows() == 9);
    REQUIRE(information.cols() == 9);
    const math::matrix<double, 0, 0> wide = optimisation::edges::gyroscope_intrinsics_prior::information(0.01);
    for (size_t r = 0; r < 9; ++r) {
        for (size_t c = 0; c < 9; ++c) {
            if (r != c) {
                REQUIRE(information[r][c] == 0.0);
            }
        }
        REQUIRE(is_value_approx(information[r][r], 250000.0));
        REQUIRE(is_value_approx(wide[r][r], 10000.0));
    }

    // The residual is the matrix itself and its jacobian the identity, the analytic one agreeing with central differences.
    optimisation::vertex vertex{ optimisation::vertices::gyroscope_intrinsics() };
    const double values[9] = { 0.011, -0.006, 0.004, 0.007, -0.009, -0.005, -0.003, 0.008, 0.012 };
    REQUIRE(vertex.set_parameters(&values[0], 9));
    optimisation::edge edge{ optimisation::edges::gyroscope_intrinsics_prior() };
    REQUIRE(edge.required_vertices() == 1);
    REQUIRE(edge.add_vertex(&vertex));
    edge.set_information(information);
    edge.compute_residual();
    edge.compute_jacobians();
    REQUIRE(edge.has_analytic_jacobians());
    REQUIRE(edge.get_residual().rows() == 9);
    REQUIRE(edge.get_jacobians().size() == 1);
    REQUIRE(edge.get_jacobians()[0].rows() == 9);
    REQUIRE(edge.get_jacobians()[0].cols() == 9);
    const double step = 1e-6;
    for (size_t i = 0; i < 9; ++i) {
        REQUIRE(edge.get_residual()[i][0] == values[i]);
        double delta[9] = {};
        delta[i] = step;
        vertex.plus(&delta[0]);
        edge.compute_residual();
        const math::matrix<double, 0, 0> forward = edge.get_residual();
        delta[i] = -2.0 * step;
        vertex.plus(&delta[0]);
        edge.compute_residual();
        const math::matrix<double, 0, 0> backward = edge.get_residual();
        delta[i] = step;
        vertex.plus(&delta[0]);
        edge.compute_residual();
        for (size_t j = 0; j < 9; ++j) {
            REQUIRE(edge.get_jacobians()[0][j][i] == ((i == j) ? 1.0 : 0.0));
            REQUIRE(std::abs(((forward[j][0] - backward[j][0]) / (2.0 * step)) - edge.get_jacobians()[0][j][i]) < 1e-9);
        }
    }
    // Each entry costs its square over the variance.
    double expected = 0.0;
    for (size_t i = 0; i < 9; ++i) {
        expected += edge.get_residual()[i][0] * edge.get_residual()[i][0] * 250000.0;
    }
    REQUIRE(is_value_approx(edge.chi2(), expected, 1e-9));

    // Alone, the prior draws the matrix to zero.
    {
        optimisation::factor_graph graph;
        optimisation::vertex* const node = graph.add_vertex(optimisation::vertex(vertex));
        optimisation::edge prior{ optimisation::edges::gyroscope_intrinsics_prior() };
        prior.set_information(information);
        REQUIRE(prior.add_vertex(node));
        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(prior)) != nullptr);
        REQUIRE(graph.solve(10) > 0);
        for (size_t i = 0; i < 9; ++i) {
            REQUIRE(std::abs(node->get_parameters()[i]) < 1e-9);
        }
    }

    return EXIT_SUCCESS;
}
