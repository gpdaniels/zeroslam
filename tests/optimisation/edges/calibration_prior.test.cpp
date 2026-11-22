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

#include "optimisation/edges/calibration_prior.hpp"

#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/calibration.hpp"

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

    static_assert(sizeof(optimisation::edges::calibration_prior) <= optimisation::edge::maximum_size, "The calibration prior edge must fit.");

    const double values[4] = { 0.3, 2.7, 1.0, 0.02 };
    const double prior[4] = { 0.31, 2.5, 1.1, 0.0 };
    const double deviations[4] = { 0.01, 0.0, 0.05, 0.1 };

    // The information holds a prior only where a deviation is given, and the observation the prior values.
    {
        const math::matrix<double, 0, 0> information = optimisation::edges::calibration_prior::information(&deviations[0], 4);
        REQUIRE((information.rows() == 8) && (information.cols() == 8));
        REQUIRE(std::abs(information[0][0] - 10000.0) < 1e-9);
        REQUIRE(information[1][1] == 0.0);
        REQUIRE(std::abs(information[3][3] - 100.0) < 1e-9);
        REQUIRE(information[5][5] == 0.0);
        REQUIRE(information[0][3] == 0.0);
        const math::matrix<double, 0, 0> observation = optimisation::edges::calibration_prior::observation(&prior[0], 4);
        REQUIRE((observation.rows() == 8) && (observation[1][0] == 2.5) && (observation[6][0] == 0.0));
    }

    // The residual of each free value is its difference from the prior, of each held value zero, and the Jacobian picks the free
    // values' local dimensions.
    {
        optimisation::vertex vertex = optimisation::vertices::calibration::make(&values[0], 4, 0xDu);
        optimisation::edge edge{ optimisation::edges::calibration_prior() };
        edge.set_observation(optimisation::edges::calibration_prior::observation(&prior[0], 4));
        edge.set_information(optimisation::edges::calibration_prior::information(&deviations[0], 4));
        REQUIRE(edge.add_vertex(&vertex));
        edge.compute_residual();
        REQUIRE(std::abs(edge.get_residual()[0][0] - (0.3 - 0.31)) < 1e-15);
        REQUIRE(edge.get_residual()[1][0] == 0.0);
        REQUIRE(std::abs(edge.get_residual()[2][0] - (1.0 - 1.1)) < 1e-15);
        REQUIRE(std::abs(edge.get_residual()[3][0] - 0.02) < 1e-15);
        REQUIRE(edge.get_residual()[4][0] == 0.0);
        edge.compute_jacobians();
        const math::matrix<double, 0, 0>& jacobian = edge.get_jacobians()[0];
        REQUIRE((jacobian.rows() == 8) && (jacobian.cols() == 3));
        for (size_t r = 0; r < 8; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                const bool one = ((r == 0) && (c == 0)) || ((r == 2) && (c == 1)) || ((r == 3) && (c == 2));
                REQUIRE(jacobian[r][c] == (one ? 1.0 : 0.0));
            }
        }
    }

    // In a graph the prior alone draws the free values to their priors and leaves the held ones.
    {
        optimisation::factor_graph graph;
        optimisation::vertex* const vertex = graph.add_vertex(optimisation::vertices::calibration::make(&values[0], 4, 0x9u));
        optimisation::edge edge{ optimisation::edges::calibration_prior() };
        edge.set_observation(optimisation::edges::calibration_prior::observation(&prior[0], 4));
        edge.set_information(optimisation::edges::calibration_prior::information(&deviations[0], 4));
        REQUIRE(edge.add_vertex(vertex));
        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(edge)) != nullptr);
        REQUIRE(graph.solve(10) > 0);
        REQUIRE(std::abs(vertex->get_parameters()[0] - 0.31) < 1e-6);
        REQUIRE(vertex->get_parameters()[1] == 2.7);
        REQUIRE(vertex->get_parameters()[2] == 1.0);
        REQUIRE(std::abs(vertex->get_parameters()[3]) < 1e-6);
        REQUIRE(graph.get_current_chi() < 1e-6);
    }

    return EXIT_SUCCESS;
}
