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

#include "optimisation/edges/similarity_scale_prior.hpp"

#include "math/lie.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/relative_similarity.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/similarity.hpp"

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

namespace {
    void set_similarity(optimisation::vertex& vertex, const math::sim3<double>& similarity) {
        const double parameters[8] = { similarity.transformation().translation()[0], similarity.transformation().translation()[1], similarity.transformation().translation()[2], similarity.transformation().rotation().get_quaternion()[1], similarity.transformation().rotation().get_quaternion()[2], similarity.transformation().rotation().get_quaternion()[3], similarity.transformation().rotation().get_quaternion()[0], similarity.scale() };
        REQUIRE(vertex.set_parameters(&parameters[0], 8));
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const math::sim3<double> similarity(math::se3<double>(math::so3<double>::exp({ { 0.2, -0.1, 0.3 } }), { { 1.0, -2.0, 0.5 } }), 1.3);

    {
        // The residual is the difference of the logarithms of the scales, and the Jacobian matches central differences.
        optimisation::vertex vertex{ optimisation::vertices::similarity() };
        set_similarity(vertex, similarity);
        optimisation::edge edge{ optimisation::edges::similarity_scale_prior(1.0) };
        edge.add_vertex(&vertex);
        edge.compute_residual();
        REQUIRE(std::abs(edge.get_residual()[0][0] - std::log(1.3)) < 1.0e-12);
        edge.compute_jacobians();
        const math::matrix<double, 0, 0> jacobian = edge.get_jacobians()[0];
        REQUIRE((jacobian.rows() == 1) && (jacobian.cols() == 7));
        const double step = 1.0e-6;
        for (size_t column = 0; column < 7; ++column) {
            double delta[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            delta[column] = step;
            set_similarity(vertex, similarity);
            vertex.plus(&delta[0]);
            edge.compute_residual();
            const double forward = edge.get_residual()[0][0];
            delta[column] = -step;
            set_similarity(vertex, similarity);
            vertex.plus(&delta[0]);
            edge.compute_residual();
            const double backward = edge.get_residual()[0][0];
            const double numeric = (forward - backward) / (2.0 * step);
            REQUIRE(std::abs(numeric - jacobian[0][column]) < 1.0e-6);
        }
    }

    {
        // With a strong prior a relative measurement that rescales is taken as a rigid motion: the free vertex keeps the
        // measured rotation and its scale stays at one.
        optimisation::factor_graph graph;
        optimisation::vertex fixed_vertex{ optimisation::vertices::similarity() };
        set_similarity(fixed_vertex, math::sim3<double>(math::se3<double>(math::so3<double>::exp({ { 0.0, 0.0, 0.0 } }), { { 0.0, 0.0, 0.0 } }), 1.0));
        fixed_vertex.set_fixed(true);
        optimisation::vertex* const fixed_node = graph.add_vertex(static_cast<optimisation::vertex&&>(fixed_vertex));
        optimisation::vertex free_vertex{ optimisation::vertices::similarity() };
        set_similarity(free_vertex, math::sim3<double>(math::se3<double>(math::so3<double>::exp({ { 0.0, 0.0, 0.0 } }), { { 0.0, 0.0, 0.0 } }), 1.0));
        optimisation::vertex* const free_node = graph.add_vertex(static_cast<optimisation::vertex&&>(free_vertex));
        optimisation::edge relative{ optimisation::edges::relative_similarity() };
        const double measured[8] = { similarity.transformation().translation()[0], similarity.transformation().translation()[1], similarity.transformation().translation()[2], similarity.transformation().rotation().get_quaternion()[1], similarity.transformation().rotation().get_quaternion()[2], similarity.transformation().rotation().get_quaternion()[3], similarity.transformation().rotation().get_quaternion()[0], similarity.scale() };
        relative.set_observation(math::matrix<double, 0, 0>(8, 1, &measured[0]));
        relative.add_vertex(fixed_node);
        relative.add_vertex(free_node);
        graph.add_edge(static_cast<optimisation::edge&&>(relative));
        optimisation::edge prior{ optimisation::edges::similarity_scale_prior(1.0) };
        prior.set_information(math::matrix<double, 0, 0>::identity(1, 1) * 1.0e8);
        prior.add_vertex(free_node);
        graph.add_edge(static_cast<optimisation::edge&&>(prior));
        graph.solve(50, true);
        const double* const solved = free_node->get_parameters();
        REQUIRE(std::abs(solved[7] - 1.0) < 1.0e-6);
        const math::so3<double> rotation(solved[6], solved[3], solved[4], solved[5]);
        const math::matrix<double, 3, 1> difference = (rotation * similarity.transformation().rotation().inverse()).log();
        REQUIRE(std::sqrt(difference.get_length_squared()) < 1.0e-4);
    }

    return EXIT_SUCCESS;
}
