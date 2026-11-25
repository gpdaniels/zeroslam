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

#include "optimisation/edges/intrinsics_ratio_prior.hpp"

#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/intrinsics_prior.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/intrinsics.hpp"
#include "sensor/camera/model.hpp"
#include "sensor/camera/pinhole.hpp"

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

    static_assert(sizeof(optimisation::edges::intrinsics_ratio_prior) <= optimisation::edge::maximum_size, "The ratio prior edge must fit.");

    const double initial[4] = { 400.0, 410.0, 200.0, 150.0 };
    const sensor::camera::model<double> camera{ sensor::camera::pinhole<double>(&initial[0], 4) };
    const bool focal_free[4] = { true, true, false, false };
    const bool first_free[4] = { true, false, false, false };
    const int focal_together[4] = { 0, 0, -1, -1 };

    {
        // The residual is the first parameter less the ratio times the second: zero at the ratio, and in the first's units.
        optimisation::vertex vertex = optimisation::vertices::intrinsics::make(camera, &focal_free[0], 4);
        REQUIRE(vertex.is_valid());
        optimisation::edge at_ratio{ optimisation::edges::intrinsics_ratio_prior(0, 1, 400.0 / 410.0) };
        REQUIRE(at_ratio.required_vertices() == 1);
        REQUIRE(at_ratio.add_vertex(&vertex));
        at_ratio.compute_residual();
        REQUIRE(std::abs(at_ratio.get_residual()[0][0]) < 1e-12);
        optimisation::edge square{ optimisation::edges::intrinsics_ratio_prior(0, 1, 1.0) };
        REQUIRE(square.add_vertex(&vertex));
        square.compute_residual();
        REQUIRE(square.get_residual()[0][0] == -10.0);
        square.compute_jacobians();
        REQUIRE((square.get_jacobians()[0].rows() == 1) && (square.get_jacobians()[0].cols() == 2));
        REQUIRE(square.get_jacobians()[0][0][0] == 1.0);
        REQUIRE(square.get_jacobians()[0][0][1] == -1.0);

        // An index outside the parameters measures nothing.
        optimisation::edge outside{ optimisation::edges::intrinsics_ratio_prior(0, 7, 1.0) };
        REQUIRE(outside.add_vertex(&vertex));
        outside.compute_residual();
        REQUIRE(outside.get_residual()[0][0] == 0.0);
        outside.compute_jacobians();
        REQUIRE((outside.get_jacobians()[0][0][0] == 0.0) && (outside.get_jacobians()[0][0][1] == 0.0));
    }

    {
        // The analytic Jacobian against central differences through the vertex's update: with both parameters free, with one
        // held, and with the two sharing a local dimension, whose step keeps the ratio they had and so only moves the residual
        // by the prior's ratio being another.
        for (int variant = 0; variant < 3; ++variant) {
            optimisation::vertex vertex = (variant == 2) ? optimisation::vertices::intrinsics::make(camera, &focal_together[0], 4) : optimisation::vertices::intrinsics::make(camera, (variant == 0) ? &focal_free[0] : &first_free[0], 4);
            REQUIRE(vertex.is_valid());
            const size_t locals = (variant == 0) ? 2 : 1;
            optimisation::edge prior{ optimisation::edges::intrinsics_ratio_prior(0, 1, 0.9) };
            REQUIRE(prior.add_vertex(&vertex));
            prior.compute_residual();
            prior.compute_jacobians();
            REQUIRE(prior.get_jacobians()[0].cols() == locals);
            const double delta = 1e-4;
            for (size_t local = 0; local < locals; ++local) {
                double step[2] = { 0.0, 0.0 };
                step[local] = delta;
                vertex.backup();
                vertex.plus(&step[0]);
                prior.compute_residual();
                const double after = prior.get_residual()[0][0];
                vertex.restore();
                step[local] = -delta;
                vertex.backup();
                vertex.plus(&step[0]);
                prior.compute_residual();
                const double before = prior.get_residual()[0][0];
                vertex.restore();
                REQUIRE(is_value_approx(prior.get_jacobians()[0][0][local], (after - before) / (2.0 * delta), 1e-7));
            }
            if (variant == 2) {
                REQUIRE(is_value_approx(prior.get_jacobians()[0][0][0], 1.0 - (0.9 * 410.0 / 400.0)));
            }
        }
    }

    {
        // A prior on the first focal length and the ratio prior between the two: the second follows the first at the ratio.
        optimisation::factor_graph graph;
        optimisation::vertex* const node = graph.add_vertex(optimisation::vertices::intrinsics::make(camera, &focal_free[0], 4));
        REQUIRE(node != nullptr);
        optimisation::edge prior{ optimisation::edges::intrinsics_prior(0, 420.0) };
        REQUIRE(prior.add_vertex(node));
        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(prior)) != nullptr);
        optimisation::edge ratio{ optimisation::edges::intrinsics_ratio_prior(0, 1, 1.05) };
        REQUIRE(ratio.add_vertex(node));
        ratio.set_information(math::matrix<double, 0, 0>::identity(1, 1) * 100.0);
        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(ratio)) != nullptr);
        REQUIRE(graph.solve(10, true) > 0);
        REQUIRE(is_value_approx(node->get_parameters()[0], 420.0, 1e-6));
        REQUIRE(is_value_approx(node->get_parameters()[1], 400.0, 1e-6));
        REQUIRE(node->get_parameters()[2] == 200.0);
    }

    return EXIT_SUCCESS;
}
