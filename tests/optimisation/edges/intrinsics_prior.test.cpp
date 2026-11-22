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

#include "optimisation/edges/intrinsics_prior.hpp"

#include "optimisation/edge.hpp"
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

    static_assert(sizeof(optimisation::edges::intrinsics_prior) <= optimisation::edge::maximum_size, "The prior edge must fit.");

    const double initial[4] = { 400.0, 410.0, 200.0, 150.0 };
    const sensor::camera::model<double> camera{ sensor::camera::pinhole<double>(&initial[0], 4) };
    const bool focal_free[4] = { true, true, false, false };

    {
        // The residual is the prior less the parameter, and its jacobian minus one at the parameter's local dimension.
        optimisation::vertex vertex = optimisation::vertices::intrinsics::make(camera, &focal_free[0], 4);
        REQUIRE(vertex.is_valid());
        optimisation::edge prior{ optimisation::edges::intrinsics_prior(1, 400.0) };
        REQUIRE(prior.required_vertices() == 1);
        REQUIRE(prior.add_vertex(&vertex));
        prior.compute_residual();
        REQUIRE(prior.get_residual()[0][0] == -10.0);
        prior.compute_jacobians();
        REQUIRE(prior.get_jacobians()[0].rows() == 1);
        REQUIRE(prior.get_jacobians()[0].cols() == 2);
        REQUIRE(prior.get_jacobians()[0][0][0] == 0.0);
        REQUIRE(prior.get_jacobians()[0][0][1] == -1.0);

        // A prior on a parameter held fixed measures it but does not move it.
        optimisation::edge held{ optimisation::edges::intrinsics_prior(2, 190.0) };
        REQUIRE(held.add_vertex(&vertex));
        held.compute_residual();
        REQUIRE(held.get_residual()[0][0] == -10.0);
        held.compute_jacobians();
        REQUIRE(held.get_jacobians()[0][0][0] == 0.0);
        REQUIRE(held.get_jacobians()[0][0][1] == 0.0);

        // An index outside the parameters measures nothing.
        optimisation::edge outside{ optimisation::edges::intrinsics_prior(7, 1.0) };
        REQUIRE(outside.add_vertex(&vertex));
        outside.compute_residual();
        REQUIRE(outside.get_residual()[0][0] == 0.0);
    }

    {
        // The analytic jacobian matches the numeric one.
        optimisation::vertex vertex = optimisation::vertices::intrinsics::make(camera, &focal_free[0], 4);
        optimisation::edge prior{ optimisation::edges::intrinsics_prior(0, 390.0) };
        REQUIRE(prior.add_vertex(&vertex));
        prior.compute_residual();
        prior.compute_jacobians();
        const double delta = 1e-4;
        for (int local = 0; local < 2; ++local) {
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
            REQUIRE(is_value_approx(prior.get_jacobians()[0][0][static_cast<size_t>(local)], (after - before) / (2.0 * delta), 1e-7));
        }
    }

    {
        // With nothing else to say, the priors hold each focal length at its prior value, weighted by their information.
        optimisation::factor_graph graph;
        optimisation::vertex* const node = graph.add_vertex(optimisation::vertices::intrinsics::make(camera, &focal_free[0], 4));
        REQUIRE(node != nullptr);
        for (int index = 0; index < 2; ++index) {
            optimisation::edge prior{ optimisation::edges::intrinsics_prior(index, 420.0 + static_cast<double>(index)) };
            REQUIRE(prior.add_vertex(node));
            prior.set_information(math::matrix<double, 0, 0>::identity(1, 1) * (1.0 / 25.0));
            REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(prior)) != nullptr);
        }
        REQUIRE(graph.solve(10, true) > 0);
        REQUIRE(is_value_approx(node->get_parameters()[0], 420.0, 1e-6));
        REQUIRE(is_value_approx(node->get_parameters()[1], 421.0, 1e-6));
        REQUIRE(node->get_parameters()[2] == 200.0);
        REQUIRE(node->get_parameters()[3] == 150.0);
        double parameters[4] = {};
        REQUIRE(node->get_policy<optimisation::vertices::intrinsics>()->get_camera().get_parameters(&parameters[0], 4));
        REQUIRE(parameters[0] == node->get_parameters()[0]);
        REQUIRE(parameters[1] == node->get_parameters()[1]);
    }

    return EXIT_SUCCESS;
}
