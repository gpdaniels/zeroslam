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

#include "optimisation/edges/intrinsics_random_walk.hpp"

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

static sensor::camera::model<double> make_camera(const double focal_x, const double focal_y) {
    const double parameters[4] = { focal_x, focal_y, 200.0, 150.0 };
    return sensor::camera::model<double>{ sensor::camera::pinhole<double>(&parameters[0], 4) };
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::intrinsics_random_walk) <= optimisation::edge::maximum_size, "The random walk edge must fit.");

    const bool focal_free[4] = { true, true, false, false };
    const int focal_together[4] = { 0, 0, -1, -1 };

    {
        // The information is the inverse of the walk's variance, which grows with the time walked.
        const math::matrix<double, 0, 0> at_reference = optimisation::edges::intrinsics_random_walk::information(0.5, 5.0, 5.0);
        REQUIRE((at_reference.rows() == 1) && (at_reference.cols() == 1));
        REQUIRE(is_value_approx(at_reference[0][0], 4.0));
        REQUIRE(is_value_approx(optimisation::edges::intrinsics_random_walk::information(0.5, 5.0, 20.0)[0][0], 1.0));
        REQUIRE(is_value_approx(optimisation::edges::intrinsics_random_walk::information(0.5, 5.0, 1.25)[0][0], 16.0));
        REQUIRE(optimisation::edges::intrinsics_random_walk::information(0.0, 5.0, 5.0)[0][0] == 0.0);
    }

    {
        // The residual is the parameter at the second instant less the parameter at the first.
        optimisation::vertex first = optimisation::vertices::intrinsics::make(make_camera(400.0, 410.0), &focal_free[0], 4);
        optimisation::vertex second = optimisation::vertices::intrinsics::make(make_camera(403.0, 408.0), &focal_free[0], 4);
        REQUIRE(first.is_valid() && second.is_valid());
        for (int index = 0; index < 2; ++index) {
            optimisation::edge walk{ optimisation::edges::intrinsics_random_walk(index) };
            REQUIRE(walk.required_vertices() == 2);
            REQUIRE(walk.add_vertex(&first));
            REQUIRE(walk.add_vertex(&second));
            walk.compute_residual();
            REQUIRE(walk.get_residual()[0][0] == ((index == 0) ? 3.0 : -2.0));
            walk.compute_jacobians();
            REQUIRE((walk.get_jacobians()[0].rows() == 1) && (walk.get_jacobians()[0].cols() == 2));
            REQUIRE((walk.get_jacobians()[1].rows() == 1) && (walk.get_jacobians()[1].cols() == 2));
            for (size_t local = 0; local < 2; ++local) {
                const bool own = (local == static_cast<size_t>(index));
                REQUIRE(walk.get_jacobians()[0][0][local] == (own ? -1.0 : 0.0));
                REQUIRE(walk.get_jacobians()[1][0][local] == (own ? 1.0 : 0.0));
            }
        }

        // A parameter held fixed is measured and not moved, and an index outside the parameters measures nothing.
        optimisation::edge held{ optimisation::edges::intrinsics_random_walk(2) };
        REQUIRE(held.add_vertex(&first));
        REQUIRE(held.add_vertex(&second));
        held.compute_residual();
        REQUIRE(held.get_residual()[0][0] == 0.0);
        held.compute_jacobians();
        REQUIRE((held.get_jacobians()[0][0][0] == 0.0) && (held.get_jacobians()[0][0][1] == 0.0));
        REQUIRE((held.get_jacobians()[1][0][0] == 0.0) && (held.get_jacobians()[1][0][1] == 0.0));
        optimisation::edge outside{ optimisation::edges::intrinsics_random_walk(9) };
        REQUIRE(outside.add_vertex(&first));
        REQUIRE(outside.add_vertex(&second));
        outside.compute_residual();
        REQUIRE(outside.get_residual()[0][0] == 0.0);
    }

    {
        // The analytic Jacobians against central differences through each vertex's update, with the focal lengths each free
        // and with the two sharing a local dimension, which moves the second in proportion to the first.
        for (int together = 0; together < 2; ++together) {
            optimisation::vertex vertices[2] = {
                (together != 0) ? optimisation::vertices::intrinsics::make(make_camera(400.0, 410.0), &focal_together[0], 4) : optimisation::vertices::intrinsics::make(make_camera(400.0, 410.0), &focal_free[0], 4),
                (together != 0) ? optimisation::vertices::intrinsics::make(make_camera(403.0, 408.0), &focal_together[0], 4) : optimisation::vertices::intrinsics::make(make_camera(403.0, 408.0), &focal_free[0], 4)
            };
            const size_t locals = (together != 0) ? 1 : 2;
            for (int index = 0; index < 2; ++index) {
                optimisation::edge walk{ optimisation::edges::intrinsics_random_walk(index) };
                REQUIRE(walk.add_vertex(&vertices[0]));
                REQUIRE(walk.add_vertex(&vertices[1]));
                walk.compute_residual();
                walk.compute_jacobians();
                const double delta = 1e-4;
                for (size_t slot = 0; slot < 2; ++slot) {
                    REQUIRE(walk.get_jacobians()[slot].cols() == locals);
                    for (size_t local = 0; local < locals; ++local) {
                        double step[2] = { 0.0, 0.0 };
                        step[local] = delta;
                        vertices[slot].backup();
                        vertices[slot].plus(&step[0]);
                        walk.compute_residual();
                        const double after = walk.get_residual()[0][0];
                        vertices[slot].restore();
                        step[local] = -delta;
                        vertices[slot].backup();
                        vertices[slot].plus(&step[0]);
                        walk.compute_residual();
                        const double before = walk.get_residual()[0][0];
                        vertices[slot].restore();
                        REQUIRE(is_value_approx(walk.get_jacobians()[slot][0][local], (after - before) / (2.0 * delta), 1e-7));
                    }
                }
            }
        }
    }

    {
        // A chain of three instants whose ends are held by priors 12 pixels apart: the walk spreads the difference over the
        // steps in proportion to their variances, here a step of five seconds and one of fifteen.
        optimisation::factor_graph graph;
        optimisation::vertex* nodes[3] = { nullptr, nullptr, nullptr };
        for (size_t index = 0; index < 3; ++index) {
            nodes[index] = graph.add_vertex(optimisation::vertices::intrinsics::make(make_camera(400.0, 400.0), &focal_free[0], 4));
            REQUIRE(nodes[index] != nullptr);
        }
        const double ends[2] = { 400.0, 412.0 };
        for (size_t end = 0; end < 2; ++end) {
            for (int index = 0; index < 2; ++index) {
                optimisation::edge prior{ optimisation::edges::intrinsics_prior(index, ends[end]) };
                REQUIRE(prior.add_vertex(nodes[2 * end]));
                prior.set_information(math::matrix<double, 0, 0>::identity(1, 1) * 1.0e8);
                REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(prior)) != nullptr);
            }
        }
        const double durations[2] = { 5.0, 15.0 };
        for (size_t step = 0; step < 2; ++step) {
            for (int index = 0; index < 2; ++index) {
                optimisation::edge walk{ optimisation::edges::intrinsics_random_walk(index) };
                REQUIRE(walk.add_vertex(nodes[step]));
                REQUIRE(walk.add_vertex(nodes[step + 1]));
                walk.set_information(optimisation::edges::intrinsics_random_walk::information(0.1, 5.0, durations[step]));
                REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(walk)) != nullptr);
            }
        }
        REQUIRE(graph.solve(20, true) > 0);
        for (int index = 0; index < 2; ++index) {
            REQUIRE(is_value_approx(nodes[0]->get_parameters()[index], 400.0, 1e-6));
            REQUIRE(is_value_approx(nodes[1]->get_parameters()[index], 403.0, 1e-6));
            REQUIRE(is_value_approx(nodes[2]->get_parameters()[index], 412.0, 1e-6));
        }
        REQUIRE(nodes[1]->get_parameters()[2] == 200.0);
        REQUIRE(nodes[1]->get_parameters()[3] == 150.0);
    }

    return EXIT_SUCCESS;
}
