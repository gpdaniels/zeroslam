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

#include "optimisation/edges/baseline.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/pose.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-8) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

static optimisation::vertex make_pose(const math::se3<double>& pose, bool fixed) {
    optimisation::vertex vertex{ optimisation::vertices::pose() };
    const double parameters[7] = { pose.translation()[0], pose.translation()[1], pose.translation()[2], pose.rotation().get_quaternion()[1], pose.rotation().get_quaternion()[2], pose.rotation().get_quaternion()[3], pose.rotation().get_quaternion()[0] };
    REQUIRE(vertex.set_parameters(&parameters[0], 7));
    vertex.set_fixed(fixed);
    return vertex;
}

static math::matrix<double, 3, 1> centre_of(const optimisation::vertex& vertex) {
    return optimisation::edges::baseline::centre(vertex.get_parameters());
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::baseline) <= optimisation::edge::maximum_size, "The baseline edge must fit.");

    const math::se3<double> pose_a(math::so3<double>::exp({ { 0.1, -0.2, 0.3 } }), { { 0.5, -1.0, 2.0 } });
    const math::se3<double> pose_b(math::so3<double>::exp({ { -0.3, 0.1, 0.2 } }), { { -1.5, 0.5, 1.0 } });
    const double separation = std::sqrt((centre_of(make_pose(pose_a, false)) - centre_of(make_pose(pose_b, false))).get_length_squared());

    {
        optimisation::vertex vertex_a = make_pose(pose_a, false);
        optimisation::vertex vertex_b = make_pose(pose_b, false);
        optimisation::edge edge{ optimisation::edges::baseline(separation - 0.25) };
        REQUIRE(edge.add_vertex(&vertex_a));
        REQUIRE(edge.add_vertex(&vertex_b));
        edge.compute_residual();
        REQUIRE(edge.get_residual().rows() == 1);
        REQUIRE(is_value_approx(edge.get_residual()[0][0], 0.25));
        edge.compute_jacobians();
        REQUIRE(edge.get_jacobians().size() == 2);
        for (size_t v = 0; v < 2; ++v) {
            optimisation::vertex& vertex = (v == 0) ? vertex_a : vertex_b;
            const math::matrix<double, 0, 0> jacobian = edge.get_jacobians()[v];
            REQUIRE(jacobian.rows() == 1);
            REQUIRE(jacobian.cols() == 6);
            for (size_t d = 0; d < 6; ++d) {
                const double step = 1e-6;
                double delta[6] = {};
                vertex.backup();
                delta[d] = step;
                vertex.plus(&delta[0]);
                edge.compute_residual();
                const double forward = edge.get_residual()[0][0];
                vertex.restore();
                delta[d] = -step;
                vertex.plus(&delta[0]);
                edge.compute_residual();
                const double backward = edge.get_residual()[0][0];
                vertex.restore();
                REQUIRE(is_value_approx(jacobian[0][d], (forward - backward) / (2.0 * step), 1e-5));
            }
            for (size_t d = 0; d < 3; ++d) {
                REQUIRE(jacobian[0][d] == 0.0);
            }
        }
    }

    {
        optimisation::vertex vertex_a = make_pose(pose_a, false);
        optimisation::vertex vertex_b = make_pose(pose_a, false);
        optimisation::edge edge{ optimisation::edges::baseline(1.0) };
        edge.add_vertex(&vertex_a);
        edge.add_vertex(&vertex_b);
        edge.compute_residual();
        REQUIRE(is_value_approx(edge.get_residual()[0][0], -1.0));
        edge.compute_jacobians();
        for (size_t v = 0; v < 2; ++v) {
            for (size_t d = 0; d < 6; ++d) {
                REQUIRE(edge.get_jacobians()[v][0][d] == 0.0);
            }
        }
    }

    {
        optimisation::factor_graph graph;
        optimisation::vertex* fixed = graph.add_vertex(make_pose(pose_a, true));
        optimisation::vertex* free = graph.add_vertex(make_pose(pose_b, false));
        const math::matrix<double, 3, 1> centre_fixed = centre_of(*fixed);
        optimisation::edge edge{ optimisation::edges::baseline(2.0 * separation) };
        edge.add_vertex(fixed);
        edge.add_vertex(free);
        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(edge)) != nullptr);
        REQUIRE(graph.solve(50, true));
        REQUIRE(graph.get_current_chi() < 1e-12);
        const math::matrix<double, 3, 1> moved = centre_of(*free) - centre_fixed;
        REQUIRE(is_value_approx(std::sqrt(moved.get_length_squared()), 2.0 * separation, 1e-6));
        for (size_t i = 0; i < 4; ++i) {
            REQUIRE(is_value_approx(free->get_parameters()[3 + i], make_pose(pose_b, false).get_parameters()[3 + i], 1e-9));
        }
    }

    return EXIT_SUCCESS;
}
