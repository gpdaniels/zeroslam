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

#include "optimisation/edges/planar_motion.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/odometry.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/extrinsic.hpp"
#include "optimisation/vertices/pose.hpp"

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

static optimisation::vertex make_pose(const math::se3<double>& pose, const bool fixed) {
    optimisation::vertex vertex{ optimisation::vertices::pose() };
    double parameters[7] = {};
    optimisation::edges::odometry::write_pose(pose, &parameters[0]);
    REQUIRE(vertex.set_parameters(&parameters[0], 7));
    vertex.set_fixed(fixed);
    return vertex;
}

// The rig's extrinsic vertex of the odometry frame, which holds it as sensor to body, from the body to odometry transform.
static optimisation::vertex make_extrinsic(const math::se3<double>& odometry_from_body, const bool fixed) {
    optimisation::vertex vertex{ optimisation::vertices::extrinsic() };
    double parameters[7] = {};
    optimisation::vertices::extrinsic::to_parameters(odometry_from_body.inverse(), parameters);
    REQUIRE(vertex.set_parameters(&parameters[0], 7));
    vertex.set_fixed(fixed);
    return vertex;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::planar_motion) <= optimisation::edge::maximum_size, "The planar motion edge must fit.");

    // A camera looking forward along the odometry frame's x axis, 1.3 m up and 1.6 m forward of it.
    const math::so3<double> camera_axes(math::matrix<double, 3, 3>({ { 0.0, 0.0, 1.0 }, { -1.0, 0.0, 0.0 }, { 0.0, -1.0, 0.0 } }));
    const math::se3<double> extrinsic(math::so3<double>::exp({ { 0.0, 0.08, 0.05 } }) * camera_axes, { { 1.6, 0.15, 1.3 } });
    const math::se3<double> world_first(math::so3<double>::exp({ { 0.0, 0.0, 0.4 } }), { { 3.0, -2.0, 0.0 } });
    const math::se3<double> world_second(math::so3<double>::exp({ { 0.0, 0.0, 1.1 } }), { { 7.0, 1.0, 0.0 } });
    // The body poses, world to camera, of the odometry frame's planar poses in the world.
    const math::se3<double> first = (world_first * extrinsic).inverse();
    const math::se3<double> second = (world_second * extrinsic).inverse();

    {
        const math::matrix<double, 0, 0> information = optimisation::edges::planar_motion::information(4.0, 0.01, 0.02, 0.001, 0.002);
        REQUIRE(is_value_approx(information[0][0], 1.0 / ((0.0001 * 4.0) + 0.000001)));
        REQUIRE(information[1][1] == information[0][0]);
        REQUIRE(is_value_approx(information[2][2], 1.0 / ((0.0004 * 4.0) + 0.000004)));
        REQUIRE(information[0][2] == 0.0);
        const math::matrix<double, 0, 0> none = optimisation::edges::planar_motion::information(4.0, 0.0, 0.0, 0.0, 0.0);
        REQUIRE((none[0][0] == 0.0) && (none[2][2] == 0.0));
    }

    // Planar motion has no residual whatever the extrinsic, and motion off the plane is measured by the end's tilt and height.
    {
        optimisation::vertex first_vertex = make_pose(first, false);
        optimisation::vertex second_vertex = make_pose(second, false);
        optimisation::vertex extrinsic_vertex = make_extrinsic(extrinsic, false);
        optimisation::edge edge{ optimisation::edges::planar_motion() };
        REQUIRE(edge.add_vertex(&first_vertex));
        REQUIRE(edge.add_vertex(&second_vertex));
        REQUIRE(edge.add_vertex(&extrinsic_vertex));
        edge.compute_residual();
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(std::abs(edge.get_residual()[i][0]) < 1e-12);
        }

        const math::se3<double> lifted(math::so3<double>::exp({ { 0.0, 0.0, 1.1 } }) * math::so3<double>::exp({ { 0.0, 0.03, 0.0 } }), { { 7.0, 1.0, 0.2 } });
        optimisation::vertex lifted_vertex = make_pose((lifted * extrinsic).inverse(), false);
        optimisation::edge lifted_edge{ optimisation::edges::planar_motion() };
        REQUIRE(lifted_edge.add_vertex(&first_vertex));
        REQUIRE(lifted_edge.add_vertex(&lifted_vertex));
        REQUIRE(lifted_edge.add_vertex(&extrinsic_vertex));
        lifted_edge.compute_residual();
        // A pitch of 0.03 about the end's y axis tilts its z axis by sin(0.03) towards its own x, turned by 0.7 into the start's.
        REQUIRE(is_value_approx(lifted_edge.get_residual()[0][0], std::sin(0.03) * std::cos(0.7), 1e-12));
        REQUIRE(is_value_approx(lifted_edge.get_residual()[1][0], std::sin(0.03) * std::sin(0.7), 1e-12));
        REQUIRE(is_value_approx(lifted_edge.get_residual()[2][0], 0.2, 1e-12));

        // The analytic Jacobians against central differences, off the plane and with every vertex.
        const math::se3<double> tilted_extrinsic(math::so3<double>::exp({ { 0.1, -0.05, 0.2 } }) * extrinsic.rotation(), extrinsic.translation());
        REQUIRE(extrinsic_vertex.set_parameters(make_extrinsic(tilted_extrinsic, false).get_parameters(), 7));
        lifted_edge.compute_residual();
        lifted_edge.compute_jacobians();
        optimisation::vertex* const vertices[3] = { &first_vertex, &lifted_vertex, &extrinsic_vertex };
        const double step = 1e-6;
        for (size_t v = 0; v < 3; ++v) {
            const math::matrix<double, 0, 0> analytic = lifted_edge.get_jacobians()[v];
            REQUIRE((analytic.rows() == 3) && (analytic.cols() == 6));
            double backup[7] = {};
            REQUIRE(vertices[v]->get_parameters(&backup[0], 7));
            for (size_t d = 0; d < 6; ++d) {
                double delta[6] = {};
                delta[d] = step;
                vertices[v]->plus(&delta[0]);
                lifted_edge.compute_residual();
                const math::matrix<double, 0, 0> plus = lifted_edge.get_residual();
                REQUIRE(vertices[v]->set_parameters(&backup[0], 7));
                delta[d] = -step;
                vertices[v]->plus(&delta[0]);
                lifted_edge.compute_residual();
                const math::matrix<double, 0, 0> minus = lifted_edge.get_residual();
                REQUIRE(vertices[v]->set_parameters(&backup[0], 7));
                for (size_t r = 0; r < 3; ++r) {
                    REQUIRE(std::abs(analytic[r][d] - ((plus[r][0] - minus[r][0]) / (2.0 * step))) < 1e-7);
                }
            }
        }
    }

    // In a graph, the constraint brings a lifted pose back to the plane of a fixed one, along the shortest way.
    {
        optimisation::factor_graph graph;
        optimisation::vertex* const first_vertex = graph.add_vertex(make_pose(first, true));
        const math::se3<double> lifted(math::so3<double>::exp({ { 0.02, 0.03, 1.1 } }), { { 7.0, 1.0, 0.2 } });
        optimisation::vertex* const second_vertex = graph.add_vertex(make_pose((lifted * extrinsic).inverse(), false));
        optimisation::vertex* const extrinsic_vertex = graph.add_vertex(make_extrinsic(extrinsic, true));
        optimisation::edge edge{ optimisation::edges::planar_motion() };
        edge.set_information(optimisation::edges::planar_motion::information(5.0, 0.01, 0.01, 0.001, 0.001));
        REQUIRE(edge.add_vertex(first_vertex));
        REQUIRE(edge.add_vertex(second_vertex));
        REQUIRE(edge.add_vertex(extrinsic_vertex));
        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(edge)) != nullptr);
        REQUIRE(graph.solve(20) > 0);
        REQUIRE(graph.get_current_chi() < 1e-12);
        const math::se3<double> result = (optimisation::edges::odometry::read_pose(second_vertex->get_parameters()).inverse()) * extrinsic.inverse();
        REQUIRE(std::abs(result.translation()[2]) < 1e-6);
        const math::matrix<double, 3, 1> axis = result.rotation() * math::matrix<double, 3, 1>({ 0.0, 0.0, 1.0 });
        REQUIRE(std::abs(axis[2] - 1.0) < 1e-9);
    }

    return EXIT_SUCCESS;
}
