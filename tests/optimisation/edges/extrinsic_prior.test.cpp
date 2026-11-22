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

#include "optimisation/edges/extrinsic_prior.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/odometry.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
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

static optimisation::vertex make_pose(const math::se3<double>& pose) {
    optimisation::vertex vertex{ optimisation::vertices::pose() };
    double parameters[7] = {};
    optimisation::edges::odometry::write_pose(pose, &parameters[0]);
    REQUIRE(vertex.set_parameters(&parameters[0], 7));
    return vertex;
}

static optimisation::edge make_prior(const math::se3<double>& prior, const double (&deviations)[6]) {
    optimisation::edge edge{ optimisation::edges::extrinsic_prior() };
    double parameters[7] = {};
    optimisation::edges::odometry::write_pose(prior, &parameters[0]);
    edge.set_observation(optimisation::edges::extrinsic_prior::observation(&parameters[0]));
    edge.set_information(optimisation::edges::extrinsic_prior::information(deviations));
    return edge;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::extrinsic_prior) <= optimisation::edge::maximum_size, "The extrinsic prior edge must fit.");

    const math::se3<double> prior(math::so3<double>::exp({ { 0.3, -1.2, 0.4 } }), { { 1.5, 0.1, 1.2 } });
    const double deviations[6] = { 0.1, 0.1, 0.0, 1.0, 1.0, 0.01 };

    {
        const math::matrix<double, 0, 0> information = optimisation::edges::extrinsic_prior::information(deviations);
        REQUIRE(is_value_approx(information[0][0], 100.0));
        REQUIRE(information[2][2] == 0.0);
        REQUIRE(is_value_approx(information[5][5], 10000.0));
        REQUIRE(information[0][5] == 0.0);
    }

    // At the prior the residual is zero, away from it the rotation error and the translation difference in the sensor frame.
    {
        optimisation::vertex vertex = make_pose(prior);
        optimisation::edge edge = make_prior(prior, deviations);
        REQUIRE(edge.add_vertex(&vertex));
        edge.compute_residual();
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(std::abs(edge.get_residual()[i][0]) < 1e-12);
        }

        const math::matrix<double, 3, 1> turn = { { 0.02, -0.01, 0.03 } };
        const math::se3<double> moved(math::so3<double>::exp(turn) * prior.rotation(), prior.translation() + math::matrix<double, 3, 1>({ 0.1, -0.2, 0.05 }));
        optimisation::vertex moved_vertex = make_pose(moved);
        optimisation::edge moved_edge = make_prior(prior, deviations);
        REQUIRE(moved_edge.add_vertex(&moved_vertex));
        moved_edge.compute_residual();
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(is_value_approx(moved_edge.get_residual()[i][0], turn[i], 1e-12));
        }
        REQUIRE(is_value_approx(moved_edge.get_residual()[3][0], 0.1, 1e-12));
        REQUIRE(is_value_approx(moved_edge.get_residual()[4][0], -0.2, 1e-12));
        REQUIRE(is_value_approx(moved_edge.get_residual()[5][0], 0.05, 1e-12));

        // The analytic Jacobian against central differences through the pose vertex's update.
        moved_edge.compute_jacobians();
        const math::matrix<double, 0, 0> analytic = moved_edge.get_jacobians()[0];
        REQUIRE((analytic.rows() == 6) && (analytic.cols() == 6));
        double backup[7] = {};
        REQUIRE(moved_vertex.get_parameters(&backup[0], 7));
        const double step = 1e-6;
        for (size_t d = 0; d < 6; ++d) {
            double delta[6] = {};
            delta[d] = step;
            moved_vertex.plus(&delta[0]);
            moved_edge.compute_residual();
            const math::matrix<double, 0, 0> plus = moved_edge.get_residual();
            REQUIRE(moved_vertex.set_parameters(&backup[0], 7));
            delta[d] = -step;
            moved_vertex.plus(&delta[0]);
            moved_edge.compute_residual();
            const math::matrix<double, 0, 0> minus = moved_edge.get_residual();
            REQUIRE(moved_vertex.set_parameters(&backup[0], 7));
            for (size_t r = 0; r < 6; ++r) {
                REQUIRE(std::abs(analytic[r][d] - ((plus[r][0] - minus[r][0]) / (2.0 * step))) < 1e-7);
            }
        }
    }

    // In a graph the prior alone draws the extrinsic to it.
    {
        const double all[6] = { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
        optimisation::factor_graph graph;
        const math::se3<double> start(math::so3<double>::exp({ { 0.1, 0.2, -0.3 } }) * prior.rotation(), prior.translation() + math::matrix<double, 3, 1>({ 0.5, -0.5, 0.2 }));
        optimisation::vertex* const vertex = graph.add_vertex(make_pose(start));
        optimisation::edge edge = make_prior(prior, all);
        REQUIRE(edge.add_vertex(vertex));
        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(edge)) != nullptr);
        REQUIRE(graph.solve(20) > 0);
        const math::se3<double> result = optimisation::edges::odometry::read_pose(vertex->get_parameters());
        REQUIRE((result.rotation() * prior.rotation().inverse()).log().get_length_squared() < 1e-14);
        REQUIRE((result.translation() - prior.translation()).get_length_squared() < 1e-14);
    }

    return EXIT_SUCCESS;
}
