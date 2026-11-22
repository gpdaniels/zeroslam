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

#include "optimisation/edges/prior.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/imu_bias.hpp"
#include "optimisation/vertices/pose.hpp"
#include "optimisation/vertices/velocity.hpp"

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

static math::matrix<double, 0, 0> pose_parameters(const math::se3<double>& pose) {
    const double values[7] = { pose.translation()[0], pose.translation()[1], pose.translation()[2], pose.rotation().get_quaternion()[1], pose.rotation().get_quaternion()[2], pose.rotation().get_quaternion()[3], pose.rotation().get_quaternion()[0] };
    return math::matrix<double, 0, 0>(7, 1, &values[0]);
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::pose_prior) <= optimisation::edge::maximum_size, "The pose prior must fit.");
    static_assert(sizeof(optimisation::edges::velocity_prior) <= optimisation::edge::maximum_size, "The velocity prior must fit.");
    static_assert(sizeof(optimisation::edges::imu_bias_prior) <= optimisation::edge::maximum_size, "The bias prior must fit.");

    // A pose prior: zero at the prior, Log(T * T_prior^-1) away from it, its jacobian agrees with central differences, and a
    // solve brings a free pose onto the prior.
    {
        const math::se3<double> prior(math::so3<double>::exp({ { 0.3, -0.1, 0.5 } }), { { 1.0, -2.0, 0.5 } });
        const math::se3<double> pose(math::so3<double>::exp({ { 0.2, 0.1, 0.4 } }), { { 1.2, -1.7, 0.4 } });
        optimisation::vertex vertex{ optimisation::vertices::pose() };
        REQUIRE(vertex.set_parameters(pose_parameters(prior).data(), 7));
        optimisation::edge edge{ optimisation::edges::pose_prior() };
        edge.set_observation(pose_parameters(prior));
        REQUIRE(edge.add_vertex(&vertex));
        edge.compute_residual();
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(std::abs(edge.get_residual()[i][0]) < 1e-12);
        }
        REQUIRE(vertex.set_parameters(pose_parameters(pose).data(), 7));
        edge.compute_residual();
        const math::matrix<double, 6, 1> expected = (pose * prior.inverse()).log();
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(is_value_approx(edge.get_residual()[i][0], expected[i], 1e-9));
        }
        edge.compute_jacobians();
        const math::matrix<double, 0, 0> analytic = edge.get_jacobians()[0];
        const double step = 1e-6;
        for (size_t c = 0; c < 6; ++c) {
            double delta[6] = {};
            delta[c] = step;
            vertex.plus(&delta[0]);
            edge.compute_residual();
            const math::matrix<double, 0, 0> forward = edge.get_residual();
            REQUIRE(vertex.set_parameters(pose_parameters(pose).data(), 7));
            delta[c] = -step;
            vertex.plus(&delta[0]);
            edge.compute_residual();
            const math::matrix<double, 0, 0> backward = edge.get_residual();
            REQUIRE(vertex.set_parameters(pose_parameters(pose).data(), 7));
            for (size_t r = 0; r < 6; ++r) {
                REQUIRE(std::abs(analytic[r][c] - (forward[r][0] - backward[r][0]) / (2.0 * step)) < 1e-6);
            }
        }

        optimisation::factor_graph graph;
        optimisation::vertex* const free_pose = graph.add_vertex(static_cast<optimisation::vertex&&>(vertex));
        optimisation::edge factor{ optimisation::edges::pose_prior() };
        factor.set_observation(pose_parameters(prior));
        REQUIRE(factor.add_vertex(free_pose));
        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(factor)) != nullptr);
        REQUIRE(graph.solve(20) > 0);
        const math::matrix<double, 0, 0> solved = pose_parameters(prior);
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(std::abs(free_pose->get_parameters()[i] - solved[i][0]) < 1e-9);
        }
        const double alignment = (free_pose->get_parameters()[3] * solved[3][0]) + (free_pose->get_parameters()[4] * solved[4][0]) + (free_pose->get_parameters()[5] * solved[5][0]) + (free_pose->get_parameters()[6] * solved[6][0]);
        REQUIRE(std::abs(std::abs(alignment) - 1.0) < 1e-12);
    }

    // A velocity prior and a bias prior are differences with identity jacobians.
    {
        optimisation::vertex velocity{ optimisation::vertices::velocity() };
        const double velocity_values[3] = { 1.0, 2.0, 3.0 };
        const double velocity_prior[3] = { 0.5, 2.5, 3.0 };
        REQUIRE(velocity.set_parameters(&velocity_values[0], 3));
        optimisation::edge velocity_edge{ optimisation::edges::velocity_prior() };
        velocity_edge.set_observation(math::matrix<double, 0, 0>(3, 1, &velocity_prior[0]));
        REQUIRE(velocity_edge.add_vertex(&velocity));
        velocity_edge.compute_residual();
        velocity_edge.compute_jacobians();
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(is_value_approx(velocity_edge.get_residual()[i][0], velocity_values[i] - velocity_prior[i]));
            for (size_t j = 0; j < 3; ++j) {
                REQUIRE(velocity_edge.get_jacobians()[0][i][j] == ((i == j) ? 1.0 : 0.0));
            }
        }

        optimisation::vertex bias{ optimisation::vertices::imu_bias() };
        const double bias_values[6] = { 0.01, 0.02, 0.03, 0.1, 0.2, 0.3 };
        REQUIRE(bias.set_parameters(&bias_values[0], 6));
        optimisation::edge bias_edge{ optimisation::edges::imu_bias_prior() };
        bias_edge.set_observation(math::matrix<double, 0, 0>::zero(6, 1));
        const math::matrix<double, 0, 0> information = optimisation::edges::imu_bias_prior::information(0.1, 0.01);
        REQUIRE(is_value_approx(information[0][0], 100.0));
        REQUIRE(is_value_approx(information[5][5], 10000.0));
        REQUIRE(information[0][1] == 0.0);
        bias_edge.set_information(information);
        REQUIRE(bias_edge.add_vertex(&bias));
        bias_edge.compute_residual();
        bias_edge.compute_jacobians();
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(is_value_approx(bias_edge.get_residual()[i][0], bias_values[i]));
            for (size_t j = 0; j < 6; ++j) {
                REQUIRE(bias_edge.get_jacobians()[0][i][j] == ((i == j) ? 1.0 : 0.0));
            }
        }
        REQUIRE(is_value_approx(bias_edge.chi2(), 100.0 * (0.01 * 0.01 + 0.02 * 0.02 + 0.03 * 0.03) + 10000.0 * (0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3)));
    }

    return EXIT_SUCCESS;
}
