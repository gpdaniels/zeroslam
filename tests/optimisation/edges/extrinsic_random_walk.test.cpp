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

#include "optimisation/edges/extrinsic_random_walk.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/extrinsic_prior.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/extrinsic.hpp"

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

// An extrinsic vertex of a sensor to body transform.
static optimisation::vertex make_extrinsic(const math::se3<double>& body_from_sensor) {
    optimisation::vertex vertex{ optimisation::vertices::extrinsic() };
    double parameters[7] = {};
    optimisation::vertices::extrinsic::to_parameters(body_from_sensor, parameters);
    REQUIRE(vertex.set_parameters(&parameters[0], 7));
    return vertex;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::extrinsic_random_walk) <= optimisation::edge::maximum_size, "The extrinsic random walk edge must fit.");

    {
        // The information is the inverse of the walk's variances, the rotation's then the translation's, which grow with the
        // time walked; a standard deviation of zero gives none.
        const math::matrix<double, 0, 0> information = optimisation::edges::extrinsic_random_walk::information(0.001, 0.0005, 5.0, 20.0);
        REQUIRE((information.rows() == 6) && (information.cols() == 6));
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(is_value_approx(information[i][i], 1.0 / (4.0 * 0.001 * 0.001)));
            REQUIRE(is_value_approx(information[3 + i][3 + i], 1.0 / (4.0 * 0.0005 * 0.0005)));
        }
        REQUIRE(information[0][3] == 0.0);
        REQUIRE(information[4][5] == 0.0);
        const math::matrix<double, 0, 0> rotation_only = optimisation::edges::extrinsic_random_walk::information(0.001, 0.0, 5.0, 5.0);
        REQUIRE(is_value_approx(rotation_only[1][1], 1.0e6));
        REQUIRE(rotation_only[4][4] == 0.0);
    }

    const math::se3<double> first(math::so3<double>::exp({ { 0.3, -1.2, 0.4 } }), { { 0.11, 0.02, -0.03 } });

    {
        // Between an extrinsic and itself the residual is zero; a sensor turned in its own frame and moved on the body gives
        // the turn and the move.
        optimisation::vertex first_vertex = make_extrinsic(first);
        optimisation::vertex same_vertex = make_extrinsic(first);
        optimisation::edge same{ optimisation::edges::extrinsic_random_walk() };
        REQUIRE(same.required_vertices() == 2);
        REQUIRE(same.add_vertex(&first_vertex));
        REQUIRE(same.add_vertex(&same_vertex));
        same.compute_residual();
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(std::abs(same.get_residual()[i][0]) < 1e-12);
        }

        const math::matrix<double, 3, 1> turn = { { 0.02, -0.01, 0.03 } };
        const math::matrix<double, 3, 1> move = { { 0.001, -0.002, 0.0005 } };
        const math::se3<double> second(first.rotation() * math::so3<double>::exp(turn), first.translation() + move);
        optimisation::vertex second_vertex = make_extrinsic(second);
        optimisation::edge walk{ optimisation::edges::extrinsic_random_walk() };
        REQUIRE(walk.add_vertex(&first_vertex));
        REQUIRE(walk.add_vertex(&second_vertex));
        walk.compute_residual();
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(is_value_approx(walk.get_residual()[i][0], turn[i], 1e-12));
            REQUIRE(is_value_approx(walk.get_residual()[3 + i][0], move[i], 1e-12));
        }

        // The other way round the walk is the opposite one, the turn seen from the second sensor.
        optimisation::edge back{ optimisation::edges::extrinsic_random_walk() };
        REQUIRE(back.add_vertex(&second_vertex));
        REQUIRE(back.add_vertex(&first_vertex));
        back.compute_residual();
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(is_value_approx(back.get_residual()[i][0], -turn[i], 1e-12));
            REQUIRE(is_value_approx(back.get_residual()[3 + i][0], -move[i], 1e-12));
        }
    }

    {
        // The analytic Jacobians against central differences through each vertex's update, for a small walk and a large one.
        const math::se3<double> seconds[2] = {
            math::se3<double>(first.rotation() * math::so3<double>::exp({ { 0.002, -0.001, 0.003 } }), first.translation() + math::matrix<double, 3, 1>({ 0.001, -0.002, 0.0005 })),
            math::se3<double>(math::so3<double>::exp({ { -0.4, 0.2, 0.9 } }) * first.rotation() * math::so3<double>::exp({ { 0.5, -0.3, 0.2 } }), first.translation() + math::matrix<double, 3, 1>({ 0.4, -0.2, 0.7 }))
        };
        for (size_t variant = 0; variant < 2; ++variant) {
            optimisation::vertex vertices[2] = { make_extrinsic(first), make_extrinsic(seconds[variant]) };
            optimisation::edge walk{ optimisation::edges::extrinsic_random_walk() };
            REQUIRE(walk.add_vertex(&vertices[0]));
            REQUIRE(walk.add_vertex(&vertices[1]));
            walk.compute_residual();
            walk.compute_jacobians();
            const double step = 1e-6;
            for (size_t slot = 0; slot < 2; ++slot) {
                const math::matrix<double, 0, 0> analytic = walk.get_jacobians()[slot];
                REQUIRE((analytic.rows() == 6) && (analytic.cols() == 6));
                double backup[7] = {};
                REQUIRE(vertices[slot].get_parameters(&backup[0], 7));
                for (size_t d = 0; d < 6; ++d) {
                    double delta[6] = {};
                    delta[d] = step;
                    vertices[slot].plus(&delta[0]);
                    walk.compute_residual();
                    const math::matrix<double, 0, 0> plus = walk.get_residual();
                    REQUIRE(vertices[slot].set_parameters(&backup[0], 7));
                    delta[d] = -step;
                    vertices[slot].plus(&delta[0]);
                    walk.compute_residual();
                    const math::matrix<double, 0, 0> minus = walk.get_residual();
                    REQUIRE(vertices[slot].set_parameters(&backup[0], 7));
                    for (size_t r = 0; r < 6; ++r) {
                        REQUIRE(std::abs(analytic[r][d] - ((plus[r][0] - minus[r][0]) / (2.0 * step))) < 1e-7);
                    }
                }
            }
        }
    }

    {
        // A chain of three instants whose ends are held by priors: the walk spreads the turn and the move between them over
        // its steps in proportion to their variances, here a step of five seconds and one of fifteen.
        const math::matrix<double, 3, 1> turn = { { 0.008, -0.004, 0.012 } };
        const math::matrix<double, 3, 1> move = { { 0.004, 0.0, -0.008 } };
        const math::se3<double> last(first.rotation() * math::so3<double>::exp(turn), first.translation() + move);
        optimisation::factor_graph graph;
        optimisation::vertex* nodes[3] = { graph.add_vertex(make_extrinsic(first)), graph.add_vertex(make_extrinsic(first)), graph.add_vertex(make_extrinsic(first)) };
        const math::se3<double>* const ends[2] = { &first, &last };
        const double tight[6] = { 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6 };
        for (size_t end = 0; end < 2; ++end) {
            optimisation::edge prior{ optimisation::edges::extrinsic_prior() };
            double parameters[7] = {};
            optimisation::vertices::extrinsic::to_parameters(*ends[end], parameters);
            prior.set_observation(optimisation::edges::extrinsic_prior::observation(&parameters[0]));
            prior.set_information(optimisation::edges::extrinsic_prior::information(tight));
            REQUIRE(prior.add_vertex(nodes[2 * end]));
            REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(prior)) != nullptr);
        }
        const double durations[2] = { 5.0, 15.0 };
        for (size_t step = 0; step < 2; ++step) {
            optimisation::edge walk{ optimisation::edges::extrinsic_random_walk() };
            REQUIRE(walk.add_vertex(nodes[step]));
            REQUIRE(walk.add_vertex(nodes[step + 1]));
            walk.set_information(optimisation::edges::extrinsic_random_walk::information(0.001, 0.0005, 5.0, durations[step]));
            REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(walk)) != nullptr);
        }
        REQUIRE(graph.solve(30, true) > 0);
        const math::se3<double> middle = optimisation::vertices::extrinsic::from_parameters(nodes[1]->get_parameters());
        const math::matrix<double, 3, 1> middle_turn = (first.rotation().inverse() * middle.rotation()).log();
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(std::abs(middle_turn[i] - (0.25 * turn[i])) < 1e-6);
            REQUIRE(std::abs(middle.translation()[i] - (first.translation()[i] + (0.25 * move[i]))) < 1e-6);
        }
    }

    return EXIT_SUCCESS;
}
