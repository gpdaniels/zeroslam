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

#include "optimisation/edges/similarity_reprojection.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/loss.hpp"
#include "optimisation/losses/huber.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/similarity.hpp"
#include "sensor/camera/model.hpp"
#include "sensor/camera/pinhole.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
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

static void similarity_parameters(const math::sim3<double>& similarity, double (&parameters)[8]) {
    parameters[0] = similarity.transformation().translation()[0];
    parameters[1] = similarity.transformation().translation()[1];
    parameters[2] = similarity.transformation().translation()[2];
    parameters[3] = similarity.transformation().rotation().get_quaternion()[1];
    parameters[4] = similarity.transformation().rotation().get_quaternion()[2];
    parameters[5] = similarity.transformation().rotation().get_quaternion()[3];
    parameters[6] = similarity.transformation().rotation().get_quaternion()[0];
    parameters[7] = similarity.scale();
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::similarity_reprojection) <= optimisation::edge::maximum_size, "The similarity reprojection edge must fit.");

    const sensor::camera::pinhole<double> pinhole(std::vector<double>{ 500.0, 500.0, 320.0, 240.0 }.data(), 4);
    const sensor::camera::model<double> camera(pinhole);
    const math::se3<double> observer(math::so3<double>::exp({ { 0.1, -0.2, 0.05 } }), { { 0.3, 0.1, -0.2 } });
    const math::sim3<double> similarity(math::se3<double>(math::so3<double>::exp({ { 0.05, 0.1, -0.02 } }), { { 0.4, -0.2, 0.1 } }), 1.2);
    const math::matrix<double, 3, 1> location({ 0.5, -0.3, 4.0 });

    for (size_t direction = 0; direction < 2; ++direction) {
        const bool inverted = (direction == 1);
        const math::matrix<double, 3, 1> transferred = inverted ? (similarity.inverse() * location) : (similarity * location);
        const math::matrix<double, 3, 1> in_camera = observer * transferred;
        double projected[2] = {};
        REQUIRE(camera.project(in_camera.data(), &projected[0]));

        std::unique_ptr<optimisation::vertex> vertex = std::make_unique<optimisation::vertex>(optimisation::vertices::similarity());
        double parameters[8] = {};
        similarity_parameters(similarity, parameters);
        vertex->set_parameters(&parameters[0], 8);

        optimisation::edge factor{ optimisation::edges::similarity_reprojection(camera, location, observer, inverted) };
        factor.add_vertex(vertex.get());
        const double observed[2] = { projected[0] + 2.0, projected[1] - 3.0 };
        factor.set_observation(math::matrix<double, 0, 0>(2, 1, &observed[0]));
        factor.compute_residual();
        REQUIRE(is_value_approx(factor.get_residual()[0][0], 2.0, 1e-6));
        REQUIRE(is_value_approx(factor.get_residual()[1][0], -3.0, 1e-6));

        factor.set_observation(math::matrix<double, 0, 0>(2, 1, &projected[0]));
        factor.compute_residual();
        REQUIRE(is_value_approx(factor.get_residual()[0][0], 0.0, 1e-6));
        REQUIRE(is_value_approx(factor.get_residual()[1][0], 0.0, 1e-6));
    }

    for (size_t direction = 0; direction < 2; ++direction) {
        const bool inverted = (direction == 1);
        std::unique_ptr<optimisation::vertex> vertex = std::make_unique<optimisation::vertex>(optimisation::vertices::similarity());
        double parameters[8] = {};
        similarity_parameters(similarity, parameters);
        vertex->set_parameters(&parameters[0], 8);

        optimisation::edge factor{ optimisation::edges::similarity_reprojection(camera, location, observer, inverted) };
        factor.add_vertex(vertex.get());
        const double observed[2] = { 300.0, 200.0 };
        factor.set_observation(math::matrix<double, 0, 0>(2, 1, &observed[0]));
        factor.compute_residual();
        factor.compute_jacobians();
        const math::matrix<double, 0, 0> analytic = factor.get_jacobians()[0];
        REQUIRE(analytic.rows() == 2);
        REQUIRE(analytic.cols() == 7);

        const double delta = 1e-6;
        for (size_t d = 0; d < 7; ++d) {
            double step[7] = {};
            step[d] = delta;
            vertex->set_parameters(&parameters[0], 8);
            vertex->plus(&step[0]);
            factor.compute_residual();
            const math::matrix<double, 0, 0> forward = factor.get_residual();
            step[d] = -delta;
            vertex->set_parameters(&parameters[0], 8);
            vertex->plus(&step[0]);
            factor.compute_residual();
            const math::matrix<double, 0, 0> backward = factor.get_residual();
            for (size_t r = 0; r < 2; ++r) {
                REQUIRE(is_value_approx(analytic[r][d], (forward[r][0] - backward[r][0]) / (2.0 * delta), 1e-5));
            }
        }
    }

    for (size_t direction = 0; direction < 2; ++direction) {
        const bool inverted = (direction == 1);
        std::unique_ptr<optimisation::vertex> vertex = std::make_unique<optimisation::vertex>(optimisation::vertices::similarity());
        double parameters[8] = {};
        similarity_parameters(similarity, parameters);
        vertex->set_parameters(&parameters[0], 8);

        const math::matrix<double, 3, 1> behind = inverted ? (similarity * (observer.inverse() * math::matrix<double, 3, 1>({ 0.2, -0.1, -1.0e-4 }))) : (similarity.inverse() * (observer.inverse() * math::matrix<double, 3, 1>({ 0.2, -0.1, -1.0e-4 })));
        optimisation::edge factor{ optimisation::edges::similarity_reprojection(camera, behind, observer, inverted) };
        factor.add_vertex(vertex.get());
        const double observed[2] = { 320.0, 240.0 };
        factor.set_observation(math::matrix<double, 0, 0>(2, 1, &observed[0]));
        factor.compute_residual();
        REQUIRE(is_value_approx(factor.get_residual()[0][0], factor.get_residual()[1][0]));
        factor.compute_jacobians();
        const math::matrix<double, 0, 0> analytic = factor.get_jacobians()[0];
        for (size_t d = 0; d < 7; ++d) {
            REQUIRE(std::isfinite(analytic[0][d]));
            REQUIRE(std::abs(analytic[0][d]) < 10.0);
            REQUIRE(is_value_approx(analytic[0][d], analytic[1][d]));
        }
    }

    {
        std::unique_ptr<optimisation::vertex> vertex = std::make_unique<optimisation::vertex>(optimisation::vertices::similarity());
        double parameters[8] = {};
        similarity_parameters(math::sim3<double>::identity(), parameters);
        vertex->set_parameters(&parameters[0], 8);

        optimisation::edge factor{ optimisation::edges::similarity_reprojection(camera, math::matrix<double, 3, 1>({ 0.0, 0.0, -5.0 }), math::se3<double>::identity(), false) };
        factor.add_vertex(vertex.get());
        const double observed[2] = { 320.0, 240.0 };
        factor.set_observation(math::matrix<double, 0, 0>(2, 1, &observed[0]));
        factor.compute_residual();
        REQUIRE(std::isfinite(factor.get_residual()[0][0]));
        REQUIRE(std::isfinite(factor.get_residual()[1][0]));
        REQUIRE(factor.get_residual()[0][0] > 0.0);
        REQUIRE(is_value_approx(factor.get_residual()[0][0], factor.get_residual()[1][0]));
    }

    {
        const math::sim3<double> truth(math::se3<double>(math::so3<double>::exp({ { 0.03, -0.06, 0.04 } }), { { 0.25, -0.15, 0.35 } }), 1.15);
        const math::se3<double> from_pose(math::so3<double>::exp({ { 0.02, 0.01, -0.03 } }), { { 0.1, -0.05, 0.2 } });
        const math::se3<double> to_pose(math::so3<double>::exp({ { -0.04, 0.02, 0.01 } }), { { -0.2, 0.15, 0.1 } });

        optimisation::factor_graph graph;
        double parameters[8] = {};
        similarity_parameters(math::sim3<double>::identity(), parameters);
        optimisation::vertex seed{ optimisation::vertices::similarity() };
        seed.set_parameters(&parameters[0], 8);
        optimisation::vertex* const vertex = graph.add_vertex(static_cast<optimisation::vertex&&>(seed));
        const optimisation::loss lossfunction(optimisation::losses::huber(3.0));

        unsigned int state = 7u;
        const auto next = [&state]() -> double {
            state = (state * 1664525u) + 1013904223u;
            return (static_cast<double>((state >> 16) % 2000u) / 1000.0) - 1.0;
        };
        for (int i = 0; i < 30; ++i) {
            const math::matrix<double, 3, 1> point({ next() * 2.0, next() * 1.5, 6.0 + next() });
            const math::matrix<double, 3, 1> mapped = truth * point;
            double in_to[2] = {};
            double in_from[2] = {};
            REQUIRE(camera.project((to_pose * mapped).data(), &in_to[0]));
            REQUIRE(camera.project((from_pose * point).data(), &in_from[0]));
            optimisation::edge forward{ optimisation::edges::similarity_reprojection(camera, point, to_pose, false) };
            forward.add_vertex(vertex);
            forward.set_observation(math::matrix<double, 0, 0>(2, 1, &in_to[0]));
            forward.set_loss(lossfunction);
            graph.add_edge(static_cast<optimisation::edge&&>(forward));
            optimisation::edge inverse{ optimisation::edges::similarity_reprojection(camera, mapped, from_pose, true) };
            inverse.add_vertex(vertex);
            inverse.set_observation(math::matrix<double, 0, 0>(2, 1, &in_from[0]));
            inverse.set_loss(lossfunction);
            graph.add_edge(static_cast<optimisation::edge&&>(inverse));
        }
        REQUIRE(graph.solve(50, true) > 0);
        const double* const refined = vertex->get_parameters();
        const math::sim3<double> recovered(math::se3<double>(math::so3<double>(refined[6], refined[3], refined[4], refined[5]), { { refined[0], refined[1], refined[2] } }), refined[7]);
        REQUIRE(is_value_approx(recovered.scale(), truth.scale(), 1e-3));
        for (size_t axis = 0; axis < 3; ++axis) {
            REQUIRE(is_value_approx(recovered.transformation().translation()[axis], truth.transformation().translation()[axis], 1e-3));
        }
        REQUIRE(graph.get_current_chi(true) < 1e-6);
    }

    return EXIT_SUCCESS;
}
