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

#include "optimisation/edges/reprojection.hpp"

#include "math/lie.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/intrinsics.hpp"
#include "optimisation/vertices/point.hpp"
#include "optimisation/vertices/pose.hpp"
#include "sensor/camera/model.hpp"
#include "sensor/camera/pinhole.hpp"
#include "sensor/camera/pinhole_radial_tangential.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>
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

// The central difference jacobian of an edge's residual by a vertex's local dimensions.
static std::vector<double> numeric_jacobian(optimisation::edge& edge, optimisation::vertex& vertex, const double delta) {
    const size_t dimensions = static_cast<size_t>(vertex.get_local_dimensions());
    std::vector<double> jacobian(2 * dimensions, 0.0);
    for (size_t d = 0; d < dimensions; ++d) {
        std::vector<double> step(dimensions, 0.0);
        step[d] = delta;
        vertex.backup();
        vertex.plus(step.data());
        edge.compute_residual();
        const double after[2] = { edge.get_residual()[0][0], edge.get_residual()[1][0] };
        vertex.restore();
        step[d] = -delta;
        vertex.backup();
        vertex.plus(step.data());
        edge.compute_residual();
        const double before[2] = { edge.get_residual()[0][0], edge.get_residual()[1][0] };
        vertex.restore();
        for (size_t r = 0; r < 2; ++r) {
            jacobian[(r * dimensions) + d] = (after[r] - before[r]) / (2.0 * delta);
        }
    }
    edge.compute_residual();
    return jacobian;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::reprojection) <= optimisation::edge::maximum_size, "The reprojection edge must fit.");

    {
        const sensor::camera::pinhole<double> camera_model(std::vector<double>{ 1.0, 1.0, 0.0, 0.0 }.data(), 4);

        std::unique_ptr<optimisation::vertex> camera_vertex = std::make_unique<optimisation::vertex>(optimisation::vertices::pose());
        const double camera_parameters_pose[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
        camera_vertex->set_parameters(&camera_parameters_pose[0], 7);
        camera_vertex->set_fixed(false);

        {
            std::unique_ptr<optimisation::vertex> landmark_vertex = std::make_unique<optimisation::vertex>(optimisation::vertices::point());
            const double landmark_location[3] = { 0.0, 0.0, 1.0 };
            landmark_vertex->set_parameters(&landmark_location[0], 3);
            landmark_vertex->set_fixed(false);
            landmark_vertex->set_marginalised(true);

            std::unique_ptr<optimisation::edge> edge = std::make_unique<optimisation::edge>(optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)));
            edge->set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { 0.1, -0.2 } }.data()));
            edge->add_vertex(camera_vertex.get());
            edge->add_vertex(landmark_vertex.get());

            edge->compute_residual();
            edge->compute_jacobians();

            REQUIRE(edge->get_residual()[0][0] == 0.1);
            REQUIRE(edge->get_residual()[1][0] == -0.2);

            const math::matrix<double, 0, 0>& jacobian_pose = edge->get_jacobians()[0];
            const double expected_pose[2][6] = {
                { 0.0, -1.0, 0.0, -1.0, 0.0, 0.0 },
                { 1.0, 0.0, 0.0, 0.0, -1.0, 0.0 }
            };
            for (size_t r = 0; r < 2; ++r) {
                for (size_t c = 0; c < 6; ++c) {
                    REQUIRE(jacobian_pose[r][c] == expected_pose[r][c]);
                }
            }

            const math::matrix<double, 0, 0>& jacobian_landmark = edge->get_jacobians()[1];
            const double expected_landmark[2][3] = {
                { -1.0, 0.0, 0.0 },
                { 0.0, -1.0, 0.0 }
            };
            for (size_t r = 0; r < 2; ++r) {
                for (size_t c = 0; c < 3; ++c) {
                    REQUIRE(jacobian_landmark[r][c] == expected_landmark[r][c]);
                }
            }
        }

        {
            std::unique_ptr<optimisation::vertex> landmark_vertex = std::make_unique<optimisation::vertex>(optimisation::vertices::point());
            const double landmark_location[3] = { 0.1, 0.2, -10.0 };
            landmark_vertex->set_parameters(&landmark_location[0], 3);
            landmark_vertex->set_fixed(false);
            landmark_vertex->set_marginalised(true);

            std::unique_ptr<optimisation::edge> edge = std::make_unique<optimisation::edge>(optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)));
            edge->set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { 0.1, -0.2 } }.data()));
            edge->add_vertex(camera_vertex.get());
            edge->add_vertex(landmark_vertex.get());

            edge->compute_residual();

            REQUIRE(std::isfinite(edge->get_residual()[0][0]));
            REQUIRE(std::isfinite(edge->get_residual()[1][0]));
            REQUIRE(!is_value_approx(edge->get_residual()[0][0], 0.0));
            REQUIRE(!is_value_approx(edge->get_residual()[1][0], 0.0));

            const double chi2 = edge->chi2();
            REQUIRE(std::isfinite(chi2));
            REQUIRE(chi2 > 1.0);

            edge->compute_jacobians();
            const math::matrix<double, 0, 0>& jacobian_pose = edge->get_jacobians()[0];
            const math::matrix<double, 0, 0>& jacobian_landmark = edge->get_jacobians()[1];

            bool jacobian_pose_nonzero = false;
            for (size_t r = 0; r < 2; ++r) {
                for (size_t c = 0; c < 6; ++c) {
                    REQUIRE(std::isfinite(jacobian_pose[r][c]));
                    jacobian_pose_nonzero = jacobian_pose_nonzero || (jacobian_pose[r][c] != 0.0);
                }
            }
            bool jacobian_landmark_nonzero = false;
            for (size_t r = 0; r < 2; ++r) {
                for (size_t c = 0; c < 3; ++c) {
                    REQUIRE(std::isfinite(jacobian_landmark[r][c]));
                    jacobian_landmark_nonzero = jacobian_landmark_nonzero || (jacobian_landmark[r][c] != 0.0);
                }
            }
            REQUIRE(jacobian_pose_nonzero);
            REQUIRE(jacobian_landmark_nonzero);
            REQUIRE(jacobian_landmark[0][2] < 0.0);
            REQUIRE(jacobian_landmark[1][2] < 0.0);
        }
    }

    {
        // A point the camera cannot project, here one behind it, gives both rows minus the slope times its depth, a weak
        // pull back in front that costs little just behind the camera, and the Jacobian of that.
        const sensor::camera::pinhole<double> camera_model(std::vector<double>{ 500.0, 500.0, 320.0, 240.0 }.data(), 4);
        optimisation::vertex camera_vertex{ optimisation::vertices::pose() };
        const double camera_parameters_pose[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
        REQUIRE(camera_vertex.set_parameters(&camera_parameters_pose[0], 7));
        for (const double depth : { -1.0e-9, -5.0, -40.0 }) {
            optimisation::vertex landmark_vertex{ optimisation::vertices::point() };
            const double landmark_location[3] = { 0.3, -0.2, depth };
            REQUIRE(landmark_vertex.set_parameters(&landmark_location[0], 3));
            optimisation::edge edge{ optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)) };
            edge.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { 320.0, 240.0 } }.data()));
            edge.add_vertex(&camera_vertex);
            edge.add_vertex(&landmark_vertex);
            edge.compute_residual();
            REQUIRE(edge.get_residual()[0][0] == edge.get_residual()[1][0]);
            REQUIRE(is_value_approx(edge.get_residual()[0][0], -optimisation::edges::reprojection::behind_camera_slope * depth));
            // The residual is no reprojection error, so however small it is the edge is not measurable and no inlier.
            REQUIRE(!edge.is_measurable());
            REQUIRE((depth < -1.0) || (edge.chi2() < 5.991));
            edge.compute_jacobians();
            REQUIRE(edge.get_jacobians()[1][0][2] == -optimisation::edges::reprojection::behind_camera_slope);
            REQUIRE(edge.get_jacobians()[1][1][2] == -optimisation::edges::reprojection::behind_camera_slope);
            REQUIRE(edge.get_jacobians()[1][0][0] == 0.0);
            REQUIRE(edge.get_jacobians()[1][0][1] == 0.0);
        }
    }

    {
        // The same point in front of the camera is measurable, and stops being so when the pose moves the camera past it.
        const sensor::camera::pinhole<double> camera_model(std::vector<double>{ 500.0, 500.0, 320.0, 240.0 }.data(), 4);
        optimisation::vertex camera_vertex{ optimisation::vertices::pose() };
        const double camera_parameters_pose[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
        REQUIRE(camera_vertex.set_parameters(&camera_parameters_pose[0], 7));
        optimisation::vertex landmark_vertex{ optimisation::vertices::point() };
        const double landmark_location[3] = { 0.3, -0.2, 2.0 };
        REQUIRE(landmark_vertex.set_parameters(&landmark_location[0], 3));
        optimisation::edge edge{ optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)) };
        edge.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { 395.0, 190.0 } }.data()));
        edge.add_vertex(&camera_vertex);
        edge.add_vertex(&landmark_vertex);
        edge.compute_residual();
        REQUIRE(edge.is_measurable());
        REQUIRE(edge.chi2() < 1.0e-12);
        const double camera_parameters_past[7] = { 0.0, 0.0, -2.5, 0.0, 0.0, 0.0, 1.0 };
        REQUIRE(camera_vertex.set_parameters(&camera_parameters_past[0], 7));
        edge.compute_residual();
        REQUIRE(!edge.is_measurable());
        REQUIRE(edge.chi2() < 5.991);
    }

    {
        // With an intrinsics vertex the edge projects through the vertex's camera and gives the jacobian by its free
        // parameters; every jacobian matches the numeric one, for each mask, and at the camera the edge was made with
        // it measures exactly what the edge holding that camera fixed does.
        const double camera_parameters[12] = { 520.0, 515.0, 322.0, 241.0, -0.21, 0.06, 0.0012, -0.0021, 0.01, 0.0, 0.0, 0.0 };
        const sensor::camera::model<double> camera{ sensor::camera::pinhole_radial_tangential<double>(&camera_parameters[0], 12) };
        const bool masks[3][12] = {
            { true, true, false, false, false, false, false, false, false, false, false, false },
            { true, true, true, true, false, false, false, false, false, false, false, false },
            { true, true, true, true, true, true, true, true, true, false, false, false }
        };
        for (const bool* const mask : { &masks[0][0], &masks[1][0], &masks[2][0] }) {
            optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
            const math::se3<double> camera_pose(math::so3<double>::exp({ { 0.05, -0.1, 0.03 } }), { { 0.1, -0.05, 0.2 } });
            const double pose_parameters[7] = { camera_pose.translation()[0], camera_pose.translation()[1], camera_pose.translation()[2], camera_pose.rotation().get_quaternion()[1], camera_pose.rotation().get_quaternion()[2], camera_pose.rotation().get_quaternion()[3], camera_pose.rotation().get_quaternion()[0] };
            REQUIRE(pose_vertex.set_parameters(&pose_parameters[0], 7));
            optimisation::vertex landmark_vertex{ optimisation::vertices::point() };
            const double landmark_parameters[3] = { 0.4, -0.3, 3.0 };
            REQUIRE(landmark_vertex.set_parameters(&landmark_parameters[0], 3));
            landmark_vertex.set_marginalised(true);
            optimisation::vertex intrinsics_vertex = optimisation::vertices::intrinsics::make(camera, mask, 12);
            REQUIRE(intrinsics_vertex.is_valid());

            optimisation::edge edge{ optimisation::edges::reprojection(camera, true) };
            REQUIRE(edge.required_vertices() == 3);
            edge.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { 300.0, 200.0 } }.data()));
            REQUIRE(edge.add_vertex(&pose_vertex));
            REQUIRE(edge.add_vertex(&landmark_vertex));
            REQUIRE(edge.add_vertex(&intrinsics_vertex));
            REQUIRE(!edge.add_vertex(&intrinsics_vertex));
            edge.compute_residual();
            edge.compute_jacobians();
            REQUIRE(edge.is_measurable());

            optimisation::edge fixed{ optimisation::edges::reprojection(camera) };
            REQUIRE(fixed.required_vertices() == 2);
            fixed.set_observation(edge.get_observation());
            REQUIRE(fixed.add_vertex(&pose_vertex));
            REQUIRE(fixed.add_vertex(&landmark_vertex));
            REQUIRE(!fixed.add_vertex(&intrinsics_vertex));
            fixed.compute_residual();
            fixed.compute_jacobians();
            for (size_t r = 0; r < 2; ++r) {
                REQUIRE(edge.get_residual()[r][0] == fixed.get_residual()[r][0]);
                for (size_t c = 0; c < 6; ++c) {
                    REQUIRE(edge.get_jacobians()[0][r][c] == fixed.get_jacobians()[0][r][c]);
                }
                for (size_t c = 0; c < 3; ++c) {
                    REQUIRE(edge.get_jacobians()[1][r][c] == fixed.get_jacobians()[1][r][c]);
                }
            }

            optimisation::vertex* const vertices[3] = { &pose_vertex, &landmark_vertex, &intrinsics_vertex };
            const double deltas[3] = { 1e-6, 1e-6, 1e-5 };
            for (size_t slot = 0; slot < 3; ++slot) {
                const std::vector<double> numeric = numeric_jacobian(edge, *vertices[slot], deltas[slot]);
                const math::matrix<double, 0, 0>& analytic = edge.get_jacobians()[slot];
                REQUIRE(analytic.rows() == 2);
                REQUIRE(analytic.cols() == static_cast<size_t>(vertices[slot]->get_local_dimensions()));
                for (size_t r = 0; r < 2; ++r) {
                    for (size_t c = 0; c < analytic.cols(); ++c) {
                        REQUIRE(is_value_approx(analytic[r][c], numeric[(r * analytic.cols()) + c], 1e-5));
                    }
                }
            }

            // Moving the intrinsics moves the projection as a camera made from the new parameters projects.
            std::vector<double> step(static_cast<size_t>(intrinsics_vertex.get_local_dimensions()), 0.0);
            step[0] = 25.0;
            intrinsics_vertex.plus(step.data());
            edge.compute_residual();
            const sensor::camera::model<double> moved{ sensor::camera::pinhole_radial_tangential<double>(intrinsics_vertex.get_parameters(), 12) };
            optimisation::edge moved_fixed{ optimisation::edges::reprojection(moved) };
            moved_fixed.set_observation(edge.get_observation());
            REQUIRE(moved_fixed.add_vertex(&pose_vertex));
            REQUIRE(moved_fixed.add_vertex(&landmark_vertex));
            moved_fixed.compute_residual();
            REQUIRE(edge.get_residual()[0][0] == moved_fixed.get_residual()[0][0]);
            REQUIRE(edge.get_residual()[1][0] == moved_fixed.get_residual()[1][0]);
            REQUIRE(edge.get_residual()[0][0] != fixed.get_residual()[0][0]);
        }
    }

    {
        // A point behind the camera gives the behind camera residual, which the intrinsics do not change.
        const double camera_parameters[4] = { 500.0, 500.0, 320.0, 240.0 };
        const sensor::camera::model<double> camera{ sensor::camera::pinhole<double>(&camera_parameters[0], 4) };
        const bool all[4] = { true, true, true, true };
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        const double pose_parameters[7] = { 0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 1.0 };
        REQUIRE(pose_vertex.set_parameters(&pose_parameters[0], 7));
        optimisation::vertex landmark_vertex{ optimisation::vertices::point() };
        const double landmark_parameters[3] = { 0.4, -0.3, 3.0 };
        REQUIRE(landmark_vertex.set_parameters(&landmark_parameters[0], 3));
        optimisation::vertex intrinsics_vertex = optimisation::vertices::intrinsics::make(camera, &all[0], 4);
        optimisation::edge edge{ optimisation::edges::reprojection(camera, true) };
        edge.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { 320.0, 240.0 } }.data()));
        REQUIRE(edge.add_vertex(&pose_vertex));
        REQUIRE(edge.add_vertex(&landmark_vertex));
        REQUIRE(edge.add_vertex(&intrinsics_vertex));
        edge.compute_residual();
        REQUIRE(!edge.is_measurable());
        REQUIRE(edge.get_residual()[0][0] == edge.get_residual()[1][0]);
        edge.compute_jacobians();
        REQUIRE(edge.get_jacobians()[2].rows() == 2);
        REQUIRE(edge.get_jacobians()[2].cols() == 4);
        for (size_t r = 0; r < 2; ++r) {
            for (size_t c = 0; c < 4; ++c) {
                REQUIRE(edge.get_jacobians()[2][r][c] == 0.0);
            }
        }
    }

    return EXIT_SUCCESS;
}
