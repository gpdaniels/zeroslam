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

#include "optimisation/edges/reprojection_inverse_depth.hpp"

#include "math/lie.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/reprojection.hpp"
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
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

namespace {
    void set_pose(optimisation::vertex& vertex, const math::se3<double>& pose) {
        const double parameters[7] = { pose.translation()[0], pose.translation()[1], pose.translation()[2], pose.rotation().get_quaternion()[1], pose.rotation().get_quaternion()[2], pose.rotation().get_quaternion()[3], pose.rotation().get_quaternion()[0] };
        REQUIRE(vertex.set_parameters(&parameters[0], 7));
    }
}

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

    static_assert(sizeof(optimisation::edges::reprojection_inverse_depth) <= optimisation::edge::maximum_size, "The inverse depth reprojection edge must fit.");

    const sensor::camera::pinhole<double> camera_model(std::vector<double>{ 500.0, 500.0, 320.0, 240.0 }.data(), 4);
    const math::se3<double> anchor_camera_from_world(math::so3<double>::exp({ { 0.1, -0.05, 0.2 } }), { { 0.4, -0.3, 0.2 } });
    const math::se3<double> anchor = anchor_camera_from_world.inverse();
    const math::se3<double> pose(math::so3<double>::exp({ { -0.08, 0.12, 0.03 } }), { { -0.5, 0.2, 0.7 } });
    const math::matrix<double, 3, 1> world_point({ 1.2, -0.4, 6.0 });
    const math::matrix<double, 3, 1> anchored = anchor_camera_from_world * world_point;
    const double landmark[3] = { anchored[0] / anchored[2], anchored[1] / anchored[2], 1.0 / anchored[2] };

    {
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, pose);
        optimisation::vertex inverse_vertex{ optimisation::vertices::point() };
        REQUIRE(inverse_vertex.set_parameters(&landmark[0], 3));
        optimisation::vertex xyz_vertex{ optimisation::vertices::point() };
        REQUIRE(xyz_vertex.set_parameters(world_point.data(), 3));
        optimisation::edge inverse_edge{ optimisation::edges::reprojection_inverse_depth(sensor::camera::model<double>(camera_model), anchor.rotation().get_matrix(), anchor.translation()) };
        optimisation::edge xyz_edge{ optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)) };
        const double observation[2] = { 300.0, 250.0 };
        inverse_edge.set_observation(math::matrix<double, 0, 0>(2, 1, &observation[0]));
        xyz_edge.set_observation(math::matrix<double, 0, 0>(2, 1, &observation[0]));
        inverse_edge.add_vertex(&pose_vertex);
        inverse_edge.add_vertex(&inverse_vertex);
        xyz_edge.add_vertex(&pose_vertex);
        xyz_edge.add_vertex(&xyz_vertex);
        inverse_edge.compute_residual();
        xyz_edge.compute_residual();
        REQUIRE(std::abs(inverse_edge.get_residual()[0][0] - xyz_edge.get_residual()[0][0]) < 1.0e-9);
        REQUIRE(std::abs(inverse_edge.get_residual()[1][0] - xyz_edge.get_residual()[1][0]) < 1.0e-9);
        inverse_edge.compute_jacobians();
        xyz_edge.compute_jacobians();
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 6; ++column) {
                REQUIRE(std::abs(inverse_edge.get_jacobians()[0][row][column] - xyz_edge.get_jacobians()[0][row][column]) < 1.0e-9 * (1.0 + std::abs(xyz_edge.get_jacobians()[0][row][column])));
            }
        }
    }

    for (int at_infinity = 0; at_infinity < 2; ++at_infinity) {
        const double state[3] = { landmark[0], landmark[1], at_infinity ? 0.0 : landmark[2] };
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, pose);
        optimisation::vertex point_vertex{ optimisation::vertices::point() };
        REQUIRE(point_vertex.set_parameters(&state[0], 3));
        optimisation::edge edge{ optimisation::edges::reprojection_inverse_depth(sensor::camera::model<double>(camera_model), anchor.rotation().get_matrix(), anchor.translation()) };
        const double observation[2] = { 310.0, 230.0 };
        edge.set_observation(math::matrix<double, 0, 0>(2, 1, &observation[0]));
        edge.add_vertex(&pose_vertex);
        edge.add_vertex(&point_vertex);
        edge.compute_jacobians();
        const math::matrix<double, 0, 0> jacobian_pose = edge.get_jacobians()[0];
        const math::matrix<double, 0, 0> jacobian_point = edge.get_jacobians()[1];
        const double step = 1.0e-6;
        for (size_t column = 0; column < 6; ++column) {
            double delta[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            delta[column] = step;
            set_pose(pose_vertex, pose);
            pose_vertex.plus(&delta[0]);
            edge.compute_residual();
            const double forward[2] = { edge.get_residual()[0][0], edge.get_residual()[1][0] };
            delta[column] = -step;
            set_pose(pose_vertex, pose);
            pose_vertex.plus(&delta[0]);
            edge.compute_residual();
            const double backward[2] = { edge.get_residual()[0][0], edge.get_residual()[1][0] };
            for (size_t row = 0; row < 2; ++row) {
                const double numeric = (forward[row] - backward[row]) / (2.0 * step);
                REQUIRE(std::abs(numeric - jacobian_pose[row][column]) < 1.0e-4 * (1.0 + std::abs(numeric)));
            }
        }
        set_pose(pose_vertex, pose);
        for (size_t column = 0; column < 3; ++column) {
            double delta[3] = { 0.0, 0.0, 0.0 };
            delta[column] = step;
            REQUIRE(point_vertex.set_parameters(&state[0], 3));
            point_vertex.plus(&delta[0]);
            edge.compute_residual();
            const double forward[2] = { edge.get_residual()[0][0], edge.get_residual()[1][0] };
            delta[column] = -step;
            REQUIRE(point_vertex.set_parameters(&state[0], 3));
            point_vertex.plus(&delta[0]);
            edge.compute_residual();
            const double backward[2] = { edge.get_residual()[0][0], edge.get_residual()[1][0] };
            for (size_t row = 0; row < 2; ++row) {
                const double numeric = (forward[row] - backward[row]) / (2.0 * step);
                REQUIRE(std::abs(numeric - jacobian_point[row][column]) < 1.0e-4 * (1.0 + std::abs(numeric)));
            }
        }
        if (at_infinity) {
            for (size_t row = 0; row < 2; ++row) {
                for (size_t column = 3; column < 6; ++column) {
                    REQUIRE(jacobian_pose[row][column] == 0.0);
                }
            }
        }
    }

    {
        // A negative inverse depth is projected like any other, as the point beyond infinity it describes, so the cost is
        // smooth through infinity and the map clamps the inverse depth after each adjustment. Treating it as behind the
        // camera, with a step in the cost or a barrier at infinity, lost tracking on LaMAria and EuRoC.
        const double mirrored[3] = { landmark[0], landmark[1], -landmark[2] };
        const math::matrix<double, 3, 3> rotation_combined = pose.rotation().get_matrix() * anchor.rotation().get_matrix();
        const math::matrix<double, 3, 1> rho_column = (pose.rotation().get_matrix() * anchor.translation()) + pose.translation();
        const math::matrix<double, 3, 1> q = (rotation_combined * math::matrix<double, 3, 1>{ { mirrored[0], mirrored[1], 1.0 } }) + (rho_column * mirrored[2]);
        math::matrix<double, 2, 1> projected;
        REQUIRE(camera_model.project(q.data(), projected.data()));
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, pose);
        optimisation::vertex point_vertex{ optimisation::vertices::point() };
        REQUIRE(point_vertex.set_parameters(&mirrored[0], 3));
        optimisation::edge edge{ optimisation::edges::reprojection_inverse_depth(sensor::camera::model<double>(camera_model), anchor.rotation().get_matrix(), anchor.translation()) };
        const double observation[2] = { 300.0, 250.0 };
        edge.set_observation(math::matrix<double, 0, 0>(2, 1, &observation[0]));
        edge.add_vertex(&pose_vertex);
        edge.add_vertex(&point_vertex);
        edge.compute_residual();
        REQUIRE(std::abs(edge.get_residual()[0][0] - (observation[0] - projected[0])) < 1.0e-9);
        REQUIRE(std::abs(edge.get_residual()[1][0] - (observation[1] - projected[1])) < 1.0e-9);
        REQUIRE(edge.is_measurable());
        edge.compute_jacobians();
        const math::matrix<double, 0, 0> jacobian_point = edge.get_jacobians()[1];
        const double step = 1.0e-6;
        for (size_t column = 0; column < 3; ++column) {
            double delta[3] = { 0.0, 0.0, 0.0 };
            delta[column] = step;
            REQUIRE(point_vertex.set_parameters(&mirrored[0], 3));
            point_vertex.plus(&delta[0]);
            edge.compute_residual();
            const double forward[2] = { edge.get_residual()[0][0], edge.get_residual()[1][0] };
            delta[column] = -step;
            REQUIRE(point_vertex.set_parameters(&mirrored[0], 3));
            point_vertex.plus(&delta[0]);
            edge.compute_residual();
            const double backward[2] = { edge.get_residual()[0][0], edge.get_residual()[1][0] };
            for (size_t row = 0; row < 2; ++row) {
                const double numeric = (forward[row] - backward[row]) / (2.0 * step);
                REQUIRE(std::abs(numeric - jacobian_point[row][column]) < 1.0e-4 * (1.0 + std::abs(numeric)));
            }
        }
    }

    {
        // A point behind the observer does not project, and the residual that stands in for its reprojection is far within
        // the inlier bound at every inverse depth, so the edge has to say that it measures nothing there. An observer facing
        // the other way along the anchor's ray has the point behind it whatever the inverse depth.
        const math::se3<double> facing_away(math::so3<double>::exp({ { 0.0, 3.14159265358979323846, 0.0 } }) * anchor_camera_from_world.rotation(), { { 0.0, 0.0, 0.0 } });
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, facing_away);
        for (const double inverse_depth : { 1.0e-3, 0.05, 1.0 / 6.0, 0.5 }) {
            const double parameters[3] = { 0.0, 0.0, inverse_depth };
            optimisation::vertex point_vertex{ optimisation::vertices::point() };
            REQUIRE(point_vertex.set_parameters(&parameters[0], 3));
            optimisation::edge edge{ optimisation::edges::reprojection_inverse_depth(sensor::camera::model<double>(camera_model), anchor.rotation().get_matrix(), anchor.translation()) };
            const double observation[2] = { 320.0, 240.0 };
            edge.set_observation(math::matrix<double, 0, 0>(2, 1, &observation[0]));
            edge.add_vertex(&pose_vertex);
            edge.add_vertex(&point_vertex);
            edge.compute_residual();
            REQUIRE(edge.get_residual()[0][0] == edge.get_residual()[1][0]);
            REQUIRE(edge.chi2() < 5.991);
            REQUIRE(!edge.is_measurable());
        }
        // The observer at the anchor, facing the same way, measures the same points.
        set_pose(pose_vertex, anchor_camera_from_world);
        const double parameters[3] = { 0.0, 0.0, 0.05 };
        optimisation::vertex point_vertex{ optimisation::vertices::point() };
        REQUIRE(point_vertex.set_parameters(&parameters[0], 3));
        optimisation::edge edge{ optimisation::edges::reprojection_inverse_depth(sensor::camera::model<double>(camera_model), anchor.rotation().get_matrix(), anchor.translation()) };
        const double observation[2] = { 320.0, 240.0 };
        edge.set_observation(math::matrix<double, 0, 0>(2, 1, &observation[0]));
        edge.add_vertex(&pose_vertex);
        edge.add_vertex(&point_vertex);
        edge.compute_residual();
        REQUIRE(edge.is_measurable());
        REQUIRE(edge.chi2() < 1.0e-12);
    }

    {
        // With an intrinsics vertex the edge projects through the vertex's camera and gives the jacobian by its free
        // parameters; every jacobian matches the numeric one, for each mask, and at the camera the edge was made with
        // it measures exactly what the edge holding that camera fixed does.
        const double camera_parameters[12] = { 520.0, 515.0, 322.0, 241.0, -0.21, 0.06, 0.0012, -0.0021, 0.01, 0.0, 0.0, 0.0 };
        const sensor::camera::model<double> camera{ sensor::camera::pinhole_radial_tangential<double>(&camera_parameters[0], 12) };
        const math::matrix<double, 3, 3> anchor_rotation = math::so3<double>::exp({ { 0.02, -0.05, 0.01 } }).get_matrix();
        const math::matrix<double, 3, 1> anchor_translation = { { 0.1, 0.05, -0.2 } };
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
            const double landmark_parameters[3] = { 0.12, -0.08, 0.35 };
            REQUIRE(landmark_vertex.set_parameters(&landmark_parameters[0], 3));
            landmark_vertex.set_marginalised(true);
            optimisation::vertex intrinsics_vertex = optimisation::vertices::intrinsics::make(camera, mask, 12);
            REQUIRE(intrinsics_vertex.is_valid());

            optimisation::edge edge{ optimisation::edges::reprojection_inverse_depth(camera, anchor_rotation, anchor_translation, true) };
            REQUIRE(edge.required_vertices() == 3);
            edge.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { 300.0, 200.0 } }.data()));
            REQUIRE(edge.add_vertex(&pose_vertex));
            REQUIRE(edge.add_vertex(&landmark_vertex));
            REQUIRE(edge.add_vertex(&intrinsics_vertex));
            REQUIRE(!edge.add_vertex(&intrinsics_vertex));
            edge.compute_residual();
            edge.compute_jacobians();
            REQUIRE(edge.is_measurable());

            optimisation::edge fixed{ optimisation::edges::reprojection_inverse_depth(camera, anchor_rotation, anchor_translation) };
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
            optimisation::edge moved_fixed{ optimisation::edges::reprojection_inverse_depth(moved, anchor_rotation, anchor_translation) };
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
        const math::matrix<double, 3, 3> anchor_rotation = math::so3<double>::exp({ { 0.02, -0.05, 0.01 } }).get_matrix();
        const math::matrix<double, 3, 1> anchor_translation = { { 0.1, 0.05, -0.2 } };
        const bool all[4] = { true, true, true, true };
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        const double pose_parameters[7] = { 0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 1.0 };
        REQUIRE(pose_vertex.set_parameters(&pose_parameters[0], 7));
        optimisation::vertex landmark_vertex{ optimisation::vertices::point() };
        const double landmark_parameters[3] = { 0.12, -0.08, 0.35 };
        REQUIRE(landmark_vertex.set_parameters(&landmark_parameters[0], 3));
        optimisation::vertex intrinsics_vertex = optimisation::vertices::intrinsics::make(camera, &all[0], 4);
        optimisation::edge edge{ optimisation::edges::reprojection_inverse_depth(camera, anchor_rotation, anchor_translation, true) };
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
