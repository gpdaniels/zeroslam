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
#include "optimisation/vertices/point.hpp"
#include "optimisation/vertices/pose.hpp"
#include "sensor/camera/model.hpp"
#include "sensor/camera/pinhole.hpp"

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

    return EXIT_SUCCESS;
}
