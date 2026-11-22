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

#include "optimisation/edges/depth_inverse_depth.hpp"

#include "math/lie.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/depth.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/point.hpp"
#include "optimisation/vertices/pose.hpp"
#include "sensor/depth.hpp"

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

    static_assert(sizeof(optimisation::edges::depth_inverse_depth) <= optimisation::edge::maximum_size, "The inverse depth depth edge must fit.");

    const math::se3<double> anchor_camera_from_world(math::so3<double>::exp({ { 0.1, -0.05, 0.2 } }), { { 0.4, -0.3, 0.2 } });
    const math::se3<double> anchor = anchor_camera_from_world.inverse();
    const math::se3<double> pose(math::so3<double>::exp({ { -0.08, 0.12, 0.03 } }), { { -0.5, 0.2, 0.7 } });
    const math::matrix<double, 3, 1> world_point({ 1.2, -0.4, 4.0 });
    const math::matrix<double, 3, 1> anchored = anchor_camera_from_world * world_point;
    const double landmark[3] = { anchored[0] / anchored[2], anchored[1] / anchored[2], 1.0 / anchored[2] };

    for (const sensor::depth::measure_kind measure : { sensor::depth::measure_kind::axial, sensor::depth::measure_kind::radial }) {
        // The same residual and pose Jacobian as the edge of a world point at the same place.
        {
            optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
            set_pose(pose_vertex, pose);
            optimisation::vertex inverse_vertex{ optimisation::vertices::point() };
            REQUIRE(inverse_vertex.set_parameters(&landmark[0], 3));
            optimisation::vertex xyz_vertex{ optimisation::vertices::point() };
            REQUIRE(xyz_vertex.set_parameters(world_point.data(), 3));
            optimisation::edge inverse_edge{ optimisation::edges::depth_inverse_depth(measure, anchor.rotation().get_matrix(), anchor.translation()) };
            optimisation::edge xyz_edge{ optimisation::edges::depth(measure) };
            const double observed = 3.1;
            inverse_edge.set_observation(math::matrix<double, 0, 0>(1, 1, &observed));
            xyz_edge.set_observation(math::matrix<double, 0, 0>(1, 1, &observed));
            inverse_edge.add_vertex(&pose_vertex);
            inverse_edge.add_vertex(&inverse_vertex);
            xyz_edge.add_vertex(&pose_vertex);
            xyz_edge.add_vertex(&xyz_vertex);
            inverse_edge.compute_residual();
            xyz_edge.compute_residual();
            REQUIRE(std::abs(inverse_edge.get_residual()[0][0] - xyz_edge.get_residual()[0][0]) < 1.0e-12);
            REQUIRE(inverse_edge.is_measurable());
            inverse_edge.compute_jacobians();
            xyz_edge.compute_jacobians();
            for (size_t column = 0; column < 6; ++column) {
                REQUIRE(std::abs(inverse_edge.get_jacobians()[0][0][column] - xyz_edge.get_jacobians()[0][0][column]) < 1.0e-9 * (1.0 + std::abs(xyz_edge.get_jacobians()[0][0][column])));
            }
        }

        // Central differences against the analytic Jacobians, also for a landmark at infinity, where the predicted inverse
        // depth is zero and the residual stays smooth.
        for (int at_infinity = 0; at_infinity < 2; ++at_infinity) {
            const double state[3] = { landmark[0], landmark[1], at_infinity ? 0.0 : landmark[2] };
            optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
            set_pose(pose_vertex, pose);
            optimisation::vertex point_vertex{ optimisation::vertices::point() };
            REQUIRE(point_vertex.set_parameters(&state[0], 3));
            optimisation::edge edge{ optimisation::edges::depth_inverse_depth(measure, anchor.rotation().get_matrix(), anchor.translation()) };
            const double observed = 2.5;
            edge.set_observation(math::matrix<double, 0, 0>(1, 1, &observed));
            edge.add_vertex(&pose_vertex);
            edge.add_vertex(&point_vertex);
            edge.compute_residual();
            if (at_infinity) {
                REQUIRE(std::abs(edge.get_residual()[0][0] - (1.0 / observed)) < 1.0e-12);
            }
            edge.compute_jacobians();
            const math::matrix<double, 0, 0> jacobian_pose = edge.get_jacobians()[0];
            const math::matrix<double, 0, 0> jacobian_point = edge.get_jacobians()[1];
            REQUIRE((jacobian_pose.rows() == 1) && (jacobian_pose.cols() == 6));
            REQUIRE((jacobian_point.rows() == 1) && (jacobian_point.cols() == 3));
            const double step = 1.0e-6;
            for (size_t column = 0; column < 6; ++column) {
                double delta[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
                delta[column] = step;
                set_pose(pose_vertex, pose);
                pose_vertex.plus(&delta[0]);
                edge.compute_residual();
                const double forward = edge.get_residual()[0][0];
                delta[column] = -step;
                set_pose(pose_vertex, pose);
                pose_vertex.plus(&delta[0]);
                edge.compute_residual();
                const double backward = edge.get_residual()[0][0];
                const double numeric = (forward - backward) / (2.0 * step);
                REQUIRE(std::abs(numeric - jacobian_pose[0][column]) < 1.0e-6 * (1.0 + std::abs(numeric)));
            }
            set_pose(pose_vertex, pose);
            for (size_t column = 0; column < 3; ++column) {
                double delta[3] = { 0.0, 0.0, 0.0 };
                delta[column] = step;
                REQUIRE(point_vertex.set_parameters(&state[0], 3));
                point_vertex.plus(&delta[0]);
                edge.compute_residual();
                const double forward = edge.get_residual()[0][0];
                delta[column] = -step;
                REQUIRE(point_vertex.set_parameters(&state[0], 3));
                point_vertex.plus(&delta[0]);
                edge.compute_residual();
                const double backward = edge.get_residual()[0][0];
                const double numeric = (forward - backward) / (2.0 * step);
                REQUIRE(std::abs(numeric - jacobian_point[0][column]) < 1.0e-6 * (1.0 + std::abs(numeric)));
            }
            if (at_infinity) {
                // At infinity the camera's translation does not move the predicted inverse depth.
                for (size_t column = 3; column < 6; ++column) {
                    REQUIRE(jacobian_pose[0][column] == 0.0);
                }
            }
        }
    }

    {
        // A bearing that points behind the observing camera is not measurable: the landmark in front of the anchor, seen by a
        // camera turned to face away from it.
        const math::se3<double> facing_away(math::so3<double>::exp({ { 0.0, 3.0, 0.0 } }), { { 0.0, 0.0, 0.0 } });
        REQUIRE((facing_away * world_point)[2] < 0.0);
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, facing_away);
        optimisation::vertex point_vertex{ optimisation::vertices::point() };
        REQUIRE(point_vertex.set_parameters(&landmark[0], 3));
        optimisation::edge edge{ optimisation::edges::depth_inverse_depth(sensor::depth::measure_kind::axial, anchor.rotation().get_matrix(), anchor.translation()) };
        const double observed = 2.0;
        edge.set_observation(math::matrix<double, 0, 0>(1, 1, &observed));
        edge.add_vertex(&pose_vertex);
        edge.add_vertex(&point_vertex);
        REQUIRE(!edge.is_measurable());
        edge.compute_residual();
        REQUIRE(std::abs(edge.get_residual()[0][0] - 0.5) < 1.0e-12);
    }

    return EXIT_SUCCESS;
}
