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

#include "optimisation/edges/depth.hpp"

#include "math/lie.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/reprojection.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/point.hpp"
#include "optimisation/vertices/pose.hpp"
#include "sensor/camera/model.hpp"
#include "sensor/camera/pinhole.hpp"
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

    // Central differences of the residual against the analytic Jacobians, for the pose's six and the point's three local
    // coordinates.
    void check_jacobians(const sensor::depth::measure_kind measure, const math::se3<double>& pose, const math::matrix<double, 3, 1>& world_point, const double observed) {
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, pose);
        optimisation::vertex point_vertex{ optimisation::vertices::point() };
        REQUIRE(point_vertex.set_parameters(world_point.data(), 3));
        optimisation::edge edge{ optimisation::edges::depth(measure) };
        edge.set_observation(math::matrix<double, 0, 0>(1, 1, &observed));
        edge.add_vertex(&pose_vertex);
        edge.add_vertex(&point_vertex);
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
            REQUIRE(point_vertex.set_parameters(world_point.data(), 3));
            point_vertex.plus(&delta[0]);
            edge.compute_residual();
            const double forward = edge.get_residual()[0][0];
            delta[column] = -step;
            REQUIRE(point_vertex.set_parameters(world_point.data(), 3));
            point_vertex.plus(&delta[0]);
            edge.compute_residual();
            const double backward = edge.get_residual()[0][0];
            const double numeric = (forward - backward) / (2.0 * step);
            REQUIRE(std::abs(numeric - jacobian_point[0][column]) < 1.0e-6 * (1.0 + std::abs(numeric)));
        }
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::depth) <= optimisation::edge::maximum_size, "The depth edge must fit.");

    const math::se3<double> pose(math::so3<double>::exp({ { -0.08, 0.12, 0.03 } }), { { -0.5, 0.2, 0.7 } });
    const math::matrix<double, 3, 1> world_point({ 1.2, -0.4, 3.0 });
    const math::matrix<double, 3, 1> camera_point = pose * world_point;
    const double axial = camera_point[2];
    const double radial = std::sqrt(camera_point.get_length_squared());
    REQUIRE(axial > 1.0);

    // The residual is the measured inverse depth less the predicted one, zero for a measurement of the true depth.
    for (const sensor::depth::measure_kind measure : { sensor::depth::measure_kind::axial, sensor::depth::measure_kind::radial }) {
        const double truth = (measure == sensor::depth::measure_kind::axial) ? axial : radial;
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, pose);
        optimisation::vertex point_vertex{ optimisation::vertices::point() };
        REQUIRE(point_vertex.set_parameters(world_point.data(), 3));
        optimisation::edge edge{ optimisation::edges::depth(measure) };
        edge.set_observation(math::matrix<double, 0, 0>(1, 1, &truth));
        edge.add_vertex(&pose_vertex);
        edge.add_vertex(&point_vertex);
        edge.compute_residual();
        REQUIRE(std::abs(edge.get_residual()[0][0]) < 1.0e-12);
        REQUIRE(edge.is_measurable());
        const double farther = truth * 1.1;
        edge.set_observation(math::matrix<double, 0, 0>(1, 1, &farther));
        edge.compute_residual();
        REQUIRE(std::abs(edge.get_residual()[0][0] - ((1.0 / farther) - (1.0 / truth))) < 1.0e-12);

        check_jacobians(measure, pose, world_point, truth * 0.95);
        check_jacobians(measure, pose, math::matrix<double, 3, 1>({ -2.0, 1.5, 8.0 }), 7.0);
    }

    // The two measures differ off the optical axis, by the ratio of the ray's length to its z.
    REQUIRE(radial > axial);

    {
        // A point behind the camera cannot be measured: the residual is the measured inverse depth, as if the point were at
        // infinity, and the Jacobians are zero, so the reprojection pulls it back in front.
        const math::matrix<double, 3, 1> behind = pose.inverse() * math::matrix<double, 3, 1>({ 0.1, 0.2, -2.0 });
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, pose);
        optimisation::vertex point_vertex{ optimisation::vertices::point() };
        REQUIRE(point_vertex.set_parameters(behind.data(), 3));
        optimisation::edge edge{ optimisation::edges::depth(sensor::depth::measure_kind::axial) };
        const double observed = 2.0;
        edge.set_observation(math::matrix<double, 0, 0>(1, 1, &observed));
        edge.add_vertex(&pose_vertex);
        edge.add_vertex(&point_vertex);
        REQUIRE(!edge.is_measurable());
        edge.compute_residual();
        REQUIRE(std::abs(edge.get_residual()[0][0] - 0.5) < 1.0e-12);
        edge.compute_jacobians();
        for (size_t column = 0; column < 6; ++column) {
            REQUIRE(edge.get_jacobians()[0][0][column] == 0.0);
        }
    }

    {
        // A depth fixes a point a single view leaves free along its ray: one pose, the point seen once with its reprojection
        // and its depth, comes back to its measured depth from a guess along the ray.
        const sensor::depth::measure_kind measure = sensor::depth::measure_kind::axial;
        optimisation::factor_graph graph;
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, pose);
        pose_vertex.set_fixed(true);
        optimisation::vertex* const pose_node = graph.add_vertex(static_cast<optimisation::vertex&&>(pose_vertex));
        const math::matrix<double, 3, 1> guess_camera = camera_point * 1.6;
        const math::matrix<double, 3, 1> guess = pose.inverse() * guess_camera;
        optimisation::vertex point_vertex{ optimisation::vertices::point() };
        REQUIRE(point_vertex.set_parameters(guess.data(), 3));
        point_vertex.set_marginalised(true);
        optimisation::vertex* const point_node = graph.add_vertex(static_cast<optimisation::vertex&&>(point_vertex));
        const sensor::camera::pinhole<double> camera_model(std::vector<double>{ 500.0, 500.0, 320.0, 240.0 }.data(), 4);
        double pixel[2];
        REQUIRE(camera_model.project(camera_point.data(), &pixel[0]));
        optimisation::edge bearing{ optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)) };
        bearing.set_observation(math::matrix<double, 0, 0>(2, 1, &pixel[0]));
        bearing.add_vertex(pose_node);
        bearing.add_vertex(point_node);
        graph.add_edge(static_cast<optimisation::edge&&>(bearing));
        optimisation::edge edge{ optimisation::edges::depth(measure) };
        edge.set_observation(math::matrix<double, 0, 0>(1, 1, &axial));
        edge.set_information(math::matrix<double, 0, 0>::identity(1, 1) * 1.0e4);
        edge.add_vertex(pose_node);
        edge.add_vertex(point_node);
        graph.add_edge(static_cast<optimisation::edge&&>(edge));
        graph.solve(50, true);
        const double* const solved = point_node->get_parameters();
        const math::matrix<double, 3, 1> solved_camera = pose * math::matrix<double, 3, 1>({ solved[0], solved[1], solved[2] });
        REQUIRE(std::abs(solved_camera[2] - axial) < 1.0e-6);
        REQUIRE(std::sqrt((solved_camera - camera_point).get_length_squared()) < 1.0e-6);
    }

    return EXIT_SUCCESS;
}
