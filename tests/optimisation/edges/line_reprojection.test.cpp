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

#include "optimisation/edges/line_reprojection.hpp"

#include "geometry/plucker.hpp"
#include "math/lie.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/line.hpp"
#include "optimisation/vertices/pose.hpp"
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

namespace {
    void set_pose(optimisation::vertex& vertex, const math::se3<double>& pose) {
        const double parameters[7] = { pose.translation()[0], pose.translation()[1], pose.translation()[2], pose.rotation().get_quaternion()[1], pose.rotation().get_quaternion()[2], pose.rotation().get_quaternion()[3], pose.rotation().get_quaternion()[0] };
        REQUIRE(vertex.set_parameters(&parameters[0], 7));
    }

    void set_line(optimisation::vertex& vertex, const geometry::plucker& line) {
        const double parameters[6] = { line.moment[0], line.moment[1], line.moment[2], line.direction[0], line.direction[1], line.direction[2] };
        REQUIRE(vertex.set_parameters(&parameters[0], 6));
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::line_reprojection) <= optimisation::edge::maximum_size, "The line reprojection edge must fit.");

    const sensor::camera::pinhole<double> camera_model(std::vector<double>{ 500.0, 500.0, 320.0, 240.0 }.data(), 4);
    const math::se3<double> pose(math::so3<double>::exp({ { 0.05, -0.1, 0.02 } }), { { 0.3, -0.2, 0.1 } });
    geometry::plucker line;
    REQUIRE(geometry::plucker::from_points({ { -1.0, 0.5, 4.0 } }, { { 1.0, 0.7, 5.0 } }, line));

    {
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, pose);
        optimisation::vertex line_vertex{ optimisation::vertices::line() };
        set_line(line_vertex, line);
        double pixels[2][2];
        const math::matrix<double, 3, 1> points[2] = { { { -1.0, 0.5, 4.0 } }, { { 1.0, 0.7, 5.0 } } };
        for (int i = 0; i < 2; ++i) {
            const math::matrix<double, 3, 1> camera_point = pose * points[i];
            REQUIRE(camera_model.project(camera_point.data(), &pixels[i][0]));
        }
        optimisation::edges::line_reprojection policy{ sensor::camera::model<double>(camera_model) };
        REQUIRE(policy.set_measured_segment(pixels[0][0], pixels[0][1], pixels[1][0], pixels[1][1]));
        optimisation::edge edge{ policy };
        edge.add_vertex(&pose_vertex);
        edge.add_vertex(&line_vertex);
        edge.compute_residual();
        REQUIRE(std::abs(edge.get_residual()[0][0]) < 1.0e-9);
        REQUIRE(std::abs(edge.get_residual()[1][0]) < 1.0e-9);
    }

    {
        geometry::plucker perturbed = line;
        REQUIRE(perturbed.oplus(0.02, -0.03, 0.01, 0.02));
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, pose);
        optimisation::vertex line_vertex{ optimisation::vertices::line() };
        set_line(line_vertex, perturbed);
        double pixels[2][2];
        const math::matrix<double, 3, 1> points[2] = { { { -1.0, 0.5, 4.0 } }, { { 1.0, 0.7, 5.0 } } };
        for (int i = 0; i < 2; ++i) {
            const math::matrix<double, 3, 1> camera_point = pose * points[i];
            REQUIRE(camera_model.project(camera_point.data(), &pixels[i][0]));
        }
        optimisation::edges::line_reprojection policy{ sensor::camera::model<double>(camera_model) };
        REQUIRE(policy.set_measured_segment(pixels[0][0], pixels[0][1], pixels[1][0], pixels[1][1]));
        optimisation::edge edge{ policy };
        edge.add_vertex(&pose_vertex);
        edge.add_vertex(&line_vertex);
        edge.compute_residual();
        REQUIRE(std::abs(edge.get_residual()[0][0]) > 1.0);
        edge.compute_jacobians();
        const math::matrix<double, 0, 0> jacobian_pose = edge.get_jacobians()[0];
        const math::matrix<double, 0, 0> jacobian_line = edge.get_jacobians()[1];
        REQUIRE(jacobian_pose.rows() == 2);
        REQUIRE(jacobian_pose.cols() == 6);
        REQUIRE(jacobian_line.rows() == 2);
        REQUIRE(jacobian_line.cols() == 4);
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
        for (size_t column = 0; column < 4; ++column) {
            double delta[4] = { 0.0, 0.0, 0.0, 0.0 };
            delta[column] = step;
            set_line(line_vertex, perturbed);
            line_vertex.plus(&delta[0]);
            edge.compute_residual();
            const double forward[2] = { edge.get_residual()[0][0], edge.get_residual()[1][0] };
            delta[column] = -step;
            set_line(line_vertex, perturbed);
            line_vertex.plus(&delta[0]);
            edge.compute_residual();
            const double backward[2] = { edge.get_residual()[0][0], edge.get_residual()[1][0] };
            for (size_t row = 0; row < 2; ++row) {
                const double numeric = (forward[row] - backward[row]) / (2.0 * step);
                REQUIRE(std::abs(numeric - jacobian_line[row][column]) < 1.0e-4 * (1.0 + std::abs(numeric)));
            }
        }
    }

    {
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, math::se3<double>());
        optimisation::vertex line_vertex{ optimisation::vertices::line() };
        geometry::plucker through_origin;
        REQUIRE(geometry::plucker::from_points({ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 1.0 } }, through_origin));
        set_line(line_vertex, through_origin);
        optimisation::edges::line_reprojection policy{ sensor::camera::model<double>(camera_model) };
        REQUIRE(policy.set_measured_segment(300.0, 200.0, 340.0, 280.0));
        optimisation::edge edge{ policy };
        edge.add_vertex(&pose_vertex);
        edge.add_vertex(&line_vertex);
        edge.compute_residual();
        edge.compute_jacobians();
        REQUIRE(edge.get_residual()[0][0] == 0.0);
        REQUIRE(edge.get_residual()[1][0] == 0.0);
        for (size_t column = 0; column < 4; ++column) {
            REQUIRE(edge.get_jacobians()[1][0][column] == 0.0);
        }
    }

    return EXIT_SUCCESS;
}
