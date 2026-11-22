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
#include "optimisation/vertices/intrinsics.hpp"
#include "optimisation/vertices/line.hpp"
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

    for (const double focal_y : { 500.0, 400.0 }) {
        const sensor::camera::pinhole<double> camera_model_xy(std::vector<double>{ 500.0, focal_y, 320.0, 240.0 }.data(), 4);
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
            REQUIRE(camera_model_xy.project(camera_point.data(), &pixels[i][0]));
        }
        optimisation::edges::line_reprojection policy{ sensor::camera::model<double>(camera_model_xy) };
        REQUIRE(policy.set_measured_segment(pixels[0][0], pixels[0][1], pixels[1][0], pixels[1][1]));
        optimisation::edge edge{ policy };
        edge.add_vertex(&pose_vertex);
        edge.add_vertex(&line_vertex);
        edge.compute_residual();
        REQUIRE(std::abs(edge.get_residual()[0][0]) > 1.0);
        // The residual is the distance in pixels from each measured end to the projected line, also when fx differs from fy.
        {
            const math::matrix<double, 3, 3> rotation = pose.rotation().get_matrix();
            const math::matrix<double, 3, 1> closest = geometry::plucker::cross(perturbed.direction, perturbed.moment);
            double line_pixels[2][2];
            for (int i = 0; i < 2; ++i) {
                const math::matrix<double, 3, 1> on_line = closest + (perturbed.direction * (4.0 + static_cast<double>(i)));
                REQUIRE(camera_model_xy.project(((rotation * on_line) + pose.translation()).data(), &line_pixels[i][0]));
            }
            const double along[2] = { line_pixels[1][0] - line_pixels[0][0], line_pixels[1][1] - line_pixels[0][1] };
            const double length = std::sqrt((along[0] * along[0]) + (along[1] * along[1]));
            for (int i = 0; i < 2; ++i) {
                const double distance = std::abs(((pixels[i][0] - line_pixels[0][0]) * along[1]) - ((pixels[i][1] - line_pixels[0][1]) * along[0])) / length;
                REQUIRE(std::abs(std::abs(edge.get_residual()[static_cast<size_t>(i)][0]) - distance) < 1.0e-6 * (1.0 + distance));
            }
        }
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

    {
        // With the camera's intrinsics as a third vertex the segment is measured through the vertex's camera: at the camera
        // the edge was made with it measures what the edge holding the camera does, and every block's Jacobian, the
        // intrinsics' by the focal lengths moving together, the principal point and the leading distortion, is the derivative
        // through its vertex.
        const double distorted_parameters[12] = { 510.0, 505.0, 322.0, 238.0, -0.2, 0.05, 0.001, -0.002, 0.0, 0.0, 0.0, 0.0 };
        const sensor::camera::model<double> distorted{ sensor::camera::pinhole_radial_tangential<double>(&distorted_parameters[0], 12) };
        const int locals[12] = { 0, 0, 1, 2, 3, 4, 5, 6, -1, -1, -1, -1 };
        geometry::plucker perturbed = line;
        REQUIRE(perturbed.oplus(0.02, -0.03, 0.01, 0.02));
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, pose);
        optimisation::vertex line_vertex{ optimisation::vertices::line() };
        set_line(line_vertex, perturbed);
        optimisation::vertex intrinsics_vertex = optimisation::vertices::intrinsics::make(distorted, &locals[0], 12);
        REQUIRE(intrinsics_vertex.is_valid());
        optimisation::edges::line_reprojection held_policy{ distorted };
        optimisation::edges::line_reprojection calibrated_policy{ distorted, true };
        REQUIRE(held_policy.get_vertex_count() == 2);
        REQUIRE(calibrated_policy.get_vertex_count() == 3);
        REQUIRE(held_policy.set_measured_segment(250.0, 200.0, 420.0, 260.0));
        REQUIRE(calibrated_policy.set_measured_segment(250.0, 200.0, 420.0, 260.0));
        optimisation::edge held{ held_policy };
        optimisation::edge calibrated{ calibrated_policy };
        REQUIRE(held.add_vertex(&pose_vertex));
        REQUIRE(held.add_vertex(&line_vertex));
        REQUIRE(calibrated.add_vertex(&pose_vertex));
        REQUIRE(calibrated.add_vertex(&line_vertex));
        REQUIRE(calibrated.add_vertex(&intrinsics_vertex));
        held.compute_residual();
        calibrated.compute_residual();
        held.compute_jacobians();
        calibrated.compute_jacobians();
        for (size_t r = 0; r < 2; ++r) {
            REQUIRE(calibrated.get_residual()[r][0] == held.get_residual()[r][0]);
            for (size_t c = 0; c < 6; ++c) {
                REQUIRE(calibrated.get_jacobians()[0][r][c] == held.get_jacobians()[0][r][c]);
            }
            for (size_t c = 0; c < 4; ++c) {
                REQUIRE(calibrated.get_jacobians()[1][r][c] == held.get_jacobians()[1][r][c]);
            }
        }
        REQUIRE(std::abs(calibrated.get_residual()[0][0]) > 1.0);
        const math::matrix<double, 0, 0> analytic = calibrated.get_jacobians()[2];
        REQUIRE(analytic.rows() == 2);
        REQUIRE(analytic.cols() == 7);
        double backup[12] = {};
        REQUIRE(intrinsics_vertex.get_parameters(&backup[0], 12));
        for (size_t column = 0; column < 7; ++column) {
            const double step = (column < 3) ? 1.0e-4 : 1.0e-7;
            double delta[7] = {};
            delta[column] = step;
            intrinsics_vertex.plus(&delta[0]);
            calibrated.compute_residual();
            const double plus[2] = { calibrated.get_residual()[0][0], calibrated.get_residual()[1][0] };
            REQUIRE(intrinsics_vertex.set_parameters(&backup[0], 12));
            delta[column] = -step;
            intrinsics_vertex.plus(&delta[0]);
            calibrated.compute_residual();
            const double minus[2] = { calibrated.get_residual()[0][0], calibrated.get_residual()[1][0] };
            REQUIRE(intrinsics_vertex.set_parameters(&backup[0], 12));
            for (size_t r = 0; r < 2; ++r) {
                const double numeric = (plus[r] - minus[r]) / (2.0 * step);
                REQUIRE(std::abs(analytic[r][column] - numeric) < 1.0e-5 * (1.0 + std::abs(numeric)));
            }
        }
    }

    return EXIT_SUCCESS;
}
