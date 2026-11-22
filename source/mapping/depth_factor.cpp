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

#include "mapping/depth_factor.hpp"

#include "math/lie.hpp"
#include "math/math.hpp"
#include "optimisation/edges/depth.hpp"
#include "optimisation/edges/depth_inverse_depth.hpp"
#include "optimisation/losses/huber.hpp"
#include "optimisation/vertices/point.hpp"
#include "optimisation/vertices/pose.hpp"
#include "sensor/depth.hpp"

namespace mapping {
    double depth_factor::sample(const mapping::frame& frame, const double x, const double y) {
        double depth = 0.0;
        return frame.depth.sample(x, y, depth) ? depth : 0.0;
    }

    optimisation::edge depth_factor::make_edge(const mapping::frame& frame, const double depth) {
        mapping::point world_point;
        world_point.inverse_depth = false;
        return depth_factor::make_edge(frame, world_point, depth);
    }

    optimisation::edge depth_factor::make_edge(const mapping::frame& frame, const mapping::point& landmark, const double depth) {
        const sensor::depth::measure_kind measure = frame.depth.get_settings().measure;
        optimisation::edge factor = landmark.inverse_depth
                                        ? optimisation::edge{ optimisation::edges::depth_inverse_depth(measure, landmark.anchor_rotation, landmark.anchor_translation) }
                                        : optimisation::edge{ optimisation::edges::depth(measure) };
        factor.set_observation(math::matrix<double, 0, 0>(1, 1, &depth));
        const double sigma = frame.depth.inverse_sigma(depth);
        factor.set_information(math::matrix<double, 0, 0>::identity(1, 1) * (1.0 / (sigma * sigma)));
        factor.set_loss(optimisation::loss(optimisation::losses::huber(math::sqrt(depth_factor::inlier_bound_squared))));
        return factor;
    }

    void depth_factor::add_information(const mapping::frame& frame, const mapping::point& landmark, const double depth, math::matrix<double, 3, 3>& information) {
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        const math::se3<double> pose(frame.rotation, frame.translation);
        const double pose_parameters[7] = { pose.translation()[0], pose.translation()[1], pose.translation()[2], pose.rotation().get_quaternion()[1], pose.rotation().get_quaternion()[2], pose.rotation().get_quaternion()[3], pose.rotation().get_quaternion()[0] };
        pose_vertex.set_parameters(&pose_parameters[0], 7);
        optimisation::vertex point_vertex{ optimisation::vertices::point() };
        point_vertex.set_parameters(landmark.inverse_depth ? landmark.inverse_parameters.data() : landmark.location.data(), 3);
        optimisation::edge factor = depth_factor::make_edge(frame, landmark, depth);
        factor.add_vertex(&pose_vertex);
        factor.add_vertex(&point_vertex);
        factor.compute_jacobians();
        const math::matrix<double, 0, 0>& jacobian = factor.get_jacobians()[1];
        const double weight = factor.get_information()[0][0];
        for (size_t a = 0; a < 3; ++a) {
            for (size_t b = 0; b < 3; ++b) {
                information[a][b] += weight * jacobian[0][a] * jacobian[0][b];
            }
        }
    }

    void depth_factor::add_information(optimisation::edge& factor, math::matrix<double, 3, 3>& information) {
        factor.compute_residual();
        factor.compute_jacobians();
        const double weight = factor.robust_weight() * factor.get_information()[0][0];
        const math::matrix<double, 0, 0>& jacobian = factor.get_jacobians()[1];
        for (size_t a = 0; a < 3; ++a) {
            for (size_t b = 0; b < 3; ++b) {
                information[a][b] += weight * jacobian[0][a] * jacobian[0][b];
            }
        }
    }

    bool depth_factor::predicted(const mapping::frame& frame, const mapping::point& landmark, double& value) {
        if (landmark.at_infinity()) {
            return false;
        }
        const math::matrix<double, 3, 1> mapped = (frame.rotation * landmark.location) + frame.translation;
        const double point[3] = { mapped[0], mapped[1], mapped[2] };
        value = sensor::depth::measured(frame.depth.get_settings().measure, point);
        return (mapped[2] > 0.0) && (value > 0.0);
    }

    bool depth_factor::agrees(const mapping::frame& frame, const mapping::point& landmark, const double depth) {
        double predicted_value = 0.0;
        if (!depth_factor::predicted(frame, landmark, predicted_value)) {
            return false;
        }
        const double sigma = frame.depth.inverse_sigma(depth);
        const double difference = ((1.0 / depth) - (1.0 / predicted_value)) / sigma;
        return (difference * difference) <= depth_factor::inlier_bound_squared;
    }

    bool depth_factor::location(const mapping::frame& frame, const double x, const double y, const double depth, math::matrix<double, 3, 1>& world, double& axial_depth) {
        const double pixel[2] = { x, y };
        double point_camera[3] = { 0.0, 0.0, 0.0 };
        if (!sensor::depth::back_project(frame.camera, frame.depth.get_settings().measure, pixel, depth, point_camera) || !(point_camera[2] > 0.0)) {
            return false;
        }
        const math::matrix<double, 3, 1> camera_point({ point_camera[0], point_camera[1], point_camera[2] });
        world = math::transpose(frame.rotation) * (camera_point - frame.translation);
        axial_depth = point_camera[2];
        return true;
    }
}
