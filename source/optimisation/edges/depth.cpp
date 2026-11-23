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

namespace optimisation::edges {
    namespace {
        math::matrix<double, 3, 1> point_in_camera(const edge& context, math::matrix<double, 3, 3>& rotation) {
            const double* const pose_params = context.get_vertex(0)->get_parameters();
            const math::se3<double> pose(math::so3<double>(pose_params[6], pose_params[3], pose_params[4], pose_params[5]), { { pose_params[0], pose_params[1], pose_params[2] } });
            const double* const xyz_params = context.get_vertex(1)->get_parameters();
            const math::matrix<double, 3, 1> landmark_world({ xyz_params[0], xyz_params[1], xyz_params[2] });
            rotation = pose.rotation().get_matrix();
            return pose * landmark_world;
        }
    }

    depth::depth(const sensor::depth::measure_kind measure_value)
        : measure(measure_value) {
    }

    bool depth::measurable(const edge& context) const {
        math::matrix<double, 3, 3> rotation;
        const math::matrix<double, 3, 1> landmark_camera = point_in_camera(context, rotation);
        const double point[3] = { landmark_camera[0], landmark_camera[1], landmark_camera[2] };
        return sensor::depth::measured(this->measure, point) > depth::minimum_depth;
    }

    void depth::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        math::matrix<double, 3, 3> rotation;
        const math::matrix<double, 3, 1> landmark_camera = point_in_camera(context, rotation);
        const double point[3] = { landmark_camera[0], landmark_camera[1], landmark_camera[2] };
        const double predicted = sensor::depth::measured(this->measure, point);
        const double observed_inverse = 1.0 / context.get_observation()[0][0];
        residual[0][0] = (predicted > depth::minimum_depth) ? (observed_inverse - (1.0 / predicted)) : observed_inverse;
    }

    void depth::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        math::matrix<double, 3, 3> rotation;
        const math::matrix<double, 3, 1> landmark_camera = point_in_camera(context, rotation);
        const double point[3] = { landmark_camera[0], landmark_camera[1], landmark_camera[2] };
        const double predicted = sensor::depth::measured(this->measure, point);
        // The derivative of the residual, -1 / predicted, by the point in the camera frame.
        math::matrix<double, 1, 3> jacobian_camera = math::matrix<double, 1, 3>::zero();
        if (predicted > depth::minimum_depth) {
            const double inverse_squared = 1.0 / (predicted * predicted);
            if (this->measure == sensor::depth::measure_kind::axial) {
                jacobian_camera[2] = inverse_squared;
            }
            else {
                for (size_t axis = 0; axis < 3; ++axis) {
                    jacobian_camera[axis] = inverse_squared * point[axis] / predicted;
                }
            }
        }

        const double X = landmark_camera[0];
        const double Y = landmark_camera[1];
        const double Z = landmark_camera[2];
        math::matrix<double, 3, 6> jacobian_point_pose;
        jacobian_point_pose[0][0] = 0;
        jacobian_point_pose[0][1] = Z;
        jacobian_point_pose[0][2] = -Y;
        jacobian_point_pose[1][0] = -Z;
        jacobian_point_pose[1][1] = 0;
        jacobian_point_pose[1][2] = X;
        jacobian_point_pose[2][0] = Y;
        jacobian_point_pose[2][1] = -X;
        jacobian_point_pose[2][2] = 0;
        jacobian_point_pose[0][3] = 1;
        jacobian_point_pose[0][4] = 0;
        jacobian_point_pose[0][5] = 0;
        jacobian_point_pose[1][3] = 0;
        jacobian_point_pose[1][4] = 1;
        jacobian_point_pose[1][5] = 0;
        jacobian_point_pose[2][3] = 0;
        jacobian_point_pose[2][4] = 0;
        jacobian_point_pose[2][5] = 1;

        const math::matrix<double, 1, 6> jacobian_pose_res = jacobian_camera * jacobian_point_pose;
        edge::set_jacobian(jacobians[0], 1, 6, jacobian_pose_res.data());

        const math::matrix<double, 1, 3> jacobian_landmark_res = jacobian_camera * rotation;
        edge::set_jacobian(jacobians[1], 1, 3, jacobian_landmark_res.data());
    }
}
