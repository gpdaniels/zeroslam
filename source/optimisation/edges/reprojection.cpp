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

namespace optimisation::edges {
    reprojection::reprojection(const sensor::camera::model<double>& camera_model)
        : camera(camera_model) {
    }

    void reprojection::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const double* const pose_params = context.get_vertex(0)->get_parameters();
        const math::se3<double> pose(math::so3<double>(pose_params[6], pose_params[3], pose_params[4], pose_params[5]), { { pose_params[0], pose_params[1], pose_params[2] } });
        const double* const xyz_params = context.get_vertex(1)->get_parameters();
        const math::matrix<double, 3, 1> landmark_world({ xyz_params[0], xyz_params[1], xyz_params[2] });
        const math::matrix<double, 3, 1> landmark_camera = pose * landmark_world;
        math::matrix<double, 2, 1> projected = { { 0.0, 0.0 } };
        if (!this->camera.project(landmark_camera.data(), projected.data())) {
            const double penalty = -reprojection::behind_camera_penalty * landmark_camera[2];
            residual[0][0] = penalty;
            residual[1][0] = penalty;
            return;
        }
        residual[0][0] = context.get_observation()[0][0] - projected[0];
        residual[1][0] = context.get_observation()[1][0] - projected[1];
    }

    void reprojection::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        const double* const pose_params = context.get_vertex(0)->get_parameters();
        const math::se3<double> pose(math::so3<double>(pose_params[6], pose_params[3], pose_params[4], pose_params[5]), { { pose_params[0], pose_params[1], pose_params[2] } });
        const double* const xyz_params = context.get_vertex(1)->get_parameters();
        const math::matrix<double, 3, 1> landmark_world({ xyz_params[0], xyz_params[1], xyz_params[2] });
        const math::matrix<double, 3, 1> landmark_camera = pose * landmark_world;

        double projected[2];
        math::matrix<double, 2, 3> jacobian_camera;
        if (!this->camera.project(landmark_camera.data(), projected, jacobian_camera.data())) {
            jacobian_camera[0][0] = 0.0;
            jacobian_camera[0][1] = 0.0;
            jacobian_camera[0][2] = reprojection::behind_camera_penalty;
            jacobian_camera[1][0] = 0.0;
            jacobian_camera[1][1] = 0.0;
            jacobian_camera[1][2] = reprojection::behind_camera_penalty;
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

        const math::matrix<double, 2, 6> jacobian_pose_res = -(jacobian_camera * jacobian_point_pose);
        jacobians[0] = math::matrix<double, 0, 0>(2, 6, jacobian_pose_res.data());

        const math::matrix<double, 2, 3> jacobian_landmark_res = -(jacobian_camera * pose.rotation().get_matrix());
        jacobians[1] = math::matrix<double, 0, 0>(2, 3, jacobian_landmark_res.data());
    }
}
