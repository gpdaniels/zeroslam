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

namespace optimisation::edges {
    reprojection_inverse_depth::reprojection_inverse_depth(const sensor::camera::model<double>& camera_model, const math::matrix<double, 3, 3>& anchor_rotation_value, const math::matrix<double, 3, 1>& anchor_translation_value)
        : camera(camera_model)
        , anchor_rotation(anchor_rotation_value)
        , anchor_translation(anchor_translation_value) {
    }

    void reprojection_inverse_depth::compute_scaled_point(const edge& context, math::matrix<double, 3, 1>& q, math::matrix<double, 3, 3>& rotation_combined, math::matrix<double, 3, 1>& rho_column, double& rho) const {
        const double* const pose_params = context.get_vertex(0)->get_parameters();
        const math::se3<double> pose(math::so3<double>(pose_params[6], pose_params[3], pose_params[4], pose_params[5]), { { pose_params[0], pose_params[1], pose_params[2] } });
        const double* const landmark_params = context.get_vertex(1)->get_parameters();
        const math::matrix<double, 3, 1> bearing({ landmark_params[0], landmark_params[1], 1.0 });
        rho = landmark_params[2];
        const math::matrix<double, 3, 3> rotation_cw = pose.rotation().get_matrix();
        rotation_combined = rotation_cw * this->anchor_rotation;
        rho_column = (rotation_cw * this->anchor_translation) + pose.translation();
        q = (rotation_combined * bearing) + (rho_column * rho);
    }

    void reprojection_inverse_depth::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        math::matrix<double, 3, 1> q;
        math::matrix<double, 3, 3> rotation_combined;
        math::matrix<double, 3, 1> rho_column;
        double rho = 0.0;
        this->compute_scaled_point(context, q, rotation_combined, rho_column, rho);
        math::matrix<double, 2, 1> projected = { { 0.0, 0.0 } };
        if (!this->camera.project(q.data(), projected.data())) {
            const double penalty = -reprojection_inverse_depth::behind_camera_penalty * q[2];
            residual[0][0] = penalty;
            residual[1][0] = penalty;
            return;
        }
        residual[0][0] = context.get_observation()[0][0] - projected[0];
        residual[1][0] = context.get_observation()[1][0] - projected[1];
    }

    void reprojection_inverse_depth::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        math::matrix<double, 3, 1> q;
        math::matrix<double, 3, 3> rotation_combined;
        math::matrix<double, 3, 1> rho_column;
        double rho = 0.0;
        this->compute_scaled_point(context, q, rotation_combined, rho_column, rho);
        double projected[2];
        math::matrix<double, 2, 3> jacobian_camera;
        if (!this->camera.project(q.data(), projected, jacobian_camera.data())) {
            jacobian_camera[0][0] = 0.0;
            jacobian_camera[0][1] = 0.0;
            jacobian_camera[0][2] = reprojection_inverse_depth::behind_camera_penalty;
            jacobian_camera[1][0] = 0.0;
            jacobian_camera[1][1] = 0.0;
            jacobian_camera[1][2] = reprojection_inverse_depth::behind_camera_penalty;
        }
        math::matrix<double, 3, 6> jacobian_point_pose;
        jacobian_point_pose[0][0] = 0.0;
        jacobian_point_pose[0][1] = q[2];
        jacobian_point_pose[0][2] = -q[1];
        jacobian_point_pose[1][0] = -q[2];
        jacobian_point_pose[1][1] = 0.0;
        jacobian_point_pose[1][2] = q[0];
        jacobian_point_pose[2][0] = q[1];
        jacobian_point_pose[2][1] = -q[0];
        jacobian_point_pose[2][2] = 0.0;
        jacobian_point_pose[0][3] = rho;
        jacobian_point_pose[0][4] = 0.0;
        jacobian_point_pose[0][5] = 0.0;
        jacobian_point_pose[1][3] = 0.0;
        jacobian_point_pose[1][4] = rho;
        jacobian_point_pose[1][5] = 0.0;
        jacobian_point_pose[2][3] = 0.0;
        jacobian_point_pose[2][4] = 0.0;
        jacobian_point_pose[2][5] = rho;
        const math::matrix<double, 2, 6> jacobian_pose_res = -(jacobian_camera * jacobian_point_pose);
        jacobians[0] = math::matrix<double, 0, 0>(2, 6, jacobian_pose_res.data());
        math::matrix<double, 3, 3> jacobian_point_landmark;
        for (size_t row = 0; row < 3; ++row) {
            jacobian_point_landmark[row][0] = rotation_combined[row][0];
            jacobian_point_landmark[row][1] = rotation_combined[row][1];
            jacobian_point_landmark[row][2] = rho_column[row];
        }
        const math::matrix<double, 2, 3> jacobian_landmark_res = -(jacobian_camera * jacobian_point_landmark);
        jacobians[1] = math::matrix<double, 0, 0>(2, 3, jacobian_landmark_res.data());
    }
}
