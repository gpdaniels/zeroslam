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

#include "optimisation/edges/rig_reprojection.hpp"

#include "math/lie.hpp"
#include "optimisation/edges/reprojection.hpp"
#include "optimisation/vertices/extrinsic.hpp"

namespace optimisation::edges {
    rig_reprojection::rig_reprojection(const sensor::camera::model<double>& camera_model)
        : camera(camera_model)
        , landmark(landmark_kind::point)
        , anchor_rotation(math::matrix<double, 3, 3>::identity())
        , anchor_translation(math::matrix<double, 3, 1>::zero()) {
    }

    rig_reprojection::rig_reprojection(const sensor::camera::model<double>& camera_model, const math::matrix<double, 3, 3>& anchor_rotation_value, const math::matrix<double, 3, 1>& anchor_translation_value)
        : camera(camera_model)
        , landmark(landmark_kind::inverse_depth)
        , anchor_rotation(anchor_rotation_value)
        , anchor_translation(anchor_translation_value) {
    }

    void rig_reprojection::compose(const edge& context, math::matrix<double, 3, 1>& point_camera, math::matrix<double, 3, 6>* jacobian_pose, math::matrix<double, 3, 3>* jacobian_landmark, math::matrix<double, 3, 6>* jacobian_extrinsic) const {
        const double* const pose_params = context.get_vertex(rig_reprojection::pose_slot)->get_parameters();
        const math::se3<double> pose(math::so3<double>(pose_params[6], pose_params[3], pose_params[4], pose_params[5]), { { pose_params[0], pose_params[1], pose_params[2] } });
        const math::se3<double> body_from_camera = vertices::extrinsic::from_parameters(context.get_vertex(rig_reprojection::extrinsic_slot)->get_parameters());
        const double* const landmark_params = context.get_vertex(rig_reprojection::landmark_slot)->get_parameters();
        const math::matrix<double, 3, 3> rotation_bw = pose.rotation().get_matrix();
        const math::matrix<double, 3, 3> rotation_cb = math::transpose(body_from_camera.rotation().get_matrix());
        // The point in the body, and the scale of both it and the camera point: one for a world point, the inverse depth for
        // an inverse depth landmark, whose points are both scaled by it.
        math::matrix<double, 3, 1> point_body;
        double scale = 1.0;
        math::matrix<double, 3, 3> landmark_body;
        if (this->landmark == landmark_kind::point) {
            const math::matrix<double, 3, 1> point_world({ landmark_params[0], landmark_params[1], landmark_params[2] });
            point_body = (rotation_bw * point_world) + pose.translation();
            landmark_body = rotation_bw;
        }
        else {
            const math::matrix<double, 3, 1> bearing({ landmark_params[0], landmark_params[1], 1.0 });
            scale = landmark_params[2];
            const math::matrix<double, 3, 3> rotation_combined = rotation_bw * this->anchor_rotation;
            const math::matrix<double, 3, 1> rho_column = (rotation_bw * this->anchor_translation) + pose.translation();
            point_body = (rotation_combined * bearing) + (rho_column * scale);
            const math::matrix<double, 3, 1> rho_column_camera = rho_column - body_from_camera.translation();
            for (size_t row = 0; row < 3; ++row) {
                landmark_body[row][0] = rotation_combined[row][0];
                landmark_body[row][1] = rotation_combined[row][1];
                landmark_body[row][2] = rho_column_camera[row];
            }
        }
        point_camera = rotation_cb * (point_body - (body_from_camera.translation() * scale));
        if (jacobian_landmark != nullptr) {
            *jacobian_landmark = rotation_cb * landmark_body;
        }
        if ((jacobian_pose == nullptr) && (jacobian_extrinsic == nullptr)) {
            return;
        }
        // A step on the left of the pose moves the body point by -[point]x omega + scale * upsilon; one on the left of the
        // extrinsic moves it by as much the other way, seen from the camera.
        math::matrix<double, 3, 6> body_step;
        body_step[0][0] = 0.0;
        body_step[0][1] = point_body[2];
        body_step[0][2] = -point_body[1];
        body_step[1][0] = -point_body[2];
        body_step[1][1] = 0.0;
        body_step[1][2] = point_body[0];
        body_step[2][0] = point_body[1];
        body_step[2][1] = -point_body[0];
        body_step[2][2] = 0.0;
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                body_step[row][3 + column] = (row == column) ? scale : 0.0;
            }
        }
        const math::matrix<double, 3, 6> camera_step = rotation_cb * body_step;
        if (jacobian_pose != nullptr) {
            *jacobian_pose = camera_step;
        }
        if (jacobian_extrinsic != nullptr) {
            *jacobian_extrinsic = -camera_step;
        }
    }

    bool rig_reprojection::measurable(const edge& context) const {
        math::matrix<double, 3, 1> point_camera;
        this->compose(context, point_camera, nullptr, nullptr, nullptr);
        math::matrix<double, 2, 1> projected = { { 0.0, 0.0 } };
        return this->camera.project(point_camera.data(), projected.data());
    }

    void rig_reprojection::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        math::matrix<double, 3, 1> point_camera;
        this->compose(context, point_camera, nullptr, nullptr, nullptr);
        math::matrix<double, 2, 1> projected = { { 0.0, 0.0 } };
        if (!this->camera.project(point_camera.data(), projected.data())) {
            const double penalty = reprojection::behind_camera_residual(point_camera[2]);
            residual[0][0] = penalty;
            residual[1][0] = penalty;
            return;
        }
        residual[0][0] = context.get_observation()[0][0] - projected[0];
        residual[1][0] = context.get_observation()[1][0] - projected[1];
    }

    void rig_reprojection::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        math::matrix<double, 3, 1> point_camera;
        math::matrix<double, 3, 6> jacobian_point_pose;
        math::matrix<double, 3, 3> jacobian_point_landmark;
        math::matrix<double, 3, 6> jacobian_point_extrinsic;
        this->compose(context, point_camera, &jacobian_point_pose, &jacobian_point_landmark, &jacobian_point_extrinsic);
        double projected[2];
        math::matrix<double, 2, 3> jacobian_camera;
        if (!this->camera.project(point_camera.data(), projected, jacobian_camera.data())) {
            jacobian_camera[0][0] = 0.0;
            jacobian_camera[0][1] = 0.0;
            jacobian_camera[0][2] = reprojection::behind_camera_slope;
            jacobian_camera[1][0] = 0.0;
            jacobian_camera[1][1] = 0.0;
            jacobian_camera[1][2] = reprojection::behind_camera_slope;
        }
        const math::matrix<double, 2, 6> jacobian_pose = -(jacobian_camera * jacobian_point_pose);
        edge::set_jacobian(jacobians[rig_reprojection::pose_slot], 2, 6, jacobian_pose.data());
        const math::matrix<double, 2, 3> jacobian_landmark = -(jacobian_camera * jacobian_point_landmark);
        edge::set_jacobian(jacobians[rig_reprojection::landmark_slot], 2, 3, jacobian_landmark.data());
        const math::matrix<double, 2, 6> jacobian_extrinsic = -(jacobian_camera * jacobian_point_extrinsic);
        edge::set_jacobian(jacobians[rig_reprojection::extrinsic_slot], 2, 6, jacobian_extrinsic.data());
    }
}
