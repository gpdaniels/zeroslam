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
#include "math/math.hpp"

namespace optimisation::edges {
    line_reprojection::line_reprojection(const sensor::camera::model<double>& camera_model)
        : camera(camera_model) {
        double parameters[32] = {};
        const size_t count = math::min<size_t>(this->camera.get_parameter_count(), 32);
        if ((count > 0) && this->camera.get_parameters(parameters, count)) {
            this->focal = parameters[0];
        }
    }

    bool line_reprojection::set_measured_segment(const double x1, const double y1, const double x2, const double y2) {
        const double pixels[2][2] = { { x1, y1 }, { x2, y2 } };
        for (int i = 0; i < 2; ++i) {
            double ray[3];
            if (!this->camera.unproject(&pixels[i][0], &ray[0]) || !(ray[2] > 1.0e-12)) {
                this->measured_valid = false;
                return false;
            }
            this->measured_normalised[i][0] = ray[0] / ray[2];
            this->measured_normalised[i][1] = ray[1] / ray[2];
        }
        this->measured_valid = true;
        return true;
    }

    void line_reprojection::compute_camera_line(const edge& context, math::matrix<double, 3, 1>& moment_camera, math::matrix<double, 3, 1>& direction_camera, math::matrix<double, 3, 3>& rotation, math::matrix<double, 3, 1>& translation) const {
        const double* const pose_params = context.get_vertex(0)->get_parameters();
        const math::se3<double> pose(math::so3<double>(pose_params[6], pose_params[3], pose_params[4], pose_params[5]), { { pose_params[0], pose_params[1], pose_params[2] } });
        const double* const line_params = context.get_vertex(1)->get_parameters();
        const math::matrix<double, 3, 1> moment_world({ line_params[0], line_params[1], line_params[2] });
        const math::matrix<double, 3, 1> direction_world({ line_params[3], line_params[4], line_params[5] });
        rotation = pose.rotation().get_matrix();
        translation = pose.translation();
        direction_camera = rotation * direction_world;
        moment_camera = (rotation * moment_world) + geometry::plucker::cross(translation, direction_camera);
    }

    void line_reprojection::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        math::matrix<double, 3, 1> moment_camera;
        math::matrix<double, 3, 1> direction_camera;
        math::matrix<double, 3, 3> rotation;
        math::matrix<double, 3, 1> translation;
        this->compute_camera_line(context, moment_camera, direction_camera, rotation, translation);
        const double line_norm = math::sqrt((moment_camera[0] * moment_camera[0]) + (moment_camera[1] * moment_camera[1]));
        if (!this->measured_valid || (line_norm < line_reprojection::minimum_line_norm)) {
            residual[0][0] = 0.0;
            residual[1][0] = 0.0;
            return;
        }
        for (size_t i = 0; i < 2; ++i) {
            const double alignment = (moment_camera[0] * this->measured_normalised[i][0]) + (moment_camera[1] * this->measured_normalised[i][1]) + moment_camera[2];
            residual[i][0] = this->focal * alignment / line_norm;
        }
    }

    void line_reprojection::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        math::matrix<double, 3, 1> moment_camera;
        math::matrix<double, 3, 1> direction_camera;
        math::matrix<double, 3, 3> rotation;
        math::matrix<double, 3, 1> translation;
        this->compute_camera_line(context, moment_camera, direction_camera, rotation, translation);
        const double line_norm_squared = (moment_camera[0] * moment_camera[0]) + (moment_camera[1] * moment_camera[1]);
        const double line_norm = math::sqrt(line_norm_squared);
        if (!this->measured_valid || (line_norm < line_reprojection::minimum_line_norm)) {
            jacobians[0] = math::matrix<double, 0, 0>::zero(2, 6);
            jacobians[1] = math::matrix<double, 0, 0>::zero(2, 4);
            return;
        }
        math::matrix<double, 2, 3> jacobian_residual_line;
        for (size_t i = 0; i < 2; ++i) {
            const double x[3] = { this->measured_normalised[i][0], this->measured_normalised[i][1], 1.0 };
            const double alignment = (moment_camera[0] * x[0]) + (moment_camera[1] * x[1]) + moment_camera[2];
            const double alignment_over_cubed = alignment / (line_norm_squared * line_norm);
            jacobian_residual_line[i][0] = this->focal * ((x[0] / line_norm) - (moment_camera[0] * alignment_over_cubed));
            jacobian_residual_line[i][1] = this->focal * ((x[1] / line_norm) - (moment_camera[1] * alignment_over_cubed));
            jacobian_residual_line[i][2] = this->focal * (x[2] / line_norm);
        }
        math::matrix<double, 3, 6> jacobian_line_pose;
        jacobian_line_pose[0][0] = 0.0;
        jacobian_line_pose[0][1] = moment_camera[2];
        jacobian_line_pose[0][2] = -moment_camera[1];
        jacobian_line_pose[1][0] = -moment_camera[2];
        jacobian_line_pose[1][1] = 0.0;
        jacobian_line_pose[1][2] = moment_camera[0];
        jacobian_line_pose[2][0] = moment_camera[1];
        jacobian_line_pose[2][1] = -moment_camera[0];
        jacobian_line_pose[2][2] = 0.0;
        jacobian_line_pose[0][3] = 0.0;
        jacobian_line_pose[0][4] = direction_camera[2];
        jacobian_line_pose[0][5] = -direction_camera[1];
        jacobian_line_pose[1][3] = -direction_camera[2];
        jacobian_line_pose[1][4] = 0.0;
        jacobian_line_pose[1][5] = direction_camera[0];
        jacobian_line_pose[2][3] = direction_camera[1];
        jacobian_line_pose[2][4] = -direction_camera[0];
        jacobian_line_pose[2][5] = 0.0;
        const math::matrix<double, 2, 6> jacobian_pose_res = jacobian_residual_line * jacobian_line_pose;
        jacobians[0] = math::matrix<double, 0, 0>(2, 6, jacobian_pose_res.data());
        const double* const line_params = context.get_vertex(1)->get_parameters();
        const geometry::plucker line_world(
            math::matrix<double, 3, 1>({ line_params[0], line_params[1], line_params[2] }),
            math::matrix<double, 3, 1>({ line_params[3], line_params[4], line_params[5] })
        );
        math::matrix<double, 3, 3> u;
        double w1 = 0.0;
        double w2 = 0.0;
        double scale = 0.0;
        line_world.to_orthonormal(u, w1, w2, scale);
        const math::matrix<double, 3, 1> u1({ u[0][0], u[1][0], u[2][0] });
        const math::matrix<double, 3, 1> u2({ u[0][1], u[1][1], u[2][1] });
        const math::matrix<double, 3, 1> u3({ u[0][2], u[1][2], u[2][2] });
        const math::matrix<double, 3, 1> zero = math::matrix<double, 3, 1>::zero();
        const math::matrix<double, 3, 1> moment_columns[4] = { zero, u3 * (-scale * w1), u2 * (scale * w1), u1 * (-scale * w2) };
        const math::matrix<double, 3, 1> direction_columns[4] = { u3 * (scale * w2), zero, u1 * (-scale * w2), u2 * (scale * w1) };
        math::matrix<double, 2, 4> jacobian_landmark_res;
        for (size_t column = 0; column < 4; ++column) {
            const math::matrix<double, 3, 1> rotated_direction = rotation * direction_columns[column];
            const math::matrix<double, 3, 1> line_column = (rotation * moment_columns[column]) + geometry::plucker::cross(translation, rotated_direction);
            for (size_t i = 0; i < 2; ++i) {
                jacobian_landmark_res[i][column] = (jacobian_residual_line[i][0] * line_column[0]) + (jacobian_residual_line[i][1] * line_column[1]) + (jacobian_residual_line[i][2] * line_column[2]);
            }
        }
        jacobians[1] = math::matrix<double, 0, 0>(2, 4, jacobian_landmark_res.data());
    }
}
