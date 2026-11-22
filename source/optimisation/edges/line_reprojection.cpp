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
#include "optimisation/edges/reprojection.hpp"
#include "optimisation/vertices/intrinsics.hpp"
#include "sensor/camera.hpp"

namespace optimisation::edges {
    line_reprojection::line_reprojection(const sensor::camera::model<double>& camera_model, const bool with_intrinsics)
        : camera(camera_model) {
        // Residuals are distances on the image plane scaled to pixels by the focal lengths of the camera's pinhole.
        double pinhole[4] = {};
        if (this->camera.get_pinhole(&pinhole[0], 4)) {
            this->focal_x = pinhole[0];
            this->focal_y = pinhole[1];
        }
        this->intrinsics_slot = with_intrinsics ? line_reprojection::vertex_count : -1;
    }

    int line_reprojection::get_vertex_count() const {
        return line_reprojection::vertex_count + ((this->intrinsics_slot >= 0) ? 1 : 0);
    }

    bool line_reprojection::measurement(const edge& context, double (&normalised)[2][2], double& fx, double& fy) const {
        const vertices::intrinsics* const calibration = reprojection::intrinsics_at(context, this->intrinsics_slot);
        fx = this->focal_x;
        fy = this->focal_y;
        if (calibration == nullptr) {
            for (int i = 0; i < 2; ++i) {
                normalised[i][0] = this->measured_normalised[i][0];
                normalised[i][1] = this->measured_normalised[i][1];
            }
            return this->measured_valid;
        }
        if (!this->measured_valid) {
            return false;
        }
        for (int i = 0; i < 2; ++i) {
            double ray[3];
            if (!calibration->get_camera().unproject(&this->measured_pixels[i][0], &ray[0]) || !(ray[2] > 1.0e-12)) {
                return false;
            }
            normalised[i][0] = ray[0] / ray[2];
            normalised[i][1] = ray[1] / ray[2];
        }
        return true;
    }

    bool line_reprojection::set_measured_segment(const double x1, const double y1, const double x2, const double y2) {
        const double pixels[2][2] = { { x1, y1 }, { x2, y2 } };
        for (int i = 0; i < 2; ++i) {
            double ray[3];
            if (!this->camera.unproject(&pixels[i][0], &ray[0]) || !sensor::camera::on_image_plane(&ray[0])) {
                this->measured_valid = false;
                return false;
            }
            this->measured_normalised[i][0] = ray[0] / ray[2];
            this->measured_normalised[i][1] = ray[1] / ray[2];
            this->measured_pixels[i][0] = pixels[i][0];
            this->measured_pixels[i][1] = pixels[i][1];
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
        double normalised[2][2];
        double fx = 1.0;
        double fy = 1.0;
        const bool valid = this->measurement(context, normalised, fx, fy);
        const double line_norm = math::sqrt((moment_camera[0] * moment_camera[0]) + (moment_camera[1] * moment_camera[1]));
        if (!valid || (line_norm < line_reprojection::minimum_line_norm)) {
            residual[0][0] = 0.0;
            residual[1][0] = 0.0;
            return;
        }
        // The distance in pixels to the image line (m0 / fx, m1 / fy, ...) of the camera line m0 x + m1 y + m2 = 0.
        const double pixel_x = moment_camera[0] / fx;
        const double pixel_y = moment_camera[1] / fy;
        const double pixel_norm = math::sqrt((pixel_x * pixel_x) + (pixel_y * pixel_y));
        for (size_t i = 0; i < 2; ++i) {
            const double alignment = (moment_camera[0] * normalised[i][0]) + (moment_camera[1] * normalised[i][1]) + moment_camera[2];
            residual[i][0] = alignment / pixel_norm;
        }
    }

    void line_reprojection::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        math::matrix<double, 3, 1> moment_camera;
        math::matrix<double, 3, 1> direction_camera;
        math::matrix<double, 3, 3> rotation;
        math::matrix<double, 3, 1> translation;
        this->compute_camera_line(context, moment_camera, direction_camera, rotation, translation);
        double normalised[2][2];
        double fx = 1.0;
        double fy = 1.0;
        const bool valid = this->measurement(context, normalised, fx, fy);
        const double line_norm_squared = (moment_camera[0] * moment_camera[0]) + (moment_camera[1] * moment_camera[1]);
        const double line_norm = math::sqrt(line_norm_squared);
        if (!valid || (line_norm < line_reprojection::minimum_line_norm)) {
            const double zeros[12] = {};
            edge::set_jacobian(jacobians[0], 2, 6, &zeros[0]);
            edge::set_jacobian(jacobians[1], 2, 4, &zeros[0]);
            if (this->intrinsics_slot >= 0) {
                reprojection::set_intrinsics_jacobian(context, this->intrinsics_slot, false, nullptr, jacobians[static_cast<size_t>(this->intrinsics_slot)]);
            }
            return;
        }
        const double pixel_x = moment_camera[0] / fx;
        const double pixel_y = moment_camera[1] / fy;
        const double pixel_norm_squared = (pixel_x * pixel_x) + (pixel_y * pixel_y);
        const double pixel_norm = math::sqrt(pixel_norm_squared);
        math::matrix<double, 2, 3> jacobian_residual_line;
        for (size_t i = 0; i < 2; ++i) {
            const double x[3] = { normalised[i][0], normalised[i][1], 1.0 };
            const double alignment = (moment_camera[0] * x[0]) + (moment_camera[1] * x[1]) + moment_camera[2];
            const double alignment_over_cubed = alignment / (pixel_norm_squared * pixel_norm);
            jacobian_residual_line[i][0] = (x[0] / pixel_norm) - ((pixel_x / fx) * alignment_over_cubed);
            jacobian_residual_line[i][1] = (x[1] / pixel_norm) - ((pixel_y / fy) * alignment_over_cubed);
            jacobian_residual_line[i][2] = x[2] / pixel_norm;
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
        edge::set_jacobian(jacobians[0], 2, 6, jacobian_pose_res.data());
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
        edge::set_jacobian(jacobians[1], 2, 4, jacobian_landmark_res.data());

        if (this->intrinsics_slot >= 0) {
            // Each endpoint's normalised coordinates n hold project((n, 1), parameters) at the measured pixel, so they move by
            // -A^-1 J_parameters with A the projection's Jacobian by n. The scale to pixels is the pinhole of the camera the edge
            // was made with, a weight held through the adjustment.
            const vertices::intrinsics* const calibration = reprojection::intrinsics_at(context, this->intrinsics_slot);
            const size_t count = (calibration != nullptr) ? static_cast<size_t>(calibration->get_parameter_count()) : 0;
            double jacobian_full[2 * vertex::maximum_parameters] = {};
            bool measured = calibration != nullptr;
            for (size_t i = 0; measured && (i < 2); ++i) {
                const double point[3] = { normalised[i][0], normalised[i][1], 1.0 };
                double pixel[2];
                double jacobian_point[2 * 3];
                double jacobian_parameters[2 * vertex::maximum_parameters];
                measured = calibration->get_camera().project(&point[0], &pixel[0], &jacobian_point[0], &jacobian_parameters[0]);
                const double determinant = (jacobian_point[0] * jacobian_point[4]) - (jacobian_point[1] * jacobian_point[3]);
                measured = measured && ((determinant > 1.0e-12) || (determinant < -1.0e-12));
                if (!measured) {
                    break;
                }
                for (size_t k = 0; k < count; ++k) {
                    const double parameter_x = jacobian_parameters[k];
                    const double parameter_y = jacobian_parameters[count + k];
                    const double moved_x = -((jacobian_point[4] * parameter_x) - (jacobian_point[1] * parameter_y)) / determinant;
                    const double moved_y = -((jacobian_point[0] * parameter_y) - (jacobian_point[3] * parameter_x)) / determinant;
                    const double value = ((moment_camera[0] * moved_x) + (moment_camera[1] * moved_y)) / pixel_norm;
                    jacobian_full[(i * count) + k] = value;
                }
            }
            const size_t columns = static_cast<size_t>(context.get_vertex(this->intrinsics_slot)->get_local_dimensions());
            double jacobian_local[2 * vertex::maximum_parameters] = {};
            if (measured) {
                calibration->select_free_columns(&jacobian_full[0], 2, 1.0, &jacobian_local[0]);
            }
            edge::set_jacobian(jacobians[static_cast<size_t>(this->intrinsics_slot)], 2, columns, &jacobian_local[0]);
        }
    }
}
