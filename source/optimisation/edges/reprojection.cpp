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
    reprojection::reprojection(const sensor::camera::model<double>& camera_model, const bool with_intrinsics)
        : camera(camera_model)
        , intrinsics_slot(with_intrinsics ? reprojection::vertex_count : -1) {
    }

    int reprojection::get_vertex_count() const {
        return reprojection::vertex_count + ((this->intrinsics_slot >= 0) ? 1 : 0);
    }

    const vertices::intrinsics* reprojection::intrinsics_at(const edge& context, const int slot) {
        if ((slot < 0) || (static_cast<size_t>(slot) >= context.num_vertices())) {
            return nullptr;
        }
        return context.get_vertex(slot)->get_policy<vertices::intrinsics>();
    }

    void reprojection::set_intrinsics_jacobian(const edge& context, const int slot, const bool projected, const double* const jacobian_parameters, math::matrix<double, 0, 0>& jacobian) {
        const vertices::intrinsics* const calibration = reprojection::intrinsics_at(context, slot);
        const size_t columns = static_cast<size_t>(context.get_vertex(slot)->get_local_dimensions());
        double values[2 * vertex::maximum_parameters] = {};
        if ((calibration != nullptr) && projected) {
            calibration->select_free_columns(jacobian_parameters, 2, -1.0, &values[0]);
        }
        edge::set_jacobian(jacobian, 2, columns, &values[0]);
    }

    double reprojection::behind_camera_residual(const double depth) {
        return -reprojection::behind_camera_slope * depth;
    }

    bool reprojection::measurable(const edge& context) const {
        const double* const pose_params = context.get_vertex(0)->get_parameters();
        const math::se3<double> pose(math::so3<double>(pose_params[6], pose_params[3], pose_params[4], pose_params[5]), { { pose_params[0], pose_params[1], pose_params[2] } });
        const double* const xyz_params = context.get_vertex(1)->get_parameters();
        const math::matrix<double, 3, 1> landmark_world({ xyz_params[0], xyz_params[1], xyz_params[2] });
        const math::matrix<double, 3, 1> landmark_camera = pose * landmark_world;
        math::matrix<double, 2, 1> projected = { { 0.0, 0.0 } };
        const vertices::intrinsics* const calibration = reprojection::intrinsics_at(context, this->intrinsics_slot);
        const sensor::camera::model<double>& projection = (calibration != nullptr) ? calibration->get_camera() : this->camera;
        return projection.project(landmark_camera.data(), projected.data());
    }

    void reprojection::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const double* const pose_params = context.get_vertex(0)->get_parameters();
        const math::se3<double> pose(math::so3<double>(pose_params[6], pose_params[3], pose_params[4], pose_params[5]), { { pose_params[0], pose_params[1], pose_params[2] } });
        const double* const xyz_params = context.get_vertex(1)->get_parameters();
        const math::matrix<double, 3, 1> landmark_world({ xyz_params[0], xyz_params[1], xyz_params[2] });
        const math::matrix<double, 3, 1> landmark_camera = pose * landmark_world;
        math::matrix<double, 2, 1> projected = { { 0.0, 0.0 } };
        const vertices::intrinsics* const calibration = reprojection::intrinsics_at(context, this->intrinsics_slot);
        const sensor::camera::model<double>& projection = (calibration != nullptr) ? calibration->get_camera() : this->camera;
        if (!projection.project(landmark_camera.data(), projected.data())) {
            const double penalty = reprojection::behind_camera_residual(landmark_camera[2]);
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
        const vertices::intrinsics* const calibration = reprojection::intrinsics_at(context, this->intrinsics_slot);
        const sensor::camera::model<double>& projection = (calibration != nullptr) ? calibration->get_camera() : this->camera;
        double jacobian_parameters[2 * vertex::maximum_parameters];
        const bool projects = projection.project(landmark_camera.data(), projected, jacobian_camera.data(), (calibration != nullptr) ? &jacobian_parameters[0] : nullptr);
        if (!projects) {
            jacobian_camera[0][0] = 0.0;
            jacobian_camera[0][1] = 0.0;
            jacobian_camera[0][2] = reprojection::behind_camera_slope;
            jacobian_camera[1][0] = 0.0;
            jacobian_camera[1][1] = 0.0;
            jacobian_camera[1][2] = reprojection::behind_camera_slope;
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
        edge::set_jacobian(jacobians[0], 2, 6, jacobian_pose_res.data());

        const math::matrix<double, 2, 3> jacobian_landmark_res = -(jacobian_camera * pose.rotation().get_matrix());
        edge::set_jacobian(jacobians[1], 2, 3, jacobian_landmark_res.data());

        if (this->intrinsics_slot >= 0) {
            reprojection::set_intrinsics_jacobian(context, this->intrinsics_slot, projects, &jacobian_parameters[0], jacobians[static_cast<size_t>(this->intrinsics_slot)]);
        }
    }
}
