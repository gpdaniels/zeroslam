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

#include "optimisation/edges/depth_inverse_depth.hpp"

#include "math/lie.hpp"

namespace optimisation::edges {
    depth_inverse_depth::depth_inverse_depth(const sensor::depth::measure_kind measure_value, const math::matrix<double, 3, 3>& anchor_rotation_value, const math::matrix<double, 3, 1>& anchor_translation_value)
        : measure(measure_value)
        , anchor_rotation(anchor_rotation_value)
        , anchor_translation(anchor_translation_value) {
    }

    void depth_inverse_depth::compute_scaled_point(const edge& context, math::matrix<double, 3, 1>& q, math::matrix<double, 3, 3>& rotation_combined, math::matrix<double, 3, 1>& rho_column, double& rho) const {
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

    bool depth_inverse_depth::measurable(const edge& context) const {
        math::matrix<double, 3, 1> q;
        math::matrix<double, 3, 3> rotation_combined;
        math::matrix<double, 3, 1> rho_column;
        double rho = 0.0;
        this->compute_scaled_point(context, q, rotation_combined, rho_column, rho);
        const double scaled[3] = { q[0], q[1], q[2] };
        return sensor::depth::measured(this->measure, scaled) > depth_inverse_depth::minimum_scaled_depth;
    }

    void depth_inverse_depth::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        math::matrix<double, 3, 1> q;
        math::matrix<double, 3, 3> rotation_combined;
        math::matrix<double, 3, 1> rho_column;
        double rho = 0.0;
        this->compute_scaled_point(context, q, rotation_combined, rho_column, rho);
        const double scaled[3] = { q[0], q[1], q[2] };
        const double scaled_measure = sensor::depth::measured(this->measure, scaled);
        const double observed_inverse = 1.0 / context.get_observation()[0][0];
        residual[0][0] = (scaled_measure > depth_inverse_depth::minimum_scaled_depth) ? (observed_inverse - (rho / scaled_measure)) : observed_inverse;
    }

    void depth_inverse_depth::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        math::matrix<double, 3, 1> q;
        math::matrix<double, 3, 3> rotation_combined;
        math::matrix<double, 3, 1> rho_column;
        double rho = 0.0;
        this->compute_scaled_point(context, q, rotation_combined, rho_column, rho);
        const double scaled[3] = { q[0], q[1], q[2] };
        const double scaled_measure = sensor::depth::measured(this->measure, scaled);
        // The residual is the observed inverse depth less rho / m(q): its derivative is rho / m^2 dm/dq by q, and -1 / m by
        // rho where rho appears outside q.
        math::matrix<double, 1, 3> jacobian_scaled = math::matrix<double, 1, 3>::zero();
        double jacobian_rho_direct = 0.0;
        if (scaled_measure > depth_inverse_depth::minimum_scaled_depth) {
            const double factor = rho / (scaled_measure * scaled_measure);
            if (this->measure == sensor::depth::measure_kind::axial) {
                jacobian_scaled[2] = factor;
            }
            else {
                for (size_t axis = 0; axis < 3; ++axis) {
                    jacobian_scaled[axis] = factor * q[axis] / scaled_measure;
                }
            }
            jacobian_rho_direct = -1.0 / scaled_measure;
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
        const math::matrix<double, 1, 6> jacobian_pose_res = jacobian_scaled * jacobian_point_pose;
        edge::set_jacobian(jacobians[0], 1, 6, jacobian_pose_res.data());
        math::matrix<double, 3, 3> jacobian_point_landmark;
        for (size_t row = 0; row < 3; ++row) {
            jacobian_point_landmark[row][0] = rotation_combined[row][0];
            jacobian_point_landmark[row][1] = rotation_combined[row][1];
            jacobian_point_landmark[row][2] = rho_column[row];
        }
        math::matrix<double, 1, 3> jacobian_landmark_res = jacobian_scaled * jacobian_point_landmark;
        jacobian_landmark_res[2] += jacobian_rho_direct;
        edge::set_jacobian(jacobians[1], 1, 3, jacobian_landmark_res.data());
    }
}
