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

#include "optimisation/edges/inertial_gravity_scale.hpp"

#include "optimisation/edges/inertial.hpp"
#include "sensor/imu/measurement.hpp"
#include "sensor/imu/preintegration.hpp"

namespace {
    sensor::imu::bias bias_from_parameters(const double* const parameters) {
        sensor::imu::bias result;
        result.gyroscope = { { parameters[0], parameters[1], parameters[2] } };
        result.accelerometer = { { parameters[3], parameters[4], parameters[5] } };
        return result;
    }

    math::so3<double> direction_from_parameters(const double* const parameters) {
        return math::so3<double>(parameters[3], parameters[0], parameters[1], parameters[2]);
    }
}

namespace optimisation::edges {
    inertial_gravity_scale::inertial_gravity_scale(const double gravity, const math::se3<double>& imu_on_body)
        : gravity_magnitude(gravity)
        , body_from_imu(imu_on_body) {
    }

    math::matrix<double, 3, 1> inertial_gravity_scale::gravity_from_direction(const double* const direction, const double gravity) {
        return direction_from_parameters(direction) * math::matrix<double, 3, 1>({ { 0.0, 0.0, -gravity } });
    }

    void inertial_gravity_scale::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const inertial::measurement preintegrated = inertial::measurement::from_observation(context.get_observation());
        const double scale = context.get_vertex(6)->get_parameters()[0];
        const sensor::imu::preintegration::state first = inertial::imu_state(context.get_vertex(0)->get_parameters(), context.get_vertex(1)->get_parameters(), this->body_from_imu, scale);
        const sensor::imu::preintegration::state second = inertial::imu_state(context.get_vertex(3)->get_parameters(), context.get_vertex(4)->get_parameters(), this->body_from_imu, scale);
        const math::matrix<double, 3, 1> gravity = inertial_gravity_scale::gravity_from_direction(context.get_vertex(5)->get_parameters(), this->gravity_magnitude);
        double values[9];
        inertial::evaluate(preintegrated, first, second, bias_from_parameters(context.get_vertex(2)->get_parameters()), gravity, &values[0], nullptr);
        for (size_t i = 0; i < 9; ++i) {
            residual[i][0] = values[i];
        }
    }

    void inertial_gravity_scale::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        const inertial::measurement preintegrated = inertial::measurement::from_observation(context.get_observation());
        const double* const first_pose = context.get_vertex(0)->get_parameters();
        const double* const second_pose = context.get_vertex(3)->get_parameters();
        const double* const direction = context.get_vertex(5)->get_parameters();
        const double scale = context.get_vertex(6)->get_parameters()[0];
        const sensor::imu::preintegration::state first = inertial::imu_state(first_pose, context.get_vertex(1)->get_parameters(), this->body_from_imu, scale);
        const sensor::imu::preintegration::state second = inertial::imu_state(second_pose, context.get_vertex(4)->get_parameters(), this->body_from_imu, scale);
        const math::matrix<double, 3, 1> gravity = inertial_gravity_scale::gravity_from_direction(direction, this->gravity_magnitude);
        double values[9];
        inertial::state_jacobians blocks;
        inertial::evaluate(preintegrated, first, second, bias_from_parameters(context.get_vertex(2)->get_parameters()), gravity, &values[0], &blocks);

        const math::matrix<double, 9, 6> first_pose_jacobian = inertial::pose_jacobian(blocks.rotation_i, blocks.position_i, first_pose, this->body_from_imu, scale);
        const math::matrix<double, 9, 6> second_pose_jacobian = inertial::pose_jacobian(blocks.rotation_j, blocks.position_j, second_pose, this->body_from_imu, scale);

        // Gravity R_wg Exp(delta) (0, 0, -g) moves by -R_wg [(0, 0, -g)]x delta, of which the update takes the first two columns.
        const math::matrix<double, 3, 3> direction_matrix = direction_from_parameters(direction).get_matrix();
        const double g = this->gravity_magnitude;
        const math::matrix<double, 3, 3> down_skew = { { { 0.0, g, 0.0 }, { -g, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } } };
        const math::matrix<double, 3, 3> gravity_change = -1.0 * (direction_matrix * down_skew);
        const math::matrix<double, 9, 3> gravity_jacobian = blocks.gravity * gravity_change;
        math::matrix<double, 9, 2> direction_jacobian;
        for (size_t r = 0; r < 9; ++r) {
            direction_jacobian[r][0] = gravity_jacobian[r][0];
            direction_jacobian[r][1] = gravity_jacobian[r][1];
        }

        // The scale update s exp(delta) moves each imu position by s p_wb delta.
        const math::se3<double> first_body = math::se3<double>(math::so3<double>(first_pose[6], first_pose[3], first_pose[4], first_pose[5]), { { first_pose[0], first_pose[1], first_pose[2] } }).inverse();
        const math::se3<double> second_body = math::se3<double>(math::so3<double>(second_pose[6], second_pose[3], second_pose[4], second_pose[5]), { { second_pose[0], second_pose[1], second_pose[2] } }).inverse();
        const math::matrix<double, 9, 1> scale_jacobian = (blocks.position_i * (scale * first_body.translation())) + (blocks.position_j * (scale * second_body.translation()));

        edge::set_jacobian(jacobians[0], 9, 6, first_pose_jacobian.data());
        edge::set_jacobian(jacobians[1], 9, 3, blocks.velocity_i.data());
        edge::set_jacobian(jacobians[2], 9, 6, blocks.bias.data());
        edge::set_jacobian(jacobians[3], 9, 6, second_pose_jacobian.data());
        edge::set_jacobian(jacobians[4], 9, 3, blocks.velocity_j.data());
        edge::set_jacobian(jacobians[5], 9, 2, direction_jacobian.data());
        edge::set_jacobian(jacobians[6], 9, 1, scale_jacobian.data());
    }
}
