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

#include "optimisation/edges/inertial.hpp"

#include "core/assert.hpp"

namespace {
    math::matrix<double, 3, 3> skew(const math::matrix<double, 3, 1>& vector) {
        return { { { 0.0, -vector[2], vector[1] },
                   { vector[2], 0.0, -vector[0] },
                   { -vector[1], vector[0], 0.0 } } };
    }

    math::se3<double> pose_from_parameters(const double* const parameters) {
        return math::se3<double>(math::so3<double>(parameters[6], parameters[3], parameters[4], parameters[5]), { { parameters[0], parameters[1], parameters[2] } });
    }

    // Writes the rows of a 3x3 block into a 9x3 or 9x6 matrix at (row, column).
    template <size_t columns>
    void put(math::matrix<double, 9, columns>& target, const size_t row, const size_t column, const math::matrix<double, 3, 3>& block) {
        math::set_block<double, 3, 3>(target, row, column, block);
    }

    void read_matrix(const math::matrix<double, 0, 0>& observation, const size_t offset, math::matrix<double, 3, 3>& value) {
        for (size_t i = 0; i < 9; ++i) {
            value.data()[i] = observation[offset + i][0];
        }
    }

    void write_matrix(math::matrix<double, 0, 0>& observation, const size_t offset, const math::matrix<double, 3, 3>& value) {
        for (size_t i = 0; i < 9; ++i) {
            observation[offset + i][0] = value.data()[i];
        }
    }
}

namespace optimisation::edges {
    inertial::measurement inertial::measurement::from_observation(const math::matrix<double, 0, 0>& observation) {
        ASSERT(observation.rows() == static_cast<size_t>(inertial::observation_count), "An inertial edge's observation holds the preintegrated measurement.");
        measurement result;
        result.delta_rotation = math::so3<double>(observation[3][0], observation[0][0], observation[1][0], observation[2][0]);
        result.delta_velocity = { { observation[4][0], observation[5][0], observation[6][0] } };
        result.delta_position = { { observation[7][0], observation[8][0], observation[9][0] } };
        result.delta_time = observation[10][0];
        result.linearisation.gyroscope = { { observation[11][0], observation[12][0], observation[13][0] } };
        result.linearisation.accelerometer = { { observation[14][0], observation[15][0], observation[16][0] } };
        read_matrix(observation, 17, result.rotation_gyroscope);
        read_matrix(observation, 26, result.velocity_gyroscope);
        read_matrix(observation, 35, result.velocity_accelerometer);
        read_matrix(observation, 44, result.position_gyroscope);
        read_matrix(observation, 53, result.position_accelerometer);
        return result;
    }

    inertial::inertial(const math::matrix<double, 3, 1>& gravity_world, const math::se3<double>& imu_on_body)
        : gravity(gravity_world)
        , body_from_imu(imu_on_body) {
    }

    math::matrix<double, 0, 0> inertial::observation(const sensor::imu::preintegration& preintegration) {
        math::matrix<double, 0, 0> result = math::matrix<double, 0, 0>::zero(static_cast<size_t>(inertial::observation_count), 1);
        const math::matrix<double, 4, 1> quaternion = preintegration.get_delta_rotation().get_quaternion();
        result[0][0] = quaternion[1];
        result[1][0] = quaternion[2];
        result[2][0] = quaternion[3];
        result[3][0] = quaternion[0];
        for (size_t i = 0; i < 3; ++i) {
            result[4 + i][0] = preintegration.get_delta_velocity()[i];
            result[7 + i][0] = preintegration.get_delta_position()[i];
            result[11 + i][0] = preintegration.get_linearisation().gyroscope[i];
            result[14 + i][0] = preintegration.get_linearisation().accelerometer[i];
        }
        result[10][0] = preintegration.get_delta_time();
        write_matrix(result, 17, preintegration.get_rotation_gyroscope_jacobian());
        write_matrix(result, 26, preintegration.get_velocity_gyroscope_jacobian());
        write_matrix(result, 35, preintegration.get_velocity_accelerometer_jacobian());
        write_matrix(result, 44, preintegration.get_position_gyroscope_jacobian());
        write_matrix(result, 53, preintegration.get_position_accelerometer_jacobian());
        return result;
    }

    bool inertial::information(const sensor::imu::preintegration& preintegration, math::matrix<double, 0, 0>& information_matrix) {
        math::matrix<double, 9, 9> inverse;
        if (!math::invert(preintegration.get_covariance(), inverse)) {
            return false;
        }
        information_matrix = math::matrix<double, 0, 0>(9, 9);
        for (size_t r = 0; r < 9; ++r) {
            for (size_t c = 0; c < 9; ++c) {
                information_matrix[r][c] = 0.5 * (inverse[r][c] + inverse[c][r]);
            }
        }
        return true;
    }

    void inertial::evaluate(
        const measurement& preintegrated,
        const sensor::imu::preintegration::state& first,
        const sensor::imu::preintegration::state& second,
        const sensor::imu::bias& estimate,
        const math::matrix<double, 3, 1>& gravity_world,
        double* const residual,
        state_jacobians* const jacobians
    ) {
        const math::matrix<double, 3, 1> gyroscope_change = estimate.gyroscope - preintegrated.linearisation.gyroscope;
        const math::matrix<double, 3, 1> accelerometer_change = estimate.accelerometer - preintegrated.linearisation.accelerometer;
        const math::matrix<double, 3, 1> rotation_correction = preintegrated.rotation_gyroscope * gyroscope_change;
        const math::so3<double> delta_rotation = (preintegrated.delta_rotation * math::so3<double>::exp(rotation_correction)).normalised();
        const math::matrix<double, 3, 1> delta_velocity = preintegrated.delta_velocity + (preintegrated.velocity_gyroscope * gyroscope_change) + (preintegrated.velocity_accelerometer * accelerometer_change);
        const math::matrix<double, 3, 1> delta_position = preintegrated.delta_position + (preintegrated.position_gyroscope * gyroscope_change) + (preintegrated.position_accelerometer * accelerometer_change);
        const double duration = preintegrated.delta_time;

        const math::matrix<double, 3, 3> first_rotation_transpose = math::transpose(first.rotation.get_matrix());
        const math::so3<double> rotation_error = (delta_rotation.inverse() * first.rotation.inverse() * second.rotation).normalised();
        const math::matrix<double, 3, 1> rotation_residual = rotation_error.log();
        const math::matrix<double, 3, 1> velocity_term = first_rotation_transpose * (second.velocity - first.velocity - (duration * gravity_world));
        const math::matrix<double, 3, 1> position_term = first_rotation_transpose * (second.position - first.position - (duration * first.velocity) - ((0.5 * duration * duration) * gravity_world));
        const math::matrix<double, 3, 1> velocity_residual = velocity_term - delta_velocity;
        const math::matrix<double, 3, 1> position_residual = position_term - delta_position;
        for (size_t i = 0; i < 3; ++i) {
            residual[i] = rotation_residual[i];
            residual[3 + i] = velocity_residual[i];
            residual[6 + i] = position_residual[i];
        }
        if (jacobians == nullptr) {
            return;
        }

        // The right jacobian of SO(3) is the left jacobian of the negated rotation vector, and so for their inverses.
        const math::matrix<double, 3, 3> right_jacobian_inverse = math::so3<double>::left_jacobian_inverse(-1.0 * rotation_residual);
        const math::matrix<double, 3, 3> correction_right_jacobian = math::so3<double>::left_jacobian(-1.0 * rotation_correction);
        const math::matrix<double, 3, 3> zero = math::matrix<double, 3, 3>::zero();
        *jacobians = state_jacobians();
        jacobians->rotation_i = math::matrix<double, 9, 3>::zero();
        jacobians->position_i = math::matrix<double, 9, 3>::zero();
        jacobians->velocity_i = math::matrix<double, 9, 3>::zero();
        jacobians->bias = math::matrix<double, 9, 6>::zero();
        jacobians->rotation_j = math::matrix<double, 9, 3>::zero();
        jacobians->position_j = math::matrix<double, 9, 3>::zero();
        jacobians->velocity_j = math::matrix<double, 9, 3>::zero();
        jacobians->gravity = math::matrix<double, 9, 3>::zero();

        put(jacobians->rotation_i, 0, 0, -1.0 * (right_jacobian_inverse * math::transpose(second.rotation.get_matrix()) * first.rotation.get_matrix()));
        put(jacobians->rotation_i, 3, 0, skew(velocity_term));
        put(jacobians->rotation_i, 6, 0, skew(position_term));
        put(jacobians->rotation_j, 0, 0, right_jacobian_inverse);

        put(jacobians->position_i, 6, 0, -1.0 * first_rotation_transpose);
        put(jacobians->position_j, 6, 0, first_rotation_transpose);

        put(jacobians->velocity_i, 3, 0, -1.0 * first_rotation_transpose);
        put(jacobians->velocity_i, 6, 0, -duration * first_rotation_transpose);
        put(jacobians->velocity_j, 3, 0, first_rotation_transpose);

        put(jacobians->bias, 0, 0, -1.0 * (right_jacobian_inverse * math::transpose(rotation_error.get_matrix()) * correction_right_jacobian * preintegrated.rotation_gyroscope));
        put(jacobians->bias, 0, 3, zero);
        put(jacobians->bias, 3, 0, -1.0 * preintegrated.velocity_gyroscope);
        put(jacobians->bias, 3, 3, -1.0 * preintegrated.velocity_accelerometer);
        put(jacobians->bias, 6, 0, -1.0 * preintegrated.position_gyroscope);
        put(jacobians->bias, 6, 3, -1.0 * preintegrated.position_accelerometer);

        put(jacobians->gravity, 3, 0, -duration * first_rotation_transpose);
        put(jacobians->gravity, 6, 0, (-0.5 * duration * duration) * first_rotation_transpose);
    }

    sensor::imu::preintegration::state inertial::imu_state(const double* const pose, const double* const velocity, const math::se3<double>& imu_on_body, const double scale) {
        const math::se3<double> body = pose_from_parameters(pose).inverse();
        sensor::imu::preintegration::state result;
        result.rotation = (body.rotation() * imu_on_body.rotation()).normalised();
        result.position = (scale * body.translation()) + (body.rotation() * imu_on_body.translation());
        result.velocity = { { velocity[0], velocity[1], velocity[2] } };
        return result;
    }

    math::matrix<double, 9, 6> inertial::pose_jacobian(const math::matrix<double, 9, 3>& rotation, const math::matrix<double, 9, 3>& position, const double* const pose, const math::se3<double>& imu_on_body, const double scale) {
        // The update exp((phi, rho)) * T_bw moves the body to the world as T_wb * exp(-(phi, rho)): R_wb * Exp(-phi) and
        // p_wb - R_wb rho at first order. So the imu turns by -R_bi^T phi on its right, and moves by R_wb [p_bi]x phi - s R_wb rho.
        const math::matrix<double, 3, 3> world_from_body = math::transpose(pose_from_parameters(pose).rotation().get_matrix());
        const math::matrix<double, 3, 3> imu_from_body = math::transpose(imu_on_body.rotation().get_matrix());
        const math::matrix<double, 9, 3> turn = (rotation * (-1.0 * imu_from_body)) + (position * (world_from_body * skew(imu_on_body.translation())));
        const math::matrix<double, 9, 3> move = position * (-scale * world_from_body);
        math::matrix<double, 9, 6> result;
        for (size_t r = 0; r < 9; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                result[r][c] = turn[r][c];
                result[r][3 + c] = move[r][c];
            }
        }
        return result;
    }

    void inertial::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const measurement preintegrated = measurement::from_observation(context.get_observation());
        const sensor::imu::preintegration::state first = inertial::imu_state(context.get_vertex(0)->get_parameters(), context.get_vertex(1)->get_parameters(), this->body_from_imu);
        const sensor::imu::preintegration::state second = inertial::imu_state(context.get_vertex(3)->get_parameters(), context.get_vertex(4)->get_parameters(), this->body_from_imu);
        const double* const bias_parameters = context.get_vertex(2)->get_parameters();
        sensor::imu::bias estimate;
        estimate.gyroscope = { { bias_parameters[0], bias_parameters[1], bias_parameters[2] } };
        estimate.accelerometer = { { bias_parameters[3], bias_parameters[4], bias_parameters[5] } };
        double values[9];
        inertial::evaluate(preintegrated, first, second, estimate, this->gravity, &values[0], nullptr);
        for (size_t i = 0; i < 9; ++i) {
            residual[i][0] = values[i];
        }
    }

    void inertial::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        const measurement preintegrated = measurement::from_observation(context.get_observation());
        const double* const first_pose = context.get_vertex(0)->get_parameters();
        const double* const second_pose = context.get_vertex(3)->get_parameters();
        const sensor::imu::preintegration::state first = inertial::imu_state(first_pose, context.get_vertex(1)->get_parameters(), this->body_from_imu);
        const sensor::imu::preintegration::state second = inertial::imu_state(second_pose, context.get_vertex(4)->get_parameters(), this->body_from_imu);
        const double* const bias_parameters = context.get_vertex(2)->get_parameters();
        sensor::imu::bias estimate;
        estimate.gyroscope = { { bias_parameters[0], bias_parameters[1], bias_parameters[2] } };
        estimate.accelerometer = { { bias_parameters[3], bias_parameters[4], bias_parameters[5] } };
        double values[9];
        state_jacobians blocks;
        inertial::evaluate(preintegrated, first, second, estimate, this->gravity, &values[0], &blocks);
        const math::matrix<double, 9, 6> first_pose_jacobian = inertial::pose_jacobian(blocks.rotation_i, blocks.position_i, first_pose, this->body_from_imu);
        const math::matrix<double, 9, 6> second_pose_jacobian = inertial::pose_jacobian(blocks.rotation_j, blocks.position_j, second_pose, this->body_from_imu);
        edge::set_jacobian(jacobians[0], 9, 6, first_pose_jacobian.data());
        edge::set_jacobian(jacobians[1], 9, 3, blocks.velocity_i.data());
        edge::set_jacobian(jacobians[2], 9, 6, blocks.bias.data());
        edge::set_jacobian(jacobians[3], 9, 6, second_pose_jacobian.data());
        edge::set_jacobian(jacobians[4], 9, 3, blocks.velocity_j.data());
    }
}
