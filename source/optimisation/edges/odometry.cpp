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

#include "optimisation/edges/odometry.hpp"

#include "core/assert.hpp"
#include "math/math.hpp"
#include "optimisation/vertices/calibration.hpp"
#include "optimisation/vertices/extrinsic.hpp"

namespace optimisation::edges {
    math::matrix<double, 0, 0> odometry::observation(const sensor::odometry::preintegration& motion) {
        const size_t parameters = motion.get_parameter_count();
        math::matrix<double, 0, 0> result = math::matrix<double, 0, 0>::zero(3 + (4 * parameters), 1);
        for (size_t i = 0; i < 3; ++i) {
            result[i][0] = motion.get_delta()[i];
        }
        for (size_t i = 0; i < parameters; ++i) {
            result[3 + i][0] = motion.get_linearisation()[i];
        }
        for (size_t row = 0; row < 3; ++row) {
            for (size_t i = 0; i < parameters; ++i) {
                result[3 + parameters + (row * parameters) + i][0] = motion.get_jacobian()[row][i];
            }
        }
        return result;
    }

    math::matrix<double, 0, 0> odometry::information(const sensor::odometry::preintegration& motion) {
        const sensor::odometry::noise& noise = motion.get_noise();
        const double travelled = motion.get_distance();
        const double minimum_translation = noise.minimum_translation * noise.minimum_translation;
        const double minimum_rotation = noise.minimum_rotation * noise.minimum_rotation;
        const math::matrix<double, 3, 3>& covariance = motion.get_covariance();

        // The planar block in the order of its residual rows (theta, x, y).
        math::matrix<double, 3, 3> planar;
        const size_t order[3] = { 2, 0, 1 };
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                planar[r][c] = covariance[order[r]][order[c]];
            }
        }
        planar[0][0] += minimum_rotation;
        planar[1][1] += minimum_translation;
        planar[2][2] += minimum_translation;
        math::matrix<double, 3, 3> planar_information;
        if (!math::invert(planar, planar_information)) {
            planar_information = math::matrix<double, 3, 3>::zero();
        }

        math::matrix<double, 0, 0> result = math::matrix<double, 0, 0>::zero(6, 6);
        const double tilt = (noise.tilt * noise.tilt * travelled) + minimum_rotation;
        const double vertical = (noise.vertical * noise.vertical * travelled) + minimum_translation;
        result[0][0] = 1.0 / tilt;
        result[1][1] = 1.0 / tilt;
        result[5][5] = 1.0 / vertical;
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                result[2 + r][2 + c] = planar_information[r][c];
            }
        }
        return result;
    }

    math::matrix<double, 3, 1> odometry::measurement(const edge& context) {
        const math::matrix<double, 0, 0>& observation = context.get_observation();
        ASSERT((observation.rows() >= 3) && (((observation.rows() - 3) % 4) == 0), "An odometry observation holds the motion, the linearisation point and the Jacobian.");
        const size_t parameters = (observation.rows() - 3) / 4;
        const double* const values = context.get_vertex(static_cast<int>(odometry::calibration))->get_parameters();
        math::matrix<double, 3, 1> result = { { observation[0][0], observation[1][0], observation[2][0] } };
        for (size_t row = 0; row < 3; ++row) {
            for (size_t i = 0; (i < parameters) && (i < vertices::calibration::maximum_values); ++i) {
                result[row] += observation[3 + parameters + (row * parameters) + i][0] * (values[i] - observation[3 + i][0]);
            }
        }
        return result;
    }

    math::se3<double> odometry::read_pose(const double* const parameters) {
        return math::se3<double>(math::so3<double>(parameters[6], parameters[3], parameters[4], parameters[5]), { { parameters[0], parameters[1], parameters[2] } });
    }

    void odometry::write_pose(const math::se3<double>& pose, double* const parameters) {
        const math::so3<double> rotation = pose.rotation().normalised();
        parameters[0] = pose.translation()[0];
        parameters[1] = pose.translation()[1];
        parameters[2] = pose.translation()[2];
        parameters[3] = rotation.get_quaternion()[1];
        parameters[4] = rotation.get_quaternion()[2];
        parameters[5] = rotation.get_quaternion()[3];
        parameters[6] = rotation.get_quaternion()[0];
    }

    math::matrix<double, 3, 3> odometry::skew(const math::matrix<double, 3, 1>& vector) {
        return { { { 0.0, -vector[2], vector[1] },
                   { vector[2], 0.0, -vector[0] },
                   { -vector[1], vector[0], 0.0 } } };
    }

    math::matrix<double, 6, 6> odometry::adjoint(const math::se3<double>& pose) {
        const math::matrix<double, 3, 3> rotation = pose.rotation().get_matrix();
        const math::matrix<double, 3, 3> coupling = odometry::skew(pose.translation()) * rotation;
        math::matrix<double, 6, 6> result = math::matrix<double, 6, 6>::zero();
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                result[r][c] = rotation[r][c];
                result[3 + r][c] = coupling[r][c];
                result[3 + r][3 + c] = rotation[r][c];
            }
        }
        return result;
    }

    math::se3<double> odometry::relative(const math::se3<double>& first_pose, const math::se3<double>& second_pose, const math::se3<double>& body_from_odometry) {
        return body_from_odometry.inverse() * first_pose * second_pose.inverse() * body_from_odometry;
    }

    math::se3<double> odometry::lift(const math::matrix<double, 3, 1>& planar) {
        return math::se3<double>(math::so3<double>::exp({ { 0.0, 0.0, planar[2] } }), { { planar[0], planar[1], 0.0 } });
    }

    math::se3<double> odometry::predict(const math::se3<double>& first_pose, const math::se3<double>& body_from_odometry, const math::matrix<double, 3, 1>& planar) {
        return body_from_odometry * odometry::lift(planar).inverse() * body_from_odometry.inverse() * first_pose;
    }

    void odometry::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const math::se3<double> first = odometry::read_pose(context.get_vertex(static_cast<int>(odometry::first_pose))->get_parameters());
        const math::se3<double> second = odometry::read_pose(context.get_vertex(static_cast<int>(odometry::second_pose))->get_parameters());
        const math::se3<double> body_from_odometry = vertices::extrinsic::from_parameters(context.get_vertex(static_cast<int>(odometry::extrinsic))->get_parameters());
        const math::se3<double> motion = odometry::relative(first, second, body_from_odometry);
        const math::matrix<double, 3, 1> planar = odometry::measurement(context);
        const math::matrix<double, 3, 1> rotation_error = (math::so3<double>::exp({ { 0.0, 0.0, -planar[2] } }) * motion.rotation()).log();
        residual[0][0] = rotation_error[0];
        residual[1][0] = rotation_error[1];
        residual[2][0] = rotation_error[2];
        residual[3][0] = motion.translation()[0] - planar[0];
        residual[4][0] = motion.translation()[1] - planar[1];
        residual[5][0] = motion.translation()[2];
    }

    void odometry::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        const math::se3<double> first = odometry::read_pose(context.get_vertex(static_cast<int>(odometry::first_pose))->get_parameters());
        const math::se3<double> second = odometry::read_pose(context.get_vertex(static_cast<int>(odometry::second_pose))->get_parameters());
        const math::se3<double> body_from_odometry = vertices::extrinsic::from_parameters(context.get_vertex(static_cast<int>(odometry::extrinsic))->get_parameters());
        const math::se3<double> odometry_from_body = body_from_odometry.inverse();
        const math::se3<double> motion = odometry::relative(first, second, body_from_odometry);
        const math::matrix<double, 3, 1> planar = odometry::measurement(context);
        const math::matrix<double, 3, 1> rotation_error = (math::so3<double>::exp({ { 0.0, 0.0, -planar[2] } }) * motion.rotation()).log();

        // The residual by a perturbation exp(phi) * T_oo of the motion, phi = (omega, upsilon): the rotation error moves by
        // Jr^-1(error) * R_oo^T * omega, the translation by upsilon - [t_oo]x * omega.
        const math::matrix<double, 3, 3> rotation_block = math::so3<double>::left_jacobian_inverse(-rotation_error) * math::transpose(motion.rotation().get_matrix());
        const math::matrix<double, 3, 3> translation_block = -odometry::skew(motion.translation());
        math::matrix<double, 6, 6> residual_motion = math::matrix<double, 6, 6>::zero();
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                residual_motion[r][c] = rotation_block[r][c];
                residual_motion[3 + r][c] = translation_block[r][c];
            }
            residual_motion[3 + r][3 + r] = 1.0;
        }

        // The start perturbs the motion by Ad(T_ob), the end by -Ad(T_oo * T_ob). A step exp(epsilon) * T_bo of the extrinsic is
        // the step exp(-Ad(T_ob) * epsilon) * T_ob of its inverse, which perturbs the motion by (I - Ad(T_oo)).
        const math::matrix<double, 6, 6> jacobian_first = residual_motion * odometry::adjoint(odometry_from_body);
        const math::matrix<double, 6, 6> jacobian_second = -(residual_motion * odometry::adjoint(motion * odometry_from_body));
        const math::matrix<double, 6, 6> jacobian_extrinsic = -(residual_motion * (math::matrix<double, 6, 6>::identity() - odometry::adjoint(motion)) * odometry::adjoint(odometry_from_body));
        edge::set_jacobian(jacobians[odometry::first_pose], 6, 6, jacobian_first.data());
        edge::set_jacobian(jacobians[odometry::second_pose], 6, 6, jacobian_second.data());
        edge::set_jacobian(jacobians[odometry::extrinsic], 6, 6, jacobian_extrinsic.data());

        // The calibration moves the measurement: its heading turns the rotation error by -Jl^-1(error) * e_z, its position moves
        // the translation error back.
        const vertex* const calibration_vertex = context.get_vertex(static_cast<int>(odometry::calibration));
        const double* const values = calibration_vertex->get_parameters();
        const size_t columns = static_cast<size_t>(calibration_vertex->get_local_dimensions());
        const math::matrix<double, 0, 0>& observation = context.get_observation();
        const size_t parameters = (observation.rows() - 3) / 4;
        const math::matrix<double, 3, 3> heading_block = math::so3<double>::left_jacobian_inverse(rotation_error);
        double jacobian_calibration[6 * vertices::calibration::maximum_values] = {};
        for (size_t i = 0; (i < parameters) && (i < vertices::calibration::maximum_values); ++i) {
            const int local = vertices::calibration::local_index(values, i);
            if ((local < 0) || (static_cast<size_t>(local) >= columns)) {
                continue;
            }
            const size_t column = static_cast<size_t>(local);
            const double x_parameter = observation[3 + parameters + i][0];
            const double y_parameter = observation[3 + parameters + parameters + i][0];
            const double heading_parameter = observation[3 + parameters + (2 * parameters) + i][0];
            for (size_t r = 0; r < 3; ++r) {
                jacobian_calibration[(r * columns) + column] = -heading_block[r][2] * heading_parameter;
            }
            jacobian_calibration[(3 * columns) + column] = -x_parameter;
            jacobian_calibration[(4 * columns) + column] = -y_parameter;
        }
        edge::set_jacobian(jacobians[odometry::calibration], 6, columns, &jacobian_calibration[0]);
    }
}
