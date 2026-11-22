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

#include "optimisation/edges/prior.hpp"

#include "math/lie.hpp"

namespace {
    math::matrix<double, 6, 1> pose_error(const optimisation::edge& context) {
        const double* const parameters = context.get_vertex(0)->get_parameters();
        const math::se3<double> pose(math::so3<double>(parameters[6], parameters[3], parameters[4], parameters[5]), { { parameters[0], parameters[1], parameters[2] } });
        const math::matrix<double, 0, 0>& observation = context.get_observation();
        const math::se3<double> prior(math::so3<double>(observation[6][0], observation[3][0], observation[4][0], observation[5][0]), { { observation[0][0], observation[1][0], observation[2][0] } });
        math::se3<double> difference = pose * prior.inverse();
        difference.rotation() = difference.rotation().normalised();
        return difference.log();
    }

    template <size_t dimensions>
    void identity_jacobian(math::matrix<double, 0, 0>& jacobian) {
        double values[dimensions * dimensions] = {};
        for (size_t i = 0; i < dimensions; ++i) {
            values[(i * dimensions) + i] = 1.0;
        }
        optimisation::edge::set_jacobian(jacobian, dimensions, dimensions, &values[0]);
    }
}

namespace optimisation::edges {
    void pose_prior::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const math::matrix<double, 6, 1> error = pose_error(context);
        for (size_t i = 0; i < 6; ++i) {
            residual[i][0] = error[i];
        }
    }

    void pose_prior::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        // Log(exp(delta) * T * T_prior^-1) = Log(exp(delta) * exp(r)), which moves by J_l(r)^-1 delta.
        const math::matrix<double, 6, 6> jacobian = math::se3<double>::left_jacobian_inverse(pose_error(context));
        edge::set_jacobian(jacobians[0], 6, 6, jacobian.data());
    }

    void velocity_prior::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const double* const parameters = context.get_vertex(0)->get_parameters();
        for (size_t i = 0; i < 3; ++i) {
            residual[i][0] = parameters[i] - context.get_observation()[i][0];
        }
    }

    void velocity_prior::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        static_cast<void>(context);
        identity_jacobian<3>(jacobians[0]);
    }

    math::matrix<double, 0, 0> imu_bias_prior::information(const double gyroscope_deviation, const double accelerometer_deviation) {
        math::matrix<double, 0, 0> result = math::matrix<double, 0, 0>::zero(6, 6);
        for (size_t i = 0; i < 3; ++i) {
            result[i][i] = 1.0 / (gyroscope_deviation * gyroscope_deviation);
            result[3 + i][3 + i] = 1.0 / (accelerometer_deviation * accelerometer_deviation);
        }
        return result;
    }

    void imu_bias_prior::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const double* const parameters = context.get_vertex(0)->get_parameters();
        for (size_t i = 0; i < 6; ++i) {
            residual[i][0] = parameters[i] - context.get_observation()[i][0];
        }
    }

    void imu_bias_prior::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        static_cast<void>(context);
        identity_jacobian<6>(jacobians[0]);
    }
}
