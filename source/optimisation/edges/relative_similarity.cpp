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

#include "optimisation/edges/relative_similarity.hpp"

#include "math/lie.hpp"

namespace optimisation::edges {
    void relative_similarity::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const double* const first_parameters = context.get_vertex(0)->get_parameters();
        const math::sim3<double> first(math::se3<double>(math::so3<double>(first_parameters[6], first_parameters[3], first_parameters[4], first_parameters[5]), { { first_parameters[0], first_parameters[1], first_parameters[2] } }), first_parameters[7]);
        const double* const second_parameters = context.get_vertex(1)->get_parameters();
        const math::sim3<double> second(math::se3<double>(math::so3<double>(second_parameters[6], second_parameters[3], second_parameters[4], second_parameters[5]), { { second_parameters[0], second_parameters[1], second_parameters[2] } }), second_parameters[7]);
        const math::matrix<double, 0, 0>& observation = context.get_observation();
        const math::sim3<double> measurement(math::se3<double>(math::so3<double>(observation[6][0], observation[3][0], observation[4][0], observation[5][0]), { { observation[0][0], observation[1][0], observation[2][0] } }), observation[7][0]);
        const math::matrix<double, 7, 1> error = (measurement.inverse() * (first.inverse() * second)).log();
        for (size_t i = 0; i < 7; ++i) {
            residual[i][0] = error[i];
        }
    }

    void relative_similarity::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        const double* const first_parameters = context.get_vertex(0)->get_parameters();
        const math::sim3<double> first(math::se3<double>(math::so3<double>(first_parameters[6], first_parameters[3], first_parameters[4], first_parameters[5]), { { first_parameters[0], first_parameters[1], first_parameters[2] } }), first_parameters[7]);
        const double* const second_parameters = context.get_vertex(1)->get_parameters();
        const math::sim3<double> second(math::se3<double>(math::so3<double>(second_parameters[6], second_parameters[3], second_parameters[4], second_parameters[5]), { { second_parameters[0], second_parameters[1], second_parameters[2] } }), second_parameters[7]);
        const math::matrix<double, 0, 0>& observation = context.get_observation();
        const math::sim3<double> measurement(math::se3<double>(math::so3<double>(observation[6][0], observation[3][0], observation[4][0], observation[5][0]), { { observation[0][0], observation[1][0], observation[2][0] } }), observation[7][0]);

        const math::matrix<double, 7, 1> error = (measurement.inverse() * (first.inverse() * second)).log();
        const math::matrix<double, 7, 7> left_jacobian_inverse = math::sim3<double>::left_jacobian_inverse(error);
        const math::matrix<double, 7, 7> adjoint = math::sim3<double>::adjoint(measurement.inverse() * first.inverse());

        const math::matrix<double, 7, 7> jacobian_first = -(left_jacobian_inverse * adjoint);
        const math::matrix<double, 7, 7> jacobian_second = +(left_jacobian_inverse * adjoint);
        jacobians[0] = math::matrix<double, 0, 0>(7, 7, jacobian_first.data());
        jacobians[1] = math::matrix<double, 0, 0>(7, 7, jacobian_second.data());
    }
}
