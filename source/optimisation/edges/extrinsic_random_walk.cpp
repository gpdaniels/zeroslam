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

#include "optimisation/edges/extrinsic_random_walk.hpp"

#include "math/lie.hpp"
#include "optimisation/vertices/extrinsic.hpp"

namespace optimisation::edges {
    math::matrix<double, 0, 0> extrinsic_random_walk::information(const double rotation_sigma, const double translation_sigma, const double reference_duration, const double duration) {
        // The variance of a walk grows with the time walked.
        const double rotation_variance = rotation_sigma * rotation_sigma * (duration / reference_duration);
        const double translation_variance = translation_sigma * translation_sigma * (duration / reference_duration);
        math::matrix<double, 0, 0> result = math::matrix<double, 0, 0>::zero(6, 6);
        for (size_t i = 0; i < 3; ++i) {
            result[i][i] = (rotation_variance > 0.0) ? (1.0 / rotation_variance) : 0.0;
            result[3 + i][3 + i] = (translation_variance > 0.0) ? (1.0 / translation_variance) : 0.0;
        }
        return result;
    }

    void extrinsic_random_walk::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const math::se3<double> first = vertices::extrinsic::from_parameters(context.get_vertex(0)->get_parameters());
        const math::se3<double> second = vertices::extrinsic::from_parameters(context.get_vertex(1)->get_parameters());
        const math::matrix<double, 3, 1> turn = (first.rotation().inverse() * second.rotation()).log();
        for (size_t i = 0; i < 3; ++i) {
            residual[i][0] = turn[i];
            residual[3 + i][0] = second.translation()[i] - first.translation()[i];
        }
    }

    void extrinsic_random_walk::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        const math::se3<double> first = vertices::extrinsic::from_parameters(context.get_vertex(0)->get_parameters());
        const math::se3<double> second = vertices::extrinsic::from_parameters(context.get_vertex(1)->get_parameters());
        const math::matrix<double, 3, 1> turn = (first.rotation().inverse() * second.rotation()).log();
        // A step exp([omega upsilon]) on the left of an extrinsic turns it by omega in the body's frame and moves its
        // translation by omega x t + upsilon. In R_i^T * R_j a turn of the first is -R_i^T * omega on the left, and a turn
        // of the second is R_j^T * omega on the right, where the logarithm's right Jacobian is its left one transposed.
        const math::matrix<double, 3, 3> left_inverse = math::so3<double>::left_jacobian_inverse(turn);
        const math::matrix<double, 3, 3> turn_first = -(left_inverse * math::transpose(first.rotation().get_matrix()));
        const math::matrix<double, 3, 3> turn_second = math::transpose(left_inverse) * math::transpose(second.rotation().get_matrix());
        const math::se3<double>* const extrinsics[2] = { &first, &second };
        const math::matrix<double, 3, 3>* const turns[2] = { &turn_first, &turn_second };
        for (size_t slot = 0; slot < 2; ++slot) {
            const double sign = (slot == 0) ? -1.0 : 1.0;
            const math::matrix<double, 3, 1>& t = extrinsics[slot]->translation();
            // The derivative of omega x t by omega is -[t]x.
            const double skew[3][3] = { { 0.0, -t[2], t[1] }, { t[2], 0.0, -t[0] }, { -t[1], t[0], 0.0 } };
            math::matrix<double, 6, 6> jacobian = math::matrix<double, 6, 6>::zero();
            for (size_t r = 0; r < 3; ++r) {
                for (size_t c = 0; c < 3; ++c) {
                    jacobian[r][c] = (*turns[slot])[r][c];
                    jacobian[3 + r][c] = -sign * skew[r][c];
                }
                jacobian[3 + r][3 + r] = sign;
            }
            edge::set_jacobian(jacobians[slot], 6, 6, jacobian.data());
        }
    }
}
