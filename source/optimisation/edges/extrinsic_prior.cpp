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

#include "optimisation/edges/extrinsic_prior.hpp"

#include "math/lie.hpp"
#include "optimisation/edges/odometry.hpp"

namespace optimisation::edges {
    math::matrix<double, 0, 0> extrinsic_prior::observation(const double* const pose_parameters) {
        return math::matrix<double, 0, 0>(7, 1, pose_parameters);
    }

    math::matrix<double, 0, 0> extrinsic_prior::information(const double* const standard_deviations) {
        math::matrix<double, 0, 0> result = math::matrix<double, 0, 0>::zero(6, 6);
        for (size_t i = 0; i < 6; ++i) {
            const double deviation = standard_deviations[i];
            result[i][i] = (deviation > 0.0) ? (1.0 / (deviation * deviation)) : 0.0;
        }
        return result;
    }

    void extrinsic_prior::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const math::se3<double> current = odometry::read_pose(context.get_vertex(0)->get_parameters());
        const math::se3<double> prior = odometry::read_pose(context.get_observation().data());
        const math::matrix<double, 3, 1> rotation_error = (current.rotation() * prior.rotation().inverse()).log();
        for (size_t i = 0; i < 3; ++i) {
            residual[i][0] = rotation_error[i];
            residual[3 + i][0] = current.translation()[i] - prior.translation()[i];
        }
    }

    void extrinsic_prior::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        const math::se3<double> current = odometry::read_pose(context.get_vertex(0)->get_parameters());
        const math::se3<double> prior = odometry::read_pose(context.get_observation().data());
        const math::matrix<double, 3, 1> rotation_error = (current.rotation() * prior.rotation().inverse()).log();

        // A perturbation exp(phi) * T moves the rotation error by Jl^-1(error) * omega and the translation by upsilon - [t]x * omega.
        const math::matrix<double, 3, 3> rotation_block = math::so3<double>::left_jacobian_inverse(rotation_error);
        const math::matrix<double, 3, 3> translation_block = -odometry::skew(current.translation());
        math::matrix<double, 6, 6> jacobian = math::matrix<double, 6, 6>::zero();
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                jacobian[r][c] = rotation_block[r][c];
                jacobian[3 + r][c] = translation_block[r][c];
            }
            jacobian[3 + r][3 + r] = 1.0;
        }
        edge::set_jacobian(jacobians[0], 6, 6, jacobian.data());
    }
}
