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

#include "optimisation/edges/baseline.hpp"

#include "math/lie.hpp"
#include "math/math.hpp"

namespace optimisation::edges {
    baseline::baseline(const double target_distance_value)
        : target_distance(target_distance_value) {
    }

    math::matrix<double, 3, 1> baseline::centre(const double* const parameters) {
        const math::so3<double> rotation(parameters[6], parameters[3], parameters[4], parameters[5]);
        const math::matrix<double, 3, 1> translation = { { parameters[0], parameters[1], parameters[2] } };
        return -(math::transpose(rotation.get_matrix()) * translation);
    }

    void baseline::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const math::matrix<double, 3, 1> difference = baseline::centre(context.get_vertex(0)->get_parameters()) - baseline::centre(context.get_vertex(1)->get_parameters());
        residual[0][0] = math::sqrt(difference.get_length_squared()) - this->target_distance;
    }

    void baseline::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        const double* const parameters_a = context.get_vertex(0)->get_parameters();
        const double* const parameters_b = context.get_vertex(1)->get_parameters();
        const math::matrix<double, 3, 1> difference = baseline::centre(parameters_a) - baseline::centre(parameters_b);
        const double distance = math::sqrt(difference.get_length_squared());
        math::matrix<double, 1, 6> jacobian_a = math::matrix<double, 1, 6>::zero();
        math::matrix<double, 1, 6> jacobian_b = math::matrix<double, 1, 6>::zero();
        if (distance > baseline::minimum_distance) {
            const math::matrix<double, 3, 1> unit = difference * (1.0 / distance);
            const math::matrix<double, 3, 1> rotated_a = math::so3<double>(parameters_a[6], parameters_a[3], parameters_a[4], parameters_a[5]).get_matrix() * unit;
            const math::matrix<double, 3, 1> rotated_b = math::so3<double>(parameters_b[6], parameters_b[3], parameters_b[4], parameters_b[5]).get_matrix() * unit;
            for (size_t axis = 0; axis < 3; ++axis) {
                jacobian_a[3 + axis] = -rotated_a[axis];
                jacobian_b[3 + axis] = rotated_b[axis];
            }
        }
        jacobians[0] = math::matrix<double, 0, 0>(1, 6, jacobian_a.data());
        jacobians[1] = math::matrix<double, 0, 0>(1, 6, jacobian_b.data());
    }
}
