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

#include "optimisation/edges/intrinsics_ratio_prior.hpp"

#include "optimisation/vertices/intrinsics.hpp"

namespace optimisation::edges {
    intrinsics_ratio_prior::intrinsics_ratio_prior(const int first_index_value, const int second_index_value, const double ratio_value)
        : first_index(first_index_value)
        , second_index(second_index_value)
        , ratio(ratio_value) {
    }

    void intrinsics_ratio_prior::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const vertex* const node = context.get_vertex(0);
        const int count = node->get_parameter_dimensions();
        const bool inside = (this->first_index >= 0) && (this->first_index < count) && (this->second_index >= 0) && (this->second_index < count);
        residual[0][0] = inside ? (node->get_parameters()[this->first_index] - (this->ratio * node->get_parameters()[this->second_index])) : 0.0;
    }

    void intrinsics_ratio_prior::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        const vertex* const node = context.get_vertex(0);
        const int count = node->get_parameter_dimensions();
        const bool inside = (this->first_index >= 0) && (this->first_index < count) && (this->second_index >= 0) && (this->second_index < count);
        double values[vertex::maximum_parameters] = {};
        const vertices::intrinsics* const calibration = node->get_policy<vertices::intrinsics>();
        if (inside && (calibration != nullptr)) {
            // The two may share a local dimension, whose step then moves both.
            const int first_local = calibration->get_local_index(this->first_index);
            const int second_local = calibration->get_local_index(this->second_index);
            if (first_local >= 0) {
                values[first_local] += calibration->get_weight(this->first_index);
            }
            if (second_local >= 0) {
                values[second_local] -= this->ratio * calibration->get_weight(this->second_index);
            }
        }
        edge::set_jacobian(jacobians[0], 1, static_cast<size_t>(node->get_local_dimensions()), &values[0]);
    }
}
