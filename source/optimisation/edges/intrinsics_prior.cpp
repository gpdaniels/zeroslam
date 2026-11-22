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

#include "optimisation/edges/intrinsics_prior.hpp"

#include "optimisation/vertices/intrinsics.hpp"

namespace optimisation::edges {
    intrinsics_prior::intrinsics_prior(const int parameter_index_value, const double prior_value_value)
        : parameter_index(parameter_index_value)
        , prior_value(prior_value_value) {
    }

    void intrinsics_prior::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const vertex* const node = context.get_vertex(0);
        const bool inside = (this->parameter_index >= 0) && (this->parameter_index < node->get_parameter_dimensions());
        residual[0][0] = inside ? (this->prior_value - node->get_parameters()[this->parameter_index]) : 0.0;
    }

    void intrinsics_prior::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        const vertex* const node = context.get_vertex(0);
        const size_t columns = static_cast<size_t>(node->get_local_dimensions());
        double values[vertex::maximum_parameters] = {};
        const vertices::intrinsics* const calibration = node->get_policy<vertices::intrinsics>();
        const int local_index = (calibration != nullptr) ? calibration->get_local_index(this->parameter_index) : -1;
        if (local_index >= 0) {
            values[local_index] = -calibration->get_weight(this->parameter_index);
        }
        edge::set_jacobian(jacobians[0], 1, columns, &values[0]);
    }
}
