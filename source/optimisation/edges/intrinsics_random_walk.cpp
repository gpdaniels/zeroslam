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

#include "optimisation/edges/intrinsics_random_walk.hpp"

#include "optimisation/vertices/intrinsics.hpp"

namespace optimisation::edges {
    intrinsics_random_walk::intrinsics_random_walk(const int parameter_index_value)
        : parameter_index(parameter_index_value) {
    }

    math::matrix<double, 0, 0> intrinsics_random_walk::information(const double sigma, const double reference_duration, const double duration) {
        // The variance of a walk grows with the time walked.
        const double variance = sigma * sigma * (duration / reference_duration);
        return math::matrix<double, 0, 0>::identity(1, 1) * ((variance > 0.0) ? (1.0 / variance) : 0.0);
    }

    void intrinsics_random_walk::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const vertex* const first = context.get_vertex(0);
        const vertex* const second = context.get_vertex(1);
        const bool inside = (this->parameter_index >= 0) && (this->parameter_index < first->get_parameter_dimensions()) && (this->parameter_index < second->get_parameter_dimensions());
        residual[0][0] = inside ? (second->get_parameters()[this->parameter_index] - first->get_parameters()[this->parameter_index]) : 0.0;
    }

    void intrinsics_random_walk::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        const bool inside = (this->parameter_index >= 0) && (this->parameter_index < context.get_vertex(0)->get_parameter_dimensions()) && (this->parameter_index < context.get_vertex(1)->get_parameter_dimensions());
        for (int slot = 0; slot < 2; ++slot) {
            const vertex* const node = context.get_vertex(slot);
            double values[vertex::maximum_parameters] = {};
            const vertices::intrinsics* const calibration = node->get_policy<vertices::intrinsics>();
            const int local_index = (inside && (calibration != nullptr)) ? calibration->get_local_index(this->parameter_index) : -1;
            if (local_index >= 0) {
                values[local_index] = ((slot == 0) ? -1.0 : 1.0) * calibration->get_weight(this->parameter_index);
            }
            edge::set_jacobian(jacobians[static_cast<size_t>(slot)], 1, static_cast<size_t>(node->get_local_dimensions()), &values[0]);
        }
    }
}
