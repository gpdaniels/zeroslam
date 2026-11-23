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

#include "optimisation/edges/calibration_prior.hpp"

#include "optimisation/vertices/calibration.hpp"

namespace optimisation::edges {
    static_assert(static_cast<size_t>(calibration_prior::residual_count) == vertices::calibration::maximum_values, "A calibration prior has a residual for each value of a calibration vertex.");

    math::matrix<double, 0, 0> calibration_prior::observation(const double* const values, const size_t count) {
        math::matrix<double, 0, 0> result = math::matrix<double, 0, 0>::zero(vertices::calibration::maximum_values, 1);
        for (size_t i = 0; (i < count) && (i < vertices::calibration::maximum_values); ++i) {
            result[i][0] = values[i];
        }
        return result;
    }

    math::matrix<double, 0, 0> calibration_prior::information(const double* const standard_deviations, const size_t count) {
        math::matrix<double, 0, 0> result = math::matrix<double, 0, 0>::zero(vertices::calibration::maximum_values, vertices::calibration::maximum_values);
        for (size_t i = 0; (i < count) && (i < vertices::calibration::maximum_values); ++i) {
            const double deviation = standard_deviations[i];
            result[i][i] = (deviation > 0.0) ? (1.0 / (deviation * deviation)) : 0.0;
        }
        return result;
    }

    void calibration_prior::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const double* const parameters = context.get_vertex(0)->get_parameters();
        const math::matrix<double, 0, 0>& prior = context.get_observation();
        for (size_t i = 0; i < vertices::calibration::maximum_values; ++i) {
            const bool is_free = vertices::calibration::local_index(parameters, i) >= 0;
            residual[i][0] = (is_free && (i < prior.rows())) ? (parameters[i] - prior[i][0]) : 0.0;
        }
    }

    void calibration_prior::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        const vertex* const node = context.get_vertex(0);
        const double* const parameters = node->get_parameters();
        const size_t columns = static_cast<size_t>(node->get_local_dimensions());
        const size_t rows = vertices::calibration::maximum_values;
        double jacobian[vertices::calibration::maximum_values * vertices::calibration::maximum_values] = {};
        for (size_t i = 0; i < rows; ++i) {
            const int local = vertices::calibration::local_index(parameters, i);
            if ((local >= 0) && (static_cast<size_t>(local) < columns) && (i < context.get_observation().rows())) {
                jacobian[(i * columns) + static_cast<size_t>(local)] = 1.0;
            }
        }
        edge::set_jacobian(jacobians[0], rows, columns, &jacobian[0]);
    }
}
