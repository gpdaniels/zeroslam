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

#include "optimisation/edges/gyroscope_intrinsics_prior.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace optimisation::edges {
    math::matrix<double, 0, 0> gyroscope_intrinsics_prior::information(const double deviation) {
        return math::matrix<double, 0, 0>::identity(9, 9) * (1.0 / (deviation * deviation));
    }

    void gyroscope_intrinsics_prior::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const double* const parameters = context.get_vertex(0)->get_parameters();
        for (size_t i = 0; i < 9; ++i) {
            residual[i][0] = parameters[i];
        }
    }

    void gyroscope_intrinsics_prior::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        static_cast<void>(context);
        double values[81] = {};
        for (size_t i = 0; i < 9; ++i) {
            values[(i * 9) + i] = 1.0;
        }
        edge::set_jacobian(jacobians[0], 9, 9, &values[0]);
    }
}
