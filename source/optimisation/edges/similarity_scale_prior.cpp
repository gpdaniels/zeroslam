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

#include "optimisation/edges/similarity_scale_prior.hpp"

#include "math/math.hpp"

namespace optimisation::edges {
    similarity_scale_prior::similarity_scale_prior(const double target_scale_value)
        : target_scale(target_scale_value) {
    }

    void similarity_scale_prior::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const double scale = context.get_vertex(0)->get_parameters()[7];
        residual[0][0] = math::log(scale) - math::log(this->target_scale);
    }

    void similarity_scale_prior::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        static_cast<void>(context);
        // The similarity vertex steps by exp(delta) times itself, and the scale of exp(delta) is exp(sigma), the last of its
        // seven coordinates: the logarithm of the scale moves by sigma alone.
        const double jacobian[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
        edge::set_jacobian(jacobians[0], 1, 7, &jacobian[0]);
    }
}
