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

#include "optimisation/losses/tukey.hpp"

#include "math/math.hpp"

namespace optimisation::losses {
    tukey::tukey(double delta_value)
        : delta(delta_value) {
    }

    void tukey::compute(const double error_squared, math::matrix<double, 3, 1>& rho) const {
        const double delta_squared = this->delta * this->delta;
        if (error_squared <= delta_squared) {
            const double aux = error_squared / delta_squared;
            const double one_minus_aux = 1.0 - aux;
            const double one_minus_aux_squared = one_minus_aux * one_minus_aux;
            const double one_minus_aux_cubed = one_minus_aux_squared * one_minus_aux;
            rho[0] = delta_squared * (1.0 - one_minus_aux_cubed) / 3.0;
            rho[1] = one_minus_aux_squared;
            rho[2] = -2.0 * one_minus_aux / delta_squared;
        }
        else {
            rho[0] = delta_squared / 3.0;
            rho[1] = 0;
            rho[2] = 0;
        }
    }
}
