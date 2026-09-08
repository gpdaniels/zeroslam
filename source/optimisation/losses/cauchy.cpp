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

#include "optimisation/losses/cauchy.hpp"

#include "math/math.hpp"

namespace optimisation::losses {
    cauchy::cauchy(double delta_value)
        : delta(delta_value) {
    }

    void cauchy::compute(const double error_squared, math::matrix<double, 3, 1>& rho) const {
        const double delta_squared = this->delta * this->delta;
        const double delta_squared_reciprocal = 1.0 / delta_squared;
        const double aux = delta_squared_reciprocal * error_squared + 1.0;
        rho[0] = delta_squared * math::log(aux);
        rho[1] = 1.0 / aux;
        rho[2] = -delta_squared_reciprocal * math::sqr(rho[1]);
    }
}
