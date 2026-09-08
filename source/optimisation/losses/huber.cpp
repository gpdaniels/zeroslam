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

#include "optimisation/losses/huber.hpp"

#include "math/math.hpp"

namespace optimisation::losses {
    huber::huber(double delta_value)
        : delta(delta_value) {
    }

    void huber::compute(const double error_squared, math::matrix<double, 3, 1>& rho) const {
        const double delta_squared = this->delta * this->delta;
        if (error_squared <= delta_squared) {
            rho[0] = error_squared;
            rho[1] = 1.0;
            rho[2] = 0.0;
        }
        else {
            const double error = math::sqrt(error_squared);
            rho[0] = 2.0 * error * this->delta - delta_squared;
            rho[1] = this->delta / error;
            rho[2] = -0.5 * rho[1] / error_squared;
        }
    }
}
