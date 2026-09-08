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

#include "optimisation/losses/trivial.hpp"

#include "math/math.hpp"

namespace optimisation::losses {
    void trivial::compute(const double error_squared, math::matrix<double, 3, 1>& rho) const {
        rho[0] = error_squared;
        rho[1] = 1;
        rho[2] = 0;
    }
}
