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

#include "optimisation/vertices/line.hpp"

#include "geometry/plucker.hpp"

namespace optimisation::vertices {
    void line::plus(double* const parameters, const double* const delta) const {
        geometry::plucker current(
            math::matrix<double, 3, 1>({ parameters[0], parameters[1], parameters[2] }),
            math::matrix<double, 3, 1>({ parameters[3], parameters[4], parameters[5] })
        );
        if (!current.oplus(delta[0], delta[1], delta[2], delta[3])) {
            return;
        }
        parameters[0] = current.moment[0];
        parameters[1] = current.moment[1];
        parameters[2] = current.moment[2];
        parameters[3] = current.direction[0];
        parameters[4] = current.direction[1];
        parameters[5] = current.direction[2];
    }
}
