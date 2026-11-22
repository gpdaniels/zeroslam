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

#include "optimisation/edges/gravity_alignment.hpp"

#include "math/lie.hpp"
#include "math/math.hpp"

namespace optimisation::edges {
    gravity_alignment::gravity_alignment(const math::matrix<double, 3, 1>& gravity_direction, const bool inverse_rotation)
        : direction{ 0.0, 0.0, 0.0 }
        , inverse(inverse_rotation) {
        const double length = math::sqrt(gravity_direction.get_length_squared());
        for (size_t i = 0; i < 3; ++i) {
            this->direction[i] = (length > 0.0) ? (gravity_direction[i] / length) : 0.0;
        }
    }

    math::matrix<double, 3, 1> gravity_alignment::turned(const double* const parameters) const {
        const math::so3<double> rotation = math::so3<double>(parameters[6], parameters[3], parameters[4], parameters[5]).normalised();
        const math::matrix<double, 3, 1> unit = { { this->direction[0], this->direction[1], this->direction[2] } };
        return this->inverse ? (rotation.inverse() * unit) : (rotation * unit);
    }

    void gravity_alignment::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const math::matrix<double, 3, 1> now = this->turned(context.get_vertex(0)->get_parameters());
        for (size_t i = 0; i < 3; ++i) {
            residual[i][0] = now[i] - context.get_observation()[i][0];
        }
    }
}
