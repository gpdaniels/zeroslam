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

#pragma once
#ifndef ZEROSLAM_OPTIMISATION_VERTICES_GRAVITY_DIRECTION_HPP
#define ZEROSLAM_OPTIMISATION_VERTICES_GRAVITY_DIRECTION_HPP

#include "math/matrix.hpp"

namespace optimisation::vertices {
    // The direction of gravity in a world, as the rotation R_wg that takes a gravity aligned frame, whose gravity is (0, 0, -g),
    // into the world: gravity in the world is R_wg * (0, 0, -g). The parameters are the quaternion (x, y, z, w), and the update
    // R_wg * Exp((delta_x, delta_y, 0)) leaves out the turn about the gravity aligned frame's z axis, which does not move gravity.
    class gravity_direction final {
    public:
        constexpr static const char* name = "gravity_direction";
        constexpr static const int parameter_count = 4;
        constexpr static const int local_count = 2;

    public:
        void plus(double* const parameters, const double* const delta) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_VERTICES_GRAVITY_DIRECTION_HPP
