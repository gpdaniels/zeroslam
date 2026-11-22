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
#ifndef ZEROSLAM_OPTIMISATION_VERTICES_VELOCITY_HPP
#define ZEROSLAM_OPTIMISATION_VERTICES_VELOCITY_HPP

#include "math/matrix.hpp"

namespace optimisation::vertices {
    // A velocity, or any other vector of three that is updated by adding to it, such as a linear velocity in the world.
    class velocity final {
    public:
        constexpr static const char* name = "velocity";
        constexpr static const int parameter_count = 3;
        constexpr static const int local_count = 3;

    public:
        void plus(double* const parameters, const double* const delta) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_VERTICES_VELOCITY_HPP
