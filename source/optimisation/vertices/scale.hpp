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
#ifndef ZEROSLAM_OPTIMISATION_VERTICES_SCALE_HPP
#define ZEROSLAM_OPTIMISATION_VERTICES_SCALE_HPP

#include "math/matrix.hpp"

namespace optimisation::vertices {
    // A positive scale, such as the metric scale of a monocular map, updated as scale * exp(delta) so that it stays positive and a
    // step is relative to it.
    class scale final {
    public:
        constexpr static const char* name = "scale";
        constexpr static const int parameter_count = 1;
        constexpr static const int local_count = 1;

    public:
        void plus(double* const parameters, const double* const delta) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_VERTICES_SCALE_HPP
