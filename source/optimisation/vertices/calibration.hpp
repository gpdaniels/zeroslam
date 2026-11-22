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
#ifndef ZEROSLAM_OPTIMISATION_VERTICES_CALIBRATION_HPP
#define ZEROSLAM_OPTIMISATION_VERTICES_CALIBRATION_HPP

#include "optimisation/vertex.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace optimisation::vertices {
    // A vector of up to eight calibration parameters, such as a wheel odometer's radii and steering gain, of which only those
    // in a mask are free. The parameters are the values followed by the mask (bit i for value i, as a double), so that the edges,
    // which see only a vertex's parameters, read which values move and in which order they take the local dimensions: one local
    // dimension per free value, in the order of the values, so that the solver sees a block of exactly the free values and no
    // singular dimensions for the held ones. A vertex with no free value is made fixed.
    class calibration final {
    public:
        constexpr static const char* name = "calibration";
        constexpr static const size_t maximum_values = 8;
        constexpr static const size_t mask_index = maximum_values;
        constexpr static const int parameter_count = static_cast<int>(maximum_values) + 1;

    public:
        // The vertex of the given values (at most maximum_values, the rest zero) with the values in the mask free.
        static vertex make(const double* const values, const size_t count, const unsigned int free_mask);

        // The mask of a calibration vertex's parameters.
        static unsigned int mask(const double* const parameters);

        // The local dimension of a value of a calibration vertex's parameters, or -1 when the value is held.
        static int local_index(const double* const parameters, const size_t value);
    };
}

#endif // ZEROSLAM_OPTIMISATION_VERTICES_CALIBRATION_HPP
