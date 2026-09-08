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
#ifndef ZEROSLAM_GEOMETRY_SAMPSON_HPP
#define ZEROSLAM_GEOMETRY_SAMPSON_HPP

#include "math/matrix.hpp"

namespace geometry {
    // First order approximation of the squared geometric distance of a correspondence from the epipolar constraint rhs^T M lhs = 0, for an essential or fundamental matrix M.
    template <typename type>
    class sampson final {
    public:
        static type distance_squared(
            const math::matrix<type, 3, 3>& epipolar,
            const math::matrix<type, 2, 1>& lhs,
            const math::matrix<type, 2, 1>& rhs
        );
    };

    extern template class sampson<float>;
    extern template class sampson<double>;
}

#endif // ZEROSLAM_GEOMETRY_SAMPSON_HPP
