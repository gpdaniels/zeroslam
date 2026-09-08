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
#ifndef ZEROSLAM_GEOMETRY_TRIANGULATION_COST_FUNCTION_HPP
#define ZEROSLAM_GEOMETRY_TRIANGULATION_COST_FUNCTION_HPP

#include "math/matrix.hpp"

namespace geometry::triangulation {
    // Poses are 3 by 4 world to camera transforms [R | t]; rays are bearings in camera coordinates of any length or direction, normalised points are rays with unit depth.
    template <typename type>
    class cost_function final {
    public:
        constexpr static const type determinant_tolerance = type(1e-6);

    public:
        static bool triangulate(
            const math::matrix<type, 3, 1>& lhs_ray,
            const math::matrix<type, 3, 4>& lhs_pose,
            const math::matrix<type, 3, 1>& rhs_ray,
            const math::matrix<type, 3, 4>& rhs_pose,
            math::matrix<type, 3, 1>& result
        );

        static bool triangulate(
            const math::matrix<type, 2, 1>& lhs_point_normalised,
            const math::matrix<type, 3, 4>& lhs_pose,
            const math::matrix<type, 2, 1>& rhs_point_normalised,
            const math::matrix<type, 3, 4>& rhs_pose,
            math::matrix<type, 3, 1>& result
        );
    };

    extern template class cost_function<float>;
    extern template class cost_function<double>;
}

#endif // ZEROSLAM_GEOMETRY_TRIANGULATION_COST_FUNCTION_HPP
