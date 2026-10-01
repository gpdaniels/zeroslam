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
#ifndef ZEROSLAM_GEOMETRY_TRIANGULATION_DIRECT_HPP
#define ZEROSLAM_GEOMETRY_TRIANGULATION_DIRECT_HPP

#include "math/matrix.hpp"

namespace geometry::triangulation {
    // Poses are 3 by 4 world to camera transforms [R | t]; rays are bearings in camera coordinates of any length or sign, normalised points are rays with unit depth.
    // The depth comes from the x and y rows of rhs_ray x (R lhs_ray depth + t) = 0 alone, so it fails when the rhs ray and the rotated lhs ray both lie in the rhs camera's xy-plane; linear_least_squares takes rays of any direction.
    template <typename type>
    class direct final {
    public:
        constexpr static const type tolerance = type(1e-8);

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

    extern template class direct<float>;
    extern template class direct<double>;
}

#endif // ZEROSLAM_GEOMETRY_TRIANGULATION_DIRECT_HPP
