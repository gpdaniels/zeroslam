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
#ifndef ZEROSLAM_GEOMETRY_CHEIRALITY_HPP
#define ZEROSLAM_GEOMETRY_CHEIRALITY_HPP

#include "math/matrix.hpp"

namespace geometry {
    // Whether points lie in front of their cameras; poses are 3 by 4 world to camera transforms and rays are bearings of any direction.
    template <typename type>
    class cheirality final {
    public:
        // True when the point, transformed into the camera, lies along the ray rather than behind it.
        static bool in_front(
            const math::matrix<type, 3, 4>& pose,
            const math::matrix<type, 3, 1>& ray,
            const math::matrix<type, 3, 1>& point
        );

        // Triangulate the rays with the given triangulator and test both cameras; the point is written whenever the triangulation succeeds.
        template <typename triangulator>
        static bool triangulate(
            const math::matrix<type, 3, 1>& lhs_ray,
            const math::matrix<type, 3, 4>& lhs_pose,
            const math::matrix<type, 3, 1>& rhs_ray,
            const math::matrix<type, 3, 4>& rhs_pose,
            math::matrix<type, 3, 1>& point
        ) {
            if (!triangulator::triangulate(lhs_ray, lhs_pose, rhs_ray, rhs_pose, point)) {
                return false;
            }
            return cheirality::in_front(lhs_pose, lhs_ray, point) && cheirality::in_front(rhs_pose, rhs_ray, point);
        }
    };

    extern template class cheirality<float>;
    extern template class cheirality<double>;
}

#endif // ZEROSLAM_GEOMETRY_CHEIRALITY_HPP
