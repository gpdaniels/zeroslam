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
#ifndef ZEROSLAM_GEOMETRY_TRIANGULATION_ITERATIVELY_REWEIGHTED_LINEAR_LEAST_SQUARES_HPP
#define ZEROSLAM_GEOMETRY_TRIANGULATION_ITERATIVELY_REWEIGHTED_LINEAR_LEAST_SQUARES_HPP

#include "math/matrix.hpp"

namespace geometry::triangulation {
    // Poses are 3 by 4 world to camera transforms [R | t]; rays are bearings in camera coordinates of any length or direction, normalised points are rays with unit depth.
    template <typename type>
    class iteratively_reweighted_linear_least_squares final {
    public:
        constexpr static const type tolerance = type(1e-8);
        constexpr static const unsigned int maximum_iterations = 10;
        constexpr static const type minimum_weight_change = type(1e-4);

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

    private:
        static bool rows(
            const math::matrix<type, 3, 1>& ray,
            const math::matrix<type, 3, 4>& pose,
            type* const row_a,
            type* const row_b
        );
    };

    extern template class iteratively_reweighted_linear_least_squares<float>;
    extern template class iteratively_reweighted_linear_least_squares<double>;
}

#endif // ZEROSLAM_GEOMETRY_TRIANGULATION_ITERATIVELY_REWEIGHTED_LINEAR_LEAST_SQUARES_HPP
