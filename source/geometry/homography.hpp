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
#ifndef ZEROSLAM_GEOMETRY_HOMOGRAPHY_HPP
#define ZEROSLAM_GEOMETRY_HOMOGRAPHY_HPP

#include "math/math.hpp"
#include "math/matrix.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace geometry {
    // Homography convention: maps pixels of image 1 (lhs) to pixels of image 2 (rhs), y ~ H x, for a plane n^T X = d in camera 1 coordinates.
    template <typename type>
    class homography final {
    public:
        constexpr static const type depth_tolerance = math::epsilon<type>() * type(100);
        constexpr static const type rotation_tolerance = math::epsilon<type>() * type(1e4);

    public:
        // From the intrinsics, the pose of camera 2 relative to camera 1, and the plane.
        static bool from_plane(
            const math::matrix<type, 3, 3>& intrinsics_lhs,
            const math::matrix<type, 3, 3>& intrinsics_rhs,
            const math::matrix<type, 3, 3>& rotation,
            const math::matrix<type, 3, 1>& translation,
            const math::matrix<type, 3, 1>& plane_normal,
            const type plane_distance,
            math::matrix<type, 3, 3>& homography
        );

        // The relative poses and unit plane normals consistent with the matrix, translations divided by the plane distance; returns how many were written (at most four).
        static size_t decompose(
            const math::matrix<type, 3, 3>& intrinsics_lhs,
            const math::matrix<type, 3, 3>& intrinsics_rhs,
            const math::matrix<type, 3, 3>& homography,
            math::matrix<type, 3, 3>* const rotations,
            math::matrix<type, 3, 1>* const translations,
            math::matrix<type, 3, 1>* const normals
        );

        // Apply the homography to a point; false if the mapped point lies at infinity.
        static bool transform(
            const math::matrix<type, 3, 3>& homography,
            const type x,
            const type y,
            type& transformed_x,
            type& transformed_y
        );
    };

    extern template class homography<float>;
    extern template class homography<double>;
}

#endif // ZEROSLAM_GEOMETRY_HOMOGRAPHY_HPP
