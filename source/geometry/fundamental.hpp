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
#ifndef ZEROSLAM_GEOMETRY_FUNDAMENTAL_HPP
#define ZEROSLAM_GEOMETRY_FUNDAMENTAL_HPP

#include "math/math.hpp"
#include "math/matrix.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace geometry {
    // Fundamental matrix convention: for a pixel x in image 1 (lhs) and its match y in image 2 (rhs), y^T F x = 0 and F x is the epipolar line in image 2.
    template <typename type>
    class fundamental final {
    public:
        constexpr static const type rank_tolerance = math::epsilon<type>() * type(100);

    public:
        // From the intrinsics of both cameras and the pose of camera 2 relative to camera 1.
        static bool from_poses(
            const math::matrix<type, 3, 3>& intrinsics_lhs,
            const math::matrix<type, 3, 3>& intrinsics_rhs,
            const math::matrix<type, 3, 3>& rotation,
            const math::matrix<type, 3, 1>& translation,
            math::matrix<type, 3, 3>& fundamental
        );

        static bool from_essential(
            const math::matrix<type, 3, 3>& intrinsics_lhs,
            const math::matrix<type, 3, 3>& intrinsics_rhs,
            const math::matrix<type, 3, 3>& essential,
            math::matrix<type, 3, 3>& fundamental
        );

        // The four relative poses of camera 2 consistent with the matrix, translations of unit length; returns how many were written.
        static size_t decompose(
            const math::matrix<type, 3, 3>& intrinsics_lhs,
            const math::matrix<type, 3, 3>& intrinsics_rhs,
            const math::matrix<type, 3, 3>& fundamental,
            math::matrix<type, 3, 3>* const rotations,
            math::matrix<type, 3, 1>* const translations
        );

        // The epipole in image 1 (F e = 0), homogeneous and not normalised; false if F is rank deficient.
        static bool epipole_lhs(
            const math::matrix<type, 3, 3>& fundamental,
            type& epipole_x,
            type& epipole_y,
            type& epipole_z
        );

        // The epipole in image 2 (F^T e = 0), homogeneous and not normalised; false if F is rank deficient.
        static bool epipole_rhs(
            const math::matrix<type, 3, 3>& fundamental,
            type& epipole_x,
            type& epipole_y,
            type& epipole_z
        );

    private:
        static bool null_vector(
            const math::matrix<type, 3, 3>& matrix,
            type& x,
            type& y,
            type& z
        );
    };

    extern template class fundamental<float>;
    extern template class fundamental<double>;
}

#endif // ZEROSLAM_GEOMETRY_FUNDAMENTAL_HPP
