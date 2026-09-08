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
#ifndef ZEROSLAM_GEOMETRY_ESSENTIAL_HPP
#define ZEROSLAM_GEOMETRY_ESSENTIAL_HPP

#include "math/matrix.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace geometry {
    // Essential matrix convention: for normalised coordinates x in camera 1 (lhs) and y in camera 2 (rhs), y^T E x = 0 with E = [t]x R.
    template <typename type>
    class essential final {
    public:
        // From the pose of camera 2 relative to camera 1.
        static void from_poses(
            const math::matrix<type, 3, 3>& rotation,
            const math::matrix<type, 3, 1>& translation,
            math::matrix<type, 3, 3>& essential
        );

        static void from_fundamental(
            const math::matrix<type, 3, 3>& intrinsics_lhs,
            const math::matrix<type, 3, 3>& intrinsics_rhs,
            const math::matrix<type, 3, 3>& fundamental,
            math::matrix<type, 3, 3>& essential
        );

        // The four relative poses of camera 2 consistent with the matrix, translations of unit length; returns how many were written.
        static size_t decompose(
            const math::matrix<type, 3, 3>& essential,
            math::matrix<type, 3, 3>* const rotations,
            math::matrix<type, 3, 1>* const translations
        );

        // The epipole in camera 1 (E e = 0) in normalised coordinates, homogeneous and not normalised; false if E is rank deficient.
        static bool epipole_lhs(
            const math::matrix<type, 3, 3>& essential,
            type& epipole_x,
            type& epipole_y,
            type& epipole_z
        );

        // The epipole in camera 2 (E^T e = 0) in normalised coordinates, homogeneous and not normalised; false if E is rank deficient.
        static bool epipole_rhs(
            const math::matrix<type, 3, 3>& essential,
            type& epipole_x,
            type& epipole_y,
            type& epipole_z
        );

    private:
        static bool singular_vectors(
            const math::matrix<type, 3, 3>& essential,
            math::matrix<type, 3, 3>& matrix_u,
            math::matrix<type, 3, 3>& matrix_vt
        );
    };

    extern template class essential<float>;
    extern template class essential<double>;
}

#endif // ZEROSLAM_GEOMETRY_ESSENTIAL_HPP
