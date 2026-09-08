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

#include "geometry/fundamental.hpp"

#include "geometry/essential.hpp"

namespace geometry {
    template <typename type>
    bool fundamental<type>::from_poses(
        const math::matrix<type, 3, 3>& intrinsics_lhs,
        const math::matrix<type, 3, 3>& intrinsics_rhs,
        const math::matrix<type, 3, 3>& rotation,
        const math::matrix<type, 3, 1>& translation,
        math::matrix<type, 3, 3>& fundamental
    ) {
        math::matrix<type, 3, 3> intrinsics_lhs_inverse;
        if (!math::invert(intrinsics_lhs, intrinsics_lhs_inverse)) {
            return false;
        }
        math::matrix<type, 3, 3> intrinsics_rhs_inverse;
        if (!math::invert(intrinsics_rhs, intrinsics_rhs_inverse)) {
            return false;
        }
        const math::matrix<type, 3, 3> translation_skew{ { { type(0), -translation[2], +translation[1] },
                                                           { +translation[2], type(0), -translation[0] },
                                                           { -translation[1], +translation[0], type(0) } } };
        fundamental = math::transpose(intrinsics_rhs_inverse) * translation_skew * rotation * intrinsics_lhs_inverse;
        return true;
    }

    template <typename type>
    bool fundamental<type>::from_essential(
        const math::matrix<type, 3, 3>& intrinsics_lhs,
        const math::matrix<type, 3, 3>& intrinsics_rhs,
        const math::matrix<type, 3, 3>& essential,
        math::matrix<type, 3, 3>& fundamental
    ) {
        math::matrix<type, 3, 3> intrinsics_lhs_inverse;
        if (!math::invert(intrinsics_lhs, intrinsics_lhs_inverse)) {
            return false;
        }
        math::matrix<type, 3, 3> intrinsics_rhs_inverse;
        if (!math::invert(intrinsics_rhs, intrinsics_rhs_inverse)) {
            return false;
        }
        fundamental = math::transpose(intrinsics_rhs_inverse) * essential * intrinsics_lhs_inverse;
        return true;
    }

    template <typename type>
    size_t fundamental<type>::decompose(
        const math::matrix<type, 3, 3>& intrinsics_lhs,
        const math::matrix<type, 3, 3>& intrinsics_rhs,
        const math::matrix<type, 3, 3>& fundamental,
        math::matrix<type, 3, 3>* const rotations,
        math::matrix<type, 3, 1>* const translations
    ) {
        math::matrix<type, 3, 3> essential;
        geometry::essential<type>::from_fundamental(intrinsics_lhs, intrinsics_rhs, fundamental, essential);
        return geometry::essential<type>::decompose(essential, rotations, translations);
    }

    template <typename type>
    bool fundamental<type>::epipole_lhs(
        const math::matrix<type, 3, 3>& fundamental,
        type& epipole_x,
        type& epipole_y,
        type& epipole_z
    ) {
        return fundamental::null_vector(math::transpose(fundamental), epipole_x, epipole_y, epipole_z);
    }

    template <typename type>
    bool fundamental<type>::epipole_rhs(
        const math::matrix<type, 3, 3>& fundamental,
        type& epipole_x,
        type& epipole_y,
        type& epipole_z
    ) {
        return fundamental::null_vector(fundamental, epipole_x, epipole_y, epipole_z);
    }

    template <typename type>
    bool fundamental<type>::null_vector(
        const math::matrix<type, 3, 3>& matrix,
        type& x,
        type& y,
        type& z
    ) {
        type frobenius_squared = type(0);
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                frobenius_squared += matrix[row][col] * matrix[row][col];
            }
        }
        if (frobenius_squared <= type(0)) {
            x = type(0);
            y = type(0);
            z = type(0);
            return false;
        }

        // The left null vector is orthogonal to every column, so it is the largest cross product of two columns.
        type best[3] = { type(0), type(0), type(0) };
        type best_norm_squared = type(-1);
        constexpr static const size_t pairs[3][2] = { { 0, 1 }, { 0, 2 }, { 1, 2 } };
        for (size_t pair = 0; pair < 3; ++pair) {
            const size_t lhs_col = pairs[pair][0];
            const size_t rhs_col = pairs[pair][1];
            const type lhs[3] = { matrix[0][lhs_col], matrix[1][lhs_col], matrix[2][lhs_col] };
            const type rhs[3] = { matrix[0][rhs_col], matrix[1][rhs_col], matrix[2][rhs_col] };
            const type cross[3] = {
                lhs[1] * rhs[2] - lhs[2] * rhs[1],
                lhs[2] * rhs[0] - lhs[0] * rhs[2],
                lhs[0] * rhs[1] - lhs[1] * rhs[0]
            };
            const type norm_squared = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
            if (norm_squared > best_norm_squared) {
                best_norm_squared = norm_squared;
                best[0] = cross[0];
                best[1] = cross[1];
                best[2] = cross[2];
            }
        }

        x = best[0];
        y = best[1];
        z = best[2];

        return (best_norm_squared > (fundamental::rank_tolerance * frobenius_squared) * (fundamental::rank_tolerance * frobenius_squared));
    }

    template class fundamental<float>;
    template class fundamental<double>;
}
