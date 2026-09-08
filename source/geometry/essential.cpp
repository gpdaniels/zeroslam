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

#include "geometry/essential.hpp"

#include "geometry/fundamental.hpp"
#include "math/math.hpp"

namespace geometry {
    template <typename type>
    void essential<type>::from_poses(
        const math::matrix<type, 3, 3>& rotation,
        const math::matrix<type, 3, 1>& translation,
        math::matrix<type, 3, 3>& essential
    ) {
        const math::matrix<type, 3, 3> translation_skew{ { { type(0), -translation[2], +translation[1] },
                                                           { +translation[2], type(0), -translation[0] },
                                                           { -translation[1], +translation[0], type(0) } } };
        essential = translation_skew * rotation;
    }

    template <typename type>
    void essential<type>::from_fundamental(
        const math::matrix<type, 3, 3>& intrinsics_lhs,
        const math::matrix<type, 3, 3>& intrinsics_rhs,
        const math::matrix<type, 3, 3>& fundamental,
        math::matrix<type, 3, 3>& essential
    ) {
        essential = math::transpose(intrinsics_rhs) * fundamental * intrinsics_lhs;
    }

    template <typename type>
    size_t essential<type>::decompose(
        const math::matrix<type, 3, 3>& essential,
        math::matrix<type, 3, 3>* const rotations,
        math::matrix<type, 3, 1>* const translations
    ) {
        math::matrix<type, 3, 3> matrix_u;
        math::matrix<type, 3, 3> matrix_vt;
        if (!essential::singular_vectors(essential, matrix_u, matrix_vt)) {
            return 0;
        }

        const type determinant_u = matrix_u[0][0] * (matrix_u[1][1] * matrix_u[2][2] - matrix_u[2][1] * matrix_u[1][2]) - matrix_u[0][1] * (matrix_u[1][0] * matrix_u[2][2] - matrix_u[1][2] * matrix_u[2][0]) + matrix_u[0][2] * (matrix_u[1][0] * matrix_u[2][1] - matrix_u[1][1] * matrix_u[2][0]);
        if (determinant_u < type(0)) {
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    matrix_u[row][col] *= type(-1);
                }
            }
        }

        const type determinant_vt = matrix_vt[0][0] * (matrix_vt[1][1] * matrix_vt[2][2] - matrix_vt[2][1] * matrix_vt[1][2]) - matrix_vt[0][1] * (matrix_vt[1][0] * matrix_vt[2][2] - matrix_vt[1][2] * matrix_vt[2][0]) + matrix_vt[0][2] * (matrix_vt[1][0] * matrix_vt[2][1] - matrix_vt[1][1] * matrix_vt[2][0]);
        if (determinant_vt < type(0)) {
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    matrix_vt[row][col] *= type(-1);
                }
            }
        }

        // The two rotations are U W V^T and U W^T V^T for the ninety degree rotation W about z.
        const math::matrix<type, 3, 3> matrix_w{ { { type(0), type(1), type(0) }, { type(-1), type(0), type(0) }, { type(0), type(0), type(1) } } };
        const math::matrix<type, 3, 3> matrix_wt{ { { type(0), type(-1), type(0) }, { type(1), type(0), type(0) }, { type(0), type(0), type(1) } } };

        math::matrix<type, 3, 3> rotation_0;
        math::matrix<type, 3, 3> rotation_1;
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                type sum_uw = type(0);
                type sum_uwt = type(0);
                for (size_t k = 0; k < 3; ++k) {
                    sum_uw += matrix_u[row][k] * matrix_w[k][col];
                    sum_uwt += matrix_u[row][k] * matrix_wt[k][col];
                }
                rotation_0[row][col] = sum_uw;
                rotation_1[row][col] = sum_uwt;
            }
        }
        const math::matrix<type, 3, 3> matrix_uw = rotation_0;
        const math::matrix<type, 3, 3> matrix_uwt = rotation_1;
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                type sum_uw = type(0);
                type sum_uwt = type(0);
                for (size_t k = 0; k < 3; ++k) {
                    sum_uw += matrix_uw[row][k] * matrix_vt[k][col];
                    sum_uwt += matrix_uwt[row][k] * matrix_vt[k][col];
                }
                rotation_0[row][col] = sum_uw;
                rotation_1[row][col] = sum_uwt;
            }
        }

        const math::matrix<type, 3, 1> translation_0{ { matrix_u[0][2], matrix_u[1][2], matrix_u[2][2] } };
        const math::matrix<type, 3, 1> translation_1{ { -matrix_u[0][2], -matrix_u[1][2], -matrix_u[2][2] } };

        rotations[0] = rotation_0;
        translations[0] = translation_0;
        rotations[1] = rotation_0;
        translations[1] = translation_1;
        rotations[2] = rotation_1;
        translations[2] = translation_0;
        rotations[3] = rotation_1;
        translations[3] = translation_1;
        return 4;
    }

    template <typename type>
    bool essential<type>::epipole_lhs(
        const math::matrix<type, 3, 3>& essential,
        type& epipole_x,
        type& epipole_y,
        type& epipole_z
    ) {
        return fundamental<type>::epipole_lhs(essential, epipole_x, epipole_y, epipole_z);
    }

    template <typename type>
    bool essential<type>::epipole_rhs(
        const math::matrix<type, 3, 3>& essential,
        type& epipole_x,
        type& epipole_y,
        type& epipole_z
    ) {
        return fundamental<type>::epipole_rhs(essential, epipole_x, epipole_y, epipole_z);
    }

    template <typename type>
    bool essential<type>::singular_vectors(
        const math::matrix<type, 3, 3>& essential,
        math::matrix<type, 3, 3>& matrix_u,
        math::matrix<type, 3, 3>& matrix_vt
    ) {
        // A rank two matrix has a closed form decomposition: the third right singular vector is the largest cross product of two rows.
        const type cross_0[3] = {
            essential[0][1] * essential[1][2] - essential[0][2] * essential[1][1],
            essential[0][2] * essential[1][0] - essential[0][0] * essential[1][2],
            essential[0][0] * essential[1][1] - essential[0][1] * essential[1][0]
        };
        const type cross_1[3] = {
            essential[0][1] * essential[2][2] - essential[0][2] * essential[2][1],
            essential[0][2] * essential[2][0] - essential[0][0] * essential[2][2],
            essential[0][0] * essential[2][1] - essential[0][1] * essential[2][0]
        };
        const type cross_2[3] = {
            essential[1][1] * essential[2][2] - essential[1][2] * essential[2][1],
            essential[1][2] * essential[2][0] - essential[1][0] * essential[2][2],
            essential[1][0] * essential[2][1] - essential[1][1] * essential[2][0]
        };

        const type cross_0_norm_squared = cross_0[0] * cross_0[0] + cross_0[1] * cross_0[1] + cross_0[2] * cross_0[2];
        const type cross_1_norm_squared = cross_1[0] * cross_1[0] + cross_1[1] * cross_1[1] + cross_1[2] * cross_1[2];
        const type cross_2_norm_squared = cross_2[0] * cross_2[0] + cross_2[1] * cross_2[1] + cross_2[2] * cross_2[2];
        if (!((cross_0_norm_squared > type(0)) || (cross_1_norm_squared > type(0)) || (cross_2_norm_squared > type(0)))) {
            return false;
        }

        const type factor_0 = type(1) / math::sqrt(cross_0_norm_squared);
        const type factor_1 = type(1) / math::sqrt(cross_1_norm_squared);
        const type factor_2 = type(1) / math::sqrt(cross_2_norm_squared);

        if ((factor_0 <= factor_1) && (factor_0 <= factor_2)) {
            for (size_t i = 0; i < 3; ++i) {
                matrix_vt[2][i] = cross_0[i] * factor_0;
            }
        }
        else if ((factor_1 <= factor_0) && (factor_1 <= factor_2)) {
            for (size_t i = 0; i < 3; ++i) {
                matrix_vt[2][i] = cross_1[i] * factor_1;
            }
        }
        else {
            for (size_t i = 0; i < 3; ++i) {
                matrix_vt[2][i] = cross_2[i] * factor_2;
            }
        }

        const type row_0_norm_squared = essential[0][0] * essential[0][0] + essential[0][1] * essential[0][1] + essential[0][2] * essential[0][2];
        const type row_1_norm_squared = essential[1][0] * essential[1][0] + essential[1][1] * essential[1][1] + essential[1][2] * essential[1][2];
        const type row_2_norm_squared = essential[2][0] * essential[2][0] + essential[2][1] * essential[2][1] + essential[2][2] * essential[2][2];
        const type row_max_norm_squared = math::max(row_0_norm_squared, math::max(row_1_norm_squared, row_2_norm_squared));

        if (row_0_norm_squared > type(1.0e-6) * row_max_norm_squared) {
            const type normalisation = type(1) / math::sqrt(row_0_norm_squared);
            matrix_vt[0][0] = essential[0][0] * normalisation;
            matrix_vt[0][1] = essential[0][1] * normalisation;
            matrix_vt[0][2] = essential[0][2] * normalisation;
        }
        else if (row_1_norm_squared >= row_2_norm_squared) {
            const type normalisation = type(1) / math::sqrt(row_1_norm_squared);
            matrix_vt[0][0] = essential[1][0] * normalisation;
            matrix_vt[0][1] = essential[1][1] * normalisation;
            matrix_vt[0][2] = essential[1][2] * normalisation;
        }
        else {
            const type normalisation = type(1) / math::sqrt(row_2_norm_squared);
            matrix_vt[0][0] = essential[2][0] * normalisation;
            matrix_vt[0][1] = essential[2][1] * normalisation;
            matrix_vt[0][2] = essential[2][2] * normalisation;
        }

        matrix_vt[1][0] = matrix_vt[2][1] * matrix_vt[0][2] - matrix_vt[2][2] * matrix_vt[0][1];
        matrix_vt[1][1] = matrix_vt[2][2] * matrix_vt[0][0] - matrix_vt[2][0] * matrix_vt[0][2];
        matrix_vt[1][2] = matrix_vt[2][0] * matrix_vt[0][1] - matrix_vt[2][1] * matrix_vt[0][0];

        matrix_u[0][0] = essential[0][0] * matrix_vt[0][0] + essential[0][1] * matrix_vt[0][1] + essential[0][2] * matrix_vt[0][2];
        matrix_u[1][0] = essential[1][0] * matrix_vt[0][0] + essential[1][1] * matrix_vt[0][1] + essential[1][2] * matrix_vt[0][2];
        matrix_u[2][0] = essential[2][0] * matrix_vt[0][0] + essential[2][1] * matrix_vt[0][1] + essential[2][2] * matrix_vt[0][2];
        matrix_u[0][1] = essential[0][0] * matrix_vt[1][0] + essential[0][1] * matrix_vt[1][1] + essential[0][2] * matrix_vt[1][2];
        matrix_u[1][1] = essential[1][0] * matrix_vt[1][0] + essential[1][1] * matrix_vt[1][1] + essential[1][2] * matrix_vt[1][2];
        matrix_u[2][1] = essential[2][0] * matrix_vt[1][0] + essential[2][1] * matrix_vt[1][1] + essential[2][2] * matrix_vt[1][2];

        const type column_0_normalisation = type(1) / math::sqrt(matrix_u[0][0] * matrix_u[0][0] + matrix_u[1][0] * matrix_u[1][0] + matrix_u[2][0] * matrix_u[2][0]);
        const type column_1_normalisation = type(1) / math::sqrt(matrix_u[0][1] * matrix_u[0][1] + matrix_u[1][1] * matrix_u[1][1] + matrix_u[2][1] * matrix_u[2][1]);

        matrix_u[0][0] *= column_0_normalisation;
        matrix_u[1][0] *= column_0_normalisation;
        matrix_u[2][0] *= column_0_normalisation;
        matrix_u[0][1] *= column_1_normalisation;
        matrix_u[1][1] *= column_1_normalisation;
        matrix_u[2][1] *= column_1_normalisation;

        matrix_u[0][2] = matrix_u[1][0] * matrix_u[2][1] - matrix_u[2][0] * matrix_u[1][1];
        matrix_u[1][2] = matrix_u[2][0] * matrix_u[0][1] - matrix_u[0][0] * matrix_u[2][1];
        matrix_u[2][2] = matrix_u[0][0] * matrix_u[1][1] - matrix_u[1][0] * matrix_u[0][1];
        return true;
    }

    template class essential<float>;
    template class essential<double>;
}
