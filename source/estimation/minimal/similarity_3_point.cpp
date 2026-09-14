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

#include "estimation/minimal/similarity_3_point.hpp"

#include "math/matrix_decomposition_singular_value.hpp"

namespace estimation::minimal {
    template <typename type>
    bool similarity_3_point<type>::solve(
        const type* const __restrict source_points,
        const type* const __restrict target_points,
        const size_t count,
        type rotation[9],
        type translation[3],
        type& scale
    ) {
        if (count < 3) {
            return false;
        }

        constexpr static const type tolerance = (sizeof(type) < sizeof(double)) ? static_cast<type>(1e-5) : static_cast<type>(1e-9);

        type source_centroid[3] = { 0, 0, 0 };
        type target_centroid[3] = { 0, 0, 0 };
        for (size_t index = 0; index < count; ++index) {
            for (size_t axis = 0; axis < 3; ++axis) {
                source_centroid[axis] += source_points[(index * 3) + axis];
                target_centroid[axis] += target_points[(index * 3) + axis];
            }
        }
        const type inverse_count = type(1) / static_cast<type>(count);
        for (size_t axis = 0; axis < 3; ++axis) {
            source_centroid[axis] *= inverse_count;
            target_centroid[axis] *= inverse_count;
        }

        type covariance[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
        type source_variance = 0;
        type source_magnitude_squared = 0;
        for (size_t index = 0; index < count; ++index) {
            type source_delta[3];
            type target_delta[3];
            for (size_t axis = 0; axis < 3; ++axis) {
                source_delta[axis] = source_points[(index * 3) + axis] - source_centroid[axis];
                target_delta[axis] = target_points[(index * 3) + axis] - target_centroid[axis];
                source_variance += source_delta[axis] * source_delta[axis];
                source_magnitude_squared += source_points[(index * 3) + axis] * source_points[(index * 3) + axis];
            }
            for (size_t row = 0; row < 3; ++row) {
                for (size_t column = 0; column < 3; ++column) {
                    covariance[row][column] += source_delta[row] * target_delta[column];
                }
            }
        }
        if (source_variance <= tolerance * tolerance * source_magnitude_squared) {
            return false;
        }

        type matrix_u[3][3];
        type matrix_s[3][3];
        type matrix_vt[3][3];
        if (!math::decompose_singular_value(&covariance[0][0], 3, 3, &matrix_u[0][0], &matrix_s[0][0], &matrix_vt[0][0])) {
            return false;
        }
        if (matrix_s[1][1] <= tolerance * matrix_s[0][0]) {
            return false;
        }

        type rotation_candidate[3][3];
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                rotation_candidate[row][column] = (matrix_vt[0][row] * matrix_u[column][0]) + (matrix_vt[1][row] * matrix_u[column][1]) + (matrix_vt[2][row] * matrix_u[column][2]);
            }
        }
        const type determinant =
            rotation_candidate[0][0] * ((rotation_candidate[1][1] * rotation_candidate[2][2]) - (rotation_candidate[2][1] * rotation_candidate[1][2])) -
            rotation_candidate[0][1] * ((rotation_candidate[1][0] * rotation_candidate[2][2]) - (rotation_candidate[1][2] * rotation_candidate[2][0])) +
            rotation_candidate[0][2] * ((rotation_candidate[1][0] * rotation_candidate[2][1]) - (rotation_candidate[1][1] * rotation_candidate[2][0]));
        const type sign = (determinant > 0) ? type(1) : type(-1);
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                rotation[(row * 3) + column] = (matrix_vt[0][row] * matrix_u[column][0]) + (matrix_vt[1][row] * matrix_u[column][1]) + (sign * matrix_vt[2][row] * matrix_u[column][2]);
            }
        }

        scale = (matrix_s[0][0] + matrix_s[1][1] + (sign * matrix_s[2][2])) / source_variance;

        for (size_t row = 0; row < 3; ++row) {
            const type rotated = (rotation[(row * 3) + 0] * source_centroid[0]) + (rotation[(row * 3) + 1] * source_centroid[1]) + (rotation[(row * 3) + 2] * source_centroid[2]);
            translation[row] = target_centroid[row] - (scale * rotated);
        }
        return true;
    }

    template class similarity_3_point<float>;
    template class similarity_3_point<double>;
}
