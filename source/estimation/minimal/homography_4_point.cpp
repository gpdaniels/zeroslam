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

#include "estimation/minimal/homography_4_point.hpp"

#include "math/math.hpp"
#include "math/matrix_decomposition_singular_value.hpp"

namespace estimation::minimal {
    template <typename type>
    bool homography_4_point<type>::solve(
        const type* const __restrict lhs_points,
        const type* const __restrict rhs_points,
        type* const __restrict homography
    ) {
        constexpr static const auto matrix_multiply = [](const type* lhs, int lhs_width, int lhs_height, const type* rhs, int rhs_width, int rhs_height, type* result) {
            static_cast<void>(rhs_height);
            for (int lhs_y = 0; lhs_y < lhs_height; ++lhs_y) {
                for (int rhs_x = 0; rhs_x < rhs_width; ++rhs_x) {
                    type sum = 0;
                    for (int lhs_x_rhs_y = 0; lhs_x_rhs_y < lhs_width; ++lhs_x_rhs_y) {
                        sum += lhs[lhs_y * lhs_width + lhs_x_rhs_y] * rhs[lhs_x_rhs_y * rhs_width + rhs_x];
                    }
                    result[lhs_y * rhs_width + rhs_x] = sum;
                }
            }
        };

        constexpr static const auto hartley_normalize = [](const type* const __restrict points, type* const __restrict normalized, type* const __restrict transform) -> bool {
            type centroid_x = 0;
            type centroid_y = 0;
            for (int i = 0; i < 4; ++i) {
                centroid_x += points[i * 2 + 0];
                centroid_y += points[i * 2 + 1];
            }
            centroid_x *= type(0.25);
            centroid_y *= type(0.25);

            type mean_distance = 0;
            for (int i = 0; i < 4; ++i) {
                const type dx = points[i * 2 + 0] - centroid_x;
                const type dy = points[i * 2 + 1] - centroid_y;
                mean_distance += math::sqrt(dx * dx + dy * dy);
            }
            mean_distance *= type(0.25);

            if (mean_distance < type(1.0e-12)) {
                return false;
            }

            const type scale = math::sqrt(type(2.0)) / mean_distance;
            for (int i = 0; i < 4; ++i) {
                normalized[i * 2 + 0] = scale * (points[i * 2 + 0] - centroid_x);
                normalized[i * 2 + 1] = scale * (points[i * 2 + 1] - centroid_y);
            }

            transform[0] = scale;
            transform[1] = 0;
            transform[2] = -scale * centroid_x;
            transform[3] = 0;
            transform[4] = scale;
            transform[5] = -scale * centroid_y;
            transform[6] = 0;
            transform[7] = 0;
            transform[8] = 1;
            return true;
        };

        type lhs_normalized[8];
        type rhs_normalized[8];
        type transform_lhs[9];
        type transform_rhs[9];
        if (!hartley_normalize(lhs_points, lhs_normalized, transform_lhs)) {
            return false;
        }
        if (!hartley_normalize(rhs_points, rhs_normalized, transform_rhs)) {
            return false;
        }

        type constraint[8][9];
        for (int i = 0; i < 4; ++i) {
            const type u = rhs_normalized[i * 2 + 0];
            const type v = rhs_normalized[i * 2 + 1];
            const type up = lhs_normalized[i * 2 + 0];
            const type vp = lhs_normalized[i * 2 + 1];

            constraint[i * 2 + 0][0] = 0;
            constraint[i * 2 + 0][1] = 0;
            constraint[i * 2 + 0][2] = 0;
            constraint[i * 2 + 0][3] = -u;
            constraint[i * 2 + 0][4] = -v;
            constraint[i * 2 + 0][5] = -1;
            constraint[i * 2 + 0][6] = vp * u;
            constraint[i * 2 + 0][7] = vp * v;
            constraint[i * 2 + 0][8] = vp;

            constraint[i * 2 + 1][0] = u;
            constraint[i * 2 + 1][1] = v;
            constraint[i * 2 + 1][2] = 1;
            constraint[i * 2 + 1][3] = 0;
            constraint[i * 2 + 1][4] = 0;
            constraint[i * 2 + 1][5] = 0;
            constraint[i * 2 + 1][6] = -up * u;
            constraint[i * 2 + 1][7] = -up * v;
            constraint[i * 2 + 1][8] = -up;
        }

        type constraint_t[9][8];
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 9; ++x) {
                constraint_t[x][y] = constraint[y][x];
            }
        }
        type block[9][9];
        matrix_multiply(&constraint_t[0][0], 8, 9, &constraint[0][0], 9, 8, &block[0][0]);
        type u_matrix[9][9];
        type s_matrix[9][9];
        type vt_matrix[9][9];
        if (!math::decompose_singular_value(&block[0][0], 9, 9, &u_matrix[0][0], &s_matrix[0][0], &vt_matrix[0][0])) {
            return false;
        }

        if (s_matrix[7][7] < type(1.0e-9) * s_matrix[0][0]) {
            return false;
        }

        type homography_normalized[9];
        for (int i = 0; i < 9; ++i) {
            homography_normalized[i] = vt_matrix[8][i];
        }

        const type transform_lhs_inverse[9] = {
            type(1.0) / transform_lhs[0],
            0,
            -transform_lhs[2] / transform_lhs[0],
            0,
            type(1.0) / transform_lhs[4],
            -transform_lhs[5] / transform_lhs[4],
            0,
            0,
            type(1.0)
        };
        type homography_temp[9];
        matrix_multiply(&homography_normalized[0], 3, 3, &transform_rhs[0], 3, 3, &homography_temp[0]);
        matrix_multiply(&transform_lhs_inverse[0], 3, 3, &homography_temp[0], 3, 3, &homography[0]);

        type frobenius_squared = 0;
        for (int i = 0; i < 9; ++i) {
            frobenius_squared += homography[i] * homography[i];
        }
        if (frobenius_squared < type(1.0e-30)) {
            return false;
        }
        const type frobenius_inverse = type(1.0) / math::sqrt(frobenius_squared);
        for (int i = 0; i < 9; ++i) {
            homography[i] *= frobenius_inverse;
            if (math::isnan(homography[i]) || math::isinf(homography[i])) {
                return false;
            }
        }

        return true;
    }

    template class homography_4_point<float>;
    template class homography_4_point<double>;
}
