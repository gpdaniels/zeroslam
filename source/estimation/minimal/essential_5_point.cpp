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

#include "estimation/minimal/essential_5_point.hpp"

#include "math/math.hpp"
#include "math/matrix_decomposition_lower_upper.hpp"
#include "math/matrix_decomposition_singular_value.hpp"
#include "math/matrix_eigen_solver.hpp"

namespace estimation::minimal {
    template <typename type>
    int essential_5_point<type>::solve(
        const type* const __restrict lhs_points,
        const type* const __restrict rhs_points,
        type* const __restrict essentials
    ) {
        constexpr static const auto multiply_one_deg_poly =
            [](
                const type* __restrict lhs_one_deg_poly,
                const type* __restrict rhs_one_deg_poly,
                type* __restrict result
            ) -> void {
            result[0] = lhs_one_deg_poly[0] * rhs_one_deg_poly[0];
            result[1] = lhs_one_deg_poly[0] * rhs_one_deg_poly[1] + lhs_one_deg_poly[1] * rhs_one_deg_poly[0];
            result[2] = lhs_one_deg_poly[1] * rhs_one_deg_poly[1];
            result[3] = lhs_one_deg_poly[0] * rhs_one_deg_poly[2] + lhs_one_deg_poly[2] * rhs_one_deg_poly[0];
            result[4] = lhs_one_deg_poly[1] * rhs_one_deg_poly[2] + lhs_one_deg_poly[2] * rhs_one_deg_poly[1];
            result[5] = lhs_one_deg_poly[2] * rhs_one_deg_poly[2];
            result[6] = lhs_one_deg_poly[0] * rhs_one_deg_poly[3] + lhs_one_deg_poly[3] * rhs_one_deg_poly[0];
            result[7] = lhs_one_deg_poly[1] * rhs_one_deg_poly[3] + lhs_one_deg_poly[3] * rhs_one_deg_poly[1];
            result[8] = lhs_one_deg_poly[2] * rhs_one_deg_poly[3] + lhs_one_deg_poly[3] * rhs_one_deg_poly[2];
            result[9] = lhs_one_deg_poly[3] * rhs_one_deg_poly[3];
        };

        constexpr static const auto multiply_one_deg_two_deg_poly =
            [](
                const type* __restrict lhs_one_deg_poly,
                const type* __restrict rhs_two_deg_poly,
                type* __restrict result
            ) -> void {
            result[0] = lhs_one_deg_poly[0] * rhs_two_deg_poly[0];
            result[1] = lhs_one_deg_poly[0] * rhs_two_deg_poly[1] + lhs_one_deg_poly[1] * rhs_two_deg_poly[0];
            result[2] = lhs_one_deg_poly[1] * rhs_two_deg_poly[1] + lhs_one_deg_poly[2] * rhs_two_deg_poly[0];
            result[3] = lhs_one_deg_poly[2] * rhs_two_deg_poly[1];
            result[4] = lhs_one_deg_poly[0] * rhs_two_deg_poly[2] + lhs_one_deg_poly[3] * rhs_two_deg_poly[0];
            result[5] = lhs_one_deg_poly[1] * rhs_two_deg_poly[2] + lhs_one_deg_poly[3] * rhs_two_deg_poly[1] + lhs_one_deg_poly[4] * rhs_two_deg_poly[0];
            result[6] = lhs_one_deg_poly[2] * rhs_two_deg_poly[2] + lhs_one_deg_poly[4] * rhs_two_deg_poly[1];
            result[7] = lhs_one_deg_poly[3] * rhs_two_deg_poly[2] + lhs_one_deg_poly[5] * rhs_two_deg_poly[0];
            result[8] = lhs_one_deg_poly[4] * rhs_two_deg_poly[2] + lhs_one_deg_poly[5] * rhs_two_deg_poly[1];
            result[9] = lhs_one_deg_poly[5] * rhs_two_deg_poly[2];
            result[10] = lhs_one_deg_poly[0] * rhs_two_deg_poly[3] + lhs_one_deg_poly[6] * rhs_two_deg_poly[0];
            result[11] = lhs_one_deg_poly[1] * rhs_two_deg_poly[3] + lhs_one_deg_poly[6] * rhs_two_deg_poly[1] + lhs_one_deg_poly[7] * rhs_two_deg_poly[0];
            result[12] = lhs_one_deg_poly[2] * rhs_two_deg_poly[3] + lhs_one_deg_poly[7] * rhs_two_deg_poly[1];
            result[13] = lhs_one_deg_poly[3] * rhs_two_deg_poly[3] + lhs_one_deg_poly[6] * rhs_two_deg_poly[2] + lhs_one_deg_poly[8] * rhs_two_deg_poly[0];
            result[14] = lhs_one_deg_poly[4] * rhs_two_deg_poly[3] + lhs_one_deg_poly[7] * rhs_two_deg_poly[2] + lhs_one_deg_poly[8] * rhs_two_deg_poly[1];
            result[15] = lhs_one_deg_poly[5] * rhs_two_deg_poly[3] + lhs_one_deg_poly[8] * rhs_two_deg_poly[2];
            result[16] = lhs_one_deg_poly[6] * rhs_two_deg_poly[3] + lhs_one_deg_poly[9] * rhs_two_deg_poly[0];
            result[17] = lhs_one_deg_poly[7] * rhs_two_deg_poly[3] + lhs_one_deg_poly[9] * rhs_two_deg_poly[1];
            result[18] = lhs_one_deg_poly[8] * rhs_two_deg_poly[3] + lhs_one_deg_poly[9] * rhs_two_deg_poly[2];
            result[19] = lhs_one_deg_poly[9] * rhs_two_deg_poly[3];
        };

        constexpr static const auto matrix_transpose = [](const type* matrix, int width, int height, type* result) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    result[x * height + y] = matrix[y * width + x];
                }
            }
        };

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

        type epipolar_constraint[5][9];
        for (int i = 0; i < 5; ++i) {
            epipolar_constraint[i][0] = rhs_points[i * 2 + 0] * lhs_points[i * 2 + 0];
            epipolar_constraint[i][1] = rhs_points[i * 2 + 1] * lhs_points[i * 2 + 0];
            epipolar_constraint[i][2] = lhs_points[i * 2 + 0];
            epipolar_constraint[i][3] = rhs_points[i * 2 + 0] * lhs_points[i * 2 + 1];
            epipolar_constraint[i][4] = rhs_points[i * 2 + 1] * lhs_points[i * 2 + 1];
            epipolar_constraint[i][5] = lhs_points[i * 2 + 1];
            epipolar_constraint[i][6] = rhs_points[i * 2 + 0];
            epipolar_constraint[i][7] = rhs_points[i * 2 + 1];
            epipolar_constraint[i][8] = 1.0;
        }
        type epipolar_constraint_t[9][5];
        matrix_transpose(&epipolar_constraint[0][0], 9, 5, &epipolar_constraint_t[0][0]);
        type block[9][9];
        matrix_multiply(&epipolar_constraint_t[0][0], 5, 9, &epipolar_constraint[0][0], 9, 5, &block[0][0]);
        type u[9][9];
        type s[9][9];
        type vt[9][9];
        if (!math::decompose_singular_value(&block[0][0], 9, 9, &u[0][0], &s[0][0], &vt[0][0])) {
            return 0;
        }
        const type null_space[9][4] = {
            { vt[5][0], vt[6][0], vt[7][0], vt[8][0] },
            { vt[5][1], vt[6][1], vt[7][1], vt[8][1] },
            { vt[5][2], vt[6][2], vt[7][2], vt[8][2] },
            { vt[5][3], vt[6][3], vt[7][3], vt[8][3] },
            { vt[5][4], vt[6][4], vt[7][4], vt[8][4] },
            { vt[5][5], vt[6][5], vt[7][5], vt[8][5] },
            { vt[5][6], vt[6][6], vt[7][6], vt[8][6] },
            { vt[5][7], vt[6][7], vt[7][7], vt[8][7] },
            { vt[5][8], vt[6][8], vt[7][8], vt[8][8] }
        };
        const type null_space_matrix[3][3][4] = {
            { { null_space[0][0], null_space[0][1], null_space[0][2], null_space[0][3] },
              { null_space[3][0], null_space[3][1], null_space[3][2], null_space[3][3] },
              { null_space[6][0], null_space[6][1], null_space[6][2], null_space[6][3] } },
            { { null_space[1][0], null_space[1][1], null_space[1][2], null_space[1][3] },
              { null_space[4][0], null_space[4][1], null_space[4][2], null_space[4][3] },
              { null_space[7][0], null_space[7][1], null_space[7][2], null_space[7][3] } },
            { { null_space[2][0], null_space[2][1], null_space[2][2], null_space[2][3] },
              { null_space[5][0], null_space[5][1], null_space[5][2], null_space[5][3] },
              { null_space[8][0], null_space[8][1], null_space[8][2], null_space[8][3] } }
        };

        type constraint_matrix[10][20];
        {
            {
                type* trace_constraint = &constraint_matrix[0][0];

                type eet[3][3][10];
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        {
                            type result_parts[3][10];
                            multiply_one_deg_poly(&null_space_matrix[i][0][0], &null_space_matrix[j][0][0], &result_parts[0][0]);
                            multiply_one_deg_poly(&null_space_matrix[i][1][0], &null_space_matrix[j][1][0], &result_parts[1][0]);
                            multiply_one_deg_poly(&null_space_matrix[i][2][0], &null_space_matrix[j][2][0], &result_parts[2][0]);
                            for (int index = 0; index < 10; ++index) {
                                eet[i][j][index] = result_parts[0][index] + result_parts[1][index] + result_parts[2][index];
                            }
                        }
                        for (int index = 0; index < 10; ++index) {
                            eet[i][j][index] *= type(2);
                        }
                    }
                }

                type trace[10];
                for (int index = 0; index < 10; ++index) {
                    trace[index] = eet[0][0][index] + eet[1][1][index] + eet[2][2][index];
                }

                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        type result_parts[4][20];
                        multiply_one_deg_two_deg_poly(&eet[i][0][0], &null_space_matrix[0][j][0], &result_parts[0][0]);
                        multiply_one_deg_two_deg_poly(&eet[i][1][0], &null_space_matrix[1][j][0], &result_parts[1][0]);
                        multiply_one_deg_two_deg_poly(&eet[i][2][0], &null_space_matrix[2][j][0], &result_parts[2][0]);
                        multiply_one_deg_two_deg_poly(&trace[0], &null_space_matrix[i][j][0], &result_parts[3][0]);
                        for (int index = 0; index < 20; ++index) {
                            trace_constraint[(3 * i + j) * 20 + index] = result_parts[0][index] + result_parts[1][index] + result_parts[2][index] - (type(0.5) * result_parts[3][index]);
                        }
                    }
                }
            }

            {
                type* determinant_constraint = &constraint_matrix[9][0];

                type null_space_01_12[10];
                multiply_one_deg_poly(&null_space_matrix[0][1][0], &null_space_matrix[1][2][0], &null_space_01_12[0]);
                type null_space_02_11[10];
                multiply_one_deg_poly(&null_space_matrix[0][2][0], &null_space_matrix[1][1][0], &null_space_02_11[0]);
                type null_space_01_12_minus_02_11[10];
                for (int i = 0; i < 10; ++i) {
                    null_space_01_12_minus_02_11[i] = null_space_01_12[i] - null_space_02_11[i];
                }
                type determinant_0[20];
                multiply_one_deg_two_deg_poly(&null_space_01_12_minus_02_11[0], &null_space_matrix[2][0][0], &determinant_0[0]);

                type null_space_02_10[10];
                multiply_one_deg_poly(&null_space_matrix[0][2][0], &null_space_matrix[1][0][0], &null_space_02_10[0]);
                type null_space_00_12[10];
                multiply_one_deg_poly(&null_space_matrix[0][0][0], &null_space_matrix[1][2][0], &null_space_00_12[0]);
                type null_space_02_10_minus_00_12[10];
                for (int i = 0; i < 10; ++i) {
                    null_space_02_10_minus_00_12[i] = null_space_02_10[i] - null_space_00_12[i];
                }
                type determinant_1[20];
                multiply_one_deg_two_deg_poly(&null_space_02_10_minus_00_12[0], &null_space_matrix[2][1][0], &determinant_1[0]);

                type null_space_00_11[10];
                multiply_one_deg_poly(&null_space_matrix[0][0][0], &null_space_matrix[1][1][0], &null_space_00_11[0]);
                type null_space_01_10[10];
                multiply_one_deg_poly(&null_space_matrix[0][1][0], &null_space_matrix[1][0][0], &null_space_01_10[0]);
                type null_space_00_11_minus_01_10[10];
                for (int i = 0; i < 10; ++i) {
                    null_space_00_11_minus_01_10[i] = null_space_00_11[i] - null_space_01_10[i];
                }
                type determinant_2[20];
                multiply_one_deg_two_deg_poly(&null_space_00_11_minus_01_10[0], &null_space_matrix[2][2][0], &determinant_2[0]);

                for (int i = 0; i < 20; ++i) {
                    determinant_constraint[i] = determinant_0[i] + determinant_1[i] + determinant_2[i];
                }
            }
        }

        type constraint_matrix_lhs[10][10];
        for (int y = 0; y < 10; ++y) {
            for (int x = 0; x < 10; ++x) {
                constraint_matrix_lhs[y][x] = constraint_matrix[y][x];
            }
        }
        type matrix_l[10][10];
        type matrix_u[10][10];
        type matrix_p[10][10];
        if (!math::decompose_lower_upper<type>(&constraint_matrix_lhs[0][0], 10, 10, &matrix_l[0][0], &matrix_u[0][0], &matrix_p[0][0])) {
            return 0;
        }

        type eliminated_matrix[10][10];
        for (int x = 0; x < 10; ++x) {
            type constraint_matrix_rhs_column[10][1];
            for (int y = 0; y < 10; ++y) {
                constraint_matrix_rhs_column[y][0] = constraint_matrix[y][x + 10];
            }
            type eliminated_matrix_column[10][1];
            if (!math::solve_lower_upper<type>(&matrix_l[0][0], &matrix_u[0][0], &matrix_p[0][0], &constraint_matrix_rhs_column[0][0], 10, 10, &eliminated_matrix_column[0][0])) {
                return 0;
            }
            for (int y = 0; y < 10; ++y) {
                eliminated_matrix[y][x] = eliminated_matrix_column[y][0];
            }
        }

        type action_matrix[10][10] = {};
        for (int x = 0; x < 10; ++x) {
            action_matrix[0][x] = eliminated_matrix[0][x];
            action_matrix[1][x] = eliminated_matrix[1][x];
            action_matrix[2][x] = eliminated_matrix[2][x];
            action_matrix[3][x] = eliminated_matrix[4][x];
            action_matrix[4][x] = eliminated_matrix[5][x];
            action_matrix[5][x] = eliminated_matrix[7][x];
        }
        action_matrix[6][0] = type(-1.0);
        action_matrix[7][1] = type(-1.0);
        action_matrix[8][3] = type(-1.0);
        action_matrix[9][6] = type(-1.0);

        type eigen_values[10][2] = {};
        type eigen_vectors[10][10] = {};
        math::eigen_solver(&action_matrix[0][0], 10, &eigen_values[0][0], &eigen_vectors[0][0]);

        int count = 0;
        for (int i = 0; i < 10; i++) {
            if (eigen_values[i][1] != 0) {
                continue;
            }
            const type eigen_vector_part[4] = {
                eigen_vectors[6][i],
                eigen_vectors[7][i],
                eigen_vectors[8][i],
                eigen_vectors[9][i]
            };
            type ematrix[3][3];
            matrix_multiply(&null_space[0][0], 4, 9, &eigen_vector_part[0], 1, 4, &ematrix[0][0]);
            matrix_transpose(&ematrix[0][0], 3, 3, &essentials[count++ * 9]);
        }

        return count;
    }

    template class essential_5_point<float>;
    template class essential_5_point<double>;
}
