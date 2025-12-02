/*
Copyright (C) 2025 Geoffrey Daniels. https://gpdaniels.com/

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
#ifndef POSE_ESTIMATION_HPP
#define POSE_ESTIMATION_HPP

#include "geometry.hpp"
#include "math.hpp"
#include "matrix.hpp"
#include "matrix_decomposition_lower_upper.hpp"
#include "matrix_decomposition_singular_value.hpp"
#include "matrix_eigen_solver.hpp"

namespace pose_estimation {

    // Computes the relative pose between two cameras using 5 corresponding points.
    // The relative pose is computed such that y * E * x = 0, where E = tx * R and tx is the cross product matrix of t.
    // Returns the number of solutions.
    template <typename type>
    int essential_5_point(
        const type* const __restrict lhs_points, // Assumed to be exactly 5 two dimensional points, so a 10 element erray [x,y,x,y,x,y,x,y,x,y].
        const type* const __restrict rhs_points, // Assumed to be exactly 5 two dimensional points, so a 10 element erray [x,y,x,y,x,y,x,y,x,y].
        type* const __restrict essentials        // Assumed to be an already allocated set of 10 matrices which are 3x3 => 90 elements.
    ) {
        // Multiply two one-degree polynomials (in x, y, z).
        // E.g. p1 = a[0]x + a[1]y + a[2]z + a[3]
        // Output order: x^2 xy y^2 xz yz z^2 x y z c (grevlex order)
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

        // Multiply a two-degree polynomial (in x, y, z) and a one-degree polynomial (in x, y, z).
        // Output order: x^3 x^2y xy^2 y^3 x^2z xyz y^2z xz^2 yz^2 z^3 x^2 xy y^2 xz yz z^2 x y z c
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

        // Step 1. Create the 5x9 matrix containing epipolar constraints.
        // The essential matrix is a linear combination of the four vectors spanning the null space of this matrix.
        type epipolar_constraint[5][9];
        for (int i = 0; i < 5; ++i) {
            // Fill matrix with the epipolar constraint.
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
        matrix::decompose_singular_value(&block[0][0], 9, 9, &u[0][0], &s[0][0], &vt[0][0]);
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

        // Step 2. Expansion of the epipolar constraints on the determinant and trace.
        type constraint_matrix[10][20];
        {
            // Builds the trace constraint: "2 * E*Et*E - trace(E*Et)*E = 0".
            {
                type* trace_constraint = &constraint_matrix[0][0];

                // Compute E*Et first.
                type eet[3][3][10];
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        // Shorthand for multiplying the essential matrix with its transpose.
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

                // Compute the trace of E*Et.
                type trace[10];
                for (int index = 0; index < 10; ++index) {
                    trace[index] = eet[0][0][index] + eet[1][1][index] + eet[2][2][index];
                }

                // Multiply E*Et and trace(E*Et) by E.
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

            // Singularity constraint.
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

        // Step 3. Eliminate part of the matrix to isolate polynomials in z.
        type constraint_matrix_lhs[10][10];
        for (int y = 0; y < 10; ++y) {
            for (int x = 0; x < 10; ++x) {
                constraint_matrix_lhs[y][x] = constraint_matrix[y][x];
            }
        }
        type eliminated_matrix[10][10];
        for (int x = 0; x < 10; ++x) {
            type constraint_matrix_rhs_column[10][1];
            for (int y = 0; y < 10; ++y) {
                constraint_matrix_rhs_column[y][0] = constraint_matrix[y][x + 10];
            }
            type eliminated_matrix_column[10][1];
            matrix::solve_lower_upper<type>(&constraint_matrix_lhs[0][0], &constraint_matrix_rhs_column[0][0], 10, 10, &eliminated_matrix_column[0][0]);
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
        matrix::eigen_solver(&action_matrix[0][0], 10, &eigen_values[0][0], &eigen_vectors[0][0]);

        // Now that we have x, y, and z we need to substitute them back into the null space to get a valid essential matrix solution.
        int count = 0;
        for (int i = 0; i < 10; i++) {
            // Only consider real solutions.
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

    template <typename type>
    class essential_matrix final {
    private:
        constexpr static void multiply(const type* __restrict lhs, const type* __restrict rhs, type* __restrict result) {
            for (int y = 0; y < 3; ++y) {
                for (int x = 0; x < 3; ++x) {
                    result[y * 3 + x] = 0;
                }
                for (int z = 0; z < 3; ++z) {
                    for (int x = 0; x < 3; ++x) {
                        result[y * 3 + x] += lhs[y * 3 + z] * rhs[z * 3 + x];
                    }
                }
            }
        };

        constexpr static void decompose_essential(const double essential[9], double matrix_u[9], double matrix_vt[9]) {
            const double E0[3] = {
                essential[1] * essential[5] - essential[2] * essential[4],
                essential[2] * essential[3] - essential[0] * essential[5],
                essential[0] * essential[4] - essential[1] * essential[3]
            };
            const double E1[3] = {
                essential[1] * essential[8] - essential[2] * essential[7],
                essential[2] * essential[6] - essential[0] * essential[8],
                essential[0] * essential[7] - essential[1] * essential[6]
            };
            const double E2[3] = {
                essential[4] * essential[8] - essential[5] * essential[7],
                essential[5] * essential[6] - essential[3] * essential[8],
                essential[3] * essential[7] - essential[4] * essential[6]
            };
            const double matrix_vt_factor[3] = {
                1.0 / math::sqrt(E0[0] * E0[0] + E0[1] * E0[1] + E0[2] * E0[2]),
                1.0 / math::sqrt(E1[0] * E1[0] + E1[1] * E1[1] + E1[2] * E1[2]),
                1.0 / math::sqrt(E2[0] * E2[0] + E2[1] * E2[1] + E2[2] * E2[2])
            };

            if ((matrix_vt_factor[0] <= matrix_vt_factor[1]) && (matrix_vt_factor[0] <= matrix_vt_factor[2])) {
                for (int i = 6; i < 9; ++i) {
                    matrix_vt[i] = E0[i - 6] * matrix_vt_factor[0];
                }
            }
            else if ((matrix_vt_factor[1] <= matrix_vt_factor[0]) && (matrix_vt_factor[1] <= matrix_vt_factor[2])) {
                for (int i = 6; i < 9; ++i) {
                    matrix_vt[i] = E1[i - 6] * matrix_vt_factor[1];
                }
            }
            else {
                for (int i = 6; i < 9; ++i) {
                    matrix_vt[i] = E2[i - 6] * matrix_vt_factor[2];
                }
            }

            const double matrix_vt_normalisation_factor = 1.0 / math::sqrt(essential[0] * essential[0] + essential[1] * essential[1] + essential[2] * essential[2]);
            matrix_vt[0] = essential[0] * matrix_vt_normalisation_factor;
            matrix_vt[1] = essential[1] * matrix_vt_normalisation_factor;
            matrix_vt[2] = essential[2] * matrix_vt_normalisation_factor;
            matrix_vt[3] = matrix_vt[7] * matrix_vt[2] - matrix_vt[8] * matrix_vt[1];
            matrix_vt[4] = matrix_vt[8] * matrix_vt[0] - matrix_vt[6] * matrix_vt[2];
            matrix_vt[5] = matrix_vt[6] * matrix_vt[1] - matrix_vt[7] * matrix_vt[0];

            matrix_u[0] = essential[0] * matrix_vt[0] + essential[1] * matrix_vt[1] + essential[2] * matrix_vt[2];
            matrix_u[3] = essential[3] * matrix_vt[0] + essential[4] * matrix_vt[1] + essential[5] * matrix_vt[2];
            matrix_u[6] = essential[6] * matrix_vt[0] + essential[7] * matrix_vt[1] + essential[8] * matrix_vt[2];
            matrix_u[1] = essential[0] * matrix_vt[3] + essential[1] * matrix_vt[4] + essential[2] * matrix_vt[5];
            matrix_u[4] = essential[3] * matrix_vt[3] + essential[4] * matrix_vt[4] + essential[5] * matrix_vt[5];
            matrix_u[7] = essential[6] * matrix_vt[3] + essential[7] * matrix_vt[4] + essential[8] * matrix_vt[5];

            const double matrix_u_normalisation_factors[2] = {
                1.0 / math::sqrt(matrix_u[0] * matrix_u[0] + matrix_u[3] * matrix_u[3] + matrix_u[6] * matrix_u[6]),
                1.0 / math::sqrt(matrix_u[1] * matrix_u[1] + matrix_u[4] * matrix_u[4] + matrix_u[7] * matrix_u[7])
            };
            matrix_u[0] *= matrix_u_normalisation_factors[0];
            matrix_u[3] *= matrix_u_normalisation_factors[0];
            matrix_u[6] *= matrix_u_normalisation_factors[0];
            matrix_u[1] *= matrix_u_normalisation_factors[1];
            matrix_u[4] *= matrix_u_normalisation_factors[1];
            matrix_u[7] *= matrix_u_normalisation_factors[1];
            matrix_u[2] = matrix_u[3] * matrix_u[7] - matrix_u[6] * matrix_u[4];
            matrix_u[5] = matrix_u[6] * matrix_u[1] - matrix_u[0] * matrix_u[7];
            matrix_u[8] = matrix_u[0] * matrix_u[4] - matrix_u[3] * matrix_u[1];
        }

    public:
        static void compose(
            const type* __restrict rotation,
            const type* __restrict translation,
            type* __restrict essential
        ) {
            const type translation_length_inverse = type(1.0) / math::sqrt(translation[0] * translation[0] + translation[1] * translation[1] + translation[2] * translation[2]);
            const type translation_normalised[3] = {
                translation[0] * translation_length_inverse,
                translation[1] * translation_length_inverse,
                translation[2] * translation_length_inverse
            };
            const type translation_skew[3][3] = {
                { 0, -translation_normalised[2], translation_normalised[1] },
                { translation_normalised[2], 0, -translation_normalised[0] },
                { -translation_normalised[1], translation_normalised[0], 0 }
            };
            multiply(&translation_skew[0][0], rotation, essential);
        }

        static void decompose(
            const type* __restrict essential,
            type* __restrict rotation_0,
            type* __restrict rotation_1,
            type* __restrict translation_0,
            type* __restrict translation_1
        ) {
            type u[3][3];
            type vt[3][3];
            decompose_essential(&essential[0], &u[0][0], &vt[0][0]);

            const type determinant_u = u[0][0] * (u[1][1] * u[2][2] - u[2][1] * u[1][2]) - u[0][1] * (u[1][0] * u[2][2] - u[1][2] * u[2][0]) + u[0][2] * (u[1][0] * u[2][1] - u[1][1] * u[2][0]);
            if (determinant_u < 0) {
                for (int y = 0; y < 3; ++y) {
                    for (int x = 0; x < 3; ++x) {
                        u[y][x] *= type(-1);
                    }
                }
            }

            const type determinant_vt = vt[0][0] * (vt[1][1] * vt[2][2] - vt[2][1] * vt[1][2]) - vt[0][1] * (vt[1][0] * vt[2][2] - vt[1][2] * vt[2][0]) + vt[0][2] * (vt[1][0] * vt[2][1] - vt[1][1] * vt[2][0]);
            if (determinant_vt < 0) {
                for (int y = 0; y < 3; ++y) {
                    for (int x = 0; x < 3; ++x) {
                        vt[y][x] *= type(-1);
                    }
                }
            }

            const type w[3][3] = {
                { 0, 1, 0 },
                { -1, 0, 0 },
                { 0, 0, 1 },
            };

            const type wt[3][3] = {
                { 0, -1, 0 },
                { 1, 0, 0 },
                { 0, 0, 1 },
            };

            type uw[3][3];
            multiply(&u[0][0], &w[0][0], &uw[0][0]);
            multiply(&uw[0][0], &vt[0][0], &rotation_0[0]);

            type uwt[3][3];
            multiply(&u[0][0], &wt[0][0], &uwt[0][0]);
            multiply(&uwt[0][0], &vt[0][0], &rotation_1[0]);

            translation_0[0] = u[0][2];
            translation_0[1] = u[1][2];
            translation_0[2] = u[2][2];

            translation_1[0] = -u[0][2];
            translation_1[1] = -u[1][2];
            translation_1[2] = -u[2][2];
        }

        // Note: Assuming all points are normalised.
        static void recover_pose(
            const type* __restrict essential,
            const type* __restrict lhs_points, // X, Y, X, Y, X, Y, ...
            const type* __restrict rhs_points, // X, Y, X, Y, X, Y, ...
            const size_t point_count,
            type* __restrict rotation,
            type* __restrict translation,
            type* __restrict trigulated_points // X, Y, Z, X, Y, Z, X, Y, Z, ...
        ) {
            constexpr static const type distance_threshold = type(50);

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

            // Extract the four possible transformations.
            type rotations[2][3][3];
            type translations[2][3];
            essential_matrix::decompose(essential, &rotations[0][0][0], &rotations[1][0][0], &translations[0][0], &translations[1][0]);

            // Generate a lhs pose:
            const type lhs_pose[3][4] = {
                { 1, 0, 0, 0 },
                { 0, 1, 0, 0 },
                { 0, 0, 1, 0 },
            };

            // Triangulate all points using the four transformations.
            type* trigulated_points_all[4]{
                new type[point_count * 3]{},
                new type[point_count * 3]{},
                new type[point_count * 3]{},
                new type[point_count * 3]{}
            };
            size_t trigulated_points_valid[4] = {};
            for (size_t r = 0; r < 2; ++r) {
                for (size_t t = 0; t < 2; ++t) {
                    const type rhs_pose[3][4] = {
                        { rotations[r][0][0], rotations[r][0][1], rotations[r][0][2], translations[t][0] },
                        { rotations[r][1][0], rotations[r][1][1], rotations[r][1][2], translations[t][1] },
                        { rotations[r][2][0], rotations[r][2][1], rotations[r][2][2], translations[t][2] },
                    };
                    for (size_t i = 0; i < point_count; ++i) {
                        // Triangulate.
                        matrix::matrix<double, 3, 1> result = matrix::matrix<double, 3, 1>::zero();
                        bool valid = geometry::triangulate(
                            matrix::matrix<double, 2, 1>(&lhs_points[i * 2]),
                            matrix::matrix<double, 3, 4>(&lhs_pose[0][0]),
                            matrix::matrix<double, 2, 1>(&rhs_points[i * 2]),
                            matrix::matrix<double, 3, 4>(&rhs_pose[0][0]),
                            result
                        );
                        trigulated_points_all[r * 2 + t][i * 3 + 0] = result[0];
                        trigulated_points_all[r * 2 + t][i * 3 + 1] = result[1];
                        trigulated_points_all[r * 2 + t][i * 3 + 2] = result[2];
                        // Check the point is infront of the camera.
                        valid &= (trigulated_points_all[r * 2 + t][i * 3 + 2] > 0);
                        // Check the point is not too far away.
                        valid &= (trigulated_points_all[r * 2 + t][i * 3 + 2] < distance_threshold);
                        if (!valid) {
                            continue;
                        }
                        // Reproject the point using the potential transformation.
                        const type homogeneous[4] = {
                            trigulated_points_all[r * 2 + t][i * 3 + 0],
                            trigulated_points_all[r * 2 + t][i * 3 + 1],
                            trigulated_points_all[r * 2 + t][i * 3 + 2],
                            1
                        };
                        type reprojected[3];
                        matrix_multiply(&rhs_pose[0][0], 4, 3, &homogeneous[0], 1, 4, &reprojected[0]);
                        // Check the point is infront of the camera.
                        valid &= (reprojected[2] > 0);
                        // Check the point is not too far away.
                        valid &= (reprojected[2] < distance_threshold);
                        // If it's still valid add one to the count.
                        trigulated_points_valid[r * 2 + t] += valid;
                    }
                }
            }

            // Find best.
            size_t index_best = 0;
            for (size_t index = 1; index < 4; ++index) {
                if (trigulated_points_valid[index] > trigulated_points_valid[index_best]) {
                    index_best = index;
                }
            }

            // Return best transformation and triangulated points.
            for (size_t y = 0; y < 3; ++y) {
                for (size_t x = 0; x < 3; ++x) {
                    rotation[y * 3 + x] = rotations[index_best / 2][y][x];
                }
                translation[y] = translations[index_best % 2][y];
            }
            for (size_t i = 0; i < point_count; ++i) {
                trigulated_points[i * 3 + 0] = trigulated_points_all[index_best][i * 3 + 0];
                trigulated_points[i * 3 + 1] = trigulated_points_all[index_best][i * 3 + 1];
                trigulated_points[i * 3 + 2] = trigulated_points_all[index_best][i * 3 + 2];
            }

            delete[] trigulated_points_all[0];
            delete[] trigulated_points_all[1];
            delete[] trigulated_points_all[2];
            delete[] trigulated_points_all[3];
        }
    };
}

#endif // POSE_ESTIMATION_HPP
