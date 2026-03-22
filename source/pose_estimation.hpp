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
    static inline int perspective_3_point(
        const type image_points_in[3][3],
        const type world_points_in[3][3],
        type rotations[4][9],
        type translations[4][3]
    ) {
        // Solves x^2 + b*x + c = 0 for real roots.
        // Returns true if real roots exist.
        constexpr static const auto solve_quadratic_real_roots = [](
            const type b,
            const type c,
            type& r1,
            type& r2
        ) -> bool {
            const type threshold = type(-1.0e-12);
            const type v = b * b - type(4.0) * c;
            if (v < threshold) {
                r1 = r2 = type(-0.5) * b;
                return v >= type(0);
            }
            if (v > threshold && v < type(0)) {
                r1 = type(-0.5) * b;
                r2 = type(-2);
                return true;
            }
            const type y = math::sqrt(v);
            if (b < type(0)) {
                r1 = type(0.5) * (-b + y);
                r2 = type(0.5) * (-b - y);
            } else {
                r1 = type(2.0) * c / (-b + y);
                r2 = type(2.0) * c / (-b - y);
            }
            return true;
        };

        // Finds a single real root of x^3 + c2*x^2 + c1*x + c0 = 0.
        // Returns true if the cubic has a single real root (discriminant > 0).
        constexpr static const auto solve_cubic_single_real_root = [](
            const type c2,
            const type c1,
            const type c0,
            type& root
        ) -> bool {
            const type a = c1 - c2 * c2 / type(3.0);
            const type b = (type(2.0) * c2 * c2 * c2 - type(9.0) * c2 * c1) / type(27.0) + c0;
            type c = b * b / type(4.0) + a * a * a / type(27.0);
            if (c != type(0)) {
                if (c > type(0)) {
                    c = math::sqrt(c);
                    const type bh = -b * type(0.5);
                    const type val1 = bh + c;
                    const type val2 = bh - c;
                    const type cbrt1 = math::copysign(math::pow(math::abs(val1), type(1.0) / type(3.0)), val1);
                    const type cbrt2 = math::copysign(math::pow(math::abs(val2), type(1.0) / type(3.0)), val2);
                    root = cbrt1 + cbrt2 - c2 / type(3.0);
                    return true;
                } else {
                    c = type(3.0) * b / (type(2.0) * a) * math::sqrt(type(-3.0) / a);
                    root = type(2.0) * math::sqrt(-a / type(3.0)) * math::cos(math::acos(c) / type(3.0)) - c2 / type(3.0);
                }
            } else {
                root = -c2 / type(3.0) + (a != type(0) ? (type(3.0) * b / a) : type(0));
            }
            return false;
        };

        // Computes the null-space factorization of C via its adjugate.
        // Returns 2 vectors p and q such that C has rank 1 in the null space.
        // pq[2][3] receives the two 3-vectors. C is a 3x3 matrix (modified in-place).
        constexpr static const auto compute_null_space_factorization = [](
            type matrix_c[3][3],
            type pq[2][3]
        ) -> void {
            type matrix_c_adj[3][3];

            matrix_c_adj[0][0] = matrix_c[1][2] * matrix_c[2][1] - matrix_c[1][1] * matrix_c[2][2];
            matrix_c_adj[1][1] = matrix_c[0][2] * matrix_c[2][0] - matrix_c[0][0] * matrix_c[2][2];
            matrix_c_adj[2][2] = matrix_c[0][1] * matrix_c[1][0] - matrix_c[0][0] * matrix_c[1][1];
            matrix_c_adj[0][1] = matrix_c[0][1] * matrix_c[2][2] - matrix_c[0][2] * matrix_c[2][1];
            matrix_c_adj[0][2] = matrix_c[0][2] * matrix_c[1][1] - matrix_c[0][1] * matrix_c[1][2];
            matrix_c_adj[1][0] = matrix_c_adj[0][1];
            matrix_c_adj[1][2] = matrix_c[0][0] * matrix_c[1][2] - matrix_c[0][2] * matrix_c[1][0];
            matrix_c_adj[2][0] = matrix_c_adj[0][2];
            matrix_c_adj[2][1] = matrix_c_adj[1][2];

            type v[3];
            if (matrix_c_adj[0][0] > matrix_c_adj[1][1]) {
                if (matrix_c_adj[0][0] > matrix_c_adj[2][2]) {
                    const type inv = type(1.0) / math::sqrt(matrix_c_adj[0][0]);
                    v[0] = matrix_c_adj[0][0] * inv;
                    v[1] = matrix_c_adj[1][0] * inv;
                    v[2] = matrix_c_adj[2][0] * inv;
                } else {
                    const type inv = type(1.0) / math::sqrt(matrix_c_adj[2][2]);
                    v[0] = matrix_c_adj[0][2] * inv;
                    v[1] = matrix_c_adj[1][2] * inv;
                    v[2] = matrix_c_adj[2][2] * inv;
                }
            } else if (matrix_c_adj[1][1] > matrix_c_adj[2][2]) {
                const type inv = type(1.0) / math::sqrt(matrix_c_adj[1][1]);
                v[0] = matrix_c_adj[0][1] * inv;
                v[1] = matrix_c_adj[1][1] * inv;
                v[2] = matrix_c_adj[2][1] * inv;
            } else {
                const type inv = type(1.0) / math::sqrt(matrix_c_adj[2][2]);
                v[0] = matrix_c_adj[0][2] * inv;
                v[1] = matrix_c_adj[1][2] * inv;
                v[2] = matrix_c_adj[2][2] * inv;
            }

            matrix_c[0][1] -= v[2];
            matrix_c[0][2] += v[1];
            matrix_c[1][2] -= v[0];
            matrix_c[1][0] += v[2];
            matrix_c[2][0] -= v[1];
            matrix_c[2][1] += v[0];

            pq[0][0] = matrix_c[0][0]; pq[0][1] = matrix_c[1][0]; pq[0][2] = matrix_c[2][0];
            pq[1][0] = matrix_c[0][0]; pq[1][1] = matrix_c[0][1]; pq[1][2] = matrix_c[0][2];
        };

        // Performs Newton refinement on the distance constraints.
        constexpr static const auto refine_distances_newton = [](
            type& lambda1,
            type& lambda2,
            type& lambda3,
            const type a12,
            const type a13,
            const type a23,
            const type b12,
            const type b13,
            const type b23
        ) -> void {
            for (int iter = 0; iter < 5; ++iter) {
                const type r1 = (lambda1 * lambda1 - type(2.0) * lambda1 * lambda2 * b12 + lambda2 * lambda2 - a12);
                const type r2 = (lambda1 * lambda1 - type(2.0) * lambda1 * lambda3 * b13 + lambda3 * lambda3 - a13);
                const type r3 = (lambda2 * lambda2 - type(2.0) * lambda2 * lambda3 * b23 + lambda3 * lambda3 - a23);
                if (math::abs(r1) + math::abs(r2) + math::abs(r3) < type(1.0e-10)) {
                    return;
                }
                const type x11 = lambda1 - lambda2 * b12;
                const type x12 = lambda2 - lambda1 * b12;
                const type x21 = lambda1 - lambda3 * b13;
                const type x23 = lambda3 - lambda1 * b13;
                const type x32 = lambda2 - lambda3 * b23;
                const type x33 = lambda3 - lambda2 * b23;
                const type determinant_j_inv = type(0.5) / (x11 * x23 * x32 + x12 * x21 * x33);
                lambda1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * determinant_j_inv;
                lambda2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * determinant_j_inv;
                lambda3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * determinant_j_inv;
            }
        };
        type world_points[3][3];
        type image_points[3][3];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                world_points[i][j] = world_points_in[i][j];
                image_points[i][j] = image_points_in[i][j];
            }
        }

        type world_vector_01[3] = {
            world_points[0][0] - world_points[1][0],
            world_points[0][1] - world_points[1][1],
            world_points[0][2] - world_points[1][2]
        };
        type world_vector_02[3] = {
            world_points[0][0] - world_points[2][0],
            world_points[0][1] - world_points[2][1],
            world_points[0][2] - world_points[2][2]
        };
        type world_vector_12[3] = {
            world_points[1][0] - world_points[2][0],
            world_points[1][1] - world_points[2][1],
            world_points[1][2] - world_points[2][2]
        };

        type world_dist_sq_01 = world_vector_01[0] * world_vector_01[0] + world_vector_01[1] * world_vector_01[1] + world_vector_01[2] * world_vector_01[2];
        type world_dist_sq_02 = world_vector_02[0] * world_vector_02[0] + world_vector_02[1] * world_vector_02[1] + world_vector_02[2] * world_vector_02[2];
        type world_dist_sq_12 = world_vector_12[0] * world_vector_12[0] + world_vector_12[1] * world_vector_12[1] + world_vector_12[2] * world_vector_12[2];

        // Switch world_points, image_points so that BC (indices 1,2) is the largest distance.
        if (world_dist_sq_01 > world_dist_sq_02) {
            if (world_dist_sq_01 > world_dist_sq_12) {
                // Swap points 0 and 2.
                for (int j = 0; j < 3; ++j) {
                    const type temp_image = image_points[0][j];
                    image_points[0][j] = image_points[2][j];
                    image_points[2][j] = temp_image;
                }
                for (int j = 0; j < 3; ++j) {
                    const type temp_world = world_points[0][j];
                    world_points[0][j] = world_points[2][j];
                    world_points[2][j] = temp_world;
                }
                const type temp_dist = world_dist_sq_01;
                world_dist_sq_01 = world_dist_sq_12;
                world_dist_sq_12 = temp_dist;
                // world_vector_01 = -world_vector_12, world_vector_02 = -world_vector_02
                for (int j = 0; j < 3; ++j) {
                    world_vector_01[j] = -world_vector_12[j];
                    world_vector_02[j] = -world_vector_02[j];
                }
            }
        } else if (world_dist_sq_02 > world_dist_sq_12) {
            // Swap points 0 and 1.
            for (int j = 0; j < 3; ++j) {
                const type temp_image = image_points[0][j];
                image_points[0][j] = image_points[1][j];
                image_points[1][j] = temp_image;
            }
            for (int j = 0; j < 3; ++j) {
                const type temp_world = world_points[0][j];
                world_points[0][j] = world_points[1][j];
                world_points[1][j] = temp_world;
            }
            const type temp_dist = world_dist_sq_02;
            world_dist_sq_02 = world_dist_sq_12;
            world_dist_sq_12 = temp_dist;
            // world_vector_01 = -world_vector_01, world_vector_02 = world_vector_12
            for (int j = 0; j < 3; ++j) {
                world_vector_01[j] = -world_vector_01[j];
                world_vector_02[j] = world_vector_12[j];
            }
        }

        const type world_dist_sq_12_inv = type(1.0) / world_dist_sq_12;
        const type ratio_a = world_dist_sq_01 * world_dist_sq_12_inv;
        const type ratio_b = world_dist_sq_02 * world_dist_sq_12_inv;

        const type image_cos_01 = image_points[0][0] * image_points[1][0] + image_points[0][1] * image_points[1][1] + image_points[0][2] * image_points[1][2];
        const type image_cos_02 = image_points[0][0] * image_points[2][0] + image_points[0][1] * image_points[2][1] + image_points[0][2] * image_points[2][2];
        const type image_cos_12 = image_points[1][0] * image_points[2][0] + image_points[1][1] * image_points[2][1] + image_points[1][2] * image_points[2][2];

        const type image_sin_sq_12 = type(1.0) - image_cos_12 * image_cos_12;
        const type image_cos_sq_02_minus_1 = image_cos_02 * image_cos_02 - type(1.0);
        const type image_cos_sq_01_minus_1 = image_cos_01 * image_cos_01 - type(1.0);
        const type ratio_ab = ratio_a * ratio_b;
        const type ratio_b_sq = ratio_b * ratio_b;
        const type ratio_a_sq = ratio_a * ratio_a;
        const type term_m013 = type(-2.0) + type(2.0) * image_cos_01 * image_cos_02 * image_cos_12;
        const type ratio_b_sq_sin_sq_12 = ratio_b_sq * image_sin_sq_12;
        const type ratio_a_sq_sin_sq_12 = ratio_a_sq * image_sin_sq_12;
        const type ratio_ab_sin_sq_12_2 = type(2.0) * ratio_ab * image_sin_sq_12;

        const type k3_inv = type(1.0) / (ratio_b_sq_sin_sq_12 + ratio_b * image_cos_sq_02_minus_1);
        const type k2 = k3_inv * ((type(-1.0) + ratio_a) * image_cos_sq_02_minus_1 + ratio_ab_sin_sq_12_2 + ratio_b_sq_sin_sq_12 + ratio_b * term_m013);
        const type k1 = k3_inv * (ratio_a_sq_sin_sq_12 + ratio_ab_sin_sq_12_2 + ratio_a * term_m013 + (type(-1.0) + ratio_b) * image_cos_sq_01_minus_1);
        const type k0 = k3_inv * (ratio_a_sq_sin_sq_12 + ratio_a * image_cos_sq_01_minus_1);

        type real_root_s;
        const bool found_single_real_root = solve_cubic_single_real_root(k2, k1, k0, real_root_s);

        type matrix_c[3][3];
        matrix_c[0][0] = -ratio_a + real_root_s * (type(1.0) - ratio_b);
        matrix_c[0][1] = -image_cos_02 * real_root_s;
        matrix_c[0][2] = ratio_a * image_cos_12 + ratio_b * image_cos_12 * real_root_s;
        matrix_c[1][0] = matrix_c[0][1];
        matrix_c[1][1] = real_root_s + type(1.0);
        matrix_c[1][2] = -image_cos_01;
        matrix_c[2][0] = matrix_c[0][2];
        matrix_c[2][1] = matrix_c[1][2];
        matrix_c[2][2] = -ratio_a - ratio_b * real_root_s + type(1.0);

        type pq_factors[2][3];
        compute_null_space_factorization(matrix_c, pq_factors);

        // Compute world_basis_matrix = [world_vector_01 | world_vector_02 | world_vector_01 x world_vector_02] and its inverse.
        type world_vector_01_cross_02[3];
        world_vector_01_cross_02[0] = world_vector_01[1] * world_vector_02[2] - world_vector_01[2] * world_vector_02[1];
        world_vector_01_cross_02[1] = world_vector_01[2] * world_vector_02[0] - world_vector_01[0] * world_vector_02[2];
        world_vector_01_cross_02[2] = world_vector_01[0] * world_vector_02[1] - world_vector_01[1] * world_vector_02[0];

        // world_basis_matrix is a 3x3 matrix with columns world_vector_01, world_vector_02, world_vector_01x02 (stored row-major).
        type world_basis_matrix[3][3] = {
            { world_vector_01[0], world_vector_02[0], world_vector_01_cross_02[0] },
            { world_vector_01[1], world_vector_02[1], world_vector_01_cross_02[1] },
            { world_vector_01[2], world_vector_02[2], world_vector_01_cross_02[2] }
        };

        // Compute inverse of world_basis_matrix using the 3x3 formula.
        const type det_world_basis =
            world_basis_matrix[0][0] * (world_basis_matrix[1][1] * world_basis_matrix[2][2] - world_basis_matrix[2][1] * world_basis_matrix[1][2])
          - world_basis_matrix[0][1] * (world_basis_matrix[1][0] * world_basis_matrix[2][2] - world_basis_matrix[1][2] * world_basis_matrix[2][0])
          + world_basis_matrix[0][2] * (world_basis_matrix[1][0] * world_basis_matrix[2][1] - world_basis_matrix[1][1] * world_basis_matrix[2][0]);

        if (math::abs(det_world_basis) < type(1.0e-15)) {
            return 0;
        }

        const type det_world_basis_inv = type(1.0) / det_world_basis;
        type world_basis_matrix_inv[3][3];
        world_basis_matrix_inv[0][0] = (world_basis_matrix[1][1] * world_basis_matrix[2][2] - world_basis_matrix[2][1] * world_basis_matrix[1][2]) * det_world_basis_inv;
        world_basis_matrix_inv[0][1] = (world_basis_matrix[0][2] * world_basis_matrix[2][1] - world_basis_matrix[0][1] * world_basis_matrix[2][2]) * det_world_basis_inv;
        world_basis_matrix_inv[0][2] = (world_basis_matrix[0][1] * world_basis_matrix[1][2] - world_basis_matrix[0][2] * world_basis_matrix[1][1]) * det_world_basis_inv;
        world_basis_matrix_inv[1][0] = (world_basis_matrix[1][2] * world_basis_matrix[2][0] - world_basis_matrix[1][0] * world_basis_matrix[2][2]) * det_world_basis_inv;
        world_basis_matrix_inv[1][1] = (world_basis_matrix[0][0] * world_basis_matrix[2][2] - world_basis_matrix[0][2] * world_basis_matrix[2][0]) * det_world_basis_inv;
        world_basis_matrix_inv[1][2] = (world_basis_matrix[1][0] * world_basis_matrix[0][2] - world_basis_matrix[0][0] * world_basis_matrix[1][2]) * det_world_basis_inv;
        world_basis_matrix_inv[2][0] = (world_basis_matrix[1][0] * world_basis_matrix[2][1] - world_basis_matrix[2][0] * world_basis_matrix[1][1]) * det_world_basis_inv;
        world_basis_matrix_inv[2][1] = (world_basis_matrix[2][0] * world_basis_matrix[0][1] - world_basis_matrix[0][0] * world_basis_matrix[2][1]) * det_world_basis_inv;
        world_basis_matrix_inv[2][2] = (world_basis_matrix[0][0] * world_basis_matrix[1][1] - world_basis_matrix[1][0] * world_basis_matrix[0][1]) * det_world_basis_inv;

        int num_solutions = 0;

        for (int i = 0; i < 2; ++i) {
            const type factor_p0 = pq_factors[i][0];
            const type factor_p1 = pq_factors[i][1];
            const type factor_p2 = pq_factors[i][2];

            const bool should_switch_12 = math::abs(factor_p0) <= math::abs(factor_p1);

            if (should_switch_12) {
                // Eliminate d0.
                const type weight_w0 = -factor_p0 / factor_p1;
                const type weight_w1 = -factor_p2 / factor_p1;
                const type coeff_ca = type(1.0) / (weight_w1 * weight_w1 - ratio_b);
                const type coeff_cb = type(2.0) * (ratio_b * image_cos_12 - image_cos_02 * weight_w1 + weight_w0 * weight_w1) * coeff_ca;
                const type coeff_cc = (weight_w0 * weight_w0 - type(2.0) * image_cos_02 * weight_w0 - ratio_b + type(1.0)) * coeff_ca;
                type solution_taus[2];
                if (!solve_quadratic_real_roots(coeff_cb, coeff_cc, solution_taus[0], solution_taus[1])) {
                    continue;
                }
                for (int ti = 0; ti < 2; ++ti) {
                    const type tau = solution_taus[ti];
                    if (tau <= type(0)) {
                        continue;
                    }
                    type dist_d2 = math::sqrt(world_dist_sq_12 / (tau * (tau - type(2.0) * image_cos_12) + type(1.0)));
                    type dist_d1 = tau * dist_d2;
                    type dist_d0 = weight_w0 * dist_d2 + weight_w1 * dist_d1;
                    if (dist_d0 < type(0)) {
                        continue;
                    }

                    refine_distances_newton(dist_d0, dist_d1, dist_d2, world_dist_sq_01, world_dist_sq_02, world_dist_sq_12, image_cos_01, image_cos_02, image_cos_12);

                    // vector_v1 = dist_d0*image_points[0] - dist_d1*image_points[1], vector_v2 = dist_d0*image_points[0] - dist_d2*image_points[2]
                    type vector_v1[3];
                    type vector_v2[3];
                    for (int j = 0; j < 3; ++j) {
                        vector_v1[j] = dist_d0 * image_points[0][j] - dist_d1 * image_points[1][j];
                        vector_v2[j] = dist_d0 * image_points[0][j] - dist_d2 * image_points[2][j];
                    }
                    // vector_v1_cross_v2
                    type vector_v1_cross_v2[3];
                    vector_v1_cross_v2[0] = vector_v1[1] * vector_v2[2] - vector_v1[2] * vector_v2[1];
                    vector_v1_cross_v2[1] = vector_v1[2] * vector_v2[0] - vector_v1[0] * vector_v2[2];
                    vector_v1_cross_v2[2] = vector_v1[0] * vector_v2[1] - vector_v1[1] * vector_v2[0];

                    // image_basis_matrix has columns vector_v1, vector_v2, vector_v1_cross_v2.
                    type image_basis_matrix[3][3] = {
                        { vector_v1[0], vector_v2[0], vector_v1_cross_v2[0] },
                        { vector_v1[1], vector_v2[1], vector_v1_cross_v2[1] },
                        { vector_v1[2], vector_v2[2], vector_v1_cross_v2[2] }
                    };

                    // rotation = image_basis_matrix * world_basis_matrix_inv
                    type solution_rotation[9];
                    for (int row = 0; row < 3; ++row) {
                        for (int col = 0; col < 3; ++col) {
                            solution_rotation[row * 3 + col] = image_basis_matrix[row][0] * world_basis_matrix_inv[0][col] + image_basis_matrix[row][1] * world_basis_matrix_inv[1][col] + image_basis_matrix[row][2] * world_basis_matrix_inv[2][col];
                        }
                    }

                    // translation = dist_d0*image_points[0] - rotation*world_points[0]
                    type solution_translation[3];
                    for (int row = 0; row < 3; ++row) {
                        solution_translation[row] = dist_d0 * image_points[0][row] - (solution_rotation[row * 3 + 0] * world_points[0][0] + solution_rotation[row * 3 + 1] * world_points[0][1] + solution_rotation[row * 3 + 2] * world_points[0][2]);
                    }

                    for (int k = 0; k < 9; ++k) rotations[num_solutions][k] = solution_rotation[k];
                    for (int k = 0; k < 3; ++k) translations[num_solutions][k] = solution_translation[k];
                    ++num_solutions;
                }
            }
            else {
                const type weight_w0 = -factor_p1 / factor_p0;
                const type weight_w1 = -factor_p2 / factor_p0;
                const type coeff_ca = type(1.0) / (-ratio_a * weight_w1 * weight_w1 + type(2.0) * ratio_a * image_cos_12 * weight_w1 - ratio_a + type(1.0));
                const type coeff_cb = type(2.0) * (ratio_a * image_cos_12 * weight_w0 - image_cos_01 - ratio_a * weight_w0 * weight_w1) * coeff_ca;
                const type coeff_cc = (type(1.0) - ratio_a * weight_w0 * weight_w0) * coeff_ca;
                type solution_taus[2];
                if (!solve_quadratic_real_roots(coeff_cb, coeff_cc, solution_taus[0], solution_taus[1])) {
                    continue;
                }
                for (int ti = 0; ti < 2; ++ti) {
                    const type tau = solution_taus[ti];
                    if (tau <= type(0)) {
                        continue;
                    }
                    type dist_d0 = math::sqrt(world_dist_sq_01 / (tau * (tau - type(2.0) * image_cos_01) + type(1.0)));
                    type dist_d1 = tau * dist_d0;
                    type dist_d2 = weight_w0 * dist_d0 + weight_w1 * dist_d1;

                    if (dist_d2 < type(0)) {
                        continue;
                    }

                    refine_distances_newton(dist_d0, dist_d1, dist_d2, world_dist_sq_01, world_dist_sq_02, world_dist_sq_12, image_cos_01, image_cos_02, image_cos_12);

                    type vector_v1[3];
                    type vector_v2[3];
                    for (int j = 0; j < 3; ++j) {
                        vector_v1[j] = dist_d0 * image_points[0][j] - dist_d1 * image_points[1][j];
                        vector_v2[j] = dist_d0 * image_points[0][j] - dist_d2 * image_points[2][j];
                    }
                    type vector_v1_cross_v2[3];
                    vector_v1_cross_v2[0] = vector_v1[1] * vector_v2[2] - vector_v1[2] * vector_v2[1];
                    vector_v1_cross_v2[1] = vector_v1[2] * vector_v2[0] - vector_v1[0] * vector_v2[2];
                    vector_v1_cross_v2[2] = vector_v1[0] * vector_v2[1] - vector_v1[1] * vector_v2[0];

                    type image_basis_matrix[3][3] = {
                        { vector_v1[0], vector_v2[0], vector_v1_cross_v2[0] },
                        { vector_v1[1], vector_v2[1], vector_v1_cross_v2[1] },
                        { vector_v1[2], vector_v2[2], vector_v1_cross_v2[2] }
                    };

                    type solution_rotation[9];
                    for (int row = 0; row < 3; ++row) {
                        for (int col = 0; col < 3; ++col) {
                            solution_rotation[row * 3 + col] = image_basis_matrix[row][0] * world_basis_matrix_inv[0][col] + image_basis_matrix[row][1] * world_basis_matrix_inv[1][col] + image_basis_matrix[row][2] * world_basis_matrix_inv[2][col];
                        }
                    }

                    type solution_translation[3];
                    for (int row = 0; row < 3; ++row) {
                        solution_translation[row] = dist_d0 * image_points[0][row] - (solution_rotation[row * 3 + 0] * world_points[0][0] + solution_rotation[row * 3 + 1] * world_points[0][1] + solution_rotation[row * 3 + 2] * world_points[0][2]);
                    }

                    for (int k = 0; k < 9; ++k) rotations[num_solutions][k] = solution_rotation[k];
                    for (int k = 0; k < 3; ++k) translations[num_solutions][k] = solution_translation[k];
                    ++num_solutions;
                }
            }

            if (num_solutions > 0 && found_single_real_root) {
                break;
            }
        }

        return num_solutions;
    }

    template <typename type>
    class essential_matrix final {
    private:
        constexpr static void matrix_multiply(
            const type* const __restrict lhs_matrix,
            const type* const __restrict rhs_matrix,
            type* const __restrict result_matrix
        ) {
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    result_matrix[row * 3 + col] = type(0);
                }
                for (int k = 0; k < 3; ++k) {
                    for (int col = 0; col < 3; ++col) {
                        result_matrix[row * 3 + col] += lhs_matrix[row * 3 + k] * rhs_matrix[k * 3 + col];
                    }
                }
            }
        }

        constexpr static void decompose_essential(
            const type* const __restrict essential,
            type* const __restrict matrix_u,
            type* const __restrict matrix_vt
        ) {
            const type essential_cross_0[3] = {
                essential[1] * essential[5] - essential[2] * essential[4],
                essential[2] * essential[3] - essential[0] * essential[5],
                essential[0] * essential[4] - essential[1] * essential[3]
            };
            const type essential_cross_1[3] = {
                essential[1] * essential[8] - essential[2] * essential[7],
                essential[2] * essential[6] - essential[0] * essential[8],
                essential[0] * essential[7] - essential[1] * essential[6]
            };
            const type essential_cross_2[3] = {
                essential[4] * essential[8] - essential[5] * essential[7],
                essential[5] * essential[6] - essential[3] * essential[8],
                essential[3] * essential[7] - essential[4] * essential[6]
            };

            const type matrix_vt_factor_0 = type(1.0) / math::sqrt(essential_cross_0[0] * essential_cross_0[0] + essential_cross_0[1] * essential_cross_0[1] + essential_cross_0[2] * essential_cross_0[2]);
            const type matrix_vt_factor_1 = type(1.0) / math::sqrt(essential_cross_1[0] * essential_cross_1[0] + essential_cross_1[1] * essential_cross_1[1] + essential_cross_1[2] * essential_cross_1[2]);
            const type matrix_vt_factor_2 = type(1.0) / math::sqrt(essential_cross_2[0] * essential_cross_2[0] + essential_cross_2[1] * essential_cross_2[1] + essential_cross_2[2] * essential_cross_2[2]);

            if ((matrix_vt_factor_0 <= matrix_vt_factor_1) && (matrix_vt_factor_0 <= matrix_vt_factor_2)) {
                for (int i = 6; i < 9; ++i) {
                    matrix_vt[i] = essential_cross_0[i - 6] * matrix_vt_factor_0;
                }
            } else if ((matrix_vt_factor_1 <= matrix_vt_factor_0) && (matrix_vt_factor_1 <= matrix_vt_factor_2)) {
                for (int i = 6; i < 9; ++i) {
                    matrix_vt[i] = essential_cross_1[i - 6] * matrix_vt_factor_1;
                }
            } else {
                for (int i = 6; i < 9; ++i) {
                    matrix_vt[i] = essential_cross_2[i - 6] * matrix_vt_factor_2;
                }
            }

            const type matrix_vt_row_v0_normalization = type(1.0) / math::sqrt(essential[0] * essential[0] + essential[1] * essential[1] + essential[2] * essential[2]);
            matrix_vt[0] = essential[0] * matrix_vt_row_v0_normalization;
            matrix_vt[1] = essential[1] * matrix_vt_row_v0_normalization;
            matrix_vt[2] = essential[2] * matrix_vt_row_v0_normalization;

            matrix_vt[3] = matrix_vt[7] * matrix_vt[2] - matrix_vt[8] * matrix_vt[1];
            matrix_vt[4] = matrix_vt[8] * matrix_vt[0] - matrix_vt[6] * matrix_vt[2];
            matrix_vt[5] = matrix_vt[6] * matrix_vt[1] - matrix_vt[7] * matrix_vt[0];

            matrix_u[0] = essential[0] * matrix_vt[0] + essential[1] * matrix_vt[1] + essential[2] * matrix_vt[2];
            matrix_u[3] = essential[3] * matrix_vt[0] + essential[4] * matrix_vt[1] + essential[5] * matrix_vt[2];
            matrix_u[6] = essential[6] * matrix_vt[0] + essential[7] * matrix_vt[1] + essential[8] * matrix_vt[2];
            matrix_u[1] = essential[0] * matrix_vt[3] + essential[1] * matrix_vt[4] + essential[2] * matrix_vt[5];
            matrix_u[4] = essential[3] * matrix_vt[3] + essential[4] * matrix_vt[4] + essential[5] * matrix_vt[5];
            matrix_u[7] = essential[6] * matrix_vt[3] + essential[7] * matrix_vt[4] + essential[8] * matrix_vt[5];

            const type matrix_u_col_u0_normalization = type(1.0) / math::sqrt(matrix_u[0] * matrix_u[0] + matrix_u[3] * matrix_u[3] + matrix_u[6] * matrix_u[6]);
            const type matrix_u_col_u1_normalization = type(1.0) / math::sqrt(matrix_u[1] * matrix_u[1] + matrix_u[4] * matrix_u[4] + matrix_u[7] * matrix_u[7]);

            matrix_u[0] *= matrix_u_col_u0_normalization;
            matrix_u[3] *= matrix_u_col_u0_normalization;
            matrix_u[6] *= matrix_u_col_u0_normalization;
            matrix_u[1] *= matrix_u_col_u1_normalization;
            matrix_u[4] *= matrix_u_col_u1_normalization;
            matrix_u[7] *= matrix_u_col_u1_normalization;

            matrix_u[2] = matrix_u[3] * matrix_u[7] - matrix_u[6] * matrix_u[4];
            matrix_u[5] = matrix_u[6] * matrix_u[1] - matrix_u[0] * matrix_u[7];
            matrix_u[8] = matrix_u[0] * matrix_u[4] - matrix_u[3] * matrix_u[1];
        }

    public:
        static void compose(
            const type* const __restrict rotation,
            const type* const __restrict translation,
            type* const __restrict essential
        ) {
            const type translation_length_inv = type(1.0) / math::sqrt(translation[0] * translation[0] + translation[1] * translation[1] + translation[2] * translation[2]);
            const type translation_normalized[3] = {
                translation[0] * translation_length_inv,
                translation[1] * translation_length_inv,
                translation[2] * translation_length_inv
            };
            const type translation_skew[3][3] = {
                { type(0), -translation_normalized[2], translation_normalized[1] },
                { translation_normalized[2], type(0), -translation_normalized[0] },
                { -translation_normalized[1], translation_normalized[0], type(0) }
            };
            matrix_multiply(&translation_skew[0][0], rotation, essential);
        }

        static void decompose(
            const type* const __restrict essential,
            type* const __restrict rotation_0,
            type* const __restrict rotation_1,
            type* const __restrict translation_0,
            type* const __restrict translation_1
        ) {
            type matrix_u[3][3];
            type matrix_vt[3][3];
            decompose_essential(&essential[0], &matrix_u[0][0], &matrix_vt[0][0]);

            const type determinant_u = matrix_u[0][0] * (matrix_u[1][1] * matrix_u[2][2] - matrix_u[2][1] * matrix_u[1][2]) - matrix_u[0][1] * (matrix_u[1][0] * matrix_u[2][2] - matrix_u[1][2] * matrix_u[2][0]) + matrix_u[0][2] * (matrix_u[1][0] * matrix_u[2][1] - matrix_u[1][1] * matrix_u[2][0]);
            if (determinant_u < type(0)) {
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        matrix_u[row][col] *= type(-1);
                    }
                }
            }

            const type determinant_vt = matrix_vt[0][0] * (matrix_vt[1][1] * matrix_vt[2][2] - matrix_vt[2][1] * matrix_vt[1][2]) - matrix_vt[0][1] * (matrix_vt[1][0] * matrix_vt[2][2] - matrix_vt[1][2] * matrix_vt[2][0]) + matrix_vt[0][2] * (matrix_vt[1][0] * matrix_vt[2][1] - matrix_vt[1][1] * matrix_vt[2][0]);
            if (determinant_vt < type(0)) {
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        matrix_vt[row][col] *= type(-1);
                    }
                }
            }

            const type matrix_w[3][3] = {
                { type(0), type(1), type(0) },
                { type(-1), type(0), type(0) },
                { type(0), type(0), type(1) },
            };

            const type matrix_wt[3][3] = {
                { type(0), type(-1), type(0) },
                { type(1), type(0), type(0) },
                { type(0), type(0), type(1) },
            };

            type matrix_uw[3][3];
            matrix_multiply(&matrix_u[0][0], &matrix_w[0][0], &matrix_uw[0][0]);
            matrix_multiply(&matrix_uw[0][0], &matrix_vt[0][0], &rotation_0[0]);

            type matrix_uwt[3][3];
            matrix_multiply(&matrix_u[0][0], &matrix_wt[0][0], &matrix_uwt[0][0]);
            matrix_multiply(&matrix_uwt[0][0], &matrix_vt[0][0], &rotation_1[0]);

            translation_0[0] = matrix_u[0][2];
            translation_0[1] = matrix_u[1][2];
            translation_0[2] = matrix_u[2][2];

            translation_1[0] = -matrix_u[0][2];
            translation_1[1] = -matrix_u[1][2];
            translation_1[2] = -matrix_u[2][2];
        }

        // Note: Assuming all points are normalised.
        static void recover_pose(
            const type* const __restrict essential,
            const type* const __restrict lhs_points, // X, Y, X, Y, X, Y, ...
            const type* const __restrict rhs_points, // X, Y, X, Y, X, Y, ...
            const size_t point_count,
            type* const __restrict rotation,
            type* const __restrict translation,
            type* const __restrict triangulated_points // X, Y, Z, X, Y, Z, X, Y, Z, ...
        ) {
            constexpr static const type distance_threshold = type(50);

            constexpr static const auto matrix_multiply_local = [](
                const type* const __restrict lhs_matrix,
                const int lhs_width,
                const int lhs_height,
                const type* const __restrict rhs_matrix,
                const int rhs_width,
                const int rhs_height,
                type* const __restrict result_matrix
            ) -> void {
                static_cast<void>(rhs_height);
                for (int row = 0; row < lhs_height; ++row) {
                    for (int col = 0; col < rhs_width; ++col) {
                        type sum = type(0);
                        for (int k = 0; k < lhs_width; ++k) {
                            sum += lhs_matrix[row * lhs_width + k] * rhs_matrix[k * rhs_width + col];
                        }
                        result_matrix[row * rhs_width + col] = sum;
                    }
                }
            };

            // Extract the four possible transformations.
            type rotations[2][3][3];
            type translations[2][3];
            essential_matrix::decompose(essential, &rotations[0][0][0], &rotations[1][0][0], &translations[0][0], &translations[1][0]);

            // Generate a lhs pose:
            const type lhs_camera_pose[3][4] = {
                { type(1), type(0), type(0), type(0) },
                { type(0), type(1), type(0), type(0) },
                { type(0), type(0), type(1), type(0) },
            };

            // Triangulate all points using the four transformations.
            type* const triangulated_points_set[4] = {
                new type[point_count * 3]{},
                new type[point_count * 3]{},
                new type[point_count * 3]{},
                new type[point_count * 3]{}
            };

            size_t triangulated_points_valid_counts[4] = {
                size_t(0),
                size_t(0),
                size_t(0),
                size_t(0)
            };

            for (size_t rotation_index = 0; rotation_index < 2; ++rotation_index) {
                for (size_t translation_index = 0; translation_index < 2; ++translation_index) {
                    const size_t solution_index = rotation_index * 2 + translation_index;
                    const type rhs_camera_pose[3][4] = {
                        { rotations[rotation_index][0][0], rotations[rotation_index][0][1], rotations[rotation_index][0][2], translations[translation_index][0] },
                        { rotations[rotation_index][1][0], rotations[rotation_index][1][1], rotations[rotation_index][1][2], translations[translation_index][1] },
                        { rotations[rotation_index][2][0], rotations[rotation_index][2][1], rotations[rotation_index][2][2], translations[translation_index][2] },
                    };

                    for (size_t point_index = 0; point_index < point_count; ++point_index) {
                        // Triangulate.
                        matrix::matrix<double, 3, 1> triangulated_result = matrix::matrix<double, 3, 1>::zero();
                        bool is_valid = geometry::triangulate(
                            matrix::matrix<double, 2, 1>(&lhs_points[point_index * 2]),
                            matrix::matrix<double, 3, 4>(&lhs_camera_pose[0][0]),
                            matrix::matrix<double, 2, 1>(&rhs_points[point_index * 2]),
                            matrix::matrix<double, 3, 4>(&rhs_camera_pose[0][0]),
                            triangulated_result
                        );

                        triangulated_points_set[solution_index][point_index * 3 + 0] = triangulated_result[0];
                        triangulated_points_set[solution_index][point_index * 3 + 1] = triangulated_result[1];
                        triangulated_points_set[solution_index][point_index * 3 + 2] = triangulated_result[2];

                        // Check the point is in front of the camera.
                        is_valid &= (triangulated_points_set[solution_index][point_index * 3 + 2] > type(0));
                        // Check the point is not too far away.
                        is_valid &= (triangulated_points_set[solution_index][point_index * 3 + 2] < distance_threshold);

                        if (!is_valid) {
                            continue;
                        }

                        // Reproject the point using the potential transformation.
                        const type homogeneous_point[4] = {
                            triangulated_points_set[solution_index][point_index * 3 + 0],
                            triangulated_points_set[solution_index][point_index * 3 + 1],
                            triangulated_points_set[solution_index][point_index * 3 + 2],
                            type(1)
                        };

                        type reprojected_point[3];
                        matrix_multiply_local(&rhs_camera_pose[0][0], 4, 3, &homogeneous_point[0], 1, 4, &reprojected_point[0]);

                        // Check the point is in front of the camera.
                        is_valid &= (reprojected_point[2] > type(0));
                        // Check the point is not too far away.
                        is_valid &= (reprojected_point[2] < distance_threshold);

                        // If it's still valid add one to the count.
                        if (is_valid) {
                            triangulated_points_valid_counts[solution_index] += size_t(1);
                        }
                    }
                }
            }

            // Find best solution.
            size_t best_solution_index = 0;
            for (size_t solution_index = 1; solution_index < 4; ++solution_index) {
                if (triangulated_points_valid_counts[solution_index] > triangulated_points_valid_counts[best_solution_index]) {
                    best_solution_index = solution_index;
                }
            }

            // Return best transformation and triangulated points.
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    rotation[row * 3 + col] = rotations[best_solution_index / 2][row][col];
                }
                translation[row] = translations[best_solution_index % 2][row];
            }
            for (size_t point_index = 0; point_index < point_count; ++point_index) {
                triangulated_points[point_index * 3 + 0] = triangulated_points_set[best_solution_index][point_index * 3 + 0];
                triangulated_points[point_index * 3 + 1] = triangulated_points_set[best_solution_index][point_index * 3 + 1];
                triangulated_points[point_index * 3 + 2] = triangulated_points_set[best_solution_index][point_index * 3 + 2];
            }

            delete[] triangulated_points_set[0];
            delete[] triangulated_points_set[1];
            delete[] triangulated_points_set[2];
            delete[] triangulated_points_set[3];
        }
    };
}

#endif // POSE_ESTIMATION_HPP
