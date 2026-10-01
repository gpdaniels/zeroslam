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
                const double* __restrict lhs_one_deg_poly,
                const double* __restrict rhs_one_deg_poly,
                double* __restrict result
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
                const double* __restrict lhs_one_deg_poly,
                const double* __restrict rhs_two_deg_poly,
                double* __restrict result
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

        // Every precision is solved in double, in single precision the elimination loses the true solution in a large fraction of samples.
        double lhs[5 * 2];
        double rhs[5 * 2];
        for (int i = 0; i < 5 * 2; ++i) {
            lhs[i] = static_cast<double>(lhs_points[i]);
            rhs[i] = static_cast<double>(rhs_points[i]);
        }

        double epipolar_constraint[5][9];
        for (int i = 0; i < 5; ++i) {
            epipolar_constraint[i][0] = rhs[i * 2 + 0] * lhs[i * 2 + 0];
            epipolar_constraint[i][1] = rhs[i * 2 + 1] * lhs[i * 2 + 0];
            epipolar_constraint[i][2] = lhs[i * 2 + 0];
            epipolar_constraint[i][3] = rhs[i * 2 + 0] * lhs[i * 2 + 1];
            epipolar_constraint[i][4] = rhs[i * 2 + 1] * lhs[i * 2 + 1];
            epipolar_constraint[i][5] = lhs[i * 2 + 1];
            epipolar_constraint[i][6] = rhs[i * 2 + 0];
            epipolar_constraint[i][7] = rhs[i * 2 + 1];
            epipolar_constraint[i][8] = 1.0;
        }
        // The null space is taken from the constraint matrix itself, the normal matrix would square its condition number.
        double u[5][5];
        double s[5][9];
        double vt[9][9];
        if (!math::decompose_singular_value(&epipolar_constraint[0][0], 9, 5, &u[0][0], &s[0][0], &vt[0][0])) {
            return 0;
        }
        // Repeated or otherwise dependent correspondences leave a larger null space, from which any basis gives arbitrary solutions.
        if (!(s[4][4] > 1.0e-10 * s[0][0])) {
            return 0;
        }
        // The solutions are found with the weight of the last basis vector set to one, so no solution may have zero weight there. A symmetry of the data can cause that in the decomposition's own basis: y' = y under a sideways translation without rotation makes the true E the difference of two basis vectors. So the basis is turned by a fixed rotation whose last column no combination of -1, 0 and 1 weights cancels.
        constexpr static const double basis_rotation[4][4] = {
            { 0.90260917371333682, -0.04568809355300537, 0.018924429823319891, -0.42761097225384836 },
            { -0.04568809355300537, 0.97856675035938145, 0.0088778497233422318, -0.20060133843736358 },
            { 0.018924429823319891, 0.0088778497233422318, 0.99632271274623319, 0.083090924932502697 },
            { -0.42761097225384836, -0.20060133843736358, 0.083090924932502697, -0.87749863681895146 }
        };
        double null_space[9][4];
        for (int row = 0; row < 9; ++row) {
            for (int col = 0; col < 4; ++col) {
                null_space[row][col] = vt[5][row] * basis_rotation[0][col] + vt[6][row] * basis_rotation[1][col] + vt[7][row] * basis_rotation[2][col] + vt[8][row] * basis_rotation[3][col];
            }
        }
        const double null_space_matrix[3][3][4] = {
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

        double constraint_matrix[10][20];
        {
            {
                double* trace_constraint = &constraint_matrix[0][0];

                double eet[3][3][10];
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        {
                            double result_parts[3][10];
                            multiply_one_deg_poly(&null_space_matrix[i][0][0], &null_space_matrix[j][0][0], &result_parts[0][0]);
                            multiply_one_deg_poly(&null_space_matrix[i][1][0], &null_space_matrix[j][1][0], &result_parts[1][0]);
                            multiply_one_deg_poly(&null_space_matrix[i][2][0], &null_space_matrix[j][2][0], &result_parts[2][0]);
                            for (int index = 0; index < 10; ++index) {
                                eet[i][j][index] = result_parts[0][index] + result_parts[1][index] + result_parts[2][index];
                            }
                        }
                        for (int index = 0; index < 10; ++index) {
                            eet[i][j][index] *= 2.0;
                        }
                    }
                }

                double trace[10];
                for (int index = 0; index < 10; ++index) {
                    trace[index] = eet[0][0][index] + eet[1][1][index] + eet[2][2][index];
                }

                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        double result_parts[4][20];
                        multiply_one_deg_two_deg_poly(&eet[i][0][0], &null_space_matrix[0][j][0], &result_parts[0][0]);
                        multiply_one_deg_two_deg_poly(&eet[i][1][0], &null_space_matrix[1][j][0], &result_parts[1][0]);
                        multiply_one_deg_two_deg_poly(&eet[i][2][0], &null_space_matrix[2][j][0], &result_parts[2][0]);
                        multiply_one_deg_two_deg_poly(&trace[0], &null_space_matrix[i][j][0], &result_parts[3][0]);
                        for (int index = 0; index < 20; ++index) {
                            trace_constraint[(3 * i + j) * 20 + index] = result_parts[0][index] + result_parts[1][index] + result_parts[2][index] - (0.5 * result_parts[3][index]);
                        }
                    }
                }
            }

            {
                double* determinant_constraint = &constraint_matrix[9][0];

                double null_space_01_12[10];
                multiply_one_deg_poly(&null_space_matrix[0][1][0], &null_space_matrix[1][2][0], &null_space_01_12[0]);
                double null_space_02_11[10];
                multiply_one_deg_poly(&null_space_matrix[0][2][0], &null_space_matrix[1][1][0], &null_space_02_11[0]);
                double null_space_01_12_minus_02_11[10];
                for (int i = 0; i < 10; ++i) {
                    null_space_01_12_minus_02_11[i] = null_space_01_12[i] - null_space_02_11[i];
                }
                double determinant_0[20];
                multiply_one_deg_two_deg_poly(&null_space_01_12_minus_02_11[0], &null_space_matrix[2][0][0], &determinant_0[0]);

                double null_space_02_10[10];
                multiply_one_deg_poly(&null_space_matrix[0][2][0], &null_space_matrix[1][0][0], &null_space_02_10[0]);
                double null_space_00_12[10];
                multiply_one_deg_poly(&null_space_matrix[0][0][0], &null_space_matrix[1][2][0], &null_space_00_12[0]);
                double null_space_02_10_minus_00_12[10];
                for (int i = 0; i < 10; ++i) {
                    null_space_02_10_minus_00_12[i] = null_space_02_10[i] - null_space_00_12[i];
                }
                double determinant_1[20];
                multiply_one_deg_two_deg_poly(&null_space_02_10_minus_00_12[0], &null_space_matrix[2][1][0], &determinant_1[0]);

                double null_space_00_11[10];
                multiply_one_deg_poly(&null_space_matrix[0][0][0], &null_space_matrix[1][1][0], &null_space_00_11[0]);
                double null_space_01_10[10];
                multiply_one_deg_poly(&null_space_matrix[0][1][0], &null_space_matrix[1][0][0], &null_space_01_10[0]);
                double null_space_00_11_minus_01_10[10];
                for (int i = 0; i < 10; ++i) {
                    null_space_00_11_minus_01_10[i] = null_space_00_11[i] - null_space_01_10[i];
                }
                double determinant_2[20];
                multiply_one_deg_two_deg_poly(&null_space_00_11_minus_01_10[0], &null_space_matrix[2][2][0], &determinant_2[0]);

                for (int i = 0; i < 20; ++i) {
                    determinant_constraint[i] = determinant_0[i] + determinant_1[i] + determinant_2[i];
                }
            }
        }

        // Gauss-Jordan elimination of the ten cubic monomials, on rows scaled to a unit largest coefficient so the singularity test is relative to the block's own scale, which is small at low parallax.
        for (int y = 0; y < 10; ++y) {
            double row_maximum = 0.0;
            for (int x = 0; x < 10; ++x) {
                row_maximum = math::max(row_maximum, math::abs(constraint_matrix[y][x]));
            }
            if (!(row_maximum > 0.0) || !math::isfinite(row_maximum)) {
                return 0;
            }
            const double row_scale = 1.0 / row_maximum;
            for (int x = 0; x < 20; ++x) {
                constraint_matrix[y][x] *= row_scale;
            }
        }
        for (int pivot = 0; pivot < 10; ++pivot) {
            int pivot_row = pivot;
            double pivot_magnitude = math::abs(constraint_matrix[pivot][pivot]);
            for (int y = pivot + 1; y < 10; ++y) {
                const double magnitude = math::abs(constraint_matrix[y][pivot]);
                if (magnitude > pivot_magnitude) {
                    pivot_magnitude = magnitude;
                    pivot_row = y;
                }
            }
            if (!(pivot_magnitude > 1.0e-14)) {
                return 0;
            }
            if (pivot_row != pivot) {
                for (int x = pivot; x < 20; ++x) {
                    const double swapped = constraint_matrix[pivot][x];
                    constraint_matrix[pivot][x] = constraint_matrix[pivot_row][x];
                    constraint_matrix[pivot_row][x] = swapped;
                }
            }
            const double pivot_inverse = 1.0 / constraint_matrix[pivot][pivot];
            for (int x = pivot; x < 20; ++x) {
                constraint_matrix[pivot][x] *= pivot_inverse;
            }
            for (int y = 0; y < 10; ++y) {
                const double factor = constraint_matrix[y][pivot];
                if ((y == pivot) || (factor == 0.0)) {
                    continue;
                }
                for (int x = pivot; x < 20; ++x) {
                    constraint_matrix[y][x] -= factor * constraint_matrix[pivot][x];
                }
            }
        }

        double action_matrix[10][10] = {};
        for (int x = 0; x < 10; ++x) {
            action_matrix[0][x] = constraint_matrix[0][x + 10];
            action_matrix[1][x] = constraint_matrix[1][x + 10];
            action_matrix[2][x] = constraint_matrix[2][x + 10];
            action_matrix[3][x] = constraint_matrix[4][x + 10];
            action_matrix[4][x] = constraint_matrix[5][x + 10];
            action_matrix[5][x] = constraint_matrix[7][x + 10];
        }
        action_matrix[6][0] = -1.0;
        action_matrix[7][1] = -1.0;
        action_matrix[8][3] = -1.0;
        action_matrix[9][6] = -1.0;
        for (int y = 0; y < 6; ++y) {
            for (int x = 0; x < 10; ++x) {
                if (!math::isfinite(action_matrix[y][x])) {
                    return 0;
                }
            }
        }

        double eigen_values[10][2] = {};
        double eigen_vectors[10][10] = {};
        if (!math::eigen_solver(&action_matrix[0][0], 10, &eigen_values[0][0], &eigen_vectors[0][0])) {
            return 0;
        }

        int count = 0;
        for (int i = 0; i < 10; i++) {
            if (eigen_values[i][1] != 0.0) {
                continue;
            }
            const double eigen_vector_part[4] = {
                eigen_vectors[6][i],
                eigen_vectors[7][i],
                eigen_vectors[8][i],
                eigen_vectors[9][i]
            };
            // The null space combination holds E in column order, so element (row, col) is entry (col * 3 + row).
            double ematrix[9];
            double norm_squared = 0.0;
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    const double* const coefficients = &null_space[col * 3 + row][0];
                    ematrix[row * 3 + col] = coefficients[0] * eigen_vector_part[0] + coefficients[1] * eigen_vector_part[1] + coefficients[2] * eigen_vector_part[2] + coefficients[3] * eigen_vector_part[3];
                    norm_squared += ematrix[row * 3 + col] * ematrix[row * 3 + col];
                }
            }
            if (!(norm_squared > 0.0) || !math::isfinite(norm_squared)) {
                continue;
            }
            const double norm_inverse = 1.0 / math::sqrt(norm_squared);
            for (int k = 0; k < 9; ++k) {
                essentials[count * 9 + k] = static_cast<type>(ematrix[k] * norm_inverse);
            }
            ++count;
        }

        return count;
    }

    template class essential_5_point<float>;
    template class essential_5_point<double>;
}
