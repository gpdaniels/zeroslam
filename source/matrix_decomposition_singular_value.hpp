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
#ifndef MATRIX_DECOMPOSITION_SINGULAR_VALUE_HPP
#define MATRIX_DECOMPOSITION_SINGULAR_VALUE_HPP

#include "math.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace matrix {
    /// @brief  Compute the singular value decomposition of a matrix.
    /// @param  matrix_a Input matrix to decompose matrix size (width x height).
    /// @param  matrix_u Output height x height orthogonal matrix.
    /// @param  matrix_s Output width x height diagonal matrix.
    /// @param  matrix_vt Output width x width orthogonal matrix.
    /// @return True if the decomposition is successful, false otherwise.
    template <typename type>
    static bool decompose_singular_value(
        const type* __restrict matrix_a,
        size_t width,
        size_t height,
        type* __restrict matrix_u,
        type* __restrict matrix_s,
        type* __restrict matrix_vt
    ) {
        // Helper function to transpose a matrix in place.
        constexpr static const auto transpose =
            [](
                type* const __restrict matrix,
                const size_t width,
                const size_t height
            ) {
                for (size_t start_index = 0; start_index <= width * height - 1; ++start_index) {
                    size_t next_index = start_index;
                    size_t cycle_length = 0;
                    do {
                        ++cycle_length;
                        next_index = ((next_index % height) * width) + (next_index / height);
                    } while (next_index > start_index);

                    if ((next_index >= start_index) && (cycle_length != 1)) {
                        const type temp = matrix[start_index];
                        next_index = start_index;
                        do {
                            cycle_length = ((next_index % height) * width) + (next_index / height);
                            matrix[next_index] = (cycle_length == start_index) ? temp : matrix[cycle_length];
                            next_index = cycle_length;
                        } while (next_index > start_index);
                    }
                }
            };

        // Helper function to apply a left givens rotation to a matrix.
        constexpr static const auto givens_rotation_left =
            [](type* const __restrict matrix, const size_t width, const size_t height, const size_t row_index, const type alpha, const type beta) {
                static_cast<void>(height);
                const type r = math::pythag(alpha, beta);
                if (r == 0) {
                    return;
                }
                const type c = alpha / r;
                const type s = -beta / r;
                for (size_t i = 0; i < width; ++i) {
                    const type S0 = matrix[(row_index + 0) * width + i];
                    const type S1 = matrix[(row_index + 1) * width + i];
                    matrix[(row_index + 0) * width + i] += S0 * (c - 1);
                    matrix[(row_index + 0) * width + i] += S1 * (0 - s);
                    matrix[(row_index + 1) * width + i] += S0 * (s - 0);
                    matrix[(row_index + 1) * width + i] += S1 * (c - 1);
                }
            };

        // Helper function to apply a right givens rotation to a matrix.
        constexpr static const auto givens_rotation_right =
            [](type* const __restrict matrix, const size_t width, const size_t height, const size_t column_index, const type alpha, const type beta) {
                const type r = math::pythag(alpha, beta);
                if (r == 0) {
                    return;
                }
                const type c = alpha / r;
                const type s = -beta / r;
                for (size_t i = 0; i < height; ++i) {
                    const type S0 = matrix[i * width + (column_index + 0)];
                    const type S1 = matrix[i * width + (column_index + 1)];
                    matrix[i * width + (column_index + 0)] += S0 * (c - 1);
                    matrix[i * width + (column_index + 0)] += S1 * (0 - s);
                    matrix[i * width + (column_index + 1)] += S0 * (s - 0);
                    matrix[i * width + (column_index + 1)] += S1 * (c - 1);
                }
            };

        // Validate the input matrix size.
        if ((width == 0) || (height == 0)) {
            return false;
        }

        // Set initial values.
        // - matrix_u = identity
        // - matrix_s = matrix_a
        // - matrix_vt = identity
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < height; ++x) {
                matrix_u[y * height + x] = (y == x);
            }
        }
        for (size_t i = 0; i < width * height; ++i) {
            matrix_s[i] = matrix_a[i];
        }
        for (size_t y = 0; y < width; ++y) {
            for (size_t x = 0; x < width; ++x) {
                matrix_vt[y * width + x] = (y == x);
            }
        }

        // To decompose square/tall matrices we decompose the matrix: A => U * S * Vt.
        // For wide matrix we tranpose then decompose: At => Vt * St * (Ut)t (Note: matrix multiplication order is reversed).
        // The pre-decomposition transform is handled here.
        const bool wide_matrix = (height < width);
        if (wide_matrix) {
            transpose(matrix_s, width, height);
            type* temp_ptr = matrix_u;
            matrix_u = matrix_vt;
            matrix_vt = temp_ptr;
            size_t temp_dimension = width;
            width = height;
            height = temp_dimension;
        }

        const size_t max_total_iterations = 50 * height;
        constexpr static const type epsilon = math::epsilon<type>() * type(64);

        // Bi-diagonalisation.
        const size_t bidiag_columns = width;
        type* house_vector = new type[height];
        for (size_t column_index = 0; column_index < bidiag_columns; ++column_index) {
            // Column Householder.
            {
                type lead_value = matrix_s[(column_index)*width + (column_index)];
                if (lead_value < type(0))
                    lead_value = -lead_value;

                type inv_norm_squared = 0;
                for (size_t row = column_index; row < height; ++row) {
                    const type v = matrix_s[(row)*width + (column_index)];
                    inv_norm_squared += v * v;
                }
                if (inv_norm_squared > type(0))
                    inv_norm_squared = type(1) / math::sqrt(inv_norm_squared);

                type house_alpha = math::sqrt(type(1) + lead_value * inv_norm_squared);
                type house_beta = inv_norm_squared / house_alpha;

                if (inv_norm_squared == type(0)) {
                    house_alpha = type(0);
                }

                house_vector[column_index] = -house_alpha;
                for (size_t row = column_index + 1; row < height; ++row) {
                    house_vector[row] = -house_beta * matrix_s[(row)*width + (column_index)];
                }
                if (matrix_s[(column_index)*width + (column_index)] < type(0)) {
                    for (size_t row = column_index + 1; row < height; ++row) {
                        house_vector[row] = -house_vector[row];
                    }
                }
            }

            for (size_t target_col = column_index; target_col < width; ++target_col) {
                type dot_prod = type(0);
                for (size_t row = column_index; row < height; ++row) {
                    dot_prod += matrix_s[(row)*width + (target_col)] * house_vector[row];
                }
                for (size_t row = column_index; row < height; ++row) {
                    matrix_s[(row)*width + (target_col)] -= dot_prod * house_vector[row];
                }
            }

            for (size_t row_index = 0; row_index < height; ++row_index) {
                type dot_prod = type(0);
                for (size_t row = column_index; row < height; ++row) {
                    dot_prod += matrix_u[(row_index)*height + (row)] * house_vector[row];
                }
                for (size_t row = column_index; row < height; ++row) {
                    matrix_u[(row_index)*height + (row)] -= dot_prod * house_vector[row];
                }
            }

            if (column_index >= bidiag_columns - 1) {
                continue;
            }

            // Row Householder.
            {
                type lead_value = matrix_s[(column_index)*width + (column_index + 1)];
                if (lead_value < type(0))
                    lead_value = -lead_value;

                type inv_norm_squared = type(0);
                for (size_t col = column_index + 1; col < width; ++col) {
                    const type v = matrix_s[(column_index)*width + (col)];
                    inv_norm_squared += v * v;
                }
                if (inv_norm_squared > type(0))
                    inv_norm_squared = type(1) / math::sqrt(inv_norm_squared);

                type house_alpha = math::sqrt(type(1) + lead_value * inv_norm_squared);
                type house_beta = inv_norm_squared / house_alpha;

                if (inv_norm_squared == type(0)) {
                    house_alpha = type(0);
                }

                house_vector[column_index + 1] = -house_alpha;
                for (size_t col = column_index + 2; col < width; ++col) {
                    house_vector[col] = -house_beta * matrix_s[(column_index)*width + (col)];
                }
                if (matrix_s[(column_index)*width + (column_index + 1)] < type(0)) {
                    for (size_t col = column_index + 2; col < width; ++col) {
                        house_vector[col] = -house_vector[col];
                    }
                }
            }

            for (size_t target_row = column_index; target_row < height; ++target_row) {
                type dot_prod = type(0);
                for (size_t col = column_index + 1; col < width; ++col) {
                    dot_prod += matrix_s[(target_row)*width + (col)] * house_vector[col];
                }
                for (size_t col = column_index + 1; col < width; ++col) {
                    matrix_s[(target_row)*width + (col)] -= dot_prod * house_vector[col];
                }
            }

            for (size_t target_col = 0; target_col < width; ++target_col) {
                type dot_prod = type(0);
                for (size_t col = column_index + 1; col < width; ++col) {
                    dot_prod += matrix_vt[(col)*width + (target_col)] * house_vector[col];
                }
                for (size_t col = column_index + 1; col < width; ++col) {
                    matrix_vt[(col)*width + (target_col)] -= dot_prod * house_vector[col];
                }
            }
        }
        delete[] house_vector;

        // Diagonalisation.
        size_t sweep_start_index = 0;
        size_t total_iterations = 0;
        while (sweep_start_index < width - 1 && total_iterations < max_total_iterations) {
            ++total_iterations;

            type singular_max = type(0);
            for (size_t diag = 0; diag < width; ++diag) {
                singular_max = math::max(singular_max, math::abs(matrix_s[(diag)*width + (diag)]));
            }
            for (size_t off = 0; off < width - 1; ++off) {
                singular_max = math::max(singular_max, math::abs(matrix_s[(off)*width + (off + 1)]));
            }

            const type tolerance = epsilon * math::max(singular_max, type(1.0));

            while (sweep_start_index < width - 1 && math::abs(matrix_s[(sweep_start_index)*width + (sweep_start_index + 1)]) <= tolerance) {
                ++sweep_start_index;
            }

            if (sweep_start_index == width - 1) {
                continue;
            }

            size_t sweep_end_index = sweep_start_index + 2;
            while (sweep_end_index < width && math::abs(matrix_s[(sweep_end_index - 1) * width + (sweep_end_index)]) > tolerance) {
                ++sweep_end_index;
            }

            bool has_small_diagonal = false;
            for (size_t diag = sweep_start_index; diag < sweep_end_index; ++diag) {
                if (math::abs(matrix_s[(diag)*width + (diag)]) <= tolerance) {
                    has_small_diagonal = true;
                    break;
                }
            }

            type bulge_alpha = type(0);
            type bulge_beta = type(0);

            if (sweep_end_index - sweep_start_index == 2 && has_small_diagonal) {
                bulge_alpha = type(0);
                bulge_beta = type(1);
            }
            else {
                type C00 = matrix_s[(sweep_end_index - 2) * width + (sweep_end_index - 2)] * matrix_s[(sweep_end_index - 2) * width + (sweep_end_index - 2)];
                if (sweep_end_index - sweep_start_index > 2) {
                    C00 += matrix_s[(sweep_end_index - 3) * width + (sweep_end_index - 2)] * matrix_s[(sweep_end_index - 3) * width + (sweep_end_index - 2)];
                }
                const type C01 = matrix_s[(sweep_end_index - 2) * width + (sweep_end_index - 2)] * matrix_s[(sweep_end_index - 2) * width + (sweep_end_index - 1)];
                const type C10 = C01;
                const type C11 = matrix_s[(sweep_end_index - 1) * width + (sweep_end_index - 1)] * matrix_s[(sweep_end_index - 1) * width + (sweep_end_index - 1)] + matrix_s[(sweep_end_index - 2) * width + (sweep_end_index - 1)] * matrix_s[(sweep_end_index - 2) * width + (sweep_end_index - 1)];

                const type b_val = -(C00 + C11) / type(2);
                const type c_val = C00 * C11 - C01 * C10;

                type discriminant = type(0);
                if (b_val * b_val - c_val > type(0)) {
                    discriminant = math::sqrt(b_val * b_val - c_val);
                }
                else {
                    const type b_alt = (C00 - C11) / type(2);
                    const type c_alt = -C01 * C10;
                    if (b_alt * b_alt - c_alt > type(0)) {
                        discriminant = math::sqrt(b_alt * b_alt - c_alt);
                    }
                }

                const type lambda_first = -b_val + discriminant;
                const type lambda_second = -b_val - discriminant;

                const type delta_first = math::abs(lambda_first - C11);
                const type delta_second = math::abs(lambda_second - C11);
                const type selected_mu = (delta_first < delta_second) ? lambda_first : lambda_second;

                bulge_alpha = matrix_s[(sweep_start_index)*width + (sweep_start_index)] * matrix_s[(sweep_start_index)*width + (sweep_start_index)] - selected_mu;
                bulge_beta = matrix_s[(sweep_start_index)*width + (sweep_start_index + 1)] * matrix_s[(sweep_start_index)*width + (sweep_start_index)];
            }

            for (size_t sweep_index = sweep_start_index; sweep_index < sweep_end_index - 1; ++sweep_index) {
                givens_rotation_right(matrix_s, width, height, sweep_index, bulge_alpha, bulge_beta);
                givens_rotation_left(matrix_vt, width, width, sweep_index, bulge_alpha, bulge_beta);
                bulge_alpha = matrix_s[(sweep_index)*width + (sweep_index)];
                bulge_beta = matrix_s[(sweep_index + 1) * width + (sweep_index)];
                givens_rotation_left(matrix_s, width, height, sweep_index, bulge_alpha, bulge_beta);
                givens_rotation_right(matrix_u, height, height, sweep_index, bulge_alpha, bulge_beta);
                bulge_alpha = matrix_s[(sweep_index)*width + (sweep_index + 1)];
                if (sweep_index + 2 < width) {
                    bulge_beta = matrix_s[(sweep_index)*width + (sweep_index + 2)];
                }
                else {
                    bulge_beta = type(0);
                }
            }

            {
                for (size_t diag_index = sweep_start_index; diag_index < sweep_end_index - 1; ++diag_index) {
                    for (size_t col_index = 0; col_index < width; ++col_index) {
                        if (diag_index > col_index || diag_index + 1 < col_index) {
                            matrix_s[(diag_index)*width + (col_index)] = type(0);
                        }
                    }
                }
                for (size_t row_index = 0; row_index < height; ++row_index) {
                    for (size_t diag_index = sweep_start_index; diag_index < sweep_end_index - 1; ++diag_index) {
                        if (row_index > diag_index || row_index + 1 < diag_index) {
                            matrix_s[(row_index)*width + (diag_index)] = type(0);
                        }
                    }
                }
                for (size_t off = 0; off < width - 1; ++off) {
                    if (math::abs(matrix_s[(off)*width + (off + 1)]) <= epsilon * singular_max) {
                        matrix_s[(off)*width + (off + 1)] = type(0);
                    }
                }
            }
        }

        // If the total iterations exceeded the max the decomposition failed.
        if (total_iterations >= max_total_iterations) {
            return false;
        }

        // The post-decomposition transform is handled here if the matrix was wide.
        if (wide_matrix) {
            size_t temp_dimension = width;
            width = height;
            height = temp_dimension;
            type* temp_ptr = matrix_u;
            matrix_u = matrix_vt;
            matrix_vt = temp_ptr;
            transpose(matrix_u, height, height);
            transpose(matrix_s, height, width);
            transpose(matrix_vt, width, width);
        }

        // The remainder of this function rearranges the restultant matrices into a determinisic pattern.
        // - matrix_s signs are made positive.
        // - matrix_s values are sorted largest to smallest, with largest in the top left.
        // - matrix_u and matrix_vt signs are swapped to maximise the number of positive signs.

        const size_t min_dimension = math::min(width, height);

        for (size_t diag = 0; diag < min_dimension; ++diag) {
            const type sign_val = ((type(0) < matrix_s[diag * width + diag]) - (matrix_s[diag * width + diag] < type(0)));
            matrix_s[diag * width + diag] *= sign_val;
            for (size_t row = 0; row < height; ++row) {
                matrix_u[row * height + diag] *= sign_val;
            }
        }

        bool performed_swap = true;
        while (performed_swap) {
            performed_swap = false;
            for (size_t diag = 0; diag < min_dimension - 1; ++diag) {
                if (matrix_s[diag * width + diag] < matrix_s[(diag + 1) * width + (diag + 1)]) {
                    for (size_t row = 0; row < height; ++row) {
                        const type temp = matrix_u[row * height + diag];
                        matrix_u[row * height + diag] = matrix_u[row * height + diag + 1];
                        matrix_u[row * height + diag + 1] = temp;
                    }
                    const type temp = matrix_s[diag * width + diag];
                    matrix_s[diag * width + diag] = matrix_s[(diag + 1) * width + diag + 1];
                    matrix_s[(diag + 1) * width + diag + 1] = temp;
                    for (size_t col = 0; col < width; ++col) {
                        const type temp = matrix_vt[diag * width + col];
                        matrix_vt[diag * width + col] = matrix_vt[(diag + 1) * width + col];
                        matrix_vt[(diag + 1) * width + col] = temp;
                    }
                    performed_swap = true;
                    break;
                }
            }
        }

        for (size_t diag = 0; diag < min_dimension; ++diag) {
            int positive_sign_count = 0;
            int negative_sign_count = 0;
            for (size_t row = 0; row < height; ++row) {
                positive_sign_count += (matrix_u[row * height + diag] > type(0));
                negative_sign_count += (matrix_u[row * height + diag] < type(0));
            }
            for (size_t col = 0; col < width; ++col) {
                positive_sign_count += (matrix_vt[diag * width + col] > type(0));
                negative_sign_count += (matrix_vt[diag * width + col] < type(0));
            }
            const bool flip_majority = (positive_sign_count < negative_sign_count);
            const bool flip_tiebreak = ((positive_sign_count == negative_sign_count) && (matrix_u[diag] > type(0)));
            if (flip_majority || flip_tiebreak) {
                for (size_t row = 0; row < height; ++row) {
                    matrix_u[row * height + diag] = -matrix_u[row * height + diag];
                }
                for (size_t col = 0; col < width; ++col) {
                    matrix_vt[diag * width + col] = -matrix_vt[diag * width + col];
                }
            }
        }

        if (min_dimension < height) {
            for (size_t diag = min_dimension; diag < height; ++diag) {
                int positive_sign_count = 0;
                int negative_sign_count = 0;
                for (size_t row = 0; row < height; ++row) {
                    positive_sign_count += (matrix_u[row * height + diag] > type(0));
                    negative_sign_count += (matrix_u[row * height + diag] < type(0));
                }
                const bool flip_majority = (positive_sign_count < negative_sign_count);
                const bool flip_tiebreak = ((positive_sign_count == negative_sign_count) && (matrix_u[diag] > type(0)));
                if (flip_majority || flip_tiebreak) {
                    for (size_t row = 0; row < height; ++row) {
                        matrix_u[row * height + diag] = -matrix_u[row * height + diag];
                    }
                }
            }
        }
        else if (min_dimension < width) {
            for (size_t diag = min_dimension; diag < width; ++diag) {
                int positive_sign_count = 0;
                int negative_sign_count = 0;
                for (size_t col = 0; col < width; ++col) {
                    positive_sign_count += (matrix_vt[diag * width + col] > type(0));
                    negative_sign_count += (matrix_vt[diag * width + col] < type(0));
                }
                const bool flip_majority = (positive_sign_count < negative_sign_count);
                const bool flip_tiebreak = ((positive_sign_count == negative_sign_count) && (matrix_vt[diag] > type(0)));
                if (flip_majority || flip_tiebreak) {
                    for (size_t col = 0; col < width; ++col) {
                        matrix_vt[diag * width + col] = -matrix_vt[diag * width + col];
                    }
                }
            }
        }

        return true;
    }
}

#endif // MATRIX_DECOMPOSITION_SINGULAR_VALUE_HPP
