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
#ifndef ZEROSLAM_MATH_MATRIX_DECOMPOSITION_SINGULAR_VALUE_HPP
#define ZEROSLAM_MATH_MATRIX_DECOMPOSITION_SINGULAR_VALUE_HPP

#include "math/math.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace matrix {
    /// @brief  Compute the singular value decomposition of a matrix, A = U * S * Vt.
    /// @param  matrix_a Input matrix to decompose, matrix size (width x height), row-major.
    /// @param  width Number of columns of the input matrix, must be non-zero.
    /// @param  height Number of rows of the input matrix, must be non-zero.
    /// @param  matrix_u Output height x height orthogonal matrix. Signs are canonical so repeating the call on the same input reproduces the same factorisation.
    /// @param  matrix_s Output width x height matrix, zero apart from the leading diagonal with min(width, height) diagonal entries are non-negative and sorted from largest to smallest.
    /// @param  matrix_vt Output width x width orthogonal matrix. Signs are canonical so repeating the call on the same input reproduces the same factorisation.
    /// @return True if the decomposition is successful, false otherwise.
    template <typename type>
    static bool decompose_singular_value(
        const type* __restrict matrix_a,
        const size_t width,
        const size_t height,
        type* __restrict matrix_u,
        type* __restrict matrix_s,
        type* __restrict matrix_vt
    ) {
        // Helper function to transpose a matrix in place.
        constexpr static const auto transpose =
            [](
                type* const __restrict matrix,
                const size_t matrix_columns,
                const size_t matrix_rows
            ) {
                if (matrix_columns == matrix_rows) {
                    for (size_t row = 0; row < matrix_rows; ++row) {
                        for (size_t column = row + 1; column < matrix_columns; ++column) {
                            const type temp = matrix[row * matrix_columns + column];
                            matrix[row * matrix_columns + column] = matrix[column * matrix_columns + row];
                            matrix[column * matrix_columns + row] = temp;
                        }
                    }
                    return;
                }
                for (size_t start_index = 0; start_index < matrix_columns * matrix_rows; ++start_index) {
                    size_t next_index = start_index;
                    size_t cycle_length = 0;
                    do {
                        ++cycle_length;
                        next_index = ((next_index % matrix_rows) * matrix_columns) + (next_index / matrix_rows);
                    } while (next_index > start_index);

                    if ((next_index == start_index) && (cycle_length != 1)) {
                        const type temp = matrix[start_index];
                        next_index = start_index;
                        do {
                            const size_t source_index = ((next_index % matrix_rows) * matrix_columns) + (next_index / matrix_rows);
                            matrix[next_index] = (source_index == start_index) ? temp : matrix[source_index];
                            next_index = source_index;
                        } while (next_index > start_index);
                    }
                }
            };

        struct givens_rotation {
            type cosine;
            type sine;
            bool active;
        };

        // Helper function to build the rotation that maps (alpha, beta) onto (hypotenuse, 0).
        constexpr static const auto givens_build =
            [](const type alpha, const type beta) -> givens_rotation {
            const type radius = math::pythag(alpha, beta);
            // A degenerate rotation is the identity, and is skipped rather than applied.
            if (radius == type(0)) {
                return givens_rotation{ type(1), type(0), false };
            }
            const type radius_reciprocal = type(1) / radius;
            return givens_rotation{ alpha * radius_reciprocal, -beta * radius_reciprocal, true };
        };

        // Helper function to apply a rotation from the left, mixing rows row_index and row_index + 1.
        constexpr static const auto rotate_rows =
            [](type* const __restrict matrix, const size_t matrix_columns, const size_t row_index, const givens_rotation rotation) {
                if (!rotation.active) {
                    return;
                }
                for (size_t column = 0; column < matrix_columns; ++column) {
                    const type upper = matrix[(row_index + 0) * matrix_columns + column];
                    const type lower = matrix[(row_index + 1) * matrix_columns + column];
                    matrix[(row_index + 0) * matrix_columns + column] = (rotation.cosine * upper) - (rotation.sine * lower);
                    matrix[(row_index + 1) * matrix_columns + column] = (rotation.sine * upper) + (rotation.cosine * lower);
                }
            };

        // Helper function to apply a rotation from the right, mixing columns column_index and column_index + 1.
        constexpr static const auto rotate_columns =
            [](type* const __restrict matrix, const size_t matrix_columns, const size_t matrix_rows, const size_t column_index, const givens_rotation rotation) {
                if (!rotation.active) {
                    return;
                }
                for (size_t row = 0; row < matrix_rows; ++row) {
                    const type left = matrix[row * matrix_columns + (column_index + 0)];
                    const type right = matrix[row * matrix_columns + (column_index + 1)];
                    matrix[row * matrix_columns + (column_index + 0)] = (rotation.cosine * left) - (rotation.sine * right);
                    matrix[row * matrix_columns + (column_index + 1)] = (rotation.sine * left) + (rotation.cosine * right);
                }
            };

        // Validate the input matrix size.
        if ((width == 0) || (height == 0)) {
            return false;
        }

        // Validate the input matrix contents. A non-finite value poisons the shift.
        for (size_t index = 0; index < width * height; ++index) {
            if (!math::isfinite(matrix_a[index])) {
                return false;
            }
        }

        // Set initial values.
        // - matrix_u = identity
        // - matrix_s = matrix_a
        // - matrix_vt = identity
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < height; ++x) {
                matrix_u[y * height + x] = type(y == x);
            }
        }
        for (size_t index = 0; index < width * height; ++index) {
            matrix_s[index] = matrix_a[index];
        }
        for (size_t y = 0; y < width; ++y) {
            for (size_t x = 0; x < width; ++x) {
                matrix_vt[y * width + x] = type(y == x);
            }
        }

        // To decompose square/tall matrices we decompose the matrix: A => U * S * Vt.
        // For wide matrix we tranpose then decompose: At => Vt * St * (Ut)t (Note: matrix multiplication order is reversed).
        const bool wide_matrix = (height < width);
        const size_t columns = wide_matrix ? height : width;
        const size_t rows = wide_matrix ? width : height;
        type* const __restrict left_factor = wide_matrix ? matrix_vt : matrix_u;
        type* const __restrict right_factor = wide_matrix ? matrix_u : matrix_vt;
        if (wide_matrix) {
            transpose(matrix_s, width, height);
        }

        constexpr static const type epsilon = math::epsilon<type>() * type(64);

        // Bi-diagonalisation.
        constexpr static const size_t house_vector_static_size = 64;
        type house_vector_storage[house_vector_static_size];
        type* house_vector_heap = nullptr;
        type* house_vector = house_vector_storage;
        if (rows > house_vector_static_size) {
            house_vector_heap = new type[rows];
            house_vector = house_vector_heap;
        }
        for (size_t column_index = 0; column_index < columns; ++column_index) {
            // Column Householder, zeroing the entries below the diagonal of this column.
            {
                const type lead_value = matrix_s[(column_index)*columns + (column_index)];
                const type lead_magnitude = math::abs(lead_value);

                type norm_reciprocal = type(0);
                for (size_t row = column_index; row < rows; ++row) {
                    const type value = matrix_s[(row)*columns + (column_index)];
                    norm_reciprocal += value * value;
                }
                if (norm_reciprocal > type(0)) {
                    norm_reciprocal = type(1) / math::sqrt(norm_reciprocal);
                }

                // An all-zero column needs no reflection, and gives the zero vector, so H = I.
                const type house_alpha = (norm_reciprocal > type(0)) ? math::sqrt(type(1) + (lead_magnitude * norm_reciprocal)) : type(0);
                const type house_beta = (house_alpha > type(0)) ? (norm_reciprocal / house_alpha) : type(0);
                const type tail_sign = (lead_value < type(0)) ? type(1) : type(-1);

                house_vector[column_index] = -house_alpha;
                for (size_t row = column_index + 1; row < rows; ++row) {
                    house_vector[row] = tail_sign * house_beta * matrix_s[(row)*columns + (column_index)];
                }
            }

            for (size_t target_column = column_index; target_column < columns; ++target_column) {
                type projection = type(0);
                for (size_t row = column_index; row < rows; ++row) {
                    projection += matrix_s[(row)*columns + (target_column)] * house_vector[row];
                }
                for (size_t row = column_index; row < rows; ++row) {
                    matrix_s[(row)*columns + (target_column)] -= projection * house_vector[row];
                }
            }

            for (size_t factor_row = 0; factor_row < rows; ++factor_row) {
                type projection = type(0);
                for (size_t row = column_index; row < rows; ++row) {
                    projection += left_factor[(factor_row)*rows + (row)] * house_vector[row];
                }
                for (size_t row = column_index; row < rows; ++row) {
                    left_factor[(factor_row)*rows + (row)] -= projection * house_vector[row];
                }
            }

            if (column_index + 1 >= columns) {
                continue;
            }

            // Row Householder, zeroing the entries to the right of the superdiagonal of this row.
            {
                const type lead_value = matrix_s[(column_index)*columns + (column_index + 1)];
                const type lead_magnitude = math::abs(lead_value);

                type norm_reciprocal = type(0);
                for (size_t column = column_index + 1; column < columns; ++column) {
                    const type value = matrix_s[(column_index)*columns + (column)];
                    norm_reciprocal += value * value;
                }
                if (norm_reciprocal > type(0)) {
                    norm_reciprocal = type(1) / math::sqrt(norm_reciprocal);
                }

                const type house_alpha = (norm_reciprocal > type(0)) ? math::sqrt(type(1) + (lead_magnitude * norm_reciprocal)) : type(0);
                const type house_beta = (house_alpha > type(0)) ? (norm_reciprocal / house_alpha) : type(0);
                const type tail_sign = (lead_value < type(0)) ? type(1) : type(-1);

                house_vector[column_index + 1] = -house_alpha;
                for (size_t column = column_index + 2; column < columns; ++column) {
                    house_vector[column] = tail_sign * house_beta * matrix_s[(column_index)*columns + (column)];
                }
            }

            for (size_t target_row = column_index; target_row < rows; ++target_row) {
                type projection = type(0);
                for (size_t column = column_index + 1; column < columns; ++column) {
                    projection += matrix_s[(target_row)*columns + (column)] * house_vector[column];
                }
                for (size_t column = column_index + 1; column < columns; ++column) {
                    matrix_s[(target_row)*columns + (column)] -= projection * house_vector[column];
                }
            }

            for (size_t target_column = 0; target_column < columns; ++target_column) {
                type projection = type(0);
                for (size_t column = column_index + 1; column < columns; ++column) {
                    projection += right_factor[(column)*columns + (target_column)] * house_vector[column];
                }
                for (size_t column = column_index + 1; column < columns; ++column) {
                    right_factor[(column)*columns + (target_column)] -= projection * house_vector[column];
                }
            }
        }
        delete[] house_vector_heap;
        house_vector_heap = nullptr;
        house_vector = nullptr;

        // Diagonalisation.
        type singular_max = type(0);
        for (size_t diagonal = 0; diagonal < columns; ++diagonal) {
            singular_max = math::max(singular_max, math::abs(matrix_s[(diagonal)*columns + (diagonal)]));
        }
        for (size_t offset = 0; offset + 1 < columns; ++offset) {
            singular_max = math::max(singular_max, math::abs(matrix_s[(offset)*columns + (offset + 1)]));
        }
        const type tolerance = epsilon * singular_max;

        // The problem being iterated is the columns x columns bidiagonal, so the iteration budget is
        // set by columns; a very tall matrix does not make the bidiagonal any harder to converge.
        const size_t max_total_iterations = 50 * columns;
        size_t sweep_start_index = 0;
        size_t total_iterations = 0;
        while ((sweep_start_index + 1 < columns) && (total_iterations < max_total_iterations)) {
            ++total_iterations;

            // Deflate any leading superdiagonal entries that have converged.
            while ((sweep_start_index + 1 < columns) && (math::abs(matrix_s[(sweep_start_index)*columns + (sweep_start_index + 1)]) <= tolerance)) {
                matrix_s[(sweep_start_index)*columns + (sweep_start_index + 1)] = type(0);
                ++sweep_start_index;
            }
            if (sweep_start_index + 1 >= columns) {
                break;
            }

            // The active block runs to the first superdiagonal entry that has already converged.
            size_t sweep_end_index = sweep_start_index + 2;
            while ((sweep_end_index < columns) && (math::abs(matrix_s[(sweep_end_index - 1) * columns + (sweep_end_index)]) > tolerance)) {
                ++sweep_end_index;
            }

            bool has_small_diagonal = false;
            for (size_t diagonal = sweep_start_index; diagonal < sweep_end_index; ++diagonal) {
                if (math::abs(matrix_s[(diagonal)*columns + (diagonal)]) <= tolerance) {
                    has_small_diagonal = true;
                    break;
                }
            }

            type bulge_alpha = type(0);
            type bulge_beta = type(0);

            if (has_small_diagonal) {
                if (math::abs(matrix_s[(sweep_start_index)*columns + (sweep_start_index)]) <= tolerance) {
                    bulge_alpha = type(0);
                    bulge_beta = type(1);
                }
                else {
                    bulge_alpha = matrix_s[(sweep_start_index)*columns + (sweep_start_index)];
                    bulge_beta = matrix_s[(sweep_start_index)*columns + (sweep_start_index + 1)];
                }
            }
            else {
                const type diagonal_last = matrix_s[(sweep_end_index - 1) * columns + (sweep_end_index - 1)];
                const type diagonal_previous = matrix_s[(sweep_end_index - 2) * columns + (sweep_end_index - 2)];
                const type superdiagonal_last = matrix_s[(sweep_end_index - 2) * columns + (sweep_end_index - 1)];
                type block_00 = diagonal_previous * diagonal_previous;
                if (sweep_end_index - sweep_start_index > 2) {
                    const type superdiagonal_previous = matrix_s[(sweep_end_index - 3) * columns + (sweep_end_index - 2)];
                    block_00 += superdiagonal_previous * superdiagonal_previous;
                }
                const type block_01 = diagonal_previous * superdiagonal_last;
                const type block_11 = (diagonal_last * diagonal_last) + (superdiagonal_last * superdiagonal_last);
                const type block_centre = (block_00 + block_11) / type(2);
                const type block_half_difference = (block_00 - block_11) / type(2);
                const type discriminant = math::pythag(block_half_difference, block_01);
                const type lambda_first = block_centre + discriminant;
                const type lambda_second = block_centre - discriminant;
                const type selected_mu = (math::abs(lambda_first - block_11) < math::abs(lambda_second - block_11)) ? lambda_first : lambda_second;
                const type diagonal_first = matrix_s[(sweep_start_index)*columns + (sweep_start_index)];
                const type superdiagonal_first = matrix_s[(sweep_start_index)*columns + (sweep_start_index + 1)];
                bulge_alpha = (diagonal_first * diagonal_first) - selected_mu;
                bulge_beta = superdiagonal_first * diagonal_first;
            }

            for (size_t sweep_index = sweep_start_index; sweep_index + 1 < sweep_end_index; ++sweep_index) {
                const givens_rotation column_rotation = givens_build(bulge_alpha, bulge_beta);
                rotate_columns(matrix_s, columns, rows, sweep_index, column_rotation);
                rotate_rows(right_factor, columns, sweep_index, column_rotation);
                bulge_alpha = matrix_s[(sweep_index)*columns + (sweep_index)];
                bulge_beta = matrix_s[(sweep_index + 1) * columns + (sweep_index)];
                const givens_rotation row_rotation = givens_build(bulge_alpha, bulge_beta);
                rotate_rows(matrix_s, columns, sweep_index, row_rotation);
                rotate_columns(left_factor, rows, rows, sweep_index, row_rotation);
                bulge_alpha = matrix_s[(sweep_index)*columns + (sweep_index + 1)];
                bulge_beta = (sweep_index + 2 < columns) ? matrix_s[(sweep_index)*columns + (sweep_index + 2)] : type(0);
            }

            for (size_t index = sweep_start_index; index + 1 < sweep_end_index; ++index) {
                matrix_s[(index + 1) * columns + (index)] = type(0);
                if (index + 2 < columns) {
                    matrix_s[(index)*columns + (index + 2)] = type(0);
                }
                if (math::abs(matrix_s[(index)*columns + (index + 1)]) <= tolerance) {
                    matrix_s[(index)*columns + (index + 1)] = type(0);
                }
            }
        }

        // Check for convergence or fail.
        const bool converged = (sweep_start_index + 1 >= columns);
        if (!converged) {
            return false;
        }

        // Ensure off diagonal entries are zero.
        for (size_t row = 0; row < rows; ++row) {
            for (size_t column = 0; column < columns; ++column) {
                if (row != column) {
                    matrix_s[(row)*columns + (column)] = type(0);
                }
            }
        }

        // The post-decomposition transform is handled here if the matrix was wide.
        if (wide_matrix) {
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
            const type sign_val = (matrix_s[diag * width + diag] < type(0)) ? type(-1) : type(1);
            matrix_s[diag * width + diag] *= sign_val;
            for (size_t row = 0; row < height; ++row) {
                matrix_u[row * height + diag] *= sign_val;
            }
        }

        for (size_t target = 0; target + 1 < min_dimension; ++target) {
            size_t largest = target;
            for (size_t candidate = target + 1; candidate < min_dimension; ++candidate) {
                if (matrix_s[candidate * width + candidate] > matrix_s[largest * width + largest]) {
                    largest = candidate;
                }
            }
            if (largest == target) {
                continue;
            }
            {
                const type temp = matrix_s[target * width + target];
                matrix_s[target * width + target] = matrix_s[largest * width + largest];
                matrix_s[largest * width + largest] = temp;
            }
            for (size_t row = 0; row < height; ++row) {
                const type temp = matrix_u[row * height + target];
                matrix_u[row * height + target] = matrix_u[row * height + largest];
                matrix_u[row * height + largest] = temp;
            }
            for (size_t column = 0; column < width; ++column) {
                const type temp = matrix_vt[target * width + column];
                matrix_vt[target * width + column] = matrix_vt[largest * width + column];
                matrix_vt[largest * width + column] = temp;
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
            const bool flip_tiebreak = ((positive_sign_count == negative_sign_count) && (matrix_u[0 * height + diag] > type(0)));
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
                const bool flip_tiebreak = ((positive_sign_count == negative_sign_count) && (matrix_u[0 * height + diag] > type(0)));
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
                const bool flip_tiebreak = ((positive_sign_count == negative_sign_count) && (matrix_vt[diag * width + 0] > type(0)));
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

#endif // ZEROSLAM_MATH_MATRIX_DECOMPOSITION_SINGULAR_VALUE_HPP
