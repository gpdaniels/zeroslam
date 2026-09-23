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
#ifndef ZEROSLAM_MATH_MATRIX_DECOMPOSITION_QR_HPP
#define ZEROSLAM_MATH_MATRIX_DECOMPOSITION_QR_HPP

#include "core/assert.hpp"
#include "math/math.hpp"

namespace math {
    template <typename type>
    static inline bool householder_vector(
        const type* column,
        const int rows,
        const int column_stride,
        type* vector,
        type& beta,
        type& alpha
    ) {
        ASSERT(rows > 0, "A Householder vector needs at least one row.");
        beta = 0;
        alpha = column[0];
        if (rows == 1) {
            return false;
        }
        type scale = 0;
        for (int i = 0; i < rows; ++i) {
            scale = math::max(scale, math::abs(column[i * column_stride]));
        }
        if (!(scale > 0)) {
            return false;
        }
        type sum_of_squares_tail = 0;
        for (int i = 1; i < rows; ++i) {
            const type scaled = column[i * column_stride] / scale;
            sum_of_squares_tail += scaled * scaled;
        }
        if (!(sum_of_squares_tail > 0)) {
            return false;
        }
        const type scaled_head = column[0] / scale;
        const type norm = scale * math::sqrt((scaled_head * scaled_head) + sum_of_squares_tail);
        alpha = (column[0] >= 0) ? -norm : norm;
        const type head = column[0] - alpha;
        for (int i = 1; i < rows; ++i) {
            vector[i - 1] = column[i * column_stride] / head;
        }
        beta = -head / alpha;
        return true;
    }

    template <typename type>
    static inline void apply_householder_left(
        type* matrix_a,
        const int matrix_cols,
        const int row,
        const int col,
        const int rows,
        const int cols,
        const type* vector,
        const type beta
    ) {
        if (beta == 0) {
            return;
        }
        for (int c = 0; c < cols; ++c) {
            type* const column_base = matrix_a + (row * matrix_cols) + col + c;
            type projection = column_base[0];
            for (int i = 1; i < rows; ++i) {
                projection += vector[i - 1] * column_base[i * matrix_cols];
            }
            projection *= beta;
            column_base[0] -= projection;
            for (int i = 1; i < rows; ++i) {
                column_base[i * matrix_cols] -= vector[i - 1] * projection;
            }
        }
    }

    template <typename type>
    static inline void decompose_qr_householder_in_place(
        type* matrix_a,
        const int rows,
        const int cols,
        type* betas,
        type* workspace
    ) {
        ASSERT(rows > 0, "Matrix must have rows.");
        ASSERT(cols > 0, "Matrix must have cols.");
        const int reflections = (rows - 1 < cols) ? (rows - 1) : cols;
        for (int k = 0; k < reflections; ++k) {
            type beta = 0;
            type alpha = 0;
            const bool non_trivial = householder_vector(matrix_a + (k * cols) + k, rows - k, cols, workspace, beta, alpha);
            betas[k] = beta;
            if (!non_trivial) {
                continue;
            }
            if (k + 1 < cols) {
                apply_householder_left(matrix_a, cols, k, k + 1, rows - k, cols - k - 1, workspace, beta);
            }
            matrix_a[(k * cols) + k] = alpha;
            for (int i = k + 1; i < rows; ++i) {
                matrix_a[(i * cols) + k] = workspace[i - k - 1];
            }
        }
    }

    template <typename type>
    static inline void apply_householder_q_transpose_left(
        const type* matrix_qr,
        const int qr_cols,
        const type* betas,
        const int reflections,
        type* matrix_b,
        const int rows,
        const int cols,
        type* workspace
    ) {
        for (int k = 0; k < reflections; ++k) {
            if (betas[k] == 0) {
                continue;
            }
            for (int i = k + 1; i < rows; ++i) {
                workspace[i - k - 1] = matrix_qr[(i * qr_cols) + k];
            }
            apply_householder_left(matrix_b, cols, k, 0, rows - k, cols, workspace, betas[k]);
        }
    }

    template <typename type>
    static inline void decompose_qr_householder(
        const type* matrix_a,
        const int rows,
        const int cols,
        type* matrix_q,
        type* matrix_r,
        const bool thin,
        type* workspace
    ) {
        ASSERT(rows > 0, "Matrix must have rows.");
        ASSERT(cols > 0, "Matrix must have cols.");
        const int diagonal = (rows < cols) ? rows : cols;
        const int q_cols = thin ? diagonal : rows;
        type* const factored = workspace;
        type* const betas = workspace + (rows * cols);
        type* const scratch = betas + diagonal;
        for (int i = 0; i < rows * cols; ++i) {
            factored[i] = matrix_a[i];
        }
        for (int i = 0; i < diagonal; ++i) {
            betas[i] = 0;
        }
        decompose_qr_householder_in_place(factored, rows, cols, betas, scratch);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix_r[(i * cols) + j] = (j >= i) ? factored[(i * cols) + j] : static_cast<type>(0);
            }
        }
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < q_cols; ++j) {
                matrix_q[(i * q_cols) + j] = (i == j) ? static_cast<type>(1) : static_cast<type>(0);
            }
        }
        const int reflections = (rows - 1 < cols) ? (rows - 1) : cols;
        for (int k = reflections; k-- > 0;) {
            if (betas[k] == 0) {
                continue;
            }
            for (int i = k + 1; i < rows; ++i) {
                scratch[i - k - 1] = factored[(i * cols) + k];
            }
            apply_householder_left(matrix_q, q_cols, k, 0, rows - k, q_cols, scratch, betas[k]);
        }
    }

    template <typename type>
    static inline type givens(const type a, const type b, type& c, type& s) {
        if (b == 0) {
            c = 1;
            s = 0;
            return a;
        }
        if (math::abs(b) > math::abs(a)) {
            const type ratio = a / b;
            const type root = math::sqrt(static_cast<type>(1) + (ratio * ratio));
            const type magnitude = math::abs(b) * root;
            s = b / magnitude;
            c = s * ratio;
            return magnitude;
        }
        const type ratio = b / a;
        const type root = math::sqrt(static_cast<type>(1) + (ratio * ratio));
        const type magnitude = math::abs(a) * root;
        c = a / magnitude;
        s = c * ratio;
        return magnitude;
    }

    template <typename type>
    static inline void apply_givens_left(
        type* matrix_a,
        const int matrix_cols,
        const int row_a,
        const int row_b,
        const int col,
        const int cols,
        const type c,
        const type s
    ) {
        type* const a = matrix_a + (row_a * matrix_cols) + col;
        type* const b = matrix_a + (row_b * matrix_cols) + col;
        for (int i = 0; i < cols; ++i) {
            const type value_a = a[i];
            const type value_b = b[i];
            a[i] = (c * value_a) + (s * value_b);
            b[i] = (c * value_b) - (s * value_a);
        }
    }

    template <typename type>
    static inline bool solve_upper_triangular(
        const type* matrix_r,
        const int matrix_cols,
        const int size,
        const type* column_rhs,
        type* column_solution
    ) {
        type scale = 0;
        for (int i = 0; i < size; ++i) {
            scale = math::max(scale, math::abs(matrix_r[(i * matrix_cols) + i]));
        }
        const type epsilon = scale * static_cast<type>(1e-14);
        for (int i = 0; i < size; ++i) {
            if (!(math::abs(matrix_r[(i * matrix_cols) + i]) > epsilon)) {
                for (int j = 0; j < size; ++j) {
                    column_solution[j] = 0;
                }
                return false;
            }
        }
        for (int i = size; i-- > 0;) {
            type value = column_rhs[i];
            for (int j = i + 1; j < size; ++j) {
                value -= matrix_r[(i * matrix_cols) + j] * column_solution[j];
            }
            column_solution[i] = value / matrix_r[(i * matrix_cols) + i];
        }
        return true;
    }
}

#endif // ZEROSLAM_MATH_MATRIX_DECOMPOSITION_QR_HPP
