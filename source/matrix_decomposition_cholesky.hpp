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
#ifndef MATRIX_DECOMPOSITION_CHOLESKY_HPP
#define MATRIX_DECOMPOSITION_CHOLESKY_HPP

#include "assert.hpp"
#include "math.hpp"

namespace matrix {
    template <typename type>
    constexpr static inline void decompose_cholesky(
        const type* __restrict matrix_a,
        const int cols,
        const int rows,
        type* __restrict matrix_l
    ) {
        ASSERT(cols == rows, "Matrix must be square.");
        for (int i = 0; i < cols * rows; ++i) {
            matrix_l[i] = 0;
        }
        for (int i = 0; i < rows; ++i) {
            for (int k = 0; k < i; ++k) {
                type value = matrix_a[i * cols + k];
                for (int j = 0; j < k; ++j) {
                    value -= matrix_l[i * cols + j] * matrix_l[k * cols + j];
                }
                matrix_l[i * cols + k] = value / matrix_l[k * cols + k];
            }
            type value = matrix_a[i * cols + i];
            for (int j = 0; j < i; ++j) {
                value -= matrix_l[i * cols + j] * matrix_l[i * cols + j];
            }
            matrix_l[i * cols + i] = math::sqrt(value);
        }
    }

    template <typename type>
    static inline bool solve_cholesky(
        const type* __restrict matrix_l,
        const type* __restrict column_rhs,
        const int cols,
        const int rows,
        type* __restrict column_solution
    ) {
        ASSERT(cols == rows, "Matrix must be square.");
        type* lower_solution = new type[rows];
        for (int index_row = 0; index_row < rows; ++index_row) {
            lower_solution[index_row] = column_rhs[index_row];
            for (int j = 0; j < index_row; ++j) {
                lower_solution[index_row] -= matrix_l[index_row * cols + j] * lower_solution[j];
            }
            if (math::abs(matrix_l[index_row * cols + index_row]) < 1e-6) {
                delete[] lower_solution;
                return false;
            }
            lower_solution[index_row] /= matrix_l[index_row * cols + index_row];
        }
        for (int i = rows; i-- > 0;) {
            column_solution[i] = lower_solution[i];
            for (int j = i + 1; j < rows; ++j) {
                column_solution[i] -= matrix_l[j * cols + i] * column_solution[j];
            }
            if (math::abs(matrix_l[i * cols + i]) < 1e-6) {
                delete[] lower_solution;
                return false;
            }
            column_solution[i] /= matrix_l[i * cols + i];
        }
        delete[] lower_solution;
        return true;
    }
}

#endif // MATRIX_DECOMPOSITION_CHOLESKY_HPP