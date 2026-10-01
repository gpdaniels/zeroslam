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
#ifndef ZEROSLAM_MATH_MATRIX_DECOMPOSITION_LOWER_UPPER_HPP
#define ZEROSLAM_MATH_MATRIX_DECOMPOSITION_LOWER_UPPER_HPP

#include "core/assert.hpp"
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace math {
    template <typename type>
    static inline bool decompose_lower_upper(
        const type* __restrict matrix,
        const int width,
        const int height,
        type* __restrict matrix_l, // height x height
        type* __restrict matrix_u, // width x height
        type* __restrict matrix_p, // height x height
        int* swap_count = nullptr  // Number of pivot swaps made.
    ) {
        type scale = type(0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const type value = matrix[y * width + x];
                if (!math::isfinite(value)) {
                    return false;
                }
                scale = math::max(scale, math::abs(value));
                matrix_u[y * width + x] = value;
            }
            for (int x = 0; x < height; ++x) {
                matrix_l[y * height + x] = type(0);
                matrix_p[y * height + x] = static_cast<type>(x == y);
            }
        }

        if (swap_count) {
            *swap_count = 0;
        }

        // A pivot within rounding error of the largest entry is numerically zero, which makes the test independent of the scale of the matrix.
        const type pivot_tolerance = static_cast<type>(math::max(width, height)) * math::epsilon<type>() * scale;

        // Calculate the lower and upper matrices with partial pivoting.
        for (int index_row = 0; index_row < height; ++index_row) {
            matrix_l[index_row * height + index_row] = type(1);
            if (index_row < width) {
                // Calculate the already-finalized upper matrix entries above the pivot.
                for (int i = 0; i < index_row; ++i) {
                    type sum_lower_upper = 0;
                    for (int k = 0; k < i; ++k) {
                        sum_lower_upper += matrix_l[i * height + k] * matrix_u[k * width + index_row];
                    }
                    matrix_u[i * width + index_row] = matrix_u[i * width + index_row] - sum_lower_upper;
                }

                // Eliminate the column on and below the diagonal in place, and select the largest as the pivot.
                int index_swap = index_row;
                for (int index_row_remaining = index_row; index_row_remaining < height; ++index_row_remaining) {
                    type sum_lower_upper = 0;
                    for (int k = 0; k < index_row; ++k) {
                        sum_lower_upper += matrix_l[index_row_remaining * height + k] * matrix_u[k * width + index_row];
                    }
                    matrix_u[index_row_remaining * width + index_row] = matrix_u[index_row_remaining * width + index_row] - sum_lower_upper;
                    if (math::abs(matrix_u[index_row_remaining * width + index_row]) > math::abs(matrix_u[index_swap * width + index_row])) {
                        index_swap = index_row_remaining;
                    }
                }

                // If a lower row's eliminated value is larger in magnitude than the pivot row's, swap the two rows.
                if (index_row != index_swap) {
                    for (int x = 0; x < height; ++x) {
                        const type temp = matrix_p[index_row * height + x];
                        matrix_p[index_row * height + x] = matrix_p[index_swap * height + x];
                        matrix_p[index_swap * height + x] = temp;
                    }
                    for (int x = 0; x < index_row; ++x) {
                        const type temp = matrix_l[index_row * height + x];
                        matrix_l[index_row * height + x] = matrix_l[index_swap * height + x];
                        matrix_l[index_swap * height + x] = temp;
                    }
                    for (int x = 0; x < width; ++x) {
                        const type temp = matrix_u[index_row * width + x];
                        matrix_u[index_row * width + x] = matrix_u[index_swap * width + x];
                        matrix_u[index_swap * width + x] = temp;
                    }
                    if (swap_count) {
                        ++(*swap_count);
                    }
                }

                // Note the comparison is also false for a NaN pivot.
                const type pivot = matrix_u[index_row * width + index_row];
                if (!math::isfinite(pivot) || !(math::abs(pivot) > pivot_tolerance)) {
                    return false;
                }

                // Calculate the lower matrix from the eliminated values below the pivot.
                for (int i = index_row + 1; i < height; ++i) {
                    matrix_l[i * height + index_row] = matrix_u[i * width + index_row] / pivot;
                    matrix_u[i * width + index_row] = type(0);
                }
            }
        }

        if (width > height) {
            for (int col = height; col < width; ++col) {
                for (int row = 0; row < height; ++row) {
                    type sum_lower_upper = 0;
                    for (int k = 0; k < row; ++k) {
                        sum_lower_upper += matrix_l[row * height + k] * matrix_u[k * width + col];
                    }
                    matrix_u[row * width + col] = matrix_u[row * width + col] - sum_lower_upper;
                }
            }
        }

        return true;
    }

    template <typename type>
    static inline bool solve_lower_upper(
        const type* __restrict matrix_l,   // height x height
        const type* __restrict matrix_u,   // width x height
        const type* __restrict matrix_p,   // height x height
        const type* __restrict matrix_rhs, // 1 x height
        const int width,
        const int height,
        type* __restrict matrix_solution // 1 x height
    ) {
        ASSERT(width >= height, "The upper matrix must have a square leading block.");

        // Diagonal entries within rounding error of the largest entry of their triangle are numerically zero.
        type scale_l = type(0);
        type scale_u = type(0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x <= y; ++x) {
                scale_l = math::max(scale_l, math::abs(matrix_l[y * height + x]));
            }
            for (int x = y; x < height; ++x) {
                scale_u = math::max(scale_u, math::abs(matrix_u[y * width + x]));
            }
        }
        const type tolerance_l = static_cast<type>(height) * math::epsilon<type>() * scale_l;
        const type tolerance_u = static_cast<type>(height) * math::epsilon<type>() * scale_u;

        // Apply the permutation directly into the solution buffer.
        for (int index_row = 0; index_row < height; ++index_row) {
            for (int index_col = 0; index_col < height; ++index_col) {
                if (matrix_p[index_row * height + index_col] == type(1)) {
                    matrix_solution[index_row] = matrix_rhs[index_col];
                    break;
                }
            }
        }

        // Forward solve lower * matrix_solution = permuted matrix_rhs, in place.
        for (size_t index_row = 0; index_row < static_cast<size_t>(height); ++index_row) {
            for (size_t j = 0; j < index_row; ++j) {
                matrix_solution[index_row] -= matrix_l[index_row * static_cast<size_t>(height) + j] * matrix_solution[j];
            }
            const type diagonal = matrix_l[index_row * static_cast<size_t>(height) + index_row];
            if (!math::isfinite(diagonal) || !(math::abs(diagonal) > tolerance_l)) {
                return false;
            }
            matrix_solution[index_row] /= diagonal;
        }

        // Backward solve upper * solution = matrix_solution, in place.
        for (size_t i = static_cast<size_t>(height); i-- > 0;) {
            for (size_t j = i + 1; j < static_cast<size_t>(height); ++j) {
                matrix_solution[i] -= matrix_u[i * static_cast<size_t>(width) + j] * matrix_solution[j];
            }
            const type diagonal = matrix_u[i * static_cast<size_t>(width) + i];
            if (!math::isfinite(diagonal) || !(math::abs(diagonal) > tolerance_u)) {
                return false;
            }
            matrix_solution[i] /= diagonal;
        }

        return true;
    }

    template <typename type>
    static inline bool solve_lower_upper(
        const type* __restrict matrix_lhs, // width x height
        const type* __restrict matrix_rhs, // 1 x height
        const int width,
        const int height,
        type* __restrict matrix_solution // 1 x height
    ) {
        // Systems up to this size are factorised on the stack, only larger ones allocate.
        constexpr static const int stack_size = 16;
        if ((width <= stack_size) && (height <= stack_size)) {
            type matrix_l[stack_size * stack_size];
            type matrix_u[stack_size * stack_size];
            type matrix_p[stack_size * stack_size];
            if (!decompose_lower_upper(matrix_lhs, width, height, &matrix_l[0], &matrix_u[0], &matrix_p[0])) {
                return false;
            }
            return solve_lower_upper(&matrix_l[0], &matrix_u[0], &matrix_p[0], matrix_rhs, width, height, matrix_solution);
        }
        const size_t square_size = static_cast<size_t>(height) * static_cast<size_t>(height);
        std::vector<type> storage((square_size * 2) + (static_cast<size_t>(width) * static_cast<size_t>(height)));
        type* const matrix_l = storage.data();
        type* const matrix_p = matrix_l + square_size;
        type* const matrix_u = matrix_p + square_size;
        if (!decompose_lower_upper(matrix_lhs, width, height, matrix_l, matrix_u, matrix_p)) {
            return false;
        }
        return solve_lower_upper(matrix_l, matrix_u, matrix_p, matrix_rhs, width, height, matrix_solution);
    }
}

#endif // ZEROSLAM_MATH_MATRIX_DECOMPOSITION_LOWER_UPPER_HPP
