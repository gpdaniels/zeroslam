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
#ifndef ZEROSLAM_MATRIX_DECOMPOSITION_LOWER_UPPER_HPP
#define ZEROSLAM_MATRIX_DECOMPOSITION_LOWER_UPPER_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace matrix {
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
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                matrix_u[y * width + x] = matrix[y * width + x];
            }
            for (int x = 0; x < height; ++x) {
                matrix_l[y * height + x] = type(0);
                matrix_p[y * height + x] = static_cast<type>(x == y);
            }
        }

        if (swap_count) {
            *swap_count = 0;
        }

        constexpr static const auto abs = [](type value) -> type {
            if ((value + type(0)) < 0) {
                return -value;
            }
            return value;
        };

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

                // Select the pivot.
                type max_element;
                {
                    type sum_lower_upper = 0;
                    for (int k = 0; k < index_row; ++k) {
                        sum_lower_upper += matrix_l[index_row * height + k] * matrix_u[k * width + index_row];
                    }
                    max_element = matrix_u[index_row * width + index_row] - sum_lower_upper;
                }
                int index_swap = index_row;
                for (int index_row_remaining = index_row + 1; index_row_remaining < height; ++index_row_remaining) {
                    type sum_lower_upper = 0;
                    for (int k = 0; k < index_row; ++k) {
                        sum_lower_upper += matrix_l[index_row_remaining * height + k] * matrix_u[k * width + index_row];
                    }
                    const type candidate = matrix_u[index_row_remaining * width + index_row] - sum_lower_upper;
                    if (abs(candidate) > abs(max_element)) {
                        max_element = candidate;
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

                // The pivot row is now in place at index_row, store its already-computed eliminated value directly as the final upper triangular diagonal entry.
                matrix_u[index_row * width + index_row] = max_element;

                if (abs(matrix_u[index_row * width + index_row]) < type(1e-6)) {
                    return false;
                }

                // Calculate the lower matrix.
                for (int i = index_row + 1; i < height; ++i) {
                    type sum_lower_upper = 0;
                    for (int k = 0; k < index_row; ++k) {
                        sum_lower_upper += matrix_l[i * height + k] * matrix_u[k * width + index_row];
                    }
                    matrix_l[i * height + index_row] = (matrix_u[i * width + index_row] - sum_lower_upper) / matrix_u[index_row * width + index_row];
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
        constexpr static const auto abs = [](type value) -> type {
            if ((value + type(0)) < 0) {
                return -value;
            }
            return value;
        };

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
            if (abs(matrix_l[index_row * static_cast<size_t>(height) + index_row]) < type(1e-6)) {
                return false;
            }
            matrix_solution[index_row] /= matrix_l[index_row * static_cast<size_t>(height) + index_row];
        }

        // Backward solve upper * solution = matrix_solution, in place.
        for (size_t i = static_cast<size_t>(height); i-- > 0;) {
            for (size_t j = i + 1; j < static_cast<size_t>(height); ++j) {
                matrix_solution[i] -= matrix_u[i * static_cast<size_t>(width) + j] * matrix_solution[j];
            }
            if (abs(matrix_u[i * static_cast<size_t>(width) + i]) < type(1e-6)) {
                return false;
            }
            matrix_solution[i] /= matrix_u[i * static_cast<size_t>(width) + i];
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
        type* matrix_l = new type[static_cast<size_t>(height * height)];
        type* matrix_u = new type[static_cast<size_t>(width * height)];
        type* matrix_p = new type[static_cast<size_t>(height * height)];
        if (!decompose_lower_upper(matrix_lhs, width, height, matrix_l, matrix_u, matrix_p)) {
            delete[] matrix_l;
            delete[] matrix_u;
            delete[] matrix_p;
            return false;
        }
        if (!solve_lower_upper(matrix_l, matrix_u, matrix_p, matrix_rhs, width, height, matrix_solution)) {
            delete[] matrix_l;
            delete[] matrix_u;
            delete[] matrix_p;
            return false;
        }
        delete[] matrix_l;
        delete[] matrix_u;
        delete[] matrix_p;
        return true;
    }
}

#endif // ZEROSLAM_MATRIX_DECOMPOSITION_LOWER_UPPER_HPP
