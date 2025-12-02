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
#ifndef MATRIX_DECOMPOSITION_LOWER_UPPER_HPP
#define MATRIX_DECOMPOSITION_LOWER_UPPER_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace matrix {
    template <typename type>
    static bool decompose_lower_upper(
        const type* __restrict matrix,
        const int width,
        const int height,
        type* __restrict matrix_l, // height x height
        type* __restrict matrix_u, // width x height
        type* __restrict matrix_p, // height x height
        int* swap_count = nullptr  // Number of pivot swaps made.
    ) {
        // Initialise matrices: Lower and Upper to zero, and Pivot to the identity matrix.
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                matrix_u[y * width + x] = type(0);
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

        // Make a permutation matrix from the identity matrix, and count how many swaps were made.
        for (int index_row = 0; index_row < height && index_row < width; ++index_row) {
            // Save the value on the leading diagonal and the current row index as the swap index.
            type max_element = matrix[index_row * width + index_row];
            int index_swap = index_row;
            // Check remaining rows in this column for larger values.
            for (int index_row_remaining = index_row + 1; index_row_remaining < height; ++index_row_remaining) {
                // If value is larger update the swap index.
                if (abs(matrix[index_row_remaining * width + index_row]) > abs(max_element)) {
                    max_element = matrix[index_row_remaining * width + index_row];
                    index_swap = index_row_remaining;
                }
            }
            // If there is a row with a greater value than the leading diagonal, swap the rows.
            if (index_row != index_swap) {
                for (int x = 0; x < height; ++x) {
                    const type temp = matrix_p[index_row * height + x];
                    matrix_p[index_row * height + x] = matrix_p[index_swap * height + x];
                    matrix_p[index_swap * height + x] = temp;
                }
                if (swap_count) {
                    ++(*swap_count);
                }
            }
        }

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

        type* pivoted_matrix = new type[height * width];
        matrix_multiply(&matrix_p[0], height, height, &matrix[0], width, height, &pivoted_matrix[0]);

        // Calculate the lower and upper matrices.
        for (int index_row = 0; index_row < height; ++index_row) {
            matrix_l[index_row * height + index_row] = type(1);
            if (index_row < width) {
                // Calculate upper matrix.
                for (int i = 0; i <= index_row; ++i) {
                    type sum_lower_upper = 0;
                    for (int k = 0; k < i; ++k) {
                        sum_lower_upper += matrix_l[i * height + k] * matrix_u[k * width + index_row];
                    }
                    matrix_u[i * width + index_row] = pivoted_matrix[i * width + index_row] - sum_lower_upper;
                }

                // Calculate the lower matrix.
                for (int i = index_row; i < height; ++i) {
                    type sum_lower_upper = 0;
                    for (int k = 0; k < index_row; ++k) {
                        sum_lower_upper += matrix_l[i * height + k] * matrix_u[k * width + index_row];
                    }
                    if (abs(matrix_u[index_row * width + index_row]) < type(1e-6)) {
                        delete[] pivoted_matrix;
                        return false;
                    }
                    matrix_l[i * height + index_row] = (pivoted_matrix[i * width + index_row] - sum_lower_upper) / matrix_u[index_row * width + index_row];
                }
            }
        }

        if (width > height) {
            for (int row = 0; row < height; ++row) {
                for (int col = height; col < width; ++col) {
                    type sum_lower_upper = 0;
                    for (int k = 0; k < height; ++k) {
                        sum_lower_upper += matrix_l[row * height + k] * matrix_u[k * width + col];
                    }
                    matrix_u[row * width + col] = pivoted_matrix[row * width + col] - sum_lower_upper;
                }
            }
        }

        delete[] pivoted_matrix;
        return true;
    }

    template <typename type>
    static bool solve_lower_upper(
        const type* __restrict matrix_l,   // height x height
        const type* __restrict matrix_u,   // width x height
        const type* __restrict matrix_p,   // height x height
        const type* __restrict matrix_rhs, // 1 x height
        const int width,
        const int height,
        type* __restrict matrix_solution // 1 x height
    ) {
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

        type* lower_solution = new type[1 * height];
        type* pivoted_matrix = new type[1 * height];
        matrix_multiply(&matrix_p[0], height, height, &matrix_rhs[0], 1, height, &pivoted_matrix[0]);

        constexpr static const auto abs = [](type value) -> type {
            if ((value + type(0)) < 0) {
                return -value;
            }
            return value;
        };

        // Forward solve lower * lower_solution = pivoted_matrix.
        for (size_t index_row = 0; index_row < static_cast<size_t>(height); ++index_row) {
            lower_solution[index_row] = pivoted_matrix[index_row];
            for (size_t j = 0; j < index_row; ++j) {
                lower_solution[index_row] -= matrix_l[index_row * height + j] * lower_solution[j];
            }
            if (abs(matrix_l[index_row * height + index_row]) < type(1e-6)) {
                delete[] pivoted_matrix;
                delete[] lower_solution;
                return false;
            }
            lower_solution[index_row] /= matrix_l[index_row * height + index_row];
        }

        // Backward solve upper * solution = lower_solution.
        for (size_t i = static_cast<size_t>(height); i-- > 0;) {
            matrix_solution[i] = lower_solution[i];
            for (size_t j = i + 1; j < static_cast<size_t>(height); ++j) {
                matrix_solution[i] -= matrix_u[i * width + j] * matrix_solution[j];
            }
            if (abs(matrix_u[i * width + i]) < type(1e-6)) {
                delete[] pivoted_matrix;
                delete[] lower_solution;
                return false;
            }
            matrix_solution[i] /= matrix_u[i * width + i];
        }

        delete[] pivoted_matrix;
        delete[] lower_solution;
        return true;
    }

    template <typename type>
    static bool solve_lower_upper(
        const type* __restrict matrix_lhs, // width x height
        const type* __restrict matrix_rhs, // 1 x height
        const int width,
        const int height,
        type* __restrict matrix_solution // 1 x height
    ) {
        type* matrix_l = new type[height * height];
        type* matrix_u = new type[width * height];
        type* matrix_p = new type[height * height];
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

#endif // MATRIX_DECOMPOSITION_LOWER_UPPER_HPP
