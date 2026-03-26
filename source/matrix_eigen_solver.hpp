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
#ifndef MATRIX_EIGEN_SOLVER_HPP
#define MATRIX_EIGEN_SOLVER_HPP

#include "math.hpp"

namespace matrix {
    /// @brief  Computes all eigenvalues and eigenvectors of a general matrix.
    /// @tparam type The scalar/numeric type to use, normally float or double.
    /// @tparam extract_eigenvectors If true the function will extract the eigen vectors and return them in the eigenvectors argument.
    /// @tparam sort_results If true the function will sort the resultant eigenvalues_complex and eigenvectors output arguments.
    /// @param  matrix The square input matrix used for the computation of eigen values and optionally eigen vectors.
    /// @param  size The size of the square input matrix.
    /// @param[out] eigenvalues_complex An array of complex eigen values in the format [real_0, imaginary_0, real_1, imaginary_1, ...]
    /// @param[out] eigenvectors An array of eigen vectors containing no complex numbers.
    /// @return True if the eigen values and optionally eigen vectors were computed successfully, false otherwise.
    template <typename type, bool extract_eigenvectors = true, bool sort_results = true>
    bool eigen_solver(const type* __restrict matrix, const int size, type* __restrict eigenvalues_complex, type* __restrict eigenvectors) {
        struct complex_number {
            type real;
            type imaginary;

            complex_number(type real_part = 0, type imaginary_part = 0)
                : real(real_part)
                , imaginary(imaginary_part) {
            }

            complex_number conjugate() const {
                return complex_number(this->real, -this->imaginary);
            }

            complex_number operator/(const complex_number& other) const {
                const type denominator = other.real * other.real + other.imaginary * other.imaginary;
                return complex_number(
                    (this->real * other.real + this->imaginary * other.imaginary) / denominator,
                    (this->imaginary * other.real - this->real * other.imaginary) / denominator
                );
            }
        };

        // Working arrays.
        type* working_matrix = new type[size * size]{};
        complex_number* eigenvalues = new complex_number[size];
        type* scaling = new type[size];
        int* permutation = new int[size];

        // Copy input matrix into working matrix.
        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size; ++col) {
                working_matrix[row * size + col] = matrix[row * size + col];
            }
        }

        // Initialize scaling to 1.
        for (int i = 0; i < size; ++i) {
            scaling[i] = type(1);
        }

        // Balance the matrix (scale rows/columns to reduce overflow/underflow).
        bool converged = false;
        while (!converged) {
            converged = true;

            for (int row_index = 0; row_index < size; ++row_index) {
                type row_sum = 0;
                type col_sum = 0;

                // Calculate sum of absolute values in row and column (excluding diagonal).
                for (int other_index = 0; other_index < size; ++other_index) {
                    if (other_index != row_index) {
                        col_sum += math::abs(working_matrix[other_index * size + row_index]);
                        row_sum += math::abs(working_matrix[row_index * size + other_index]);
                    }
                }

                // Only balance if both row and column have non-zero elements.
                // Note: Assuming `type` is a binary type and therefore its radix is 2.
                // Note: This should be true for all standard float and double types.
                if ((col_sum != 0.0) && (row_sum != 0.0)) {
                    const type original_sum = col_sum + row_sum;
                    type scale_factor = 1;
                    type col_sum_scaled = col_sum;

                    // Scale down if column sum is too small.
                    type lower_threshold = row_sum / 2;
                    while (col_sum_scaled < lower_threshold) {
                        scale_factor *= 2;
                        col_sum_scaled *= 2 * 2;
                    }

                    // Scale up if column sum is too large.
                    type upper_threshold = row_sum * 2;
                    while (col_sum_scaled > upper_threshold) {
                        scale_factor /= 2;
                        col_sum_scaled /= 2 * 2;
                    }

                    // Apply scaling if it provides sufficient improvement.
                    if (((col_sum_scaled + row_sum) / scale_factor) < (0.95 * original_sum)) {
                        converged = false;
                        const type inverse_scale = type(1) / scale_factor;
                        scaling[row_index] *= scale_factor;

                        // Scale row elements.
                        for (int col_index = 0; col_index < size; ++col_index) {
                            working_matrix[row_index * size + col_index] *= inverse_scale;
                        }

                        // Scale column elements.
                        for (int col_index = 0; col_index < size; ++col_index) {
                            working_matrix[col_index * size + row_index] *= scale_factor;
                        }
                    }
                }
            }
        }

        // Reduce to upper Hessenberg form with partial pivoting (householder reduction variant).
        for (int m = 1; m < size - 1; ++m) {
            type pivot_abs_max = 0;
            int pivot_row = m;
            for (int candidate_row = m; candidate_row < size; ++candidate_row) {
                if (math::abs(working_matrix[candidate_row * size + (m - 1)]) > math::abs(pivot_abs_max)) {
                    pivot_abs_max = working_matrix[candidate_row * size + (m - 1)];
                    pivot_row = candidate_row;
                }
            }
            permutation[m] = pivot_row;
            if (pivot_row != m) {
                for (int col = m - 1; col < size; ++col) {
                    const type temp = working_matrix[pivot_row * size + col];
                    working_matrix[pivot_row * size + col] = working_matrix[m * size + col];
                    working_matrix[m * size + col] = temp;
                }
                for (int row = 0; row < size; ++row) {
                    const type temp = working_matrix[row * size + pivot_row];
                    working_matrix[row * size + pivot_row] = working_matrix[row * size + m];
                    working_matrix[row * size + m] = temp;
                }
            }
            if (pivot_abs_max != 0.0) {
                for (int row = m + 1; row < size; ++row) {
                    type multiplier = working_matrix[row * size + (m - 1)];
                    if (multiplier != 0.0) {
                        multiplier /= pivot_abs_max;
                        working_matrix[row * size + (m - 1)] = multiplier;
                        for (int col = m; col < size; ++col) {
                            working_matrix[row * size + col] -= multiplier * working_matrix[m * size + col];
                        }
                        for (int r = 0; r < size; ++r) {
                            working_matrix[r * size + m] += multiplier * working_matrix[r * size + row];
                        }
                    }
                }
            }
        }

        // Initialize eigenvector accumulation matrix if requested.
        if constexpr (extract_eigenvectors) {
            for (int row = 0; row < size; ++row) {
                eigenvectors[row * size + row] = 1.0;
            }
            for (int mp = size - 2; mp > 0; --mp) {
                for (int k_idx = mp + 1; k_idx < size; ++k_idx) {
                    eigenvectors[k_idx * size + mp] = working_matrix[k_idx * size + (mp - 1)];
                }
                const int perm_row = permutation[mp];
                if (perm_row != mp) {
                    for (int col = mp; col < size; ++col) {
                        eigenvectors[mp * size + col] = eigenvectors[perm_row * size + col];
                        eigenvectors[perm_row * size + col] = 0.0;
                    }
                    eigenvectors[perm_row * size + mp] = 1.0;
                }
            }
        }

        // Compute and store the type epsilon.
        constexpr static const type epsilon = math::epsilon<type>();

        // Compute and store the magnitude of the working matrix.
        type working_matrix_norm = 0;
        for (int i = 0; i < size; ++i) {
            for (int j = math::max(i - 1, 0); j < size; ++j) {
                working_matrix_norm += math::abs(working_matrix[i * size + j]);
            }
        }

        // Compute eigenvalues and optionally eigenvectors using QR algorithm.
        {
            int current_dimension = size - 1;
            type cumulative_shift = 0.0;

            while (current_dimension >= 0) {
                int iteration_count = 0;

                int split_index;
                do {
                    // Find where subdiagonal elements are negligible.
                    split_index = 0;
                    for (int l = current_dimension; l > 0; --l) {
                        const type diagonal_sum = math::abs(working_matrix[(l - 1) * size + (l - 1)]) + math::abs(working_matrix[l * size + l]);
                        const type threshold = (diagonal_sum == 0.0) ? working_matrix_norm : diagonal_sum;

                        if (math::abs(working_matrix[l * size + (l - 1)]) <= epsilon * threshold) {
                            working_matrix[l * size + (l - 1)] = 0.0;
                            split_index = l;
                            break;
                        }
                    }

                    const type diagonal_element = working_matrix[current_dimension * size + current_dimension];

                    // Case 1: Single eigenvalue converged.
                    if (split_index == current_dimension) {
                        eigenvalues[current_dimension] = working_matrix[current_dimension * size + current_dimension] = diagonal_element + cumulative_shift;
                        --current_dimension;
                    }
                    else {
                        const type prev_diagonal = working_matrix[(current_dimension - 1) * size + (current_dimension - 1)];
                        const type subdiagonal_product = working_matrix[current_dimension * size + (current_dimension - 1)] *
                                                         working_matrix[(current_dimension - 1) * size + current_dimension];

                        // Case 2: 2x2 block converged.
                        if (split_index == current_dimension - 1) {
                            const type half_trace_diff = 0.5 * (prev_diagonal - diagonal_element);
                            const type discriminant = half_trace_diff * half_trace_diff + subdiagonal_product;
                            const type sqrt_abs_discriminant = math::sqrt(math::abs(discriminant));
                            const type shifted_diagonal = diagonal_element + cumulative_shift;

                            working_matrix[current_dimension * size + current_dimension] = shifted_diagonal;
                            working_matrix[(current_dimension - 1) * size + (current_dimension - 1)] = prev_diagonal + cumulative_shift;

                            if (discriminant >= 0.0) {
                                // Real eigenvalues.
                                const type eigenvalue_sum = half_trace_diff + math::copysign(sqrt_abs_discriminant, half_trace_diff);
                                eigenvalues[current_dimension - 1] = eigenvalues[current_dimension] = shifted_diagonal + eigenvalue_sum;

                                if (eigenvalue_sum != 0.0) {
                                    eigenvalues[current_dimension] = shifted_diagonal - subdiagonal_product / eigenvalue_sum;
                                }

                                // Apply Givens rotation.
                                const type offdiag_element = working_matrix[current_dimension * size + (current_dimension - 1)];
                                const type norm_scale = math::abs(offdiag_element) + math::abs(eigenvalue_sum);
                                const type cos_unnorm = offdiag_element / norm_scale;
                                const type sin_unnorm = eigenvalue_sum / norm_scale;
                                const type rotation_norm = math::sqrt(cos_unnorm * cos_unnorm + sin_unnorm * sin_unnorm);
                                const type cos_factor = cos_unnorm / rotation_norm;
                                const type sin_factor = sin_unnorm / rotation_norm;

                                // Apply rotation to rows.
                                for (int j = current_dimension - 1; j < size; ++j) {
                                    const type temp = working_matrix[(current_dimension - 1) * size + j];
                                    working_matrix[(current_dimension - 1) * size + j] = sin_factor * temp + cos_factor * working_matrix[current_dimension * size + j];
                                    working_matrix[current_dimension * size + j] = sin_factor * working_matrix[current_dimension * size + j] - cos_factor * temp;
                                }

                                // Apply rotation to columns.
                                for (int i = 0; i <= current_dimension; ++i) {
                                    const type temp = working_matrix[i * size + (current_dimension - 1)];
                                    working_matrix[i * size + (current_dimension - 1)] = sin_factor * temp + cos_factor * working_matrix[i * size + current_dimension];
                                    working_matrix[i * size + current_dimension] = sin_factor * working_matrix[i * size + current_dimension] - cos_factor * temp;
                                }

                                if constexpr (extract_eigenvectors) {
                                    for (int i = 0; i < size; ++i) {
                                        const type temp = eigenvectors[i * size + (current_dimension - 1)];
                                        eigenvectors[i * size + (current_dimension - 1)] = sin_factor * temp + cos_factor * eigenvectors[i * size + current_dimension];
                                        eigenvectors[i * size + current_dimension] = sin_factor * eigenvectors[i * size + current_dimension] - cos_factor * temp;
                                    }
                                }
                            }
                            else {
                                // Complex conjugate eigenvalues.
                                eigenvalues[current_dimension] = complex_number(shifted_diagonal + half_trace_diff, -sqrt_abs_discriminant);
                                eigenvalues[current_dimension - 1] = eigenvalues[current_dimension].conjugate();
                            }
                            current_dimension -= 2;
                        }
                        // Case 3: No convergence - perform QR step.
                        else {
                            if (iteration_count == 30) {
                                delete[] working_matrix;
                                delete[] eigenvalues;
                                delete[] scaling;
                                delete[] permutation;
                                return false;
                            }

                            type x_shift = diagonal_element;
                            type y_shift = prev_diagonal;
                            type w_shift = subdiagonal_product;

                            // Apply exceptional shift.
                            if ((iteration_count == 10) || (iteration_count == 20)) {
                                cumulative_shift += diagonal_element;
                                for (int i = 0; i < current_dimension + 1; ++i) {
                                    working_matrix[i * size + i] -= diagonal_element;
                                }

                                const type shift_scale = math::abs(working_matrix[current_dimension * size + (current_dimension - 1)]) +
                                                         math::abs(working_matrix[(current_dimension - 1) * size + (current_dimension - 2)]);
                                x_shift = y_shift = 0.75 * shift_scale;
                                w_shift = -0.4375 * shift_scale * shift_scale;
                            }

                            ++iteration_count;

                            // Find starting position for QR step.
                            int bulge_start = current_dimension - 2;
                            for (int m = current_dimension - 2; m >= split_index; --m) {
                                const type z_elem = working_matrix[m * size + m];
                                const type r_diff = x_shift - z_elem;
                                const type s_diff = y_shift - z_elem;

                                const type p_unnorm = (r_diff * s_diff - w_shift) / working_matrix[(m + 1) * size + m] + working_matrix[m * size + (m + 1)];
                                const type q_unnorm = working_matrix[(m + 1) * size + (m + 1)] - z_elem - r_diff - s_diff;
                                const type r_unnorm = working_matrix[(m + 2) * size + (m + 1)];

                                const type norm = math::abs(p_unnorm) + math::abs(q_unnorm) + math::abs(r_unnorm);
                                const type p_norm = p_unnorm / norm;
                                const type q_norm = q_unnorm / norm;
                                const type r_norm = r_unnorm / norm;

                                if (m == split_index) {
                                    bulge_start = m;
                                    break;
                                }

                                const type subdiag_contribution = math::abs(working_matrix[m * size + (m - 1)]) * (math::abs(q_norm) + math::abs(r_norm));
                                const type diagonal_contribution = math::abs(p_norm) * (math::abs(working_matrix[(m - 1) * size + (m - 1)]) + math::abs(z_elem) + math::abs(working_matrix[(m + 1) * size + (m + 1)]));

                                if (subdiag_contribution <= epsilon * diagonal_contribution) {
                                    bulge_start = m;
                                    break;
                                }
                            }

                            // Zero out elements below bulge.
                            for (int i = bulge_start; i < current_dimension - 1; ++i) {
                                working_matrix[(i + 2) * size + i] = 0.0;
                                if (i != bulge_start) {
                                    working_matrix[(i + 2) * size + (i - 1)] = 0.0;
                                }
                            }

                            // Chase the bulge.
                            for (int k = bulge_start; k < current_dimension; ++k) {
                                type p_coef, q_coef, r_coef;

                                if (k != bulge_start) {
                                    p_coef = working_matrix[k * size + (k - 1)];
                                    q_coef = working_matrix[(k + 1) * size + (k - 1)];
                                    r_coef = (k + 1 != current_dimension) ? working_matrix[(k + 2) * size + (k - 1)] : 0.0;

                                    const type coef_norm = math::abs(p_coef) + math::abs(q_coef) + math::abs(r_coef);
                                    if (coef_norm != 0.0) {
                                        p_coef /= coef_norm;
                                        q_coef /= coef_norm;
                                        r_coef /= coef_norm;
                                    }
                                }
                                else {
                                    // First iteration uses values from bulge search.
                                    const type z_elem = working_matrix[k * size + k];
                                    const type r_diff = x_shift - z_elem;
                                    const type s_diff = y_shift - z_elem;

                                    const type p_unnorm = (r_diff * s_diff - w_shift) / working_matrix[(k + 1) * size + k] + working_matrix[k * size + (k + 1)];
                                    const type q_unnorm = working_matrix[(k + 1) * size + (k + 1)] - z_elem - r_diff - s_diff;
                                    const type r_unnorm = working_matrix[(k + 2) * size + (k + 1)];

                                    const type norm = math::abs(p_unnorm) + math::abs(q_unnorm) + math::abs(r_unnorm);
                                    p_coef = p_unnorm / norm;
                                    q_coef = q_unnorm / norm;
                                    r_coef = r_unnorm / norm;
                                }

                                const type householder_norm = math::copysign(math::sqrt(p_coef * p_coef + q_coef * q_coef + r_coef * r_coef), p_coef);

                                if (householder_norm != 0.0) {
                                    if (k == bulge_start) {
                                        if (split_index != bulge_start) {
                                            working_matrix[k * size + (k - 1)] = -working_matrix[k * size + (k - 1)];
                                        }
                                    }
                                    else {
                                        const type original_norm = math::abs(working_matrix[k * size + (k - 1)]) + math::abs(working_matrix[(k + 1) * size + (k - 1)]) +
                                                                   ((k + 1 != current_dimension) ? math::abs(working_matrix[(k + 2) * size + (k - 1)]) : 0.0);
                                        working_matrix[k * size + (k - 1)] = -householder_norm * original_norm;
                                    }

                                    const type p_normalized = (p_coef + householder_norm) / householder_norm;
                                    const type q_normalized = q_coef / householder_norm;
                                    const type r_normalized = r_coef / householder_norm;
                                    const type q_ratio = q_coef / (p_coef + householder_norm);
                                    const type r_ratio = r_coef / (p_coef + householder_norm);

                                    // Apply Householder transformation to rows.
                                    for (int j = k; j < size; ++j) {
                                        type dot_product = working_matrix[k * size + j] + q_ratio * working_matrix[(k + 1) * size + j];
                                        if (k + 1 != current_dimension) {
                                            dot_product += r_ratio * working_matrix[(k + 2) * size + j];
                                            working_matrix[(k + 2) * size + j] -= dot_product * r_normalized;
                                        }
                                        working_matrix[(k + 1) * size + j] -= dot_product * q_normalized;
                                        working_matrix[k * size + j] -= dot_product * p_normalized;
                                    }

                                    // Apply Householder transformation to columns.
                                    const int column_limit = (current_dimension < k + 3) ? current_dimension : k + 3;
                                    for (int i = 0; i <= column_limit; ++i) {
                                        type dot_product = p_normalized * working_matrix[i * size + k] + q_normalized * working_matrix[i * size + (k + 1)];
                                        if (k + 1 != current_dimension) {
                                            dot_product += r_normalized * working_matrix[i * size + (k + 2)];
                                            working_matrix[i * size + (k + 2)] -= dot_product * r_ratio;
                                        }
                                        working_matrix[i * size + (k + 1)] -= dot_product * q_ratio;
                                        working_matrix[i * size + k] -= dot_product;
                                    }

                                    if constexpr (extract_eigenvectors) {
                                        for (int i = 0; i < size; ++i) {
                                            type dot_product = p_normalized * eigenvectors[i * size + k] + q_normalized * eigenvectors[i * size + (k + 1)];
                                            if (k + 1 != current_dimension) {
                                                dot_product += r_normalized * eigenvectors[i * size + (k + 2)];
                                                eigenvectors[i * size + (k + 2)] -= dot_product * r_ratio;
                                            }
                                            eigenvectors[i * size + (k + 1)] -= dot_product * q_ratio;
                                            eigenvectors[i * size + k] -= dot_product;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } while (split_index + 1 < current_dimension);
            }

            // Back substitution to find vectors.
            if constexpr (extract_eigenvectors) {
                if (working_matrix_norm != 0.0) {
                    // Back-substitute to compute eigenvectors.
                    for (int vec_index = size - 1; vec_index >= 0; --vec_index) {
                        const type eigenvalue_real = eigenvalues[vec_index].real;
                        const type eigenvalue_imag = eigenvalues[vec_index].imaginary;

                        // Real eigenvalue: compute real eigenvector.
                        if (eigenvalue_imag == 0.0) {
                            int row_start = vec_index;
                            working_matrix[vec_index * size + vec_index] = 1.0;

                            // State variables that persist across loop iterations for complex conjugate pairs.
                            type prev_diagonal_diff = 0.0;
                            type prev_accumulator = 0.0;

                            for (int i = vec_index - 1; i >= 0; --i) {
                                const type diagonal_diff = working_matrix[i * size + i] - eigenvalue_real;

                                type accumulator = 0.0;
                                for (int j = row_start; j <= vec_index; ++j) {
                                    accumulator += working_matrix[i * size + j] * working_matrix[j * size + vec_index];
                                }

                                if (eigenvalues[i].imaginary < 0.0) {
                                    // Store for use with complex conjugate pair in next iteration.
                                    prev_diagonal_diff = diagonal_diff;
                                    prev_accumulator = accumulator;
                                }
                                else {
                                    row_start = i;

                                    if (eigenvalues[i].imaginary == 0.0) {
                                        // Simple real eigenvalue case.
                                        const type divisor = (diagonal_diff != 0.0) ? diagonal_diff : epsilon * working_matrix_norm;
                                        working_matrix[i * size + vec_index] = -accumulator / divisor;
                                    }
                                    else {
                                        // Complex conjugate pair, solve 2x2 system.
                                        const type x_elem = working_matrix[i * size + (i + 1)];
                                        const type y_elem = working_matrix[(i + 1) * size + i];
                                        const type discriminant = math::sqr(eigenvalues[i].real - eigenvalue_real) + math::sqr(eigenvalues[i].imaginary);

                                        const type solution = (x_elem * prev_accumulator - prev_diagonal_diff * accumulator) / discriminant;
                                        working_matrix[i * size + vec_index] = solution;

                                        if (math::abs(x_elem) > math::abs(prev_diagonal_diff)) {
                                            working_matrix[(i + 1) * size + vec_index] = (-accumulator - diagonal_diff * solution) / x_elem;
                                        }
                                        else {
                                            working_matrix[(i + 1) * size + vec_index] = (-prev_accumulator - y_elem * solution) / prev_diagonal_diff;
                                        }
                                    }

                                    // Normalize if magnitude is too large.
                                    const type magnitude = math::abs(working_matrix[i * size + vec_index]);
                                    if (epsilon * magnitude * magnitude > 1) {
                                        for (int j = i; j <= vec_index; ++j) {
                                            working_matrix[j * size + vec_index] /= magnitude;
                                        }
                                    }
                                }
                            }
                        }
                        // Complex eigenvalue: compute complex eigenvector (stored as two real columns).
                        else if (eigenvalue_imag < 0.0) {
                            const int prev_col = vec_index - 1;
                            int row_start = prev_col;

                            // Determine initial values based on matrix structure.
                            if (math::abs(working_matrix[vec_index * size + prev_col]) > math::abs(working_matrix[prev_col * size + vec_index])) {
                                working_matrix[prev_col * size + prev_col] = eigenvalue_imag / working_matrix[vec_index * size + prev_col];
                                working_matrix[prev_col * size + vec_index] = -(working_matrix[vec_index * size + vec_index] - eigenvalue_real) / working_matrix[vec_index * size + prev_col];
                            }
                            else {
                                const complex_number initial = complex_number(0.0, -working_matrix[prev_col * size + vec_index]) /
                                                               complex_number(working_matrix[prev_col * size + prev_col] - eigenvalue_real, eigenvalue_imag);
                                working_matrix[prev_col * size + prev_col] = initial.real;
                                working_matrix[prev_col * size + vec_index] = initial.imaginary;
                            }

                            working_matrix[vec_index * size + prev_col] = 0.0;
                            working_matrix[vec_index * size + vec_index] = 1.0;

                            // State variables that persist across loop iterations.
                            type prev_diagonal_diff = 0.0;
                            type prev_real_accum = 0.0;
                            type prev_imag_accum = 0.0;

                            for (int i = vec_index - 2; i >= 0; --i) {
                                const type diagonal_diff = working_matrix[i * size + i] - eigenvalue_real;

                                type real_accumulator = 0.0;
                                type imag_accumulator = 0.0;
                                for (int j = row_start; j <= vec_index; ++j) {
                                    real_accumulator += working_matrix[i * size + j] * working_matrix[j * size + prev_col];
                                    imag_accumulator += working_matrix[i * size + j] * working_matrix[j * size + vec_index];
                                }

                                if (eigenvalues[i].imaginary < 0.0) {
                                    // Store for use in next iteration.
                                    prev_diagonal_diff = diagonal_diff;
                                    prev_real_accum = real_accumulator;
                                    prev_imag_accum = imag_accumulator;
                                }
                                else {
                                    row_start = i;

                                    if (eigenvalues[i].imaginary == 0.0) {
                                        // Real eigenvalue in complex eigenvector computation.
                                        const complex_number solution = complex_number(-real_accumulator, -imag_accumulator) /
                                                                        complex_number(diagonal_diff, eigenvalue_imag);
                                        working_matrix[i * size + prev_col] = solution.real;
                                        working_matrix[i * size + vec_index] = solution.imaginary;
                                    }
                                    else {
                                        // Complex eigenvalue pair - solve complex 2x2 system.
                                        const type x_elem = working_matrix[i * size + (i + 1)];
                                        const type y_elem = working_matrix[(i + 1) * size + i];

                                        type vr = math::sqr(eigenvalues[i].real - eigenvalue_real) + math::sqr(eigenvalues[i].imaginary) - eigenvalue_imag * eigenvalue_imag;
                                        type vi = 2.0 * eigenvalue_imag * (eigenvalues[i].real - eigenvalue_real);

                                        // Regularize if denominator is too small.
                                        if (vr == 0.0 && vi == 0.0) {
                                            vr = epsilon * working_matrix_norm * (math::abs(diagonal_diff) + math::abs(eigenvalue_imag) + math::abs(x_elem) + math::abs(y_elem) + math::abs(prev_diagonal_diff));
                                        }

                                        const complex_number solution = complex_number(
                                                                            x_elem * prev_real_accum - prev_diagonal_diff * real_accumulator + eigenvalue_imag * imag_accumulator,
                                                                            x_elem * prev_imag_accum - prev_diagonal_diff * imag_accumulator - eigenvalue_imag * real_accumulator
                                                                        ) /
                                                                        complex_number(vr, vi);

                                        working_matrix[i * size + prev_col] = solution.real;
                                        working_matrix[i * size + vec_index] = solution.imaginary;

                                        if (math::abs(x_elem) > math::abs(prev_diagonal_diff) + math::abs(eigenvalue_imag)) {
                                            working_matrix[(i + 1) * size + prev_col] = (-real_accumulator - diagonal_diff * working_matrix[i * size + prev_col] + eigenvalue_imag * working_matrix[i * size + vec_index]) / x_elem;
                                            working_matrix[(i + 1) * size + vec_index] = (-imag_accumulator - diagonal_diff * working_matrix[i * size + vec_index] - eigenvalue_imag * working_matrix[i * size + prev_col]) / x_elem;
                                        }
                                        else {
                                            const complex_number next_solution = complex_number(
                                                                                     -prev_real_accum - y_elem * working_matrix[i * size + prev_col],
                                                                                     -prev_imag_accum - y_elem * working_matrix[i * size + vec_index]
                                                                                 ) /
                                                                                 complex_number(prev_diagonal_diff, eigenvalue_imag);

                                            working_matrix[(i + 1) * size + prev_col] = next_solution.real;
                                            working_matrix[(i + 1) * size + vec_index] = next_solution.imaginary;
                                        }
                                    }

                                    // Normalize if magnitude is too large.
                                    const type magnitude = math::max(math::abs(working_matrix[i * size + prev_col]), math::abs(working_matrix[i * size + vec_index]));
                                    if (epsilon * magnitude * magnitude > 1) {
                                        for (int j = i; j <= vec_index; ++j) {
                                            working_matrix[j * size + prev_col] /= magnitude;
                                            working_matrix[j * size + vec_index] /= magnitude;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Multiply by accumulated transformations to get final eigenvectors.
                    for (int i = size - 1; i >= 0; --i) {
                        for (int j = 0; j < size; ++j) {
                            type sum = 0;
                            for (int k = 0; k <= i; ++k) {
                                sum += eigenvectors[j * size + k] * working_matrix[k * size + i];
                            }
                            eigenvectors[j * size + i] = sum;
                        }
                    }
                }

                // Apply scaling factors to eigenvectors.
                for (int row = 0; row < size; ++row) {
                    for (int col = 0; col < size; ++col) {
                        eigenvectors[row * size + col] *= scaling[row];
                    }
                }
            }
        }

        // Optionally sort eigenvalues (and corresponding eigenvectors).
        if constexpr (sort_results) {
            for (int i = 0; i < size - 1; ++i) {
                complex_number min_value = eigenvalues[i];
                int min_index = i;
                for (int j = i; j < size; ++j) {
                    if ((eigenvalues[j].real < min_value.real) || ((eigenvalues[j].real == min_value.real) && (eigenvalues[j].imaginary < min_value.imaginary))) {
                        min_value = eigenvalues[j];
                        min_index = j;
                    }
                }
                if (min_index != i) {
                    eigenvalues[min_index] = eigenvalues[i];
                    eigenvalues[i] = min_value;
                    if constexpr (extract_eigenvectors) {
                        for (int row = 0; row < size; ++row) {
                            const type temp = eigenvectors[row * size + i];
                            eigenvectors[row * size + i] = eigenvectors[row * size + min_index];
                            eigenvectors[row * size + min_index] = temp;
                        }
                    }
                }
            }
        }

        // Write out eigenvalues in interleaved format [real, imag, real, imag, ...].
        for (int i = 0; i < size; i++) {
            eigenvalues_complex[i * 2 + 0] = eigenvalues[i].real;
            eigenvalues_complex[i * 2 + 1] = eigenvalues[i].imaginary;
        }

        // Clean up heap allocations.
        delete[] working_matrix;
        delete[] eigenvalues;
        delete[] scaling;
        delete[] permutation;

        return true;
    }
}

#endif // MATRIX_EIGEN_SOLVER_HPP
