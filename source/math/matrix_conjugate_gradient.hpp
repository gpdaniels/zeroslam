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
#ifndef ZEROSLAM_MATH_MATRIX_CONJUGATE_GRADIENT_HPP
#define ZEROSLAM_MATH_MATRIX_CONJUGATE_GRADIENT_HPP

#include "core/assert.hpp"
#include "math/math.hpp"

namespace math {
    struct conjugate_gradient_result final {
        int iterations;
        bool converged;
        double initial_residual_norm;
        double final_residual_norm;
    };

    template <typename type, typename operator_type, typename preconditioner_type>
    static inline conjugate_gradient_result conjugate_gradient(
        const operator_type& apply_operator,
        const preconditioner_type& apply_preconditioner,
        const type* column_b,
        type* column_x,
        const int size,
        const int max_iterations,
        const type relative_tolerance,
        type* scratch
    ) {
        ASSERT(size >= 0, "System size cannot be negative.");
        conjugate_gradient_result result{ 0, true, 0.0, 0.0 };
        if (size == 0) {
            return result;
        }
        type* const residual = scratch;
        type* const preconditioned = scratch + size;
        type* const direction = scratch + (2 * size);
        type* const image = scratch + (3 * size);

        bool initial_guess_is_zero = true;
        for (int i = 0; i < size; ++i) {
            if (column_x[i] != 0) {
                initial_guess_is_zero = false;
                break;
            }
        }
        if (initial_guess_is_zero) {
            for (int i = 0; i < size; ++i) {
                residual[i] = column_b[i];
            }
        }
        else {
            apply_operator(column_x, image);
            for (int i = 0; i < size; ++i) {
                residual[i] = column_b[i] - image[i];
            }
        }

        type norm_b_squared = 0;
        for (int i = 0; i < size; ++i) {
            norm_b_squared += column_b[i] * column_b[i];
        }
        type norm_residual_squared = 0;
        for (int i = 0; i < size; ++i) {
            norm_residual_squared += residual[i] * residual[i];
        }
        result.initial_residual_norm = static_cast<double>(math::sqrt(norm_residual_squared));
        result.final_residual_norm = result.initial_residual_norm;
        const type threshold_squared = relative_tolerance * relative_tolerance * norm_b_squared;
        if (norm_residual_squared <= threshold_squared) {
            return result;
        }

        apply_preconditioner(residual, preconditioned);
        for (int i = 0; i < size; ++i) {
            direction[i] = preconditioned[i];
        }
        type residual_dot_preconditioned = 0;
        for (int i = 0; i < size; ++i) {
            residual_dot_preconditioned += residual[i] * preconditioned[i];
        }

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            apply_operator(direction, image);
            ++result.iterations;
            type curvature = 0;
            for (int i = 0; i < size; ++i) {
                curvature += direction[i] * image[i];
            }
            if (!(curvature > 0)) {
                result.converged = false;
                break;
            }
            const type step = residual_dot_preconditioned / curvature;
            if (!math::isfinite(step)) {
                result.converged = false;
                break;
            }
            for (int i = 0; i < size; ++i) {
                column_x[i] += step * direction[i];
            }
            for (int i = 0; i < size; ++i) {
                residual[i] -= step * image[i];
            }
            norm_residual_squared = 0;
            for (int i = 0; i < size; ++i) {
                norm_residual_squared += residual[i] * residual[i];
            }
            result.final_residual_norm = static_cast<double>(math::sqrt(norm_residual_squared));
            if (norm_residual_squared <= threshold_squared) {
                return result;
            }
            apply_preconditioner(residual, preconditioned);
            type residual_dot_preconditioned_next = 0;
            for (int i = 0; i < size; ++i) {
                residual_dot_preconditioned_next += residual[i] * preconditioned[i];
            }
            if (!(residual_dot_preconditioned != 0)) {
                result.converged = false;
                break;
            }
            const type beta = residual_dot_preconditioned_next / residual_dot_preconditioned;
            if (!math::isfinite(beta)) {
                result.converged = false;
                break;
            }
            for (int i = 0; i < size; ++i) {
                direction[i] = preconditioned[i] + (beta * direction[i]);
            }
            residual_dot_preconditioned = residual_dot_preconditioned_next;
        }
        if (!(norm_residual_squared <= threshold_squared)) {
            result.converged = false;
        }
        return result;
    }

    template <typename type, typename operator_type>
    static inline conjugate_gradient_result conjugate_gradient(
        const operator_type& apply_operator,
        const type* column_b,
        type* column_x,
        const int size,
        const int max_iterations,
        const type relative_tolerance,
        type* scratch
    ) {
        const auto identity_preconditioner = [size](const type* input, type* output) {
            for (int i = 0; i < size; ++i) {
                output[i] = input[i];
            }
        };
        return conjugate_gradient(apply_operator, identity_preconditioner, column_b, column_x, size, max_iterations, relative_tolerance, scratch);
    }
}

#endif // ZEROSLAM_MATH_MATRIX_CONJUGATE_GRADIENT_HPP
