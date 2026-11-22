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
#ifndef ZEROSLAM_MATH_POLYNOMIAL_HPP
#define ZEROSLAM_MATH_POLYNOMIAL_HPP

#include "math/math.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

// Real polynomials of low degree, coefficients[i] multiplying value^i. The camera models find where their distortion folds
// back with these, which is where the valid region of each ends.
namespace math::polynomial {
    constexpr static const size_t maximum_degree = 8;

    inline double evaluate(const double* const coefficients, const size_t degree, const double value) {
        double result = coefficients[degree];
        for (size_t index = degree; index > 0; --index) {
            result = (result * value) + coefficients[index - 1];
        }
        return result;
    }

    // The roots above lower and up to upper where the polynomial changes sign or reaches zero, in increasing order.
    // The roots of the derivative split the range into monotonic pieces, each holds at most one root, found by bisection.
    inline size_t roots(const double* const coefficients, size_t degree, const double lower, const double upper, double* const found) {
        while ((degree > 0) && (coefficients[degree] == 0.0)) {
            --degree;
        }
        if ((degree == 0) || (degree > polynomial::maximum_degree)) {
            return 0;
        }
        double bounds[polynomial::maximum_degree + 2];
        size_t bounds_count = 0;
        bounds[bounds_count++] = lower;
        if (degree > 1) {
            double derivative[polynomial::maximum_degree];
            for (size_t index = 1; index <= degree; ++index) {
                derivative[index - 1] = static_cast<double>(index) * coefficients[index];
            }
            double critical[polynomial::maximum_degree];
            const size_t critical_count = polynomial::roots(&derivative[0], degree - 1, lower, upper, &critical[0]);
            for (size_t index = 0; index < critical_count; ++index) {
                if (critical[index] < upper) {
                    bounds[bounds_count++] = critical[index];
                }
            }
        }
        bounds[bounds_count++] = upper;
        size_t count = 0;
        for (size_t index = 1; index < bounds_count; ++index) {
            double below = bounds[index - 1];
            double above = bounds[index];
            const double value_below = polynomial::evaluate(coefficients, degree, below);
            const double value_above = polynomial::evaluate(coefficients, degree, above);
            if (value_above == 0.0) {
                found[count++] = above;
                continue;
            }
            if ((value_below == 0.0) || ((value_below < 0.0) == (value_above < 0.0))) {
                continue;
            }
            const bool rising = (value_above > 0.0);
            for (size_t iteration = 0; iteration < 256; ++iteration) {
                const double middle = 0.5 * (below + above);
                if (!(middle > below) || !(middle < above)) {
                    break;
                }
                if ((polynomial::evaluate(coefficients, degree, middle) > 0.0) == rising) {
                    above = middle;
                }
                else {
                    below = middle;
                }
            }
            found[count++] = above;
        }
        return count;
    }

    // The first value past zero and up to upper where the polynomial, positive at zero, stops being positive, infinite when
    // it does not there.
    inline double first_non_positive(const double* const coefficients, const size_t degree, const double upper) {
        double found[polynomial::maximum_degree];
        const size_t count = polynomial::roots(coefficients, degree, 0.0, upper, &found[0]);
        return (count > 0) ? found[0] : math::inf<double>();
    }
}

#endif // ZEROSLAM_MATH_POLYNOMIAL_HPP
