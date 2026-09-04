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
#ifndef ZEROSLAM_MATH_HPP
#define ZEROSLAM_MATH_HPP

namespace {
    using size_t = decltype(sizeof(0));

    template <typename lhs, typename rhs>
    struct is_same_type {
        static constexpr bool value = false;
    };

    template <typename type>
    struct is_same_type<type, type> {
        static constexpr bool value = true;
    };
}

namespace math {
    template <typename type>
    constexpr static inline type pi();
    template <typename type>
    constexpr static inline type e();
    template <typename type>
    constexpr static const type epsilon();
    template <typename type>
    constexpr static inline type nan();
    template <typename type>
    constexpr static inline type inf();
    template <typename type>
    constexpr static inline bool isnan(type value);
    template <typename type>
    constexpr static inline bool isinf(type value);
    template <typename type>
    constexpr static inline bool isfinite(type value);
    template <typename type>
    constexpr static inline type copysign(type magnitude, type sign);
    template <typename type>
    constexpr static inline bool signbit(type value);
    template <typename type>
    constexpr static inline type abs(type value);
    template <typename type>
    constexpr static inline type min(type lhs, type rhs);
    template <typename type>
    constexpr static inline type max(type lhs, type rhs);
    template <typename type>
    constexpr static inline type floor(type value);
    template <typename type>
    constexpr static inline type ceil(type value);

    constexpr static inline int round(float value);
    constexpr static inline long long int round(double value);

    template <typename type>
    constexpr static inline type fmod(type value, type modulus);
    template <typename type>
    constexpr static inline type sqr(type value);
    template <typename type>
    constexpr static inline type sqrt(type value);
    template <typename type>
    constexpr static type pythag(const type a, const type b);
    template <typename type>
    constexpr static inline type exp(type value);
    template <typename type>
    constexpr static inline type log(type value);
    template <typename type>
    constexpr static inline type pow(type value, type exponent);
    template <typename type>
    constexpr static inline type sin(type value);
    template <typename type>
    constexpr static inline type cos(type value);

    template <typename type>
    constexpr static inline type asin(type value);
    template <typename type>
    constexpr static inline type acos(type value);
    template <typename type>
    constexpr static inline type atan2(type y, type x);
}

namespace math {
    template <typename type>
    constexpr static inline type pi() {
        return 3.14159265358979323846264338327950288419716939937510582097494459230781640628;
    }

    template <typename type>
    constexpr static inline type e() {
        return 2.71828182845904523536028747135266249775724709369995957496696762772407663035;
    }

    template <typename type>
    constexpr static const type epsilon() {
        type epsilon = 1;
        while (type(1) + epsilon / type(2) != type(1)) {
            epsilon /= type(2);
        }
        return epsilon;
    }

    template <typename type>
    constexpr static inline type nan() {
#if __has_builtin(__builtin_nanf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_nanf("0");
        }
#endif
#if __has_builtin(__builtin_nan)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_nan("0");
        }
#endif
        __builtin_trap();
    }

    template <typename type>
    constexpr static inline type inf() {
#if __has_builtin(__builtin_inff)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_inff();
        }
#endif
#if __has_builtin(__builtin_inf)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_inf();
        }
#endif
        __builtin_trap();
    }

    template <typename type>
    constexpr static inline bool isnan(type value) {
#if __has_builtin(__builtin_isnan)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_isnan(value);
        }
#endif
#if __has_builtin(__builtin_isnan)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_isnan(value);
        }
#endif
        return value != value;
    }

    template <typename type>
    constexpr static inline bool isinf(type value) {
#if __has_builtin(__builtin_isinf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_isinf(value);
        }
#endif
#if __has_builtin(__builtin_isinf)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_isinf(value);
        }
#endif
        if (isnan(value))
            return false;
        if ((value > 0) && ((value / value) != (value / value)))
            return true;
        if ((value < 0) && ((value / value) != (value / value)))
            return true;
        return false;
    }

    template <typename type>
    constexpr static inline bool isfinite(type value) {
#if __has_builtin(__builtin_isfinite)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_isfinite(value);
        }
#endif
#if __has_builtin(__builtin_isfinite)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_isfinite(value);
        }
#endif
        if (isnan(value))
            return false;
        if ((value > 0) && ((value / value) != (value / value)))
            return false;
        if ((value < 0) && ((value / value) != (value / value)))
            return false;
        return true;
    }

    template <typename type>
    constexpr static inline type copysign(type magnitude, type sign) {
#if __has_builtin(__builtin_copysignf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_copysignf(magnitude, sign);
        }
#endif
#if __has_builtin(__builtin_copysign)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_copysign(magnitude, sign);
        }
#endif
        if (isnan(magnitude))
            return nan<type>();
        if (isnan(sign))
            sign = type(1);
        if ((sign == 0) && (type(1) / sign) < 0)
            sign = type(-1);
        return (sign >= 0 ? (magnitude >= 0 ? magnitude : -magnitude) : (magnitude >= 0 ? -magnitude : magnitude));
    }

    template <typename type>
    constexpr static inline bool signbit(type value) {
#if __has_builtin(__builtin_signbitf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_signbitf(value);
        }
#endif
#if __has_builtin(__builtin_signbit)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_signbit(value);
        }
#endif
        return copysign(type(1), value) < 0;
    }

    template <typename type>
    constexpr static inline type abs(type value) {
#if __has_builtin(__builtin_fabsf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_fabsf(value);
        }
#endif
#if __has_builtin(__builtin_fabs)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_fabs(value);
        }
#endif
        if ((value + type(0)) < type(0)) {
            return -value;
        }
        return value;
    }

    template <typename type>
    constexpr static inline type min(type lhs, type rhs) {
        return rhs < lhs ? rhs : lhs;
    }

    template <typename type>
    constexpr static inline type max(type lhs, type rhs) {
        return lhs < rhs ? rhs : lhs;
    }

    template <typename type>
    constexpr static inline type floor(type value) {
#if __has_builtin(__builtin_floorf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_floorf(value);
        }
#endif
#if __has_builtin(__builtin_floor)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_floor(value);
        }
#endif
        constexpr const long long int max_integer = static_cast<long long int>(static_cast<unsigned long long int>(-1) >> 1);
        constexpr const long long int min_integer = -max_integer - 1;
        constexpr const double max_integer_as_double = static_cast<double>(max_integer / 2) * type(2);
        constexpr const double min_integer_as_double = static_cast<double>(min_integer);
        if ((value >= max_integer_as_double) || (value <= min_integer_as_double) || isnan(value)) {
            return value;
        }
        const long long int casted = static_cast<long long int>(value);
        const double rounded = static_cast<double>(casted);
        return ((rounded == value) || (value >= 0)) ? rounded : rounded - 1;
    }

    template <typename type>
    constexpr static inline type ceil(type value) {
#if __has_builtin(__builtin_ceilf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_ceilf(value);
        }
#endif
#if __has_builtin(__builtin_ceil)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_ceil(value);
        }
#endif
        constexpr const long long int max_integer = static_cast<long long int>(static_cast<unsigned long long int>(-1) >> 1);
        constexpr const long long int min_integer = -max_integer - 1;
        constexpr const double max_integer_as_double = static_cast<double>(max_integer / 2) * type(2);
        constexpr const double min_integer_as_double = static_cast<double>(min_integer);
        if ((value >= max_integer_as_double) || (value <= min_integer_as_double) || isnan(value)) {
            return value;
        }
        const long long int casted = static_cast<long long int>(value);
        const double rounded = static_cast<double>(casted);
        return ((rounded == value) || (value <= 0)) ? rounded : rounded + 1;
    }

    constexpr static inline int round(float value) {
#if __has_builtin(__builtin_roundf)
        return static_cast<int>(__builtin_roundf(value));
#endif
        return (value > 0.0f) ? static_cast<int>(value + 0.5f) : static_cast<int>(value - 0.5f);
    }

    constexpr static inline long long int round(double value) {
#if __has_builtin(__builtin_round)
        return static_cast<long long int>(__builtin_round(value));
#endif
        return (value > 0.0) ? static_cast<long long int>(value + 0.5) : static_cast<long long int>(value - 0.5);
    }

    template <typename type>
    constexpr static inline type fmod(type value, type modulus) {
#if __has_builtin(__builtin_fmodf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_fmodf(value, modulus);
        }
#endif
#if __has_builtin(__builtin_fmod)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_fmod(value, modulus);
        }
#endif
        if (isnan(value) || isnan(modulus))
            return nan<type>();
        if ((value == 0) && (modulus != 0))
            return copysign(type(0), value);
        if (isinf(value) && !isnan(modulus))
            return nan<type>();
        if (!isnan(value) && (modulus == 0))
            return nan<type>();
        if (isfinite(value) && isinf(modulus))
            return value;
        type value_as_absolute = abs(value);
        const type modulus_as_absolute = abs(modulus);
        while (value_as_absolute >= modulus_as_absolute) {
            type factor = modulus_as_absolute;
            while (value_as_absolute >= (type(2) * factor)) {
                factor *= type(2);
            }
            value_as_absolute -= factor;
        }
        return copysign(value_as_absolute, value);
    }

    template <typename type>
    constexpr static inline type sqr(type value) {
        return value * value;
    }

    template <typename type>
    constexpr static inline type sqrt(type value) {
#if __has_builtin(__builtin_sqrtf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_sqrtf(value);
        }
#endif
#if __has_builtin(__builtin_sqrt)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_sqrt(value);
        }
#endif
        if ((value < 0) || isnan(value))
            return nan<type>();
        if ((value == 0) || isinf(value))
            return value;
        unsigned long long int bits = 0;
        __builtin_memcpy(&bits, &value, sizeof(bits));
        bits = (bits >> 1) + 0x1FF7A3BEA91D9B1BULL;
        double estimate = 0;
        __builtin_memcpy(&estimate, &bits, sizeof(estimate));
        double previous = 0;
        while (estimate != previous) {
            previous = estimate;
            estimate = 0.5 * (estimate + value / estimate);
        }
        return estimate;
    }

    template <typename type>
    constexpr static type pythag(const type a, const type b) {
        const type abs_a = abs(a);
        const type abs_b = abs(b);
        if (abs_a > abs_b) {
            return abs_a * sqrt(1.0 + sqr(abs_b / abs_a));
        }
        if (abs_b == 0) {
            return 0;
        }
        return abs_b * sqrt(1.0 + sqr(abs_a / abs_b));
    }

    template <typename type>
    constexpr static inline type exp(type value) {
#if __has_builtin(__builtin_expf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_expf(value);
        }
#endif
#if __has_builtin(__builtin_exp)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_exp(value);
        }
#endif
        if (isnan(value))
            return nan<type>();
        if (value == 0)
            return 1;
        if ((value < 0) && isinf(value))
            return 0;
        if ((value > 0) && isinf(value))
            return inf<type>();
        constexpr const double epsilon = 1e-9;
        const double abs_value = abs(value);
        int order = 0;
        double term = 1.0;
        double sum = term;
        while ((term > epsilon) && (isfinite(sum))) {
            term = (term * abs_value) / static_cast<double>(++order);
            sum += term;
        }
        return (value < 0) ? 1.0 / sum : sum;
    }

    template <typename type>
    constexpr static inline type log(type value) {
#if __has_builtin(__builtin_logf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_logf(value);
        }
#endif
#if __has_builtin(__builtin_log)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_log(value);
        }
#endif
        if ((value < 0) || isnan(value))
            return nan<type>();
        if (value == 0)
            return -inf<type>();
        if (value == 1)
            return 0;
        if (isinf(value))
            return inf<type>();
        constexpr const double epsilon = 1e-9;
        // Normalize the value and count how many times we divide by e.
        double working_value = (value < 1.0) ? (1.0 / value) : value;
        unsigned int exponent_count = 0;
        while ((working_value /= e<type>()) > 1.0) {
            ++exponent_count;
        }
        // Prepare for series expansion.
        working_value = 1.0 / (working_value * e<type>() - 1.0);
        working_value = 2.0 * working_value + 1.0;
        const double squared_working_value = working_value * working_value;
        // Iteratively compute using a Taylor-like series until convergence.
        unsigned int denominator = 1;
        double term_accumulator = 0.0;
        double previous_accumulator = 0.0;
        working_value /= 2.0;
        do {
            previous_accumulator = term_accumulator;
            term_accumulator += 1.0 / (denominator * working_value);
            denominator += 2;
            working_value *= squared_working_value;
        } while ((term_accumulator - previous_accumulator) > epsilon);
        // Apply sign correction for values less than one.
        const double result = exponent_count + term_accumulator;
        return (value < 1.0) ? -result : result;
    }

    template <typename type>
    constexpr static inline type pow(type value, type exponent) {
#if __has_builtin(__builtin_powf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_powf(value, exponent);
        }
#endif
#if __has_builtin(__builtin_pow)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_pow(value, exponent);
        }
#endif
        const bool is_value_positive = !signbit(value);
        const double value_as_absolute = abs(value);
        const bool is_value_inf = isinf(value);
        const bool is_exponent_positive = !signbit(exponent);
        const long long int exponent_as_integer = round(exponent);
        const bool is_exponent_integer = (static_cast<double>(exponent_as_integer) == exponent);
        const bool is_exponent_even = (exponent_as_integer != 0) && ((exponent_as_integer & 1) == 0);
        const bool is_exponent_odd = (exponent_as_integer != 0) && ((exponent_as_integer & 1) == 1);
        const bool is_exponent_inf = isinf(exponent) || (abs(exponent) > 1024);
        if (value == 1.0)
            return 1.0;
        if (exponent == 0.0)
            return 1.0;
        if (exponent == 1.0)
            return value;
        if (exponent == -1.0)
            return 1.0 / value;
        if (isnan(value) || isnan(exponent))
            return nan<type>();
        if (value == 0.0 && is_value_positive && !is_exponent_positive && is_exponent_integer && is_exponent_odd)
            return inf<type>();
        if (value == 0.0 && !is_value_positive && !is_exponent_positive && is_exponent_integer && is_exponent_odd)
            return -inf<type>();
        if (value == 0.0 && !is_exponent_positive && !is_exponent_inf && ((is_exponent_integer && is_exponent_even) || (!is_exponent_integer)))
            return inf<type>();
        if (value == 0.0 && !is_exponent_positive && is_exponent_inf)
            return inf<type>();
        if (value == 0.0 && is_value_positive && is_exponent_positive && is_exponent_integer && is_exponent_odd)
            return +0.0;
        if (value == 0.0 && !is_value_positive && is_exponent_positive && is_exponent_integer && is_exponent_odd)
            return -0.0;
        if (value == 0.0 && is_exponent_positive && (!is_exponent_integer || (is_exponent_integer && is_exponent_even)))
            return +0.0;
        if (value == -1.0 && is_exponent_inf)
            return 1.0;
        if (!is_value_inf && !is_value_positive && !is_exponent_inf && !is_exponent_integer)
            return nan<type>();
        if (value_as_absolute < 1.0 && !is_exponent_positive && is_exponent_inf)
            return inf<type>();
        if (value_as_absolute > 1.0 && !is_exponent_positive && is_exponent_inf)
            return +0.0;
        if (value_as_absolute < 1.0 && is_exponent_positive && is_exponent_inf)
            return +0.0;
        if (value_as_absolute > 1.0 && is_exponent_positive && is_exponent_inf)
            return inf<type>();
        if (!is_value_positive && is_value_inf && !is_exponent_positive && is_exponent_integer && is_exponent_odd)
            return -0.0;
        if (!is_value_positive && is_value_inf && !is_exponent_positive && (!is_exponent_integer || (is_exponent_integer && is_exponent_even)))
            return +0.0;
        if (!is_value_positive && is_value_inf && is_exponent_positive && is_exponent_integer && is_exponent_odd)
            return -inf<type>();
        if (!is_value_positive && is_value_inf && is_exponent_positive && (!is_exponent_integer || (is_exponent_integer && is_exponent_even)))
            return inf<type>();
        if (is_value_positive && is_value_inf && !is_exponent_positive)
            return +0.0;
        if (is_value_positive && is_value_inf && is_exponent_positive)
            return inf<type>();
        return (value < 0) ? (exp(log(-value) * exponent) * (-1 + 2 * is_exponent_even)) : (exp(log(value) * exponent));
    }

    template <typename type>
    constexpr static inline type sin(type value) {
#if __has_builtin(__builtin_sinf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_sinf(value);
        }
#endif
#if __has_builtin(__builtin_sin)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_sin(value);
        }
#endif
        if (value == type(0))
            return copysign(type(0), value);
        if (!isfinite(value))
            return nan<type>();
        const type angle = fmod(abs(value), (pi<type>() * type(2)));
        const type sign = (angle <= pi<type>()) ? (type(1) - type(2) * signbit(value)) : (type(-1) + type(2) * signbit(value));
        const type remapped = (angle > (pi<type>() * type(1.5))) ? ((pi<type>() * type(2)) - angle) : (angle > (pi<type>()))           ? angle - pi<type>()
                                                                                                  : (angle > (pi<type>() * type(0.5))) ? pi<type>() - angle
                                                                                                                                       : angle;
        const type remapped2 = remapped * remapped;
        type polynomial = 0;
        if constexpr (is_same_type<type, double>::value) {
            polynomial = type(1) / type(51090942171709440000.0);
            polynomial *= remapped2;
            polynomial -= type(1) / type(121645100408832000.0);
            polynomial *= remapped2;
            polynomial += type(1) / type(355687428096000.0);
            polynomial *= remapped2;
            polynomial -= type(1) / type(1307674368000.0);
            polynomial *= remapped2;
            polynomial += type(1) / type(6227020800.0);
            polynomial *= remapped2;
            polynomial -= type(1) / type(39916800.0);
        }
        else if constexpr (is_same_type<type, float>::value) {
            polynomial = type(1) / type(39916800.0);
        }
        polynomial *= remapped2;
        polynomial += type(1) / type(362880.0);
        polynomial *= remapped2;
        polynomial -= type(1) / type(5040.0);
        polynomial *= remapped2;
        polynomial += type(1) / type(120.0);
        polynomial *= remapped2;
        polynomial -= type(1) / type(6.0);
        polynomial *= remapped2;
        polynomial += type(1);
        return sign * remapped * polynomial;
    }

    template <typename type>
    constexpr static inline type cos(type value) {
#if __has_builtin(__builtin_cosf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_cosf(value);
        }
#endif
#if __has_builtin(__builtin_cos)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_cos(value);
        }
#endif
        if (value == type(0))
            return type(1);
        if (!isfinite(value))
            return nan<type>();
        const type angle = fmod(abs(value), (pi<type>() * type(2)));
        const type sign = (angle > (pi<type>() * type(1.5))) ? type(1) : (angle > (pi<type>() * type(0.5))) ? -type(1)
                                                                                                            : type(1);
        const type remapped = (angle > (pi<type>() * type(1.5))) ? ((pi<type>() * type(2)) - angle) : (angle > (pi<type>()))           ? angle - pi<type>()
                                                                                                  : (angle > (pi<type>() * type(0.5))) ? pi<type>() - angle
                                                                                                                                       : angle;
        const type remapped2 = remapped * remapped;
        type polynomial = 0;
        if constexpr (is_same_type<type, double>::value) {
            polynomial = type(1) / type(2432902008176640000.0);
            polynomial *= remapped2;
            polynomial -= type(1) / type(6402373705728000.0);
            polynomial *= remapped2;
            polynomial += type(1) / type(20922789888000.0);
            polynomial *= remapped2;
            polynomial -= type(1) / type(87178291200.0);
            polynomial *= remapped2;
            polynomial += type(1) / type(479001600.0);
        }
        else if constexpr (is_same_type<type, float>::value) {
            polynomial = type(1) / type(479001600.0);
        }
        polynomial *= remapped2;
        polynomial -= type(1) / type(3628800.0);
        polynomial *= remapped2;
        polynomial += type(1) / type(40320.0);
        polynomial *= remapped2;
        polynomial -= type(1) / type(720.0);
        polynomial *= remapped2;
        polynomial += type(1) / type(24.0);
        polynomial *= remapped2;
        polynomial -= type(1) / type(2.0);
        polynomial *= remapped2;
        polynomial += type(1);
        return sign * polynomial;
    }

    template <typename type>
    constexpr static inline void sincos(type value, type& sine, type& cosine) {
#if __has_builtin(__builtin_sincosf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_sincosf(value, &sine, &cosine);
        }
#endif
#if __has_builtin(__builtin_sincos)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_sincos(value, &sine, &cosine);
        }
#endif
        if (value == type(0)) {
            sine = copysign(type(0), value);
            cosine = type(1);
            return;
        }
        if (!isfinite(value)) {
            sine = nan<type>();
            cosine = nan<type>();
            return;
        }
        const type angle = fmod(abs(value), (pi<type>() * type(2)));
        const type sign_sin = (angle <= pi<type>()) ? (type(1) - type(2) * signbit(value)) : (type(-1) + type(2) * signbit(value));
        const type sign_cos = (angle > (pi<type>() * type(1.5))) ? type(1) : (angle > (pi<type>()))           ? type(-1)
                                                                         : (angle > (pi<type>() * type(0.5))) ? type(-1)
                                                                                                              : type(1);
        const type remapped = (angle > (pi<type>() * type(1.5))) ? ((pi<type>() * type(2)) - angle) : (angle > (pi<type>()))           ? angle - pi<type>()
                                                                                                  : (angle > (pi<type>() * type(0.5))) ? pi<type>() - angle
                                                                                                                                       : angle;
        const type remapped2 = remapped * remapped;
        type polynomial_sin = 0;
        type polynomial_cos = 0;
        if constexpr (is_same_type<type, double>::value) {
            polynomial_sin = type(1) / type(51090942171709440000.0);
            polynomial_sin *= remapped2;
            polynomial_sin -= type(1) / type(121645100408832000.0);
            polynomial_sin *= remapped2;
            polynomial_sin += type(1) / type(355687428096000.0);
            polynomial_sin *= remapped2;
            polynomial_sin -= type(1) / type(1307674368000.0);
            polynomial_sin *= remapped2;
            polynomial_sin += type(1) / type(6227020800.0);
            polynomial_sin *= remapped2;
            polynomial_sin -= type(1) / type(39916800.0);
            polynomial_cos = type(1) / type(2432902008176640000.0);
            polynomial_cos *= remapped2;
            polynomial_cos -= type(1) / type(6402373705728000.0);
            polynomial_cos *= remapped2;
            polynomial_cos += type(1) / type(20922789888000.0);
            polynomial_cos *= remapped2;
            polynomial_cos -= type(1) / type(87178291200.0);
            polynomial_cos *= remapped2;
            polynomial_cos += type(1) / type(479001600.0);
        }
        else if constexpr (is_same_type<type, float>::value) {
            polynomial_sin = type(1) / type(39916800.0);
            polynomial_cos = type(1) / type(479001600.0);
        }
        polynomial_sin *= remapped2;
        polynomial_sin += type(1) / type(362880.0);
        polynomial_sin *= remapped2;
        polynomial_sin -= type(1) / type(5040.0);
        polynomial_sin *= remapped2;
        polynomial_sin += type(1) / type(120.0);
        polynomial_sin *= remapped2;
        polynomial_sin -= type(1) / type(6.0);
        polynomial_sin *= remapped2;
        polynomial_sin += type(1);
        polynomial_cos *= remapped2;
        polynomial_cos -= type(1) / type(3628800.0);
        polynomial_cos *= remapped2;
        polynomial_cos += type(1) / type(40320.0);
        polynomial_cos *= remapped2;
        polynomial_cos -= type(1) / type(720.0);
        polynomial_cos *= remapped2;
        polynomial_cos += type(1) / type(24.0);
        polynomial_cos *= remapped2;
        polynomial_cos -= type(1) / type(2.0);
        polynomial_cos *= remapped2;
        polynomial_cos += type(1);
        sine = sign_sin * remapped * polynomial_sin;
        cosine = sign_cos * polynomial_cos;
    }

    template <typename type>
    constexpr static inline type asin(type value) {
#if __has_builtin(__builtin_asinf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_asinf(value);
        }
#endif
#if __has_builtin(__builtin_asin)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_asin(value);
        }
#endif
        if (value == 0.0)
            return value;
        if ((value < -1.0) || (value > 1.0) || isnan(value))
            return nan<type>();
        const double sign = (value < 0.0) ? -1.0 : 1.0;
        const double value_as_absolute = abs(value);
        double angle = -0.0187293;
        angle *= value_as_absolute;
        angle += 0.0742610;
        angle *= value_as_absolute;
        angle -= 0.2121144;
        angle *= value_as_absolute;
        angle += 1.5707288;
        angle *= sqrt(1.0 - value_as_absolute);
        angle = (pi<type>() * 0.5) - angle;
        if (value_as_absolute < 1.0) {
            if (value_as_absolute <= 0.70710678118654752) {
                for (int iteration = 0; iteration < 4; ++iteration) {
                    angle -= (sin(angle) - value_as_absolute) / cos(angle);
                }
            }
            else {
                const double cosine_target = sqrt((1.0 - value_as_absolute) * (1.0 + value_as_absolute));
                for (int iteration = 0; iteration < 4; ++iteration) {
                    angle += (cos(angle) - cosine_target) / sin(angle);
                }
            }
        }
        return sign * angle;
    }

    template <typename type>
    constexpr static inline type acos(type value) {
#if __has_builtin(__builtin_acosf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_acosf(value);
        }
#endif
#if __has_builtin(__builtin_acos)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_acos(value);
        }
#endif
        if (value == 1.0)
            return 0.0;
        if ((value < -1.0) || (value > 1.0) || isnan(value))
            return nan<type>();
        const double sign = (value < 0.0) ? -1.0 : 1.0;
        const double value_as_absolute = abs(value);
        double angle = -0.0187293;
        angle *= value_as_absolute;
        angle += 0.0742610;
        angle *= value_as_absolute;
        angle -= 0.2121144;
        angle *= value_as_absolute;
        angle += 1.5707288;
        angle *= sqrt(1.0 - value_as_absolute);
        if (value_as_absolute < 1.0) {
            if (value_as_absolute <= 0.70710678118654752) {
                for (int iteration = 0; iteration < 4; ++iteration) {
                    angle += (cos(angle) - value_as_absolute) / sin(angle);
                }
            }
            else {
                const double sine_target = sqrt((1.0 - value_as_absolute) * (1.0 + value_as_absolute));
                for (int iteration = 0; iteration < 4; ++iteration) {
                    angle -= (sin(angle) - sine_target) / cos(angle);
                }
            }
        }
        return (value < 0.0) ? (pi<type>() + sign * angle) : (sign * angle);
    }

    template <typename type>
    constexpr static inline type atan2(type y, type x) {
#if __has_builtin(__builtin_atan2f)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_atan2f(y, x);
        }
#endif
#if __has_builtin(__builtin_atan2)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_atan2(y, x);
        }
#endif
        if ((y == 0) && ((x < 0) || ((x == 0) && signbit(x))))
            return copysign(pi<type>(), y);
        if ((y == 0) && ((x > 0) || ((x == 0) && !signbit(x))))
            return copysign(type(0), y);
        if (isinf(y) && isfinite(x))
            return copysign(pi<type>() * type(0.5), y);
        if (isinf(y) && isinf(x) && signbit(x))
            return copysign(pi<type>() * type(0.75), y);
        if (isinf(y) && isinf(x) && !signbit(x))
            return copysign(pi<type>() * type(0.25), y);
        if ((x == 0) && (y < 0))
            return -pi<type>() * 0.5;
        if ((x == 0) && (y > 0))
            return +pi<type>() * 0.5;
        if (isinf(x) && signbit(x) && isfinite(y) && (y > 0))
            return +pi<type>();
        if (isinf(x) && signbit(x) && isfinite(y) && (y < 0))
            return -pi<type>();
        if (isinf(x) && !signbit(x) && isfinite(y) && (y > 0))
            return +0.0;
        if (isinf(x) && !signbit(x) && isfinite(y) && (y < 0))
            return -0.0;
        if (isnan(x) || isnan(y))
            return nan<type>();
        const bool swap = abs(x) < abs(y);
        const double ratio = ((swap ? x : y) / (swap ? y : x));
        const double ratio2 = ratio * ratio;
        double angle = -0.01172120;
        angle *= ratio2;
        angle += 0.05265332;
        angle *= ratio2;
        angle -= 0.11643287;
        angle *= ratio2;
        angle += 0.19354346;
        angle *= ratio2;
        angle -= 0.33262347;
        angle *= ratio2;
        angle += 0.99997726;
        angle *= ratio;
        angle = swap ? ((((ratio > 0.0) || ((ratio == 0.0) && !signbit(ratio))) ? (+pi<type>() * 0.5) : (-pi<type>() * 0.5)) - angle) : angle;
        if ((x >= 0.0) && (y >= 0.0)) {
        }
        else if ((x < 0.0) && (y >= 0.0)) {
            angle = +pi<type>() + angle;
        }
        else if ((x < 0.0) && (y < 0.0)) {
            angle = -pi<type>() + angle;
        }
        else if ((x >= 0.0) && (y < 0.0)) {
        }
        return angle;
    }
}

#endif // ZEROSLAM_MATH_HPP
