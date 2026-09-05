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
#ifndef ZEROSLAM_MATH_MATH_HPP
#define ZEROSLAM_MATH_MATH_HPP

// MSVC does not provide __has_builtin, treat every builtin as unavailable there so the fallback implementations are used.
#ifndef __has_builtin
#define __has_builtin(builtin) 0
#endif

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
    constexpr static inline void sincos(type value, type& sine, type& cosine);

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
        return static_cast<type>(3.14159265358979323846264338327950288419716939937510582097494459230781640628);
    }

    template <typename type>
    constexpr static inline type e() {
        return static_cast<type>(2.71828182845904523536028747135266249775724709369995957496696762772407663035);
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
        static_assert(is_same_type<type, float>::value || is_same_type<type, double>::value, "Only float and double are supported.");
        if constexpr (is_same_type<type, float>::value) {
#if __has_builtin(__builtin_nanf)
            return __builtin_nanf("0");
#else
            return __builtin_bit_cast(float, 0x7FC00000u);
#endif
        }
        else {
#if __has_builtin(__builtin_nan)
            return __builtin_nan("0");
#else
            return __builtin_bit_cast(double, 0x7FF8000000000000ull);
#endif
        }
    }

    template <typename type>
    constexpr static inline type inf() {
        static_assert(is_same_type<type, float>::value || is_same_type<type, double>::value, "Only float and double are supported.");
        if constexpr (is_same_type<type, float>::value) {
#if __has_builtin(__builtin_inff)
            return __builtin_inff();
#else
            return __builtin_bit_cast(float, 0x7F800000u);
#endif
        }
        else {
#if __has_builtin(__builtin_inf)
            return __builtin_inf();
#else
            return __builtin_bit_cast(double, 0x7FF0000000000000ull);
#endif
        }
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
        return abs(value) == inf<type>();
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
        return !isnan(value) && !isinf(value);
    }

    template <typename type>
    constexpr static inline type copysign(type magnitude, type sign) {
        static_assert(is_same_type<type, float>::value || is_same_type<type, double>::value, "Only float and double are supported.");
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
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_bit_cast(float, (__builtin_bit_cast(unsigned int, magnitude) & 0x7FFFFFFFu) | (__builtin_bit_cast(unsigned int, sign) & 0x80000000u));
        }
        else {
            return __builtin_bit_cast(double, (__builtin_bit_cast(unsigned long long int, magnitude) & 0x7FFFFFFFFFFFFFFFull) | (__builtin_bit_cast(unsigned long long int, sign) & 0x8000000000000000ull));
        }
    }

    template <typename type>
    constexpr static inline bool signbit(type value) {
        static_assert(is_same_type<type, float>::value || is_same_type<type, double>::value, "Only float and double are supported.");
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
        if constexpr (is_same_type<type, float>::value) {
            return (__builtin_bit_cast(unsigned int, value) >> 31u) != 0u;
        }
        else {
            return (__builtin_bit_cast(unsigned long long int, value) >> 63u) != 0u;
        }
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
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_bit_cast(float, __builtin_bit_cast(unsigned int, value) & 0x7FFFFFFFu);
        }
        else if constexpr (is_same_type<type, double>::value) {
            return __builtin_bit_cast(double, __builtin_bit_cast(unsigned long long int, value) & 0x7FFFFFFFFFFFFFFFull);
        }
        else {
            return (value < 0) ? -value : value;
        }
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
        constexpr const type max_integer_as_type = static_cast<type>(max_integer / 2) * type(2);
        constexpr const type min_integer_as_type = static_cast<type>(min_integer);
        if ((value >= max_integer_as_type) || (value <= min_integer_as_type) || isnan(value)) {
            return value;
        }
        const long long int casted = static_cast<long long int>(value);
        const type rounded = static_cast<type>(casted);
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
        constexpr const type max_integer_as_type = static_cast<type>(max_integer / 2) * type(2);
        constexpr const type min_integer_as_type = static_cast<type>(min_integer);
        if ((value >= max_integer_as_type) || (value <= min_integer_as_type) || isnan(value)) {
            return value;
        }
        const long long int casted = static_cast<long long int>(value);
        const type rounded = static_cast<type>(casted);
        return ((rounded == value) || (value <= 0)) ? rounded : rounded + 1;
    }

    constexpr static inline int round(float value) {
#if __has_builtin(__builtin_roundf) && __has_builtin(__builtin_is_constant_evaluated)
        // The rounding builtin is not usable in constant expressions on every compiler, so only use it at runtime.
        if (!__builtin_is_constant_evaluated()) {
            return static_cast<int>(__builtin_roundf(value));
        }
#endif
        const int truncated = static_cast<int>(value);
        const float remainder = value - static_cast<float>(truncated);
        return truncated + ((remainder >= 0.5f) ? 1 : ((remainder <= -0.5f) ? -1 : 0));
    }

    constexpr static inline long long int round(double value) {
#if __has_builtin(__builtin_round) && __has_builtin(__builtin_is_constant_evaluated)
        // The rounding builtin is not usable in constant expressions on every compiler, so only use it at runtime.
        if (!__builtin_is_constant_evaluated()) {
            return static_cast<long long int>(__builtin_round(value));
        }
#endif
        const long long int truncated = static_cast<long long int>(value);
        const double remainder = value - static_cast<double>(truncated);
        return truncated + ((remainder >= 0.5) ? 1 : ((remainder <= -0.5) ? -1 : 0));
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
        const double value_as_double = static_cast<double>(value);
        const unsigned long long int bits = (__builtin_bit_cast(unsigned long long int, value_as_double) >> 1) + 0x1FF7A3BEA91D9B1BULL;
        double estimate = __builtin_bit_cast(double, bits);
        estimate = 0.5 * (estimate + value_as_double / estimate);
        double previous = estimate;
        do {
            previous = estimate;
            estimate = 0.5 * (estimate + value_as_double / estimate);
        } while (estimate < previous);
        return static_cast<type>(previous);
    }

    template <typename type>
    constexpr static type pythag(const type a, const type b) {
        const type abs_a = abs(a);
        const type abs_b = abs(b);
        if (abs_a > abs_b) {
            return abs_a * sqrt(type(1) + sqr(abs_b / abs_a));
        }
        if (abs_b == 0) {
            return 0;
        }
        return abs_b * sqrt(type(1) + sqr(abs_a / abs_b));
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
        const double value_as_double = static_cast<double>(value);
        if (value_as_double > 709.782712893384)
            return inf<type>();
        if (value_as_double < -745.1332191019412)
            return 0;
        constexpr const double ln2 = 6.93147180559945309417e-01;
        constexpr const double ln2_hi = 6.93147180369123816490e-01;
        constexpr const double ln2_lo = 1.90821492927058770002e-10;
        const long long int k = round(value_as_double / ln2);
        const double k_as_double = static_cast<double>(k);
        const double r = (value_as_double - k_as_double * ln2_hi) - k_as_double * ln2_lo;
        double term = 1.0;
        double sum = 1.0;
        for (int n = 1; n < 40; ++n) {
            term *= r / static_cast<double>(n);
            sum += term;
            if (abs(term) <= sum * 1e-20) {
                break;
            }
        }
        int exponent = static_cast<int>(k);
        while (exponent > 1023) {
            sum *= __builtin_bit_cast(double, 0x7FE0000000000000ull); // 2^1023
            exponent -= 1023;
        }
        while (exponent < -1022) {
            sum *= __builtin_bit_cast(double, 0x0010000000000000ull); // 2^-1022
            exponent += 1022;
        }
        const double result = sum * __builtin_bit_cast(double, static_cast<unsigned long long int>(exponent + 1023) << 52u);
        if constexpr (is_same_type<type, float>::value) {
            if (result > 3.4028235677973366e+38) {
                return inf<float>();
            }
        }
        return static_cast<type>(result);
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
        constexpr const double ln2 = 6.93147180559945309417e-01;
        unsigned long long int value_bits = __builtin_bit_cast(unsigned long long int, static_cast<double>(value));
        int exponent = 0;
        if ((value_bits & 0x7FF0000000000000ull) == 0ull) {
            value_bits = __builtin_bit_cast(unsigned long long int, static_cast<double>(value) * 18014398509481984.0); // 2^54
            exponent = -54;
        }
        exponent += static_cast<int>((value_bits >> 52u) & 0x7FFull) - 1023;
        double mantissa = __builtin_bit_cast(double, (value_bits & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull);
        if (mantissa > 1.4142135623730951) {
            mantissa *= 0.5;
            exponent += 1;
        }
        const double s = (mantissa - 1.0) / (mantissa + 1.0);
        const double s2 = s * s;
        double power = s;
        double sum = s;
        for (int n = 3; n < 80; n += 2) {
            power *= s2;
            const double term = power / static_cast<double>(n);
            sum += term;
            if (abs(term) <= abs(sum) * 1e-19) {
                break;
            }
        }
        return static_cast<type>(static_cast<double>(exponent) * ln2 + 2.0 * sum);
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
        const double x = static_cast<double>(value);
        const double y = static_cast<double>(exponent);
        if ((y == 0.0) || (x == 1.0))
            return type(1);
        if (isnan(x) || isnan(y))
            return nan<type>();
        const double x_as_absolute = abs(x);
        const bool y_is_integer = isfinite(y) && ((abs(y) >= 9007199254740992.0) || (static_cast<double>(round(y)) == y));
        const bool y_is_odd_integer = y_is_integer && (abs(y) < 9007199254740992.0) && ((round(y) & 1) != 0);
        if (x == 0.0) {
            if (y < 0.0)
                return y_is_odd_integer ? copysign(inf<type>(), value) : inf<type>();
            return y_is_odd_integer ? value : type(0);
        }
        if (isinf(y)) {
            if (x_as_absolute == 1.0)
                return type(1);
            return ((x_as_absolute < 1.0) == (y < 0.0)) ? inf<type>() : type(0);
        }
        if (isinf(x)) {
            if (x > 0.0)
                return (y < 0.0) ? type(0) : inf<type>();
            if (y < 0.0)
                return y_is_odd_integer ? type(-0.0) : type(0);
            return y_is_odd_integer ? -inf<type>() : inf<type>();
        }
        if ((x < 0.0) && !y_is_integer)
            return nan<type>();
        const double result = exp(y * log(x_as_absolute));
        if constexpr (is_same_type<type, float>::value) {
            if (result > 3.4028235677973366e+38) {
                return ((x < 0.0) && y_is_odd_integer) ? -inf<float>() : inf<float>();
            }
        }
        return static_cast<type>(((x < 0.0) && y_is_odd_integer) ? -result : result);
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
        double sine = 0.0;
        double cosine = 0.0;
        sincos(static_cast<double>(value), sine, cosine);
        return static_cast<type>(sine);
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
        double sine = 0.0;
        double cosine = 0.0;
        sincos(static_cast<double>(value), sine, cosine);
        return static_cast<type>(cosine);
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
        constexpr const double two_over_pi = 6.36619772367581382433e-01;
        constexpr const double two_pi = 6.28318530717958647693e+00;
        constexpr const double pio2_1 = 1.57079632673412561417e+00;
        constexpr const double pio2_2 = 6.07710050630396597660e-11;
        constexpr const double pio2_3 = 2.02226624879595063154e-21;
        const double value_as_double = static_cast<double>(value);
        const double reduced = (abs(value_as_double) < 1.5e6) ? value_as_double : fmod(value_as_double, two_pi);
        const long long int k = round(reduced * two_over_pi);
        const double k_as_double = static_cast<double>(k);
        const double r = ((reduced - k_as_double * pio2_1) - k_as_double * pio2_2) - k_as_double * pio2_3;
        const double r2 = r * r;
        double s = 1.0 / 51090942171709440000.0;
        s = s * r2 - 1.0 / 121645100408832000.0;
        s = s * r2 + 1.0 / 355687428096000.0;
        s = s * r2 - 1.0 / 1307674368000.0;
        s = s * r2 + 1.0 / 6227020800.0;
        s = s * r2 - 1.0 / 39916800.0;
        s = s * r2 + 1.0 / 362880.0;
        s = s * r2 - 1.0 / 5040.0;
        s = s * r2 + 1.0 / 120.0;
        s = s * r2 - 1.0 / 6.0;
        s = s * r2 + 1.0;
        s = s * r;
        double c = 1.0 / 2432902008176640000.0;
        c = c * r2 - 1.0 / 6402373705728000.0;
        c = c * r2 + 1.0 / 20922789888000.0;
        c = c * r2 - 1.0 / 87178291200.0;
        c = c * r2 + 1.0 / 479001600.0;
        c = c * r2 - 1.0 / 3628800.0;
        c = c * r2 + 1.0 / 40320.0;
        c = c * r2 - 1.0 / 720.0;
        c = c * r2 + 1.0 / 24.0;
        c = c * r2 - 1.0 / 2.0;
        c = c * r2 + 1.0;
        const long long int quadrant = k & 3;
        if (quadrant == 0) {
            sine = static_cast<type>(s);
            cosine = static_cast<type>(c);
        }
        else if (quadrant == 1) {
            sine = static_cast<type>(c);
            cosine = static_cast<type>(-s);
        }
        else if (quadrant == 2) {
            sine = static_cast<type>(-s);
            cosine = static_cast<type>(-c);
        }
        else {
            sine = static_cast<type>(-c);
            cosine = static_cast<type>(s);
        }
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
        if (value == type(0))
            return value;
        if ((value < type(-1)) || (value > type(1)) || isnan(value))
            return nan<type>();
        const double value_as_double = static_cast<double>(value);
        return static_cast<type>(atan2(value_as_double, sqrt((1.0 - value_as_double) * (1.0 + value_as_double))));
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
        if (value == type(1))
            return type(0);
        if ((value < type(-1)) || (value > type(1)) || isnan(value))
            return nan<type>();
        const double value_as_double = static_cast<double>(value);
        return static_cast<type>(atan2(sqrt((1.0 - value_as_double) * (1.0 + value_as_double)), value_as_double));
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
            return -pi<type>() * type(0.5);
        if ((x == 0) && (y > 0))
            return +pi<type>() * type(0.5);
        if (isinf(x) && signbit(x) && isfinite(y) && (y > 0))
            return +pi<type>();
        if (isinf(x) && signbit(x) && isfinite(y) && (y < 0))
            return -pi<type>();
        if (isinf(x) && !signbit(x) && isfinite(y) && (y > 0))
            return type(+0.0);
        if (isinf(x) && !signbit(x) && isfinite(y) && (y < 0))
            return type(-0.0);
        if (isnan(x) || isnan(y))
            return nan<type>();
        const double y_as_absolute = static_cast<double>(abs(y));
        const double x_as_absolute = static_cast<double>(abs(x));
        const bool swap = y_as_absolute > x_as_absolute;
        const double ratio = swap ? (x_as_absolute / y_as_absolute) : (y_as_absolute / x_as_absolute);
        const double t1 = ratio / (1.0 + sqrt(1.0 + ratio * ratio));
        const double t2 = t1 / (1.0 + sqrt(1.0 + t1 * t1));
        const double t2_2 = t2 * t2;
        double power = t2;
        double sum = t2;
        for (int n = 3; n < 100; n += 2) {
            power *= -t2_2;
            const double term = power / static_cast<double>(n);
            sum += term;
            if (abs(term) <= abs(sum) * 1e-19) {
                break;
            }
        }
        double angle = 4.0 * sum;
        if (swap) {
            angle = pi<double>() * 0.5 - angle;
        }
        if (x < 0) {
            angle = pi<double>() - angle;
        }
        return static_cast<type>(signbit(y) ? -angle : angle);
    }
}

#endif // ZEROSLAM_MATH_MATH_HPP
