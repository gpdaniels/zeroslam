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
#ifndef MATH_HPP
#define MATH_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace math {
    constexpr static inline double pi();
    constexpr static inline double e();
    template <typename type>
    constexpr static const type epsilon();
    constexpr static inline double nan();
    constexpr static inline double inf();
    constexpr static inline bool isnan(double value);
    constexpr static inline bool isinf(double value);
    constexpr static inline bool isfinite(double value);
    template <typename type>
    constexpr static inline type copysign(type magnitude, type sign);
    constexpr static inline double copysign(double magnitude, double sign);
    constexpr static inline bool signbit(double value);
    template <typename type>
    constexpr static inline type abs(type value);
    template <typename type>
    constexpr static inline type min(type lhs, type rhs);
    template <typename type>
    constexpr static inline type max(type lhs, type rhs);
    constexpr static inline double floor(double value);
    constexpr static inline double ceil(double value);
    constexpr static inline int round(float value);
    constexpr static inline long long int round(double value);
    constexpr static inline double fmod(double value, double modulus);
    template <typename type>
    constexpr static inline type sqr(type value);
    constexpr static inline double sqrt(double value);
    template <typename type>
    constexpr static type pythag(const type a, const type b);
    constexpr static inline double exp(double value);
    constexpr static inline double log(double value);
    constexpr static inline double pow(double value, double exponent);
    constexpr static inline double sin(double value);
    constexpr static inline double cos(double value);
    constexpr static inline double asin(double value);
    constexpr static inline double acos(double value);
    constexpr static inline double atan2(double y, double x);
}

namespace math {
    constexpr static inline double pi() {
        return 3.14159265358979323846264338327950288419716939937510582097494459230781640628;
    }

    constexpr static inline double e() {
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

    constexpr static inline double nan() {
        return __builtin_nan("0");
    }

    constexpr static inline double inf() {
        return __builtin_inf();
    }

    constexpr static inline bool isnan(double value) {
        return value != value;
    }

    constexpr static inline bool isinf(double value) {
        if (value != value) {
            return false;
        }
        if ((value > 0) && ((value / value) != (value / value))) {
            return true;
        }
        if ((value < 0) && ((value / value) != (value / value))) {
            return true;
        }
        return false;
    }

    constexpr static inline bool isfinite(double value) {
        if (value != value)
            return false;
        if ((value > 0) && ((value / value) != (value / value)))
            return false;
        if ((value < 0) && ((value / value) != (value / value)))
            return false;
        return true;
    }

    template <typename type>
    constexpr static inline type copysign(type magnitude, type sign) {
        return (sign >= 0 ? (magnitude >= 0 ? magnitude : -magnitude) : (magnitude >= 0 ? -magnitude : magnitude));
    }

    constexpr static inline double copysign(double magnitude, double sign) {
        return __builtin_copysign(magnitude, sign);
    }

    constexpr static inline bool signbit(double value) {
        return __builtin_copysign(1, value) < 0;
    }

    template <typename type>
    constexpr static inline type abs(type value) {
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

    constexpr static inline double floor(double value) {
        constexpr const long long int max_integer = static_cast<long long int>(static_cast<unsigned long long int>(-1) >> 1);
        constexpr const long long int min_integer = -max_integer - 1;
        constexpr const double max_integer_as_double = static_cast<double>(max_integer / 2) * 2.0;
        constexpr const double min_integer_as_double = static_cast<double>(min_integer);
        if ((value >= max_integer_as_double) || (value <= min_integer_as_double) || isnan(value)) {
            return value;
        }
        const long long int casted = static_cast<long long int>(value);
        const double rounded = static_cast<double>(casted);
        return ((rounded == value) || (value >= 0)) ? rounded : rounded - 1;
    }

    constexpr static inline double ceil(double value) {
        constexpr const long long int max_integer = static_cast<long long int>(static_cast<unsigned long long int>(-1) >> 1);
        constexpr const long long int min_integer = -max_integer - 1;
        constexpr const double max_integer_as_double = static_cast<double>(max_integer / 2) * 2.0;
        constexpr const double min_integer_as_double = static_cast<double>(min_integer);
        if ((value >= max_integer_as_double) || (value <= min_integer_as_double) || isnan(value)) {
            return value;
        }
        const long long int casted = static_cast<long long int>(value);
        const double rounded = static_cast<double>(casted);
        return ((rounded == value) || (value <= 0)) ? rounded : rounded + 1;
    }

    constexpr static inline int round(float value) {
        return (value > 0.0f) ? static_cast<int>(value + 0.5f) : static_cast<int>(value - 0.5f);
    }

    constexpr static inline long long int round(double value) {
        return (value > 0.0) ? static_cast<long long int>(value + 0.5) : static_cast<long long int>(value - 0.5);
    }

    constexpr static inline double fmod(double value, double modulus) {
        if (isnan(value) || isnan(modulus))
            return nan();
        if ((value == 0.0) && (modulus != 0.0))
            return copysign(0.0, value);
        if (isinf(value) && !isnan(modulus))
            return nan();
        if (!isnan(value) && (modulus == 0.0))
            return nan();
        if (isfinite(value) && isinf(modulus))
            return value;
        double value_as_absolute = abs(value);
        const double modulus_as_absolute = abs(modulus);
        while (value_as_absolute >= modulus_as_absolute) {
            double factor = modulus_as_absolute;
            while (value_as_absolute >= (2.0 * factor)) {
                factor *= 2.0;
            }
            value_as_absolute -= factor;
        }
        return copysign(value_as_absolute, value);
    }

    template <typename type>
    constexpr static inline type sqr(type value) {
        return value * value;
    }

    constexpr static inline double sqrt(double value) {
        if ((value < 0) || isnan(value))
            return nan();
        if ((value == 0) || isinf(value))
            return value;
        double estimate = value;
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

    constexpr static inline double exp(double value) {
        if (isnan(value))
            return nan();
        if (value == 0)
            return 1;
        if ((value < 0) && isinf(value))
            return 0;
        if ((value > 0) && isinf(value))
            return inf();
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

    constexpr static inline double log(double value) {
        if ((value < 0) || isnan(value))
            return nan();
        if (value == 0)
            return -inf();
        if (value == 1)
            return 0;
        if (isinf(value))
            return inf();
        constexpr const double epsilon = 1e-9;
        // Normalize the value and count how many times we divide by e.
        double working_value = (value < 1.0) ? (1.0 / value) : value;
        unsigned int exponent_count = 0;
        while ((working_value /= e()) > 1.0) {
            ++exponent_count;
        }
        // Prepare for series expansion.
        working_value = 1.0 / (working_value * e() - 1.0);
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

    constexpr static inline double pow(double value, double exponent) {
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
            return nan();
        if (value == 0.0 && is_value_positive && !is_exponent_positive && is_exponent_integer && is_exponent_odd)
            return inf();
        if (value == 0.0 && !is_value_positive && !is_exponent_positive && is_exponent_integer && is_exponent_odd)
            return -inf();
        if (value == 0.0 && !is_exponent_positive && !is_exponent_inf && ((is_exponent_integer && is_exponent_even) || (!is_exponent_integer)))
            return inf();
        if (value == 0.0 && !is_exponent_positive && is_exponent_inf)
            return inf();
        if (value == 0.0 && is_value_positive && is_exponent_positive && is_exponent_integer && is_exponent_odd)
            return +0.0;
        if (value == 0.0 && !is_value_positive && is_exponent_positive && is_exponent_integer && is_exponent_odd)
            return -0.0;
        if (value == 0.0 && is_exponent_positive && (!is_exponent_integer || (is_exponent_integer && is_exponent_even)))
            return +0.0;
        if (value == -1.0 && is_exponent_inf)
            return 1.0;
        if (!is_value_inf && !is_value_positive && !is_exponent_inf && !is_exponent_integer)
            return nan();
        if (value_as_absolute < 1.0 && !is_exponent_positive && is_exponent_inf)
            return inf();
        if (value_as_absolute > 1.0 && !is_exponent_positive && is_exponent_inf)
            return +0.0;
        if (value_as_absolute < 1.0 && is_exponent_positive && is_exponent_inf)
            return +0.0;
        if (value_as_absolute > 1.0 && is_exponent_positive && is_exponent_inf)
            return inf();
        if (!is_value_positive && is_value_inf && !is_exponent_positive && is_exponent_integer && is_exponent_odd)
            return -0.0;
        if (!is_value_positive && is_value_inf && !is_exponent_positive && (!is_exponent_integer || (is_exponent_integer && is_exponent_even)))
            return +0.0;
        if (!is_value_positive && is_value_inf && is_exponent_positive && is_exponent_integer && is_exponent_odd)
            return -inf();
        if (!is_value_positive && is_value_inf && is_exponent_positive && (!is_exponent_integer || (is_exponent_integer && is_exponent_even)))
            return inf();
        if (is_value_positive && is_value_inf && !is_exponent_positive)
            return +0.0;
        if (is_value_positive && is_value_inf && is_exponent_positive)
            return inf();
        return (value < 0) ? (exp(log(-value) * exponent) * (-1 + 2 * is_exponent_even)) : (exp(log(value) * exponent));
    }

    constexpr static inline double sin(double value) {
        if (value == 0)
            return copysign(0.0, value);
        if (!isfinite(value))
            return nan();
        const double angle = fmod(abs(value), (pi() * 2.0));
        const double sign = (angle <= pi()) ? (1.0 - 2.0 * signbit(value)) : (-1.0 + 2.0 * signbit(value));
        const double remapped = (angle > (pi() * 1.5)) ? ((pi() * 2.0) - angle) : (angle > (pi()))     ? angle - pi()
                                                                              : (angle > (pi() * 0.5)) ? pi() - angle
                                                                                                       : angle;
        const double remapped2 = remapped * remapped;
        return sign * remapped * (1.0 + remapped2 * (-0.1666666 + remapped2 * (0.0083119 + remapped2 * (-0.00018488140289))));
    }

    constexpr static inline double cos(double value) {
        if (value == 0)
            return 1.0;
        if (!isfinite(value))
            return nan();
        const double angle = fmod(abs(value), (pi() * 2.0));
        const double sign = (angle > (pi() * 1.5)) ? 1.0 : (angle > (pi() * 0.5)) ? -1.0
                                                                                  : 1.0;
        const double remapped = (angle > (pi() * 1.5)) ? ((pi() * 2.0) - angle) : (angle > (pi()))     ? angle - pi()
                                                                              : (angle > (pi() * 0.5)) ? pi() - angle
                                                                                                       : angle;
        const double remapped2 = remapped * remapped;
        return sign * 1.0 * (1.0 + remapped2 * (-0.5000000 + remapped2 * (0.0415896 + remapped2 * (-0.00129810625032))));
    }

    constexpr static inline double asin(double value) {
        if (value == 0.0)
            return value;
        if ((value < -1.0) || (value > 1.0) || isnan(value))
            return nan();
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
        angle = (pi() * 0.5) - angle;
        return sign * angle;
    }

    constexpr static inline double acos(double value) {
        if (value == 1.0)
            return 0.0;
        if ((value < -1.0) || (value > 1.0) || isnan(value))
            return nan();
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
        return (value < 0.0) ? (pi() + sign * angle) : (sign * angle);
    }

    constexpr static inline double atan2(double y, double x) {
        if ((y == 0) && ((x < 0) || ((x == 0) && signbit(x))))
            return copysign(pi(), y);
        if ((y == 0) && ((x > 0) || ((x == 0) && !signbit(x))))
            return copysign(0.0, y);
        if (isinf(y) && isfinite(x))
            return copysign(pi() * 0.5, y);
        if (isinf(y) && isinf(x) && signbit(x))
            return copysign(pi() * 0.75, y);
        if (isinf(y) && isinf(x) && !signbit(x))
            return copysign(pi() * 0.25, y);
        if ((x == 0) && (y < 0))
            return -pi() * 0.5;
        if ((x == 0) && (y > 0))
            return +pi() * 0.5;
        if (isinf(x) && signbit(x) && isfinite(y) && (y > 0))
            return +pi();
        if (isinf(x) && signbit(x) && isfinite(y) && (y < 0))
            return -pi();
        if (isinf(x) && !signbit(x) && isfinite(y) && (y > 0))
            return +0.0;
        if (isinf(x) && !signbit(x) && isfinite(y) && (y < 0))
            return -0.0;
        if (isnan(x) || isnan(y))
            return nan();
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
        angle = swap ? ((((ratio > 0.0) || ((ratio == 0.0) && !signbit(ratio))) ? (+pi() * 0.5) : (-pi() * 0.5)) - angle) : angle;
        if ((x >= 0.0) && (y >= 0.0)) {
        }
        else if ((x < 0.0) && (y >= 0.0)) {
            angle = +pi() + angle;
        }
        else if ((x < 0.0) && (y < 0.0)) {
            angle = -pi() + angle;
        }
        else if ((x >= 0.0) && (y < 0.0)) {
        }
        return angle;
    }
}

#endif // MATH_HPP