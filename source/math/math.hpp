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
// Defining ZEROSLAM_MATH_FALLBACK before including this header forces the fallbacks on every compiler, which the tests use to check them.
#if defined(__has_builtin) && !defined(ZEROSLAM_MATH_FALLBACK)
#define ZEROSLAM_MATH_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#define ZEROSLAM_MATH_HAS_BUILTIN(builtin) 0
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
    // The double precision cores of the fallback implementations, used where the compiler provides no builtins (MSVC). They rely on
    // error free transformations, so on floating point expressions not being contracted into fused multiply-adds.
    namespace fallback {
        struct double_double {
            double high;
            double low;
        };

        constexpr static inline double_double two_sum(const double lhs, const double rhs) {
            const double sum = lhs + rhs;
            const double rhs_virtual = sum - lhs;
            const double lhs_virtual = sum - rhs_virtual;
            return { sum, (lhs - lhs_virtual) + (rhs - rhs_virtual) };
        }

        // Requires |lhs| >= |rhs|, or lhs equal to zero.
        constexpr static inline double_double fast_two_sum(const double lhs, const double rhs) {
            const double sum = lhs + rhs;
            return { sum, rhs - (sum - lhs) };
        }

        // Dekker's exact product, for magnitudes below 2^996.
        constexpr static inline double_double two_product(const double lhs, const double rhs) {
            constexpr const double splitter = 134217729.0;
            const double product = lhs * rhs;
            const double lhs_scaled = splitter * lhs;
            const double lhs_high = lhs_scaled - (lhs_scaled - lhs);
            const double lhs_low = lhs - lhs_high;
            const double rhs_scaled = splitter * rhs;
            const double rhs_high = rhs_scaled - (rhs_scaled - rhs);
            const double rhs_low = rhs - rhs_high;
            return { product, (((lhs_high * rhs_high) - product) + (lhs_high * rhs_low) + (lhs_low * rhs_high)) + (lhs_low * rhs_low) };
        }

        constexpr static inline double_double add(const double_double lhs, const double_double rhs) {
            const double_double high = two_sum(lhs.high, rhs.high);
            const double_double low = two_sum(lhs.low, rhs.low);
            const double_double partial = fast_two_sum(high.high, high.low + low.high);
            return fast_two_sum(partial.high, partial.low + low.low);
        }

        constexpr static inline double_double multiply(const double_double lhs, const double_double rhs) {
            const double_double product = two_product(lhs.high, rhs.high);
            return fast_two_sum(product.high, product.low + ((lhs.high * rhs.low) + (lhs.low * rhs.high)));
        }

        constexpr static inline double_double divide(const double_double lhs, const double_double rhs) {
            const double quotient = lhs.high / rhs.high;
            const double_double remainder = add(lhs, multiply({ -quotient, 0.0 }, rhs));
            return fast_two_sum(quotient, remainder.high / rhs.high);
        }

        // Multiplies by 2^exponent, for exponents in [-1100, 1100], exactly unless the result is subnormal, which is then rounded once.
        constexpr static inline double scale(const double value, const int exponent) {
            constexpr const auto power_of_two = [](const int power) {
                return __builtin_bit_cast(double, static_cast<unsigned long long int>(power + 1023) << 52u);
            };
            if (exponent > 1023) {
                return (value * 0x1p+1023) * power_of_two(exponent - 1023);
            }
            if (exponent < -1022) {
                return (value * power_of_two(exponent + 1074)) * 0x1p-1074;
            }
            return value * power_of_two(exponent);
        }

        // exp(reduced) for |reduced| below ln(2) / 2 in double-double, as 1 + x + x^2 / 2 in double-double plus the rest of its series.
        constexpr static inline double_double exp_kernel(const double_double reduced) {
            const double x = reduced.high;
            double series = 1.0 / 355687428096000.0;
            series = (1.0 / 20922789888000.0) + (x * series);
            series = (1.0 / 1307674368000.0) + (x * series);
            series = (1.0 / 87178291200.0) + (x * series);
            series = (1.0 / 6227020800.0) + (x * series);
            series = (1.0 / 479001600.0) + (x * series);
            series = (1.0 / 39916800.0) + (x * series);
            series = (1.0 / 3628800.0) + (x * series);
            series = (1.0 / 362880.0) + (x * series);
            series = (1.0 / 40320.0) + (x * series);
            series = (1.0 / 5040.0) + (x * series);
            series = (1.0 / 720.0) + (x * series);
            series = (1.0 / 120.0) + (x * series);
            series = (1.0 / 24.0) + (x * series);
            series = (1.0 / 6.0) + (x * series);
            const double_double square = two_product(x, x);
            double_double sum = two_sum(1.0, x);
            sum = add(sum, { 0.5 * square.high, 0.5 * square.low });
            sum = add(sum, { square.high * x * series, 0.0 });
            // exp(x + low) is exp(x) * (1 + low) to double-double precision.
            return add(sum, { reduced.low * sum.high, 0.0 });
        }

        // exp(value) as mantissa * 2^exponent, for |value.high| below 746, with the mantissa in [0.7, 1.42] to double-double precision.
        constexpr static inline double_double exp_reduced(const double_double value, int& exponent) {
            constexpr const double inverse_ln2 = 0x1.71547652b82fep+0;
            // The leading part of ln(2) has 32 bits so its product with any exponent is exact, fdlibm's split.
            constexpr const double ln2_high = 0x1.62e42fee00000p-1;
            constexpr const double ln2_low = 0x1.a39ef35793c76p-33;
            const long long int power = round(value.high * inverse_ln2);
            const double power_as_double = static_cast<double>(power);
            const double_double reduced = add(add({ value.high - (power_as_double * ln2_high), 0.0 }, two_product(-power_as_double, ln2_low)), { value.low, 0.0 });
            exponent = static_cast<int>(power);
            return exp_kernel(reduced);
        }

        // ln(value) for finite positive values in double-double, by 2 atanh((m - 1) / (m + 1)) of the mantissa m in [0.7, 1.42].
        constexpr static inline double_double log_double_double(const double value) {
            constexpr const double_double ln2 = { 0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56 };
            constexpr const double_double two_thirds = divide({ 2.0, 0.0 }, { 3.0, 0.0 });
            constexpr const double_double two_fifths = divide({ 2.0, 0.0 }, { 5.0, 0.0 });
            unsigned long long int bits = __builtin_bit_cast(unsigned long long int, value);
            int exponent = 0;
            if ((bits >> 52u) == 0u) {
                bits = __builtin_bit_cast(unsigned long long int, value * 0x1p+54);
                exponent = -54;
            }
            exponent += static_cast<int>(bits >> 52u) - 1023;
            double mantissa = __builtin_bit_cast(double, (bits & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull);
            if (mantissa > 0x1.6a09e667f3bcdp+0) {
                mantissa *= 0.5;
                exponent += 1;
            }
            const double_double ratio = divide({ mantissa - 1.0, 0.0 }, two_sum(mantissa, 1.0));
            const double_double ratio_squared = multiply(ratio, ratio);
            const double_double ratio_cubed = multiply(ratio_squared, ratio);
            const double_double ratio_fifth = multiply(ratio_cubed, ratio_squared);
            // 2 atanh(s) = 2 s + 2 s^3 / 3 + 2 s^5 / 5 + 2 s^7 (1 / 7 + s^2 / 9 + ...), the first three terms in double-double.
            const double squared = ratio_squared.high;
            double series = 1.0 / 29.0;
            series = (1.0 / 27.0) + (squared * series);
            series = (1.0 / 25.0) + (squared * series);
            series = (1.0 / 23.0) + (squared * series);
            series = (1.0 / 21.0) + (squared * series);
            series = (1.0 / 19.0) + (squared * series);
            series = (1.0 / 17.0) + (squared * series);
            series = (1.0 / 15.0) + (squared * series);
            series = (1.0 / 13.0) + (squared * series);
            series = (1.0 / 11.0) + (squared * series);
            series = (1.0 / 9.0) + (squared * series);
            series = (1.0 / 7.0) + (squared * series);
            double_double result = add({ 2.0 * ratio.high, 2.0 * ratio.low }, multiply(ratio_cubed, two_thirds));
            result = add(result, multiply(ratio_fifth, two_fifths));
            result = add(result, { 2.0 * ratio_fifth.high * squared * series, 0.0 });
            if (exponent != 0) {
                const double exponent_as_double = static_cast<double>(exponent);
                result = add(add(two_product(exponent_as_double, ln2.high), { exponent_as_double * ln2.low, 0.0 }), result);
            }
            return result;
        }

        // magnitude^power for a finite positive magnitude and an integer power below 2^53, by squaring in double-double with the binary
        // exponents kept apart so that no intermediate result overflows, correctly rounded apart from near ties and exact when representable.
        constexpr static inline double pow_integer(const double magnitude, unsigned long long int power, const bool reciprocal) {
            unsigned long long int bits = __builtin_bit_cast(unsigned long long int, magnitude);
            long long int base_exponent = 0;
            if ((bits >> 52u) == 0u) {
                bits = __builtin_bit_cast(unsigned long long int, magnitude * 0x1p+54);
                base_exponent = -54;
            }
            base_exponent += static_cast<long long int>(bits >> 52u) - 1023;
            double_double base = { __builtin_bit_cast(double, (bits & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull), 0.0 };
            double_double result = { 1.0, 0.0 };
            long long int result_exponent = 0;
            // The mantissas stay in [1, 2) and the exponents keep the sign of the first, so a large exponent is already out of range.
            constexpr const long long int exponent_limit = 4096;
            while (true) {
                if ((power & 1u) != 0u) {
                    result = multiply(result, base);
                    result_exponent += base_exponent;
                    if (result.high >= 2.0) {
                        result = { 0.5 * result.high, 0.5 * result.low };
                        ++result_exponent;
                    }
                }
                power >>= 1u;
                if ((power == 0u) || (base_exponent > exponent_limit) || (base_exponent < -exponent_limit) || (result_exponent > exponent_limit) || (result_exponent < -exponent_limit)) {
                    break;
                }
                base = multiply(base, base);
                base_exponent *= 2;
                if (base.high >= 2.0) {
                    base = { 0.5 * base.high, 0.5 * base.low };
                    ++base_exponent;
                }
            }
            if (power != 0u) {
                result_exponent = (base_exponent > 0) ? exponent_limit : -exponent_limit;
            }
            if (reciprocal) {
                result = divide({ 1.0, 0.0 }, result);
                result_exponent = -result_exponent;
            }
            if (result_exponent > 1100) {
                return __builtin_bit_cast(double, 0x7FF0000000000000ull);
            }
            if (result_exponent < -1100) {
                return 0.0;
            }
            return scale(result.high + result.low, static_cast<int>(result_exponent));
        }

        // The quadrant of value and its remainder modulo pi / 2 in double-double, within [-pi / 4, pi / 4] up to rounding. Up to 2^19 pi / 2
        // by Cody and Waite's three part pi / 2 with fdlibm's cancellation checks, and beyond by Payne and Hanek's multiplication by 2 / pi.
        constexpr static inline long long int reduce_quarter_turns(const double value, double_double& reduced) {
            constexpr const double inverse_quarter_turn = 0x1.45f306dc9c883p-1;
            constexpr const double quarter_turn_1 = 0x1.921fb54400000p+0;
            constexpr const double quarter_turn_1_tail = 0x1.0b4611a626331p-34;
            constexpr const double quarter_turn_2 = 0x1.0b4611a600000p-34;
            constexpr const double quarter_turn_2_tail = 0x1.3198a2e037073p-69;
            constexpr const double quarter_turn_3 = 0x1.3198a2e000000p-69;
            constexpr const double quarter_turn_3_tail = 0x1.b839a252049c1p-104;
            constexpr const double_double quarter_turn = { 0x1.921fb54442d18p+0, 0x1.1a62633145c07p-54 };
            const bool negative = value < 0.0;
            const double magnitude = negative ? -value : value;
            long long int quadrant = 0;
            if (magnitude <= 0x1.921fb54442d18p+19) {
                const long long int turns = round(magnitude * inverse_quarter_turn);
                const double turns_as_double = static_cast<double>(turns);
                const auto exponent_of = [](const double number) {
                    return static_cast<int>((__builtin_bit_cast(unsigned long long int, number) >> 52u) & 0x7FFu);
                };
                double remainder = magnitude - (turns_as_double * quarter_turn_1);
                double correction = turns_as_double * quarter_turn_1_tail;
                double high = remainder - correction;
                if ((exponent_of(magnitude) - exponent_of(high)) > 16) {
                    const double previous = remainder;
                    correction = turns_as_double * quarter_turn_2;
                    remainder = previous - correction;
                    correction = (turns_as_double * quarter_turn_2_tail) - ((previous - remainder) - correction);
                    high = remainder - correction;
                    if ((exponent_of(magnitude) - exponent_of(high)) > 49) {
                        const double last = remainder;
                        correction = turns_as_double * quarter_turn_3;
                        remainder = last - correction;
                        correction = (turns_as_double * quarter_turn_3_tail) - ((last - remainder) - correction);
                        high = remainder - correction;
                    }
                }
                reduced = { high, (remainder - high) - correction };
                quadrant = turns;
            }
            else {
                // The bits of 2 / pi, enough for any double. The magnitude is m * 2^e with an integer m of 53 bits, and the bits of 2 / pi
                // with weight above 2^(1 - e) only add multiples of four to the product, which leave the quadrant unchanged.
                constexpr const unsigned int two_over_pi[40] = {
                    0xA2F9836Eu,
                    0x4E441529u,
                    0xFC2757D1u,
                    0xF534DDC0u,
                    0xDB629599u,
                    0x3C439041u,
                    0xFE5163ABu,
                    0xDEBBC561u,
                    0xB7246E3Au,
                    0x424DD2E0u,
                    0x06492EEAu,
                    0x09D1921Cu,
                    0xFE1DEB1Cu,
                    0xB129A73Eu,
                    0xE88235F5u,
                    0x2EBB4484u,
                    0xE99C7026u,
                    0xB45F7E41u,
                    0x3991D639u,
                    0x835339F4u,
                    0x9C845F8Bu,
                    0xBDF9283Bu,
                    0x1FF897FFu,
                    0xDE05980Fu,
                    0xEF2F118Bu,
                    0x5A0A6D1Fu,
                    0x6D367ECFu,
                    0x27CB09B7u,
                    0x4F463F66u,
                    0x9E5FEA2Du,
                    0x7527BAC7u,
                    0xEBE5F17Bu,
                    0x3D0739F7u,
                    0x8A5292EAu,
                    0x6BFB5FB1u,
                    0x1F8D5D08u,
                    0x56033046u,
                    0xFC7B6BABu,
                    0xF0CFBC20u,
                    0x9AF4361Du
                };
                const unsigned long long int bits = __builtin_bit_cast(unsigned long long int, magnitude);
                const int exponent = static_cast<int>(bits >> 52u) - 1075;
                const unsigned long long int mantissa = (bits & 0x000FFFFFFFFFFFFFull) | 0x0010000000000000ull;
                // A window of 192 bits of 2 / pi from bit first_bit after the binary point, most significant word first.
                const int first_bit = ((exponent - 1) > 1) ? (exponent - 1) : 1;
                const int word = (first_bit - 1) / 32;
                const int shift = (first_bit - 1) % 32;
                unsigned int window[6] = {};
                for (int index = 0; index < 6; ++index) {
                    const unsigned int upper = two_over_pi[word + index];
                    const unsigned int lower = two_over_pi[word + index + 1];
                    window[index] = (shift == 0) ? upper : ((upper << static_cast<unsigned int>(shift)) | (lower >> static_cast<unsigned int>(32 - shift)));
                }
                // The product of the mantissa and the window, least significant word first.
                unsigned int product[8] = {};
                const unsigned long long int mantissa_parts[2] = { mantissa & 0xFFFFFFFFull, mantissa >> 32u };
                for (int part = 0; part < 2; ++part) {
                    unsigned long long int carry = 0;
                    for (int index = 0; index < 6; ++index) {
                        const unsigned long long int sum = static_cast<unsigned long long int>(product[part + index]) + (mantissa_parts[part] * window[5 - index]) + carry;
                        product[part + index] = static_cast<unsigned int>(sum & 0xFFFFFFFFull);
                        carry = sum >> 32u;
                    }
                    product[part + 6] = static_cast<unsigned int>(static_cast<unsigned long long int>(product[part + 6]) + carry);
                }
                // The binary point of magnitude * 2 / pi lies above bit point of the product.
                const int point = first_bit + 191 - exponent;
                const auto bits_at = [&product](const int position) {
                    const int index = position / 32;
                    const int offset = position % 32;
                    if (offset == 0) {
                        return product[index];
                    }
                    const unsigned int above = (index + 1 < 8) ? product[index + 1] : 0u;
                    return (product[index] >> static_cast<unsigned int>(offset)) | (above << static_cast<unsigned int>(32 - offset));
                };
                quadrant = static_cast<long long int>(bits_at(point) & 3u);
                unsigned int fraction[4] = { bits_at(point - 32), bits_at(point - 64), bits_at(point - 96), bits_at(point - 128) };
                // A fraction of a half or more is taken from the next quadrant.
                double sign = 1.0;
                if ((fraction[0] & 0x80000000u) != 0u) {
                    ++quadrant;
                    sign = -1.0;
                    unsigned long long int borrow = 1;
                    for (int index = 3; index >= 0; --index) {
                        // The complement of a word, in the width it is kept, zero extended before the borrow is added.
                        const unsigned long long int negated = ((~static_cast<unsigned long long int>(fraction[index])) & 0xFFFFFFFFull) + borrow;
                        fraction[index] = static_cast<unsigned int>(negated & 0xFFFFFFFFull);
                        borrow = negated >> 32u;
                    }
                }
                double_double turn_fraction = { static_cast<double>(fraction[3]) * 0x1p-128, 0.0 };
                turn_fraction = add(turn_fraction, { static_cast<double>(fraction[2]) * 0x1p-96, 0.0 });
                turn_fraction = add(turn_fraction, { static_cast<double>(fraction[1]) * 0x1p-64, 0.0 });
                turn_fraction = add(turn_fraction, { static_cast<double>(fraction[0]) * 0x1p-32, 0.0 });
                reduced = multiply({ sign * turn_fraction.high, sign * turn_fraction.low }, quarter_turn);
            }
            if (negative) {
                reduced = { -reduced.high, -reduced.low };
                quadrant = -quadrant;
            }
            return quadrant;
        }

        // The square root of a non-negative double-double, by one correction of the double root.
        constexpr static inline double_double sqrt_double_double(const double_double value) {
            if (!(value.high > 0.0)) {
                return { 0.0, 0.0 };
            }
            const double root = sqrt(value.high);
            const double_double remainder = add(value, two_product(-root, root));
            return fast_two_sum(root, remainder.high / (2.0 * root));
        }

        // atan(numerator / denominator) in double-double for non-negative numerator and denominator, not both zero. The ratio is taken at
        // most one, and reduced by atan(t) = atan(c) + atan((t - c) / (1 + c t)) with fdlibm's breakpoints to below 7 / 16 for the series.
        constexpr static inline double_double atan_ratio(const double_double numerator, const double_double denominator) {
            constexpr const double_double quarter_turn = { 0x1.921fb54442d18p+0, 0x1.1a62633145c07p-54 };
            constexpr const double_double eighth_turn = { 0x1.921fb54442d18p-1, 0x1.1a62633145c07p-55 };
            constexpr const double_double atan_half = { 0x1.dac670561bb4fp-2, 0x1.a2b7f222f65e2p-56 };
            constexpr const double_double minus_one_third = divide({ -1.0, 0.0 }, { 3.0, 0.0 });
            // The ratio does not change when both are scaled by a power of two, which keeps the division's products in range.
            const double largest = (numerator.high > denominator.high) ? numerator.high : denominator.high;
            const double range_scale = (largest > 0x1p+900) ? 0x1p-600 : ((largest < 0x1p-900) ? 0x1p+600 : 1.0);
            const double_double scaled_numerator = { numerator.high * range_scale, numerator.low * range_scale };
            const double_double scaled_denominator = { denominator.high * range_scale, denominator.low * range_scale };
            const bool swap = numerator.high > denominator.high;
            const double_double ratio = swap ? divide(scaled_denominator, scaled_numerator) : divide(scaled_numerator, scaled_denominator);
            double_double base = { 0.0, 0.0 };
            double_double reduced = ratio;
            if (ratio.high > 0.6875) {
                reduced = divide(add(ratio, { -1.0, 0.0 }), add(ratio, { 1.0, 0.0 }));
                base = eighth_turn;
            }
            else if (ratio.high > 0.4375) {
                reduced = divide(add(ratio, { -0.5, 0.0 }), add({ 1.0, 0.0 }, { 0.5 * ratio.high, 0.5 * ratio.low }));
                base = atan_half;
            }
            // atan(u) = u - u^3 / 3 + u^5 (1 / 5 - u^2 / 7 + ...) with u - u^3 / 3 in double-double, and atan(u + low) = atan(u) + low (1 - u^2).
            const double u = reduced.high;
            const double_double square_exact = two_product(u, u);
            const double square = square_exact.high;
            double series = 1.0 / 45.0;
            series = (-1.0 / 43.0) + (square * series);
            series = (1.0 / 41.0) + (square * series);
            series = (-1.0 / 39.0) + (square * series);
            series = (1.0 / 37.0) + (square * series);
            series = (-1.0 / 35.0) + (square * series);
            series = (1.0 / 33.0) + (square * series);
            series = (-1.0 / 31.0) + (square * series);
            series = (1.0 / 29.0) + (square * series);
            series = (-1.0 / 27.0) + (square * series);
            series = (1.0 / 25.0) + (square * series);
            series = (-1.0 / 23.0) + (square * series);
            series = (1.0 / 21.0) + (square * series);
            series = (-1.0 / 19.0) + (square * series);
            series = (1.0 / 17.0) + (square * series);
            series = (-1.0 / 15.0) + (square * series);
            series = (1.0 / 13.0) + (square * series);
            series = (-1.0 / 11.0) + (square * series);
            series = (1.0 / 9.0) + (square * series);
            series = (-1.0 / 7.0) + (square * series);
            series = (1.0 / 5.0) + (square * series);
            const double_double cube = multiply({ u, 0.0 }, square_exact);
            const double_double cube_term = multiply(cube, minus_one_third);
            const double_double leading = two_sum(u, cube_term.high);
            const double_double angle = add(base, add(leading, { cube_term.low + ((reduced.low * (1.0 - square)) + (cube.high * square * series)), 0.0 }));
            return swap ? add(quarter_turn, { -angle.high, -angle.low }) : angle;
        }

        // sin(reduced) for |reduced| up to pi / 4, within about half an ulp, as x - x^3 / 3! + x^5 (1 / 5! - z / 7! + ...) with z = x^2
        // and x - x^3 / 3! in double-double.
        constexpr static inline double sine_kernel(const double_double reduced) {
            constexpr const double_double minus_one_sixth = divide({ -1.0, 0.0 }, { 6.0, 0.0 });
            const double x = reduced.high;
            const double_double square = two_product(x, x);
            const double z = square.high;
            double series = -1.0 / 121645100408832000.0;
            series = (1.0 / 355687428096000.0) + (z * series);
            series = (-1.0 / 1307674368000.0) + (z * series);
            series = (1.0 / 6227020800.0) + (z * series);
            series = (-1.0 / 39916800.0) + (z * series);
            series = (1.0 / 362880.0) + (z * series);
            series = (-1.0 / 5040.0) + (z * series);
            series = (1.0 / 120.0) + (z * series);
            const double_double cube = multiply({ x, 0.0 }, square);
            const double_double cube_term = multiply(cube, minus_one_sixth);
            const double_double leading = two_sum(x, cube_term.high);
            return leading.high + (leading.low + (cube_term.low + ((cube.high * z * series) + (reduced.low * (1.0 - (0.5 * z))))));
        }

        // cos(reduced) for |reduced| up to pi / 4, within about half an ulp, as 1 - z / 2 + z^2 / 4! + z^3 (-1 / 6! + z / 8! - ...) with
        // z = x^2, the rounding error of 1 - z / 2 carried into the rest and z^2 / 4! in double-double.
        constexpr static inline double cosine_kernel(const double_double reduced) {
            constexpr const double_double one_twenty_fourth = divide({ 1.0, 0.0 }, { 24.0, 0.0 });
            const double x = reduced.high;
            const double_double square = two_product(x, x);
            const double z = square.high;
            double series = 1.0 / 2432902008176640000.0;
            series = (-1.0 / 6402373705728000.0) + (z * series);
            series = (1.0 / 20922789888000.0) + (z * series);
            series = (-1.0 / 87178291200.0) + (z * series);
            series = (1.0 / 479001600.0) + (z * series);
            series = (-1.0 / 3628800.0) + (z * series);
            series = (1.0 / 40320.0) + (z * series);
            series = (-1.0 / 720.0) + (z * series);
            const double_double fourth = multiply(square, square);
            const double_double fourth_term = multiply(fourth, one_twenty_fourth);
            const double half = 0.5 * z;
            const double leading = 1.0 - half;
            const double leading_error = (1.0 - leading) - half;
            const double_double sum = two_sum(leading, fourth_term.high);
            return sum.high + (sum.low + ((leading_error - (0.5 * square.low)) + (fourth_term.low + ((fourth.high * z * series) - (x * reduced.low)))));
        }

        // sin (or cos when cosine is true) of value, from the quadrant and remainder of value modulo pi / 2.
        constexpr static inline double sine_or_cosine(const double value, const bool cosine) {
            if ((value <= 0x1.921fb54442d18p-1) && (value >= -0x1.921fb54442d18p-1)) {
                return cosine ? cosine_kernel({ value, 0.0 }) : sine_kernel({ value, 0.0 });
            }
            double_double reduced = { 0.0, 0.0 };
            const long long int quadrant = (reduce_quarter_turns(value, reduced) + (cosine ? 1 : 0)) & 3;
            // cos(x) = sin(x + pi / 2), so the cosine is the sine one quadrant on.
            const double result = ((quadrant & 1) == 0) ? sine_kernel(reduced) : cosine_kernel(reduced);
            return (quadrant >= 2) ? -result : result;
        }
    }
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
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_nanf)
            return __builtin_nanf("0");
#else
            return __builtin_bit_cast(float, 0x7FC00000u);
#endif
        }
        else {
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_nan)
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
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_inff)
            return __builtin_inff();
#else
            return __builtin_bit_cast(float, 0x7F800000u);
#endif
        }
        else {
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_inf)
            return __builtin_inf();
#else
            return __builtin_bit_cast(double, 0x7FF0000000000000ull);
#endif
        }
    }

    template <typename type>
    constexpr static inline bool isnan(type value) {
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_isnan)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_isnan(value);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_isnan)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_isnan(value);
        }
#endif
        return value != value;
    }

    template <typename type>
    constexpr static inline bool isinf(type value) {
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_isinf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_isinf(value);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_isinf)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_isinf(value);
        }
#endif
        return abs(value) == inf<type>();
    }

    template <typename type>
    constexpr static inline bool isfinite(type value) {
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_isfinite)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_isfinite(value);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_isfinite)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_isfinite(value);
        }
#endif
        return !isnan(value) && !isinf(value);
    }

    template <typename type>
    constexpr static inline type copysign(type magnitude, type sign) {
        static_assert(is_same_type<type, float>::value || is_same_type<type, double>::value, "Only float and double are supported.");
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_copysignf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_copysignf(magnitude, sign);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_copysign)
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
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_signbitf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_signbitf(value);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_signbit)
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
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_fabsf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_fabsf(value);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_fabs)
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
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_floorf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_floorf(value);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_floor)
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
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_ceilf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_ceilf(value);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_ceil)
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

    // Casting NaN or an out of range value to an integer is undefined, so out of range values saturate and NaN rounds to zero.

    constexpr static inline int round(float value) {
        constexpr const int max_integer = static_cast<int>(static_cast<unsigned int>(-1) >> 1);
        constexpr const int min_integer = -max_integer - 1;
        constexpr const float limit = 2147483648.0f;
        if (isnan(value)) {
            return 0;
        }
        if (value >= limit) {
            return max_integer;
        }
        if (value < -limit) {
            return min_integer;
        }
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_roundf) && ZEROSLAM_MATH_HAS_BUILTIN(__builtin_is_constant_evaluated)
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
        constexpr const long long int max_integer = static_cast<long long int>(static_cast<unsigned long long int>(-1) >> 1);
        constexpr const long long int min_integer = -max_integer - 1;
        constexpr const double limit = 9223372036854775808.0;
        if (isnan(value)) {
            return 0;
        }
        if (value >= limit) {
            return max_integer;
        }
        if (value < -limit) {
            return min_integer;
        }
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_round) && ZEROSLAM_MATH_HAS_BUILTIN(__builtin_is_constant_evaluated)
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
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_fmodf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_fmodf(value, modulus);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_fmod)
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
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_sqrtf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_sqrtf(value);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_sqrt)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_sqrt(value);
        }
#endif
        if ((value < 0) || isnan(value))
            return nan<type>();
        if ((value == 0) || isinf(value))
            return value;
        // Tiny and huge inputs are scaled by an even power of two, whose square root is exact, so the residuals below stay in range.
        double value_as_double = static_cast<double>(value);
        double root_scale = 1.0;
        if (value_as_double < 0x1p-900) {
            value_as_double *= 0x1p+1000;
            root_scale = 0x1p-500;
        }
        else if (value_as_double > 0x1p+900) {
            value_as_double *= 0x1p-1000;
            root_scale = 0x1p+500;
        }
        const unsigned long long int bits = (__builtin_bit_cast(unsigned long long int, value_as_double) >> 1) + 0x1FF7A3BEA91D9B1BULL;
        double estimate = __builtin_bit_cast(double, bits);
        estimate = 0.5 * (estimate + value_as_double / estimate);
        double previous = estimate;
        do {
            previous = estimate;
            estimate = 0.5 * (estimate + value_as_double / estimate);
        } while (estimate < previous);
        // The iteration can stop an ulp away from the correctly rounded root, which has the smaller exact residual of it and the neighbour
        // the residual points to. A square root is never a midpoint, so this is correct rounding.
        const auto residual = [value_as_double](const double root) {
            const fallback::double_double square = fallback::two_product(root, root);
            return (value_as_double - square.high) - square.low;
        };
        const double error = residual(previous);
        const unsigned long long int root_bits = __builtin_bit_cast(unsigned long long int, previous);
        const double neighbour = __builtin_bit_cast(double, (error < 0.0) ? (root_bits - 1u) : (root_bits + 1u));
        const double root = (abs(residual(neighbour)) < abs(error)) ? neighbour : previous;
        return static_cast<type>(root * root_scale);
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
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_expf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_expf(value);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_exp)
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
        int exponent = 0;
        const fallback::double_double mantissa = fallback::exp_reduced({ value_as_double, 0.0 }, exponent);
        const double result = fallback::scale(mantissa.high + mantissa.low, exponent);
        if constexpr (is_same_type<type, float>::value) {
            if (result > 3.4028235677973366e+38) {
                return inf<float>();
            }
        }
        return static_cast<type>(result);
    }

    template <typename type>
    constexpr static inline type log(type value) {
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_logf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_logf(value);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_log)
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
        const fallback::double_double result = fallback::log_double_double(static_cast<double>(value));
        return static_cast<type>(result.high + result.low);
    }

    template <typename type>
    constexpr static inline type pow(type value, type exponent) {
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_powf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_powf(value, exponent);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_pow)
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
        // Integer powers are exact where representable, others use a double-double logarithm and exponential.
        double result = 0.0;
        if (x_as_absolute == 1.0) {
            result = 1.0;
        }
        else if (y_is_integer && (abs(y) < 9007199254740992.0)) {
            result = fallback::pow_integer(x_as_absolute, static_cast<unsigned long long int>(abs(y)), y < 0.0);
        }
        else {
            const fallback::double_double logarithm = fallback::log_double_double(x_as_absolute);
            const double estimate = y * logarithm.high;
            if (estimate > 710.0) {
                result = inf<double>();
            }
            else if (estimate >= -746.0) {
                int binary_exponent = 0;
                const fallback::double_double mantissa = fallback::exp_reduced(fallback::add(fallback::two_product(y, logarithm.high), { y * logarithm.low, 0.0 }), binary_exponent);
                result = fallback::scale(mantissa.high + mantissa.low, binary_exponent);
            }
        }
        if constexpr (is_same_type<type, float>::value) {
            if (result > 3.4028235677973366e+38) {
                return ((x < 0.0) && y_is_odd_integer) ? -inf<float>() : inf<float>();
            }
        }
        return static_cast<type>(((x < 0.0) && y_is_odd_integer) ? -result : result);
    }

    template <typename type>
    constexpr static inline type sin(type value) {
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_sinf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_sinf(value);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_sin)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_sin(value);
        }
#endif
        if ((value == type(0)) || !isfinite(value)) {
            return (value == type(0)) ? value : nan<type>();
        }
        return static_cast<type>(fallback::sine_or_cosine(static_cast<double>(value), false));
    }

    template <typename type>
    constexpr static inline type cos(type value) {
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_cosf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_cosf(value);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_cos)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_cos(value);
        }
#endif
        if (!isfinite(value)) {
            return nan<type>();
        }
        return static_cast<type>(fallback::sine_or_cosine(static_cast<double>(value), true));
    }

    template <typename type>
    constexpr static inline void sincos(type value, type& sine, type& cosine) {
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_sincosf) && !defined(_WIN32)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_sincosf(value, &sine, &cosine);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_sincos) && !defined(_WIN32)
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
        const double value_as_double = static_cast<double>(value);
        double sine_result = 0.0;
        double cosine_result = 0.0;
        if (abs(value_as_double) <= 0x1.921fb54442d18p-1) {
            sine_result = fallback::sine_kernel({ value_as_double, 0.0 });
            cosine_result = fallback::cosine_kernel({ value_as_double, 0.0 });
        }
        else {
            fallback::double_double reduced = { 0.0, 0.0 };
            const long long int quadrant = fallback::reduce_quarter_turns(value_as_double, reduced) & 3;
            const double reduced_sine = fallback::sine_kernel(reduced);
            const double reduced_cosine = fallback::cosine_kernel(reduced);
            sine_result = ((quadrant & 1) == 0) ? reduced_sine : reduced_cosine;
            cosine_result = ((quadrant & 1) == 0) ? reduced_cosine : reduced_sine;
            if (quadrant >= 2) {
                sine_result = -sine_result;
            }
            if ((quadrant == 1) || (quadrant == 2)) {
                cosine_result = -cosine_result;
            }
        }
        sine = static_cast<type>(sine_result);
        cosine = static_cast<type>(cosine_result);
    }

    template <typename type>
    constexpr static inline type asin(type value) {
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_asinf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_asinf(value);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_asin)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_asin(value);
        }
#endif
        if (value == type(0))
            return value;
        if ((value < type(-1)) || (value > type(1)) || isnan(value))
            return nan<type>();
        // asin(x) = atan(|x| / sqrt((1 - |x|) (1 + |x|))), the root in double-double.
        const double magnitude = abs(static_cast<double>(value));
        const fallback::double_double root = fallback::sqrt_double_double(fallback::multiply(fallback::two_sum(1.0, -magnitude), fallback::two_sum(1.0, magnitude)));
        const fallback::double_double angle = fallback::atan_ratio({ magnitude, 0.0 }, root);
        const double result = angle.high + angle.low;
        return static_cast<type>((value < type(0)) ? -result : result);
    }

    template <typename type>
    constexpr static inline type acos(type value) {
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_acosf)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_acosf(value);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_acos)
        if constexpr (is_same_type<type, double>::value) {
            return __builtin_acos(value);
        }
#endif
        if (value == type(1))
            return type(0);
        if ((value < type(-1)) || (value > type(1)) || isnan(value))
            return nan<type>();
        // acos(x) = atan(sqrt((1 - |x|) (1 + |x|)) / |x|), taken from pi for negative x, the root in double-double.
        constexpr const fallback::double_double half_turn = { 0x1.921fb54442d18p+1, 0x1.1a62633145c07p-53 };
        const double magnitude = abs(static_cast<double>(value));
        const fallback::double_double root = fallback::sqrt_double_double(fallback::multiply(fallback::two_sum(1.0, -magnitude), fallback::two_sum(1.0, magnitude)));
        fallback::double_double angle = fallback::atan_ratio(root, { magnitude, 0.0 });
        if (value < type(0)) {
            angle = fallback::add(half_turn, { -angle.high, -angle.low });
        }
        return static_cast<type>(angle.high + angle.low);
    }

    template <typename type>
    constexpr static inline type atan2(type y, type x) {
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_atan2f)
        if constexpr (is_same_type<type, float>::value) {
            return __builtin_atan2f(y, x);
        }
#endif
#if ZEROSLAM_MATH_HAS_BUILTIN(__builtin_atan2)
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
        // The angle of the absolute values in double-double, taken from pi for negative x.
        constexpr const fallback::double_double half_turn = { 0x1.921fb54442d18p+1, 0x1.1a62633145c07p-53 };
        fallback::double_double angle = fallback::atan_ratio({ static_cast<double>(abs(y)), 0.0 }, { static_cast<double>(abs(x)), 0.0 });
        if (x < 0) {
            angle = fallback::add(half_turn, { -angle.high, -angle.low });
        }
        const double result = angle.high + angle.low;
        return static_cast<type>(signbit(y) ? -result : result);
    }
}

#endif // ZEROSLAM_MATH_MATH_HPP
