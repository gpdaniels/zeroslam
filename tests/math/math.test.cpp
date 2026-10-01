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

// The builtins are libm, so the fallback implementations used where there are none (MSVC) are forced here and checked against it.
#define ZEROSLAM_MATH_FALLBACK
#include "math/math.hpp"

#include "core/random_pcg.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-8) {
    if (std::isnan(lhs) && std::isnan(rhs))
        return true;
    if (std::isnan(lhs) != std::isnan(rhs))
        return false;
    if (std::isinf(lhs) != std::isinf(rhs))
        return false;
    if (lhs == 0 && rhs != 0)
        return false;
    if (lhs != 0 && rhs == 0)
        return false;
    if (lhs == 0 && rhs == 0 && std::signbit(lhs) != std::signbit(rhs))
        return false;
    if (std::signbit(lhs + epsilon) != std::signbit(rhs + epsilon))
        return false;
    if (std::isinf(lhs) && std::isinf(rhs))
        return true;
    return (std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon);
}

static inline bool is_value_approx(float lhs, float rhs, double epsilon = 1e-8) {
    return is_value_approx(static_cast<double>(lhs), static_cast<double>(rhs), epsilon);
}

// The distance from a value to a reference in units in the last place of the reference, zero for matching non-finite values.
template <typename type, typename reference_type>
static inline double ulp_distance(const type value, const reference_type reference) {
    const type rounded = static_cast<type>(reference);
    if (std::isnan(rounded) || std::isinf(rounded)) {
        return ((std::isnan(rounded) && std::isnan(value)) || (value == rounded)) ? 0.0 : std::numeric_limits<double>::infinity();
    }
    const type magnitude = std::abs(rounded);
    const type ulp = (magnitude < std::numeric_limits<type>::min()) ? std::numeric_limits<type>::denorm_min() : (std::nextafter(magnitude, std::numeric_limits<type>::infinity()) - magnitude);
    return static_cast<double>(std::abs(static_cast<long double>(value) - static_cast<long double>(reference)) / static_cast<long double>(ulp));
}

// The references in long double only have extra precision on some platforms, elsewhere the checks against them are skipped.
constexpr static const bool long_double_is_extended = std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits;

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    float test_values_float[] = {
        std::numeric_limits<float>::quiet_NaN(),
        +std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        0.0f,
        -0.0f,
        +0.0f,
        1.0f,
        -1.0f,
        0.1f,
        -0.1f,
        1.23456789f,
        -1.23456789f,
        98.7654321f,
        -98.7654321f,
        123.456f,
        456.789f,
        std::numeric_limits<float>::min(),
        -std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max()
    };

    double test_values_double[] = {
        std::numeric_limits<double>::quiet_NaN(),
        +std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        0.0,
        -0.0,
        +0.0,
        1.0,
        -1.0,
        0.1,
        -0.1,
        1.23456789,
        -1.23456789,
        98.7654321,
        -98.7654321,
        123.456,
        456.789,
        std::numeric_limits<double>::min(),
        -std::numeric_limits<double>::min(),
        std::numeric_limits<double>::max(),
        -std::numeric_limits<double>::max()
    };

    {
        REQUIRE(math::pi<float>() == static_cast<float>(M_PI));
        REQUIRE(math::e<float>() == static_cast<float>(M_E));
    }

    {
        REQUIRE(math::pi<double>() == M_PI);
        REQUIRE(math::e<double>() == M_E);
    }

    {
        REQUIRE(math::isnan(math::nan<float>()));
        REQUIRE(math::isinf(math::inf<float>()));
    }

    {
        REQUIRE(math::isnan(math::nan<double>()));
        REQUIRE(math::isinf(math::inf<double>()));
    }

    {
        for (const float value : test_values_float) {
            const bool lhs = math::isnan(value);
            const bool rhs = std::isnan(value);
            REQUIRE(lhs == rhs);
        }
    }

    {
        for (const double value : test_values_double) {
            const bool lhs = math::isnan(value);
            const bool rhs = std::isnan(value);
            REQUIRE(lhs == rhs);
        }
    }

    {
        for (const float value : test_values_float) {
            const bool lhs = math::isinf(value);
            const bool rhs = std::isinf(value);
            REQUIRE(lhs == rhs);
        }
    }

    {
        for (const double value : test_values_double) {
            const bool lhs = math::isinf(value);
            const bool rhs = std::isinf(value);
            REQUIRE(lhs == rhs);
        }
    }

    {
        for (const float value : test_values_float) {
            const bool lhs = math::isfinite(value);
            const bool rhs = std::isfinite(value);
            REQUIRE(lhs == rhs);
        }
    }

    {
        for (const double value : test_values_double) {
            const bool lhs = math::isfinite(value);
            const bool rhs = std::isfinite(value);
            REQUIRE(lhs == rhs);
        }
    }

    {
        for (const float value : test_values_float) {
            for (const float value2 : test_values_float) {
                const float lhs = math::copysign(value, value2);
                const float rhs = std::copysign(value, value2);
                REQUIRE((lhs == rhs) || (std::isnan(lhs) && std::isnan(rhs)));
            }
        }
    }

    {
        for (const double value : test_values_double) {
            for (const double value2 : test_values_double) {
                const double lhs = math::copysign(value, value2);
                const double rhs = std::copysign(value, value2);
                REQUIRE((lhs == rhs) || (std::isnan(lhs) && std::isnan(rhs)));
            }
        }
    }

    {
        for (const float value : test_values_float) {
            const bool lhs = math::signbit(value);
            const bool rhs = std::signbit(value);
            REQUIRE(lhs == rhs);
        }
    }

    {
        for (const double value : test_values_double) {
            const bool lhs = math::signbit(value);
            const bool rhs = std::signbit(value);
            REQUIRE(lhs == rhs);
        }
    }

    {
        for (const float value : test_values_float) {
            const float lhs = math::abs(value);
            const float rhs = std::abs(value);
            REQUIRE((lhs == rhs) || (std::isnan(lhs) && std::isnan(rhs)));
        }
    }

    {
        for (const double value : test_values_double) {
            const double lhs = math::abs(value);
            const double rhs = std::abs(value);
            REQUIRE((lhs == rhs) || (std::isnan(lhs) && std::isnan(rhs)));
        }
    }

    {
        for (const float value : test_values_float) {
            for (const float value2 : test_values_float) {
                const float lhs = math::min(value, value2);
                const float rhs = std::min(value, value2);
                REQUIRE((lhs == rhs) || (std::isnan(lhs) && std::isnan(rhs)));
            }
        }

        for (const float value : test_values_float) {
            for (const float value2 : test_values_float) {
                const float lhs = math::max(value, value2);
                const float rhs = std::max(value, value2);
                REQUIRE((lhs == rhs) || (std::isnan(lhs) && std::isnan(rhs)));
            }
        }
    }

    {
        for (const double value : test_values_double) {
            for (const double value2 : test_values_double) {
                const double lhs = math::min(value, value2);
                const double rhs = std::min(value, value2);
                REQUIRE((lhs == rhs) || (std::isnan(lhs) && std::isnan(rhs)));
            }
        }

        for (const double value : test_values_double) {
            for (const double value2 : test_values_double) {
                const double lhs = math::max(value, value2);
                const double rhs = std::max(value, value2);
                REQUIRE((lhs == rhs) || (std::isnan(lhs) && std::isnan(rhs)));
            }
        }
    }

    {
        for (const float value : test_values_float) {
            const float lhs = math::floor(value);
            const float rhs = std::floor(value);
            REQUIRE((lhs == rhs) || (std::isnan(lhs) && std::isnan(rhs)));
        }

        for (const float value : test_values_float) {
            const float lhs = math::ceil(value);
            const float rhs = std::ceil(value);
            REQUIRE((lhs == rhs) || (std::isnan(lhs) && std::isnan(rhs)));
        }
    }

    {
        for (const double value : test_values_double) {
            const double lhs = math::floor(value);
            const double rhs = std::floor(value);
            REQUIRE((lhs == rhs) || (std::isnan(lhs) && std::isnan(rhs)));
        }

        for (const double value : test_values_double) {
            const double lhs = math::ceil(value);
            const double rhs = std::ceil(value);
            REQUIRE((lhs == rhs) || (std::isnan(lhs) && std::isnan(rhs)));
        }
    }

    {
        for (const float value : test_values_float) {
            if (!std::isfinite(value))
                continue;
            if (std::abs(value) == std::numeric_limits<float>::max())
                continue;
            const int lhs = math::round(value);
            const int rhs = static_cast<int>(std::lround(value));
            REQUIRE(lhs == rhs);
        }
    }

    {
        for (const double value : test_values_double) {
            if (!std::isfinite(value))
                continue;
            if (std::abs(value) == std::numeric_limits<double>::max())
                continue;
            const long long int lhs = math::round(value);
            const long long int rhs = std::llround(value);
            REQUIRE(lhs == rhs);
        }
    }

    // Out of range values saturate and NaN rounds to zero, at runtime and in constant expressions.
    {
        constexpr const int int_max = std::numeric_limits<int>::max();
        constexpr const int int_min = std::numeric_limits<int>::min();
        constexpr const long long int long_max = std::numeric_limits<long long int>::max();
        constexpr const long long int long_min = std::numeric_limits<long long int>::min();
        volatile float float_values[] = { std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), 1e30f, -1e30f, 2147483648.0f, -2147483648.0f, 2147483520.0f, -2147483520.0f, -2147483904.0f, 2.5f, -2.5f, 0.49999997f };
        const int float_expected[] = { 0, 0, int_max, int_min, int_max, int_min, int_max, int_min, int_max, int_min, 2147483520, -2147483520, int_min, 3, -3, 0 };
        for (size_t i = 0; i < sizeof(float_expected) / sizeof(float_expected[0]); ++i) {
            REQUIRE(math::round(float_values[i]) == float_expected[i]);
        }
        volatile double double_values[] = { std::numeric_limits<double>::quiet_NaN(), -std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), 1e300, -1e300, 9223372036854775808.0, -9223372036854775808.0, 9223372036854774784.0, -9223372036854774784.0, -9223372036854777856.0, 2147483647.5, -2.5, 0.49999999999999994 };
        const long long int double_expected[] = { 0, 0, long_max, long_min, long_max, long_min, long_max, long_min, long_max, long_min, 9223372036854774784LL, -9223372036854774784LL, long_min, 2147483648LL, -3, 0 };
        for (size_t i = 0; i < sizeof(double_expected) / sizeof(double_expected[0]); ++i) {
            REQUIRE(math::round(double_values[i]) == double_expected[i]);
        }
        static_assert(math::round(math::nan<float>()) == 0);
        static_assert(math::round(math::inf<float>()) == int_max);
        static_assert(math::round(-math::inf<float>()) == int_min);
        static_assert(math::round(1e30f) == int_max);
        static_assert(math::round(-2147483648.0f) == int_min);
        static_assert(math::round(2147483520.0f) == 2147483520);
        static_assert(math::round(-2.5f) == -3);
        static_assert(math::round(math::nan<double>()) == 0);
        static_assert(math::round(math::inf<double>()) == long_max);
        static_assert(math::round(-math::inf<double>()) == long_min);
        static_assert(math::round(1e300) == long_max);
        static_assert(math::round(-9223372036854775808.0) == long_min);
        static_assert(math::round(9223372036854774784.0) == 9223372036854774784LL);
        static_assert(math::round(2.5) == 3);
    }

    {
        for (const float value : test_values_float) {
            for (const float value2 : test_values_float) {
                const float lhs = math::fmod(value, value2);
                const float rhs = std::fmod(value, value2);
                REQUIRE(is_value_approx(lhs, rhs));
            }
        }
    }

    {
        for (const double value : test_values_double) {
            for (const double value2 : test_values_double) {
                const double lhs = math::fmod(value, value2);
                const double rhs = std::fmod(value, value2);
                REQUIRE(is_value_approx(lhs, rhs));
            }
        }
    }

    {
        for (const float value : test_values_float) {
            const float lhs = math::sqr(value);
            const float rhs = (value * value);
            REQUIRE(is_value_approx(lhs, rhs));
        }

        for (int i = -10; i < 10000; ++i) {
            const float value = static_cast<float>(i) / 10.0f;
            const float lhs = math::sqr(value);
            const float rhs = (value * value);
            REQUIRE(is_value_approx(lhs, rhs));
        }

        for (const float value : test_values_float) {
            const float lhs = math::sqrt(value);
            const float rhs = std::sqrt(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-12));
        }

        for (int i = -10; i <= 10000; ++i) {
            const float value = static_cast<float>(i) / 10.0f;
            const float lhs = math::sqrt(value);
            const float rhs = std::sqrt(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-12));
        }
    }

    {
        for (const double value : test_values_double) {
            const double lhs = math::sqr(value);
            const double rhs = (value * value);
            REQUIRE(is_value_approx(lhs, rhs));
        }

        for (int i = -10; i < 10000; ++i) {
            const double value = static_cast<double>(i) / 10.0;
            const double lhs = math::sqr(value);
            const double rhs = (value * value);
            REQUIRE(is_value_approx(lhs, rhs));
        }

        for (const double value : test_values_double) {
            const double lhs = math::sqrt(value);
            const double rhs = std::sqrt(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-12));
        }

        for (int i = -10; i <= 10000; ++i) {
            const double value = static_cast<double>(i) / 10.0;
            const double lhs = math::sqrt(value);
            const double rhs = std::sqrt(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-12));
        }
    }

    {
        for (const float value : test_values_float) {
            const float lhs = math::exp(value);
            const float rhs = std::exp(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-6));
        }

        for (const float value : test_values_float) {
            const float lhs = math::log(value);
            const float rhs = std::log(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-6));
        }
    }

    {
        for (const double value : test_values_double) {
            const double lhs = math::exp(value);
            const double rhs = std::exp(value);
            REQUIRE(is_value_approx(lhs, rhs));
        }

        for (const double value : test_values_double) {
            const double lhs = math::log(value);
            const double rhs = std::log(value);
            REQUIRE(is_value_approx(lhs, rhs));
        }
    }

    {
        for (const float value : test_values_float) {
            for (const float value2 : test_values_float) {
                const float lhs = math::pow(value, value2);
                const float rhs = std::pow(value, value2);
                REQUIRE(is_value_approx(lhs, rhs, 1e-6));
            }
        }

        for (int i = -1000; i <= 1000; ++i) {
            const float value = static_cast<float>(i) / 10.0f;
            for (int j = -100; j <= 100; ++j) {
                const float value2 = static_cast<float>(j) / 10.0f;
                const float lhs = math::pow(value, value2);
                const float rhs = std::pow(value, value2);
                REQUIRE(is_value_approx(lhs, rhs, 1e-6));
            }
        }
    }

    {
        for (const double value : test_values_double) {
            for (const double value2 : test_values_double) {
                const double lhs = math::pow(value, value2);
                const double rhs = std::pow(value, value2);
                REQUIRE(is_value_approx(lhs, rhs));
            }
        }

        for (int i = -1000; i <= 1000; ++i) {
            const double value = static_cast<double>(i) / 10.0;
            for (int j = -100; j <= 100; ++j) {
                const double value2 = static_cast<double>(j) / 10.0;
                const double lhs = math::pow(value, value2);
                const double rhs = std::pow(value, value2);
                REQUIRE(is_value_approx(lhs, rhs));
            }
        }
    }

    {
        for (const float value : test_values_float) {
            const float lhs = math::sin(value);
            const float rhs = std::sin(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-6));
        }

        for (int i = -100000; i <= 100000; ++i) {
            const float value = static_cast<float>(i) / 100.0f;
            const float lhs = math::sin(value);
            const float rhs = std::sin(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-6));
        }
    }

    {
        for (const double value : test_values_double) {
            const double lhs = math::sin(value);
            const double rhs = std::sin(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-12));
        }

        for (int i = -100000; i <= 100000; ++i) {
            const double value = static_cast<double>(i) / 100.0;
            const double lhs = math::sin(value);
            const double rhs = std::sin(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-12));
        }
    }

    {
        for (const float value : test_values_float) {
            const float lhs = math::cos(value);
            const float rhs = std::cos(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-6));
        }

        for (int i = -100000; i <= 100000; ++i) {
            const float value = static_cast<float>(i) / 100.0f;
            const float lhs = math::cos(value);
            const float rhs = std::cos(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-6));
        }
    }

    {
        for (const double value : test_values_double) {
            const double lhs = math::cos(value);
            const double rhs = std::cos(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-12));
        }

        for (int i = -100000; i <= 100000; ++i) {
            const double value = static_cast<double>(i) / 100.0;
            const double lhs = math::cos(value);
            const double rhs = std::cos(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-12));
        }
    }

    {
        for (const float value : test_values_float) {
            float lhs_sin = 0;
            float lhs_cos = 0;
            math::sincos(value, lhs_sin, lhs_cos);
            const float rhs_sin = std::sin(value);
            const float rhs_cos = std::cos(value);
            REQUIRE(is_value_approx(lhs_sin, rhs_sin, 1e-6));
            REQUIRE(is_value_approx(lhs_cos, rhs_cos, 1e-6));
        }

        for (int i = -100000; i <= 100000; ++i) {
            const float value = static_cast<float>(i) / 100.0f;
            float lhs_sin = 0;
            float lhs_cos = 0;
            math::sincos(value, lhs_sin, lhs_cos);
            const float rhs_sin = std::sin(value);
            const float rhs_cos = std::cos(value);
            REQUIRE(is_value_approx(lhs_sin, rhs_sin, 1e-6));
            REQUIRE(is_value_approx(lhs_cos, rhs_cos, 1e-6));
        }
    }

    {
        for (const double value : test_values_double) {
            double lhs_sin = 0;
            double lhs_cos = 0;
            math::sincos(value, lhs_sin, lhs_cos);
            const double rhs_sin = std::sin(value);
            const double rhs_cos = std::cos(value);
            REQUIRE(is_value_approx(lhs_sin, rhs_sin, 1e-12));
            REQUIRE(is_value_approx(lhs_cos, rhs_cos, 1e-12));
        }

        for (int i = -100000; i <= 100000; ++i) {
            const double value = static_cast<double>(i) / 100.0;
            double lhs_sin = 0;
            double lhs_cos = 0;
            math::sincos(value, lhs_sin, lhs_cos);
            const double rhs_sin = std::sin(value);
            const double rhs_cos = std::cos(value);
            REQUIRE(is_value_approx(lhs_sin, rhs_sin, 1e-12));
            REQUIRE(is_value_approx(lhs_cos, rhs_cos, 1e-12));
        }
    }

    {
        for (const float value : test_values_float) {
            const float lhs = math::asin(value);
            const float rhs = std::asin(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-6));
        }
    }

    {
        for (const double value : test_values_double) {
            const double lhs = math::asin(value);
            const double rhs = std::asin(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-13));
        }
    }

    {
        for (const float value : test_values_float) {
            const float lhs = math::acos(value);
            const float rhs = std::acos(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-6));
        }
    }

    {
        for (const double value : test_values_double) {
            const double lhs = math::acos(value);
            const double rhs = std::acos(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-13));
        }
    }

    {
        for (const float value : test_values_float) {
            for (const float value2 : test_values_float) {
                const float lhs = math::atan2(value, value2);
                const float rhs = std::atan2(value, value2);
                REQUIRE(is_value_approx(lhs, rhs, 1e-5));
            }
        }

        for (int i = -100; i <= 100; ++i) {
            const float value = static_cast<float>(i) / 10.0f;
            for (int j = -100; j <= 100; ++j) {
                const float value2 = static_cast<float>(j) / 10.0f;
                const float lhs = math::atan2(value, value2);
                const float rhs = std::atan2(value, value2);
                REQUIRE(is_value_approx(lhs, rhs, 1e-5));
            }
        }
    }

    {
        for (const double value : test_values_double) {
            for (const double value2 : test_values_double) {
                const double lhs = math::atan2(value, value2);
                const double rhs = std::atan2(value, value2);
                REQUIRE(is_value_approx(lhs, rhs, 1e-5));
            }
        }

        for (int i = -100; i <= 100; ++i) {
            const double value = static_cast<double>(i) / 10.0;
            for (int j = -100; j <= 100; ++j) {
                const double value2 = static_cast<double>(j) / 10.0;
                const double lhs = math::atan2(value, value2);
                const double rhs = std::atan2(value, value2);
                REQUIRE(is_value_approx(lhs, rhs, 1e-5));
            }
        }
    }

    // The fallbacks against libm over wide ranges, in ulps: within two of libm (itself within about one), and where long double is wider
    // within 0.6 of the exact value. The former sin and cos reduced large arguments with a rounded 2 pi, pow lost up to 1460 ulps, and
    // asin, acos and atan2 up to 6.
    {
        core::random_pcg rng;
        const auto uniform = [&rng](const double lower, const double upper) {
            return lower + ((upper - lower) * rng.get_random_exclusive_top());
        };
        const auto log_uniform = [&uniform](const double lower_exponent, const double upper_exponent) {
            return std::ldexp(uniform(1.0, 2.0), static_cast<int>(std::floor(uniform(lower_exponent, upper_exponent))));
        };
        constexpr static const int samples = 20000;
        double worst_libm[13] = {};
        double worst_exact[13] = {};
        // Subnormal results are rounded twice, to double and then to the subnormal grid, so only normal ones are held to the exact bound.
        const auto record = [&worst_libm, &worst_exact](const int index, const double value, const double libm, const long double exact) {
            worst_libm[index] = std::fmax(worst_libm[index], ulp_distance(value, libm));
            if (long_double_is_extended && (std::abs(exact) >= static_cast<long double>(std::numeric_limits<double>::min()))) {
                worst_exact[index] = std::fmax(worst_exact[index], ulp_distance(value, exact));
            }
        };
        for (int sample = 0; sample < samples; ++sample) {
            const double root = log_uniform(-1074.0, 1024.0);
            REQUIRE(math::sqrt(root) == std::sqrt(root));
            const double exponential = uniform(-745.0, 709.7);
            record(0, math::exp(exponential), std::exp(exponential), std::exp(static_cast<long double>(exponential)));
            const double logarithm = log_uniform(-1074.0, 1024.0);
            record(1, math::log(logarithm), std::log(logarithm), std::log(static_cast<long double>(logarithm)));
            const double logarithm_near_one = 1.0 + uniform(-0.5, 0.5) * std::pow(10.0, uniform(-15.0, 0.0));
            record(2, math::log(logarithm_near_one), std::log(logarithm_near_one), std::log(static_cast<long double>(logarithm_near_one)));
            const double base = log_uniform(-30.0, 30.0);
            const double power = uniform(-20.0, 20.0);
            if (std::isfinite(std::pow(base, power)) && (std::pow(base, power) != 0.0)) {
                record(3, math::pow(base, power), std::pow(base, power), std::pow(static_cast<long double>(base), static_cast<long double>(power)));
            }
            const double integer_base = uniform(-10.0, 10.0);
            const double integer_power = std::floor(uniform(-300.0, 300.0));
            if (std::isfinite(std::pow(integer_base, integer_power)) && (std::pow(integer_base, integer_power) != 0.0)) {
                record(4, math::pow(integer_base, integer_power), std::pow(integer_base, integer_power), std::pow(static_cast<long double>(integer_base), static_cast<long double>(integer_power)));
            }
            const double wide_base = log_uniform(-1000.0, 1000.0);
            const double wide_power = uniform(-1.0, 1.0);
            if (std::isfinite(std::pow(wide_base, wide_power)) && (std::pow(wide_base, wide_power) != 0.0)) {
                record(5, math::pow(wide_base, wide_power), std::pow(wide_base, wide_power), std::pow(static_cast<long double>(wide_base), static_cast<long double>(wide_power)));
            }
            const double angle = uniform(-1.0, 1.0) * std::pow(10.0, uniform(-3.0, 6.5));
            record(6, math::sin(angle), std::sin(angle), std::sin(static_cast<long double>(angle)));
            record(7, math::cos(angle), std::cos(angle), std::cos(static_cast<long double>(angle)));
            const double huge_angle = uniform(-1.0, 1.0) * std::pow(10.0, uniform(6.5, 308.0));
            record(8, math::sin(huge_angle), std::sin(huge_angle), std::sin(static_cast<long double>(huge_angle)));
            record(9, math::cos(huge_angle), std::cos(huge_angle), std::cos(static_cast<long double>(huge_angle)));
            double sine = 0.0;
            double cosine = 0.0;
            math::sincos(huge_angle, sine, cosine);
            REQUIRE(sine == math::sin(huge_angle));
            REQUIRE(cosine == math::cos(huge_angle));
            const double ratio = (sample % 2 == 0) ? uniform(-1.0, 1.0) : std::copysign(1.0 - std::pow(10.0, uniform(-16.0, 0.0)), uniform(-1.0, 1.0));
            record(10, math::asin(ratio), std::asin(ratio), std::asin(static_cast<long double>(ratio)));
            record(11, math::acos(ratio), std::acos(ratio), std::acos(static_cast<long double>(ratio)));
            const double ordinate = uniform(-1.0, 1.0) * std::pow(2.0, uniform(-100.0, 100.0));
            const double abscissa = uniform(-1.0, 1.0) * std::pow(2.0, uniform(-100.0, 100.0));
            record(12, math::atan2(ordinate, abscissa), std::atan2(ordinate, abscissa), std::atan2(static_cast<long double>(ordinate), static_cast<long double>(abscissa)));
        }
        for (int index = 0; index < 13; ++index) {
            REQUIRE(worst_libm[index] <= 2.0);
            REQUIRE(worst_exact[index] <= 0.6);
        }

        // The float versions are the double ones rounded once more.
        for (int sample = 0; sample < samples; ++sample) {
            const float root = static_cast<float>(log_uniform(-149.0, 128.0));
            REQUIRE(math::sqrt(root) == std::sqrt(root));
            const float exponential = static_cast<float>(uniform(-103.0, 88.7));
            REQUIRE(ulp_distance(math::exp(exponential), std::exp(exponential)) <= 1.0);
            const float logarithm = static_cast<float>(log_uniform(-149.0, 128.0));
            REQUIRE(ulp_distance(math::log(logarithm), std::log(logarithm)) <= 1.0);
            const float angle = static_cast<float>(uniform(-1.0, 1.0) * std::pow(10.0, uniform(-3.0, 38.0)));
            REQUIRE(ulp_distance(math::sin(angle), std::sin(angle)) <= 1.0);
            REQUIRE(ulp_distance(math::cos(angle), std::cos(angle)) <= 1.0);
            const float base = static_cast<float>(log_uniform(-10.0, 10.0));
            const float power = static_cast<float>(uniform(-8.0, 8.0));
            REQUIRE(ulp_distance(math::pow(base, power), std::pow(base, power)) <= 1.0);
            const float ratio = static_cast<float>(uniform(-1.0, 1.0));
            REQUIRE(ulp_distance(math::asin(ratio), std::asin(ratio)) <= 1.0);
            REQUIRE(ulp_distance(math::acos(ratio), std::acos(ratio)) <= 1.0);
            const float ordinate = static_cast<float>(uniform(-1.0, 1.0) * std::pow(2.0, uniform(-60.0, 60.0)));
            const float abscissa = static_cast<float>(uniform(-1.0, 1.0) * std::pow(2.0, uniform(-60.0, 60.0)));
            REQUIRE(ulp_distance(math::atan2(ordinate, abscissa), std::atan2(ordinate, abscissa)) <= 1.0);
        }

        // Exact integer powers, and ones that only fit after the intermediate powers overflow or underflow.
        REQUIRE(math::pow(3.0, 33.0) == 5559060566555523.0);
        REQUIRE(math::pow(-2.0, 63.0) == -9223372036854775808.0);
        REQUIRE(math::pow(10.0, 22.0) == 1e22);
        REQUIRE(math::pow(0.5, 1074.0) == std::numeric_limits<double>::denorm_min());
        REQUIRE(math::pow(2.0, -1074.0) == std::numeric_limits<double>::denorm_min());
        REQUIRE(math::pow(1e-5, -61.0) == std::pow(1e-5, -61.0));
        REQUIRE(math::pow(10.0, -310.0) == std::pow(10.0, -310.0));
        REQUIRE(math::pow(1.0 + 0x1p-52, 0x1p+52) == std::pow(1.0 + 0x1p-52, 0x1p+52));
        REQUIRE(math::pow(2.0, 1024.0) == std::numeric_limits<double>::infinity());
        REQUIRE(math::pow(-2.0, 1025.0) == -std::numeric_limits<double>::infinity());
        REQUIRE(math::pow(0.9, 1e18) == 0.0);

        // Arguments close to multiples of pi / 2, and the classic huge ones.
        const double hard_angles[] = { 1.5707963267948966, 3.141592653589793, 4.71238898038469, 6.283185307179586, 355.0, 103993.0, 1e22, -1e22, 0x1.921fb54442d18p+19, 0x1.921fb54442d19p+19, 1.7976931348623157e308 };
        for (const double hard_angle : hard_angles) {
            REQUIRE(ulp_distance(math::sin(hard_angle), std::sin(hard_angle)) <= 1.0);
            REQUIRE(ulp_distance(math::cos(hard_angle), std::cos(hard_angle)) <= 1.0);
        }
    }

    // The fallbacks are still usable in constant expressions.
    {
        static_assert(math::sqrt(2.25) == 1.5);
        static_assert(math::sqrt(2.0) == 1.4142135623730951);
        static_assert(math::exp(0.0) == 1.0);
        static_assert(math::exp(1.0) == 2.718281828459045);
        static_assert(math::log(1.0) == 0.0);
        static_assert(math::log(2.0) == 0.6931471805599453);
        static_assert(math::pow(2.0, 10.0) == 1024.0);
        static_assert(math::pow(3.0, 33.0) == 5559060566555523.0);
        static_assert(math::pow(2.0, 0.5) == 1.4142135623730951);
        static_assert(math::sin(1.0) == 0.8414709848078965);
        static_assert(math::cos(1.0) == 0.5403023058681398);
        static_assert(math::sin(1e22) == -0.8522008497671888);
        static_assert(math::cos(1e22) == 0.52321478539513899);
        static_assert(math::atan2(1.0, 1.0) == 0.7853981633974483);
        static_assert(math::asin(0.5) == 0.5235987755982989);
        static_assert(math::acos(-1.0) == 3.141592653589793);
    }

    return EXIT_SUCCESS;
}
