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

#include "math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    float test_values_float[] = {
        __builtin_nanf("0"),
        +__builtin_inff(),
        -__builtin_inff(),
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
        __builtin_nan("0"),
        +__builtin_inff64(),
        -__builtin_inff64(),
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
            const long long int lhs = math::round(value);
            const long long int rhs = std::llround(value);
            REQUIRE(lhs == rhs);
        }
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
            REQUIRE(is_value_approx(lhs, rhs));
        }

        for (const float value : test_values_float) {
            const float lhs = math::log(value);
            const float rhs = std::log(value);
            REQUIRE(is_value_approx(lhs, rhs));
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
                REQUIRE(is_value_approx(lhs, rhs));
            }
        }

        for (int i = -1000; i <= 1000; ++i) {
            const float value = static_cast<float>(i) / 10.0f;
            for (int j = -100; j <= 100; ++j) {
                const float value2 = static_cast<float>(j) / 10.0f;
                const float lhs = math::pow(value, value2);
                const float rhs = std::pow(value, value2);
                REQUIRE(is_value_approx(lhs, rhs));
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
            if (std::abs(value) == std::numeric_limits<float>::max())
                continue;
            const float lhs = math::sin(value);
            const float rhs = std::sin(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-12));
        }

        for (int i = -100000; i <= 100000; ++i) {
            const float value = static_cast<float>(i) / 100.0f;
            const float lhs = math::sin(value);
            const float rhs = std::sin(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-12));
        }
    }

    {
        for (const double value : test_values_double) {
            if (std::abs(value) == std::numeric_limits<double>::max())
                continue;
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
            if (std::abs(value) == std::numeric_limits<float>::max())
                continue;
            const float lhs = math::cos(value);
            const float rhs = std::cos(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-12));
        }

        for (int i = -100000; i <= 100000; ++i) {
            const float value = static_cast<float>(i) / 100.0f;
            const float lhs = math::cos(value);
            const float rhs = std::cos(value);
            REQUIRE(is_value_approx(lhs, rhs, 1e-12));
        }
    }

    {
        for (const double value : test_values_double) {
            if (std::abs(value) == std::numeric_limits<double>::max())
                continue;
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
            if (std::abs(value) == std::numeric_limits<float>::max())
                continue;
            float lhs_sin = 0;
            float lhs_cos = 0;
            math::sincos(value, lhs_sin, lhs_cos);
            const float rhs_sin = std::sin(value);
            const float rhs_cos = std::cos(value);
            REQUIRE(is_value_approx(lhs_sin, rhs_sin, 1e-12));
            REQUIRE(is_value_approx(lhs_cos, rhs_cos, 1e-12));
        }

        for (int i = -100000; i <= 100000; ++i) {
            const float value = static_cast<float>(i) / 100.0f;
            float lhs_sin = 0;
            float lhs_cos = 0;
            math::sincos(value, lhs_sin, lhs_cos);
            const float rhs_sin = std::sin(value);
            const float rhs_cos = std::cos(value);
            REQUIRE(is_value_approx(lhs_sin, rhs_sin, 1e-12));
            REQUIRE(is_value_approx(lhs_cos, rhs_cos, 1e-12));
        }
    }

    {
        for (const double value : test_values_double) {
            if (std::abs(value) == std::numeric_limits<double>::max())
                continue;
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
            REQUIRE(is_value_approx(lhs, rhs, 1e-13));
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
            REQUIRE(is_value_approx(lhs, rhs, 1e-13));
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

    return EXIT_SUCCESS;
}
