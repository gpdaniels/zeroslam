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

#include "math/lie.hpp"

#include "core/random_pcg.hpp"

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
    if (std::signbit(lhs + epsilon) != std::signbit(rhs + epsilon))
        return false;
    if (std::isinf(lhs) && std::isinf(rhs))
        return true;
    return (std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon);
}

template <typename array_type>
static inline bool are_values_approx(const array_type& lhs, const array_type& rhs, unsigned long long int length, const double& epsilon = 1e-8) {
    for (size_t index = 0; index < length; ++index) {
        if (!is_value_approx(lhs[index], rhs[index], epsilon)) {
            return false;
        }
    }
    return true;
}

// The references are evaluated in long double, which only has extra precision on some platforms.
constexpr static const bool long_double_is_extended = std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits;

// Reference (1 - cos(theta)) / theta^2 and (theta - sin(theta)) / theta^3.
static void reference_rotation_coefficients(const long double theta, long double& first, long double& second) {
    first = 0.5L;
    second = 1.0L / 6.0L;
    if (theta == 0.0L) {
        return;
    }
    const long double half_sinc = std::sin(0.5L * theta) / (0.5L * theta);
    first = 0.5L * half_sinc * half_sinc;
    if (theta >= 1.0L) {
        second = (theta - std::sin(theta)) / (theta * theta * theta);
        return;
    }
    long double term = 1.0L / 6.0L;
    second = 0.0L;
    for (int k = 0; k < 40; ++k) {
        second += term;
        term *= -(theta * theta) / (static_cast<long double>((2 * k) + 4) * static_cast<long double>((2 * k) + 5));
    }
}

// Reference coefficients of the sim3 translation W = c * I + a * omega_hat + b * omega_hat^2, summing the series of phi(z) = (exp(z) - 1) / z
// term by term, with z = sigma + i * theta, z^k = p + i * theta * q and r = (sigma^k - p) / theta^2.
static void reference_translation_coefficients(const long double sigma, const long double theta, long double& a, long double& b, long double& c) {
    long double p = 1.0L;
    long double q = 0.0L;
    long double r = 0.0L;
    long double sigma_power = 1.0L;
    long double factorial = 1.0L;
    a = 0.0L;
    b = 0.0L;
    c = 0.0L;
    for (int k = 0; k < 150; ++k) {
        factorial *= static_cast<long double>(k + 1);
        c += sigma_power / factorial;
        a += q / factorial;
        b += r / factorial;
        const long double next_p = (sigma * p) - (theta * theta * q);
        const long double next_q = p + (sigma * q);
        const long double next_r = (sigma * r) + q;
        p = next_p;
        q = next_q;
        r = next_r;
        sigma_power *= sigma;
    }
}

// The left jacobian as the series sum_n ad^n / (n + 1)! of the adjoint, in long double and with enough terms to converge for angles up to pi.
template <size_t size>
static void reference_left_jacobian(const math::matrix<double, size, 1>& tangent, long double (&result)[size][size]) {
    long double adjoint[size][size] = {};
    const long double omega[3] = { static_cast<long double>(tangent[0]), static_cast<long double>(tangent[1]), static_cast<long double>(tangent[2]) };
    const long double upsilon[3] = { static_cast<long double>(tangent[3]), static_cast<long double>(tangent[4]), static_cast<long double>(tangent[5]) };
    const long double omega_hat[3][3] = { { 0.0L, -omega[2], omega[1] }, { omega[2], 0.0L, -omega[0] }, { -omega[1], omega[0], 0.0L } };
    const long double upsilon_hat[3][3] = { { 0.0L, -upsilon[2], upsilon[1] }, { upsilon[2], 0.0L, -upsilon[0] }, { -upsilon[1], upsilon[0], 0.0L } };
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            adjoint[i][j] = omega_hat[i][j];
            adjoint[i + 3][j] = upsilon_hat[i][j];
            adjoint[i + 3][j + 3] = omega_hat[i][j];
        }
        if constexpr (size == 7) {
            adjoint[i + 3][i + 3] += static_cast<long double>(tangent[6]);
            adjoint[i + 3][6] = -upsilon[i];
        }
    }
    long double power[size][size];
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            result[i][j] = (i == j) ? 1.0L : 0.0L;
            power[i][j] = adjoint[i][j];
        }
    }
    long double factorial = 1.0L;
    for (int n = 1; n < 60; ++n) {
        factorial *= static_cast<long double>(n + 1);
        long double next[size][size];
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                result[i][j] += power[i][j] / factorial;
                next[i][j] = 0.0L;
                for (size_t k = 0; k < size; ++k) {
                    next[i][j] += power[i][k] * adjoint[k][j];
                }
            }
        }
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                power[i][j] = next[i][j];
            }
        }
    }
}

static math::matrix<double, 3, 1> random_unit_axis(core::random_pcg& rng) {
    math::matrix<double, 3, 1> axis;
    double length_squared = 0.0;
    do {
        for (size_t i = 0; i < 3; ++i) {
            axis[i] = (2.0 * rng.get_random_exclusive_top()) - 1.0;
        }
        length_squared = axis.get_length_squared();
    } while ((length_squared < 0.01) || (length_squared > 1.0));
    return axis * (1.0 / std::sqrt(length_squared));
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        math::so3<double> so3;
        static_cast<void>(so3);
        math::se3<double> se3;
        static_cast<void>(se3);
        math::sim3<double> sim3;
        static_cast<void>(sim3);
    }

    {
        math::so3<double> so3{ -1.0, 2.0, -3.0, 4.0 };
        static_cast<void>(so3);
        math::se3<double> se3{ { -1.0, 2.0, -3.0, 4.0 }, { -5.0, 6.0, -7.0 } };
        static_cast<void>(se3);
        math::sim3<double> sim3{ { -1.0, 2.0, -3.0, 4.0 }, { -5.0, 6.0, -7.0 }, 8.0 };
        static_cast<void>(sim3);
    }

    {
        math::so3<double> so3 = { { -1.0, 2.0, -3.0, 4.0 } };
        REQUIRE(is_value_approx(so3.get_quaternion()[0], -1.0, 1e-4));
        REQUIRE(is_value_approx(so3.get_quaternion()[1], 2.0, 1e-4));
        REQUIRE(is_value_approx(so3.get_quaternion()[2], -3.0, 1e-4));
        REQUIRE(is_value_approx(so3.get_quaternion()[3], 4.0, 1e-4));
    }

    {
        math::so3<double> so3 = math::so3<double>::identity();
        math::matrix<double, 3, 3> so3_matrix = so3.get_matrix();
        REQUIRE(are_values_approx(so3_matrix.data(), math::matrix<double, 3, 3>{ { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } } }.data(), 9, 1e-4));
    }

    {
        math::so3<double> so3 = math::so3<double>::identity();
        REQUIRE(is_value_approx(so3.get_quaternion()[0], 1.0, 1e-4));
        REQUIRE(is_value_approx(so3.get_quaternion()[1], 0.0, 1e-4));
        REQUIRE(is_value_approx(so3.get_quaternion()[2], 0.0, 1e-4));
        REQUIRE(is_value_approx(so3.get_quaternion()[3], 0.0, 1e-4));
    }

    {
        math::so3<double> so3;
        so3 = math::so3<double>::rotation(0.0, 0.0, 0.0);
        REQUIRE(are_values_approx(so3.get_quaternion(), { { 1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4));
        so3 = math::so3<double>::rotation(M_PI, 0.0, 0.0);
        REQUIRE(are_values_approx(so3.get_quaternion(), { { 0.0, 1.0, 0.0, 0.0 } }, 4, 1e-4));
        so3 = math::so3<double>::rotation(0.0, M_PI, 0.0);
        REQUIRE(are_values_approx(so3.get_quaternion(), { { 0.0, 0.0, 1.0, 0.0 } }, 4, 1e-4));
        so3 = math::so3<double>::rotation(0.0, 0.0, M_PI);
        REQUIRE(are_values_approx(so3.get_quaternion(), { { 0.0, 0.0, 0.0, 1.0 } }, 4, 1e-4));
    }

    {
        const double quaternions[4][4] = {
            { 0.9, 0.1, 0.1, 0.4 },
            { 0.1, 0.9, 0.1, 0.4 },
            { 0.1, 0.1, 0.9, 0.4 },
            { 0.1, 0.1, 0.4, 0.9 },
        };
        for (const double* quaternion : quaternions) {
            const double length = math::sqrt(math::sqr(quaternion[0]) + math::sqr(quaternion[1]) + math::sqr(quaternion[2]) + math::sqr(quaternion[3]));
            const math::so3<double> so3(quaternion[0] / length, quaternion[1] / length, quaternion[2] / length, quaternion[3] / length);
            const math::so3<double> so3_round_trip(so3.get_matrix());
            const math::matrix<double, 4, 1> quaternion_expected = so3.get_quaternion();
            const math::matrix<double, 4, 1> quaternion_round_trip = so3_round_trip.get_quaternion();
            for (size_t i = 0; i < 4; ++i) {
                REQUIRE(is_value_approx(quaternion_round_trip[i], quaternion_expected[i], 1e-9));
            }
            const math::matrix<double, 3, 3> matrix_expected = so3.get_matrix();
            const math::matrix<double, 3, 3> matrix_round_trip = so3_round_trip.get_matrix();
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    REQUIRE(is_value_approx(matrix_round_trip[row][col], matrix_expected[row][col], 1e-9));
                }
            }
            const math::matrix<double, 3, 1> point = { { 1.0, 2.0, 3.0 } };
            const math::matrix<double, 3, 1> rotated_expected = so3 * point;
            const math::matrix<double, 3, 1> rotated_round_trip = so3_round_trip * point;
            for (size_t i = 0; i < 3; ++i) {
                REQUIRE(is_value_approx(rotated_round_trip[i], rotated_expected[i], 1e-9));
            }
        }
    }

    {
        {
            math::so3<double> so3 = math::so3<double>::identity();
            math::so3<double> so3_inverse = so3.inverse();
            math::so3<double> so3_inverse_expected = math::so3<double>::identity();
            REQUIRE(are_values_approx(so3_inverse.get_quaternion(), so3_inverse_expected.get_quaternion(), 4, 1e-4));
        }

        {
            math::so3<double> so3 = { { 0.0, 1.0, 0.0, 0.0 } };
            math::so3<double> so3_inverse = so3.inverse();
            math::so3<double> so3_inverse_expected = { { 0.0, -1.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(so3_inverse.get_quaternion(), so3_inverse_expected.get_quaternion(), 4, 1e-4));
        }
        {
            math::so3<double> so3 = { { 0.0, 0.0, 1.0, 0.0 } };
            math::so3<double> so3_inverse = so3.inverse();
            math::so3<double> so3_inverse_expected = { { 0.0, 0.0, -1.0, 0.0 } };
            REQUIRE(are_values_approx(so3_inverse.get_quaternion(), so3_inverse_expected.get_quaternion(), 4, 1e-4));
        }
        {
            math::so3<double> so3 = { { 0.0, 0.0, 0.0, 1.0 } };
            math::so3<double> so3_inverse = so3.inverse();
            math::so3<double> so3_inverse_expected = { { 0.0, 0.0, 0.0, -1.0 } };
            REQUIRE(are_values_approx(so3_inverse.get_quaternion(), so3_inverse_expected.get_quaternion(), 4, 1e-4));
        }
        {
            math::so3<double> so3 = { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } };
            math::so3<double> so3_inverse = so3.inverse();
            math::so3<double> so3_inverse_expected = { { std::sqrt(0.1), std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4) } };
            REQUIRE(are_values_approx(so3_inverse.get_quaternion(), so3_inverse_expected.get_quaternion(), 4, 1e-4));
        }
    }

    {
        {
            math::matrix<double, 3, 3> so3_generator = math::so3<double>::generator(0);
            REQUIRE(are_values_approx(so3_generator.data(), math::matrix<double, 3, 3>{ { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, -1.0 }, { 0.0, 1.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            math::matrix<double, 3, 3> so3_generator = math::so3<double>::generator(1);
            REQUIRE(are_values_approx(so3_generator.data(), math::matrix<double, 3, 3>{ { { 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0 }, { -1.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            math::matrix<double, 3, 3> so3_generator = math::so3<double>::generator(2);
            REQUIRE(are_values_approx(so3_generator.data(), math::matrix<double, 3, 3>{ { { 0.0, -1.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
    }

    {
        for (unsigned long long int i = 0; i < 3; ++i) {
            math::matrix<double, 3, 3> so3_generator = math::so3<double>::generator(i);
            math::matrix<double, 3, 1> point = { { 1, 2, 3 } };
            math::matrix<double, 3, 1> delta = math::so3<double>::generator_field(i, point);
            math::matrix<double, 3, 1> expected = so3_generator * point;
            REQUIRE(are_values_approx(delta.data(), expected.data(), 3, 1e-4));
        }
    }

    {
        {
            math::so3<double> so3 = math::so3<double>::identity();
            math::so3<double> so3_explog = math::so3<double>::exp(so3.log());
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_explog.get_quaternion(), 4, 1e-4));
        }
        {
            math::so3<double> so3 = { { 1.0, 0.0, 0.0, 0.0 } };
            math::so3<double> so3_explog = math::so3<double>::exp(so3.log());
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_explog.get_quaternion(), 4, 1e-4));
        }
        {
            math::so3<double> so3 = { { 0.0, 1.0, 0.0, 0.0 } };
            math::so3<double> so3_explog = math::so3<double>::exp(so3.log());
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_explog.get_quaternion(), 4, 1e-4));
        }
        {
            math::so3<double> so3 = { { 0.0, 0.0, 1.0, 0.0 } };
            math::so3<double> so3_explog = math::so3<double>::exp(so3.log());
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_explog.get_quaternion(), 4, 1e-4));
        }
        {
            math::so3<double> so3 = { { 0.0, 0.0, 0.0, 1.0 } };
            math::so3<double> so3_explog = math::so3<double>::exp(so3.log());
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_explog.get_quaternion(), 4, 1e-4));
        }
        {
            math::so3<double> so3 = { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } };
            math::so3<double> so3_explog = math::so3<double>::exp(so3.log());
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_explog.get_quaternion(), 4, 1e-4));
        }
    }

    {
        {
            math::so3<double> so3 = math::so3<double>::identity();
            math::matrix<double, 3, 1> so3_log = so3.log();
            math::matrix<double, 3, 1> so3_log_expected = { { 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(so3_log, so3_log_expected, 3, 1e-4));
        }
        {
            math::so3<double> so3 = { { 1.0, 0.0, 0.0, 0.0 } };
            math::matrix<double, 3, 1> so3_log = so3.log();
            math::matrix<double, 3, 1> so3_log_expected = { { 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(so3_log, so3_log_expected, 3, 1e-4));
        }
        {
            math::so3<double> so3 = { { 0.0, 1.0, 0.0, 0.0 } };
            math::matrix<double, 3, 1> so3_log = so3.log();
            math::matrix<double, 3, 1> so3_log_expected = { { M_PI, 0.0, 0.0 } };
            REQUIRE(are_values_approx(so3_log, so3_log_expected, 3, 1e-4));
        }
        {
            math::so3<double> so3 = { { 0.0, 0.0, 1.0, 0.0 } };
            math::matrix<double, 3, 1> so3_log = so3.log();
            math::matrix<double, 3, 1> so3_log_expected = { { 0.0, M_PI, 0.0 } };
            REQUIRE(are_values_approx(so3_log, so3_log_expected, 3, 1e-4));
        }
        {
            math::so3<double> so3 = { { 0.0, 0.0, 0.0, 1.0 } };
            math::matrix<double, 3, 1> so3_log = so3.log();
            math::matrix<double, 3, 1> so3_log_expected = { { 0.0, 0.0, M_PI } };
            REQUIRE(are_values_approx(so3_log, so3_log_expected, 3, 1e-4));
        }
        {
            math::so3<double> so3 = { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } };
            math::matrix<double, 3, 1> so3_log = so3.log();
            math::matrix<double, 3, 1> so3_log_expected = { { -1.177612, 1.442274, -1.665394 } };
            REQUIRE(are_values_approx(so3_log, so3_log_expected, 3, 1e-4));
        }
        {
            const double theta = 1.999e-6;
            const math::so3<double> so3 = math::so3<double>::exp(math::matrix<double, 3, 1>{ { theta, 0.0, 0.0 } });
            const math::matrix<double, 3, 1> so3_log = so3.log();
            REQUIRE(is_value_approx(so3_log[0], theta, 1e-19));
            REQUIRE(is_value_approx(so3_log[1], 0.0, 1e-19));
            REQUIRE(is_value_approx(so3_log[2], 0.0, 1e-19));
        }
    }

    {
        {
            math::matrix<double, 3, 1> so3_log = { { 0.0, 0.0, 0.0 } };
            math::so3<double> so3 = math::so3<double>::exp(so3_log);
            math::so3<double> so3_expected = math::so3<double>::identity();
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
        }
        {
            math::matrix<double, 3, 1> so3_log = { { 0.0, 0.0, 0.0 } };
            math::so3<double> so3 = math::so3<double>::exp(so3_log);
            math::so3<double> so3_expected = { { 1.0, 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
        }
        {
            math::matrix<double, 3, 1> so3_log = { { M_PI, 0.0, 0.0 } };
            math::so3<double> so3 = math::so3<double>::exp(so3_log);
            math::so3<double> so3_expected = { { 0.0, 1.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
        }
        {
            math::matrix<double, 3, 1> so3_log = { { 0.0, M_PI, 0.0 } };
            math::so3<double> so3 = math::so3<double>::exp(so3_log);
            math::so3<double> so3_expected = { { 0.0, 0.0, 1.0, 0.0 } };
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
        }
        {
            math::matrix<double, 3, 1> so3_log = { { 0.0, 0.0, M_PI } };
            math::so3<double> so3 = math::so3<double>::exp(so3_log);
            math::so3<double> so3_expected = { { 0.0, 0.0, 0.0, 1.0 } };
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
        }
        {
            math::matrix<double, 3, 1> so3_log = { { -1.177612, 1.442274, -1.665394 } };
            math::so3<double> so3 = math::so3<double>::exp(so3_log);
            math::so3<double> so3_expected = { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } };
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
        }
    }

    {
        REQUIRE((math::so3<double>(1, 0, 0, 0) == math::so3<double>::identity()));
        REQUIRE((math::so3<double>(1, 0, 0, 0) != math::so3<double>::identity()) == false);
        REQUIRE((math::so3<double>(0, 0, 0, 1) != math::so3<double>({ 0, 1, 0, 0 })));
        REQUIRE((math::so3<double>(0, 0, 0, 1) == math::so3<double>({ 0, 1, 0, 0 })) == false);

        REQUIRE((math::so3<double>({ 1, 0, 0, 0 }) == math::so3<double>::identity()));
        REQUIRE((math::so3<double>({ 1, 0, 0, 0 }) != math::so3<double>::identity()) == false);
        REQUIRE((math::so3<double>({ 0, 0, 0, 1 }) != math::so3<double>({ 0, 1, 0, 0 })));
        REQUIRE((math::so3<double>({ 0, 0, 0, 1 }) == math::so3<double>({ 0, 1, 0, 0 })) == false);
    }

    {
        {
            math::so3<double> so3_lhs = math::so3<double>::identity();
            math::so3<double> so3_rhs = math::so3<double>::identity();
            math::so3<double> so3 = so3_lhs * so3_rhs;
            math::so3<double> so3_expected = math::so3<double>::identity();
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(so3 * math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(so3 * math::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, -2.0, 3.0 } }, 3, 1e-4));
        }
        {
            math::so3<double> so3_lhs = math::so3<double>::identity();
            math::so3<double> so3_rhs = math::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0);
            math::so3<double> so3 = so3_lhs * so3_rhs;
            math::so3<double> so3_expected = so3_rhs;
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(so3 * math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(so3 * math::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, -3.0, -2.0 } }, 3, 1e-4));
        }
        {
            math::so3<double> so3_lhs = math::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0);
            math::so3<double> so3_rhs = math::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0);
            math::so3<double> so3 = so3_lhs * so3_rhs;
            math::so3<double> so3_expected = math::so3<double>::rotation(M_PI, 0.0, 0.0);
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(so3 * math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(so3 * math::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, 2.0, -3.0 } }, 3, 1e-4));
        }
        {
            math::so3<double> so3_lhs = math::so3<double>::rotation(M_PI, 0.0, 0.0);
            math::so3<double> so3_rhs = math::so3<double>::rotation(M_PI, 0.0, 0.0);
            math::so3<double> so3 = (so3_lhs * so3_rhs);
            math::so3<double> so3_expected = -math::so3<double>::identity();
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(so3 * math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(so3 * math::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, -2.0, 3.0 } }, 3, 1e-4));
        }
        {
            math::so3<double> so3_lhs = math::so3<double>::rotation(M_PI, 0.0, 0.0);
            math::so3<double> so3_rhs = math::so3<double>::rotation(0.0, M_PI, 0.0);
            math::so3<double> so3 = (so3_lhs * so3_rhs);
            math::so3<double> so3_expected = math::so3<double>::rotation(0.0, 0.0, M_PI);
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(so3 * math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(so3 * math::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { -1.0, 2.0, 3.0 } }, 3, 1e-4));
        }
    }

    {
        for (double omega_x = 0.0; omega_x < 0.5 + 0.01; omega_x += 0.5) {
            for (double omega_y = 0.0; omega_y > -0.3 - 0.01; omega_y -= 0.3) {
                for (double omega_z = 0.0; omega_z < 0.2 + 0.01; omega_z += 0.2) {
                    math::matrix<double, 3, 1> omega = { { omega_x, omega_y, omega_z } };
                    math::matrix<double, 3, 3> jacobian = math::so3<double>::left_jacobian(omega);
                    math::matrix<double, 3, 3> jacobian_inverse = math::so3<double>::left_jacobian_inverse(omega);
                    math::matrix<double, 3, 3> identity_product = jacobian * jacobian_inverse;
                    REQUIRE(are_values_approx(identity_product.data(), math::matrix<double, 3, 3>::identity().data(), 9, 1e-4));
                    double epsilon = 1e-7;
                    for (size_t i = 0; i < 3; ++i) {
                        math::matrix<double, 3, 1> omega_plus = omega;
                        omega_plus[i] += epsilon;
                        math::so3<double> exp_omega = math::so3<double>::exp(omega);
                        math::so3<double> exp_omega_plus = math::so3<double>::exp(omega_plus);
                        math::matrix<double, 3, 1> delta_omega = (exp_omega_plus * exp_omega.inverse()).log();
                        for (size_t j = 0; j < 3; ++j) {
                            REQUIRE(is_value_approx(delta_omega[j] / epsilon, jacobian[j][i], 1e-4));
                        }
                    }
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    {
        {
            math::se3<double> se3 = math::se3<double>::identity();
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { -1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4) == false);
            se3.rotation() = { { -1.0, 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { -1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 1.0, 0.0, 0.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, -1.0, 0.0, 0.0 } }, 4, 1e-4) == false);
            se3.rotation() = { { 0.0, -1.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 1.0, 0.0, 0.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, -1.0, 0.0, 0.0 } }, 4, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, 1.0, 0.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, -1.0, 0.0 } }, 4, 1e-4) == false);
            se3.rotation() = { { 0.0, 0.0, -1.0, 0.0 } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, 1.0, 0.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, -1.0, 0.0 } }, 4, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, 0.0, 1.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, 0.0, -1.0 } }, 4, 1e-4) == false);
            se3.rotation() = { { 0.0, 0.0, 0.0, -1.0 } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, 0.0, 1.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, 0.0, -1.0 } }, 4, 1e-4));
        }
        {
            math::se3<double> se3 = { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, 4, 1e-4));
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { -std::sqrt(0.1), std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4) } }, 4, 1e-4) == false);
            se3.rotation() = -se3.rotation();
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { -std::sqrt(0.1), std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4) } }, 4, 1e-4));
        }
    }

    {
        {
            math::se3<double> se3 = math::se3<double>::identity();
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), { { 1.0, 1.0, 1.0 } }, 3, 1e-4) == false);
            se3.translation() = { { 1.0, 1.0, 1.0 } };
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 0.0, 0.0 } }, 3, 1e-4) == false);
            REQUIRE(are_values_approx(se3.translation(), { { 1.0, 1.0, 1.0 } }, 3, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.translation(), { { 1.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), { { -1.0, 0.0, 0.0 } }, 3, 1e-4) == false);
            se3.translation() = { { -1.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(se3.translation(), { { 1.0, 0.0, 0.0 } }, 3, 1e-4) == false);
            REQUIRE(are_values_approx(se3.translation(), { { -1.0, 0.0, 0.0 } }, 3, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 1.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, -1.0, 0.0 } }, 3, 1e-4) == false);
            se3.translation() = { { 0.0, -1.0, 0.0 } };
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 1.0, 0.0 } }, 3, 1e-4) == false);
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, -1.0, 0.0 } }, 3, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } };
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 0.0, 1.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 0.0, -1.0 } }, 3, 1e-4) == false);
            se3.translation() = { { 0.0, 0.0, -1.0 } };
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 0.0, 1.0 } }, 3, 1e-4) == false);
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 0.0, -1.0 } }, 3, 1e-4));
        }
        {
            math::se3<double> se3 = { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } };
            REQUIRE(are_values_approx(se3.translation(), { { 0.5, -0.6, 0.7 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), { { -0.5, 0.6, -0.7 } }, 3, 1e-4) == false);
            se3.translation() = -se3.translation();
            REQUIRE(are_values_approx(se3.translation(), { { 0.5, -0.6, 0.7 } }, 3, 1e-4) == false);
            REQUIRE(are_values_approx(se3.translation(), { { -0.5, 0.6, -0.7 } }, 3, 1e-4));
        }
    }

    {
        math::se3<double> se3 = math::se3<double>::identity();
        REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4));
        REQUIRE(are_values_approx(se3.translation(), { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
    }

    {
        {
            math::se3<double> se3 = math::se3<double>::identity();
            math::se3<double> se3_inverse = se3.inverse();
            math::se3<double> se3_inverse_expected = math::se3<double>::identity();
            REQUIRE(are_values_approx(se3_inverse.rotation().get_quaternion(), se3_inverse_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3_inverse.translation(), se3_inverse_expected.translation(), 3, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } };
            math::se3<double> se3_inverse = se3.inverse();
            math::se3<double> se3_inverse_expected = { { { 0.0, -1.0, 0.0, 0.0 } }, { { -1.0, 0.0, 0.0 } } };
            REQUIRE(are_values_approx(se3_inverse.rotation().get_quaternion(), se3_inverse_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3_inverse.translation(), se3_inverse_expected.translation(), 3, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } };
            math::se3<double> se3_inverse = se3.inverse();
            math::se3<double> se3_inverse_expected = { { { 0.0, 0.0, -1.0, 0.0 } }, { { 0.0, -1.0, 0.0 } } };
            REQUIRE(are_values_approx(se3_inverse.rotation().get_quaternion(), se3_inverse_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3_inverse.translation(), se3_inverse_expected.translation(), 3, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } };
            math::se3<double> se3_inverse = se3.inverse();
            math::se3<double> se3_inverse_expected = { { { 0.0, 0.0, 0.0, -1.0 } }, { { 0.0, 0.0, -1.0 } } };
            REQUIRE(are_values_approx(se3_inverse.rotation().get_quaternion(), se3_inverse_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3_inverse.translation(), se3_inverse_expected.translation(), 3, 1e-4));
        }
        {
            math::se3<double> se3 = { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } };
            math::se3<double> se3_inverse = se3.inverse();
            math::se3<double> se3_inverse_expected = { { { std::sqrt(0.1), std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4) } }, { { -0.487431, 0.607913, -0.702034 } } };
            REQUIRE(are_values_approx(se3_inverse.rotation().get_quaternion(), se3_inverse_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3_inverse.translation(), se3_inverse_expected.translation(), 3, 1e-4));
        }
    }

    {
        {
            math::matrix<double, 4, 4> se3_generator = math::se3<double>::generator(0);
            REQUIRE(are_values_approx(se3_generator.data(), math::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, -1.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            math::matrix<double, 4, 4> se3_generator = math::se3<double>::generator(1);
            REQUIRE(are_values_approx(se3_generator.data(), math::matrix<double, 4, 4>{ { { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { -1.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            math::matrix<double, 4, 4> se3_generator = math::se3<double>::generator(2);
            REQUIRE(are_values_approx(se3_generator.data(), math::matrix<double, 4, 4>{ { { 0.0, -1.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            math::matrix<double, 4, 4> se3_generator = math::se3<double>::generator(3);
            REQUIRE(are_values_approx(se3_generator.data(), math::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            math::matrix<double, 4, 4> se3_generator = math::se3<double>::generator(4);
            REQUIRE(are_values_approx(se3_generator.data(), math::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            math::matrix<double, 4, 4> se3_generator = math::se3<double>::generator(5);
            REQUIRE(are_values_approx(se3_generator.data(), math::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
    }

    {
        for (unsigned long long int i = 0; i < 6; ++i) {
            math::matrix<double, 4, 4> se3_generator = math::se3<double>::generator(i);
            math::matrix<double, 4, 1> point = { { 1, 2, 3, 4 } };
            math::matrix<double, 4, 1> delta = math::se3<double>::generator_field(i, point);
            math::matrix<double, 4, 1> expected = se3_generator * point;
            REQUIRE(are_values_approx(delta.data(), expected.data(), 4, 1e-4));
        }
    }

    {
        {
            math::se3<double> se3 = math::se3<double>::identity();
            math::se3<double> se3_explog = math::se3<double>::exp(se3.log());
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_explog.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_explog.translation(), 3, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 1.0, 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } } };
            math::se3<double> se3_explog = math::se3<double>::exp(se3.log());
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_explog.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_explog.translation(), 3, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } };
            math::se3<double> se3_explog = math::se3<double>::exp(se3.log());
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_explog.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_explog.translation(), 3, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } };
            math::se3<double> se3_explog = math::se3<double>::exp(se3.log());
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_explog.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_explog.translation(), 3, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } };
            math::se3<double> se3_explog = math::se3<double>::exp(se3.log());
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_explog.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_explog.translation(), 3, 1e-4));
        }
        {
            math::se3<double> se3 = { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } };
            math::se3<double> se3_explog = math::se3<double>::exp(se3.log());
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_explog.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_explog.translation(), 3, 1e-4));
        }
    }

    {
        {
            math::se3<double> se3 = math::se3<double>::identity();
            math::matrix<double, 6, 1> se3_log = se3.log();
            math::matrix<double, 6, 1> se3_log_expected = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(se3_log, se3_log_expected, 6, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 1.0, 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } } };
            math::matrix<double, 6, 1> se3_log = se3.log();
            math::matrix<double, 6, 1> se3_log_expected = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(se3_log, se3_log_expected, 6, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } };
            math::matrix<double, 6, 1> se3_log = se3.log();
            math::matrix<double, 6, 1> se3_log_expected = { { M_PI, 0.0, 0.0, 1.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(se3_log, se3_log_expected, 6, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } };
            math::matrix<double, 6, 1> se3_log = se3.log();
            math::matrix<double, 6, 1> se3_log_expected = { { 0.0, M_PI, 0.0, 0.0, 1.0, 0.0 } };
            REQUIRE(are_values_approx(se3_log, se3_log_expected, 6, 1e-4));
        }
        {
            math::se3<double> se3 = { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } };
            math::matrix<double, 6, 1> se3_log = se3.log();
            math::matrix<double, 6, 1> se3_log_expected = { { 0.0, 0.0, M_PI, 0.0, 0.0, 1.0 } };
            REQUIRE(are_values_approx(se3_log, se3_log_expected, 6, 1e-4));
        }
        {
            math::se3<double> se3 = { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } };
            math::matrix<double, 6, 1> se3_log = se3.log();
            math::matrix<double, 6, 1> se3_log_expected = { { -1.177612, 1.442274, -1.665394, 0.491554, -0.599033, 0.706810 } };
            REQUIRE(are_values_approx(se3_log, se3_log_expected, 6, 1e-4));
        }
    }

    {
        {
            math::matrix<double, 6, 1> se3_log = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            math::se3<double> se3 = math::se3<double>::exp(se3_log);
            math::se3<double> se3_expected = math::se3<double>::identity();
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
        }
        {
            math::matrix<double, 6, 1> se3_log = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            math::se3<double> se3 = math::se3<double>::exp(se3_log);
            math::se3<double> se3_expected = { { { 1.0, 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
        }
        {
            math::matrix<double, 6, 1> se3_log = { { M_PI, 0.0, 0.0, 1.0, 0.0, 0.0 } };
            math::se3<double> se3 = math::se3<double>::exp(se3_log);
            math::se3<double> se3_expected = { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
        }
        {
            math::matrix<double, 6, 1> se3_log = { { 0.0, M_PI, 0.0, 0.0, 1.0, 0.0 } };
            math::se3<double> se3 = math::se3<double>::exp(se3_log);
            math::se3<double> se3_expected = { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
        }
        {
            math::matrix<double, 6, 1> se3_log = { { 0.0, 0.0, M_PI, 0.0, 0.0, 1.0 } };
            math::se3<double> se3 = math::se3<double>::exp(se3_log);
            math::se3<double> se3_expected = { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
        }
        {
            math::matrix<double, 6, 1> se3_log = { { -1.177612, 1.442274, -1.665394, 0.491554, -0.599033, 0.706810 } };
            math::se3<double> se3 = math::se3<double>::exp(se3_log);
            math::se3<double> se3_expected = { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
        }
    }

    {
        REQUIRE((math::se3<double>({ { 1, 0, 0, 0 } }, { { 0, 0, 0 } }) == math::se3<double>::identity()));
        REQUIRE((math::se3<double>({ { 1, 0, 0, 0 } }, { { 0, 0, 0 } }) != math::se3<double>::identity()) == false);
        REQUIRE((math::se3<double>({ { 0, 0, 0, 1 } }, { { 0, 0, 0 } }) != math::se3<double>({ { 0, 1, 0, 0 } }, { { 0, 0, 0 } })));
        REQUIRE((math::se3<double>({ { 0, 0, 0, 1 } }, { { 0, 0, 0 } }) == math::se3<double>({ { 0, 1, 0, 0 } }, { { 0, 0, 0 } })) == false);
        REQUIRE((math::se3<double>({ { 0, 0, 1, 0 } }, { { 1, 0, 0 } }) != math::se3<double>({ { 0, 0, 1, 0 } }, { { 0, 1, 0 } })));
        REQUIRE((math::se3<double>({ { 0, 0, 1, 0 } }, { { 1, 0, 0 } }) == math::se3<double>({ { 0, 0, 1, 0 } }, { { 1, 1, 0 } })) == false);
        REQUIRE((math::se3<double>({ { 0, 0, 1, 0 } }, { { 0, 1, -1 } }) == math::se3<double>({ { 0, 0, 1, 0 } }, { { 0, 1, -1 } })));
        REQUIRE((math::se3<double>({ { 0, 0, 1, 0 } }, { { 0, 1, -1 } }) != math::se3<double>({ { 0, 0, 1, 0 } }, { { 0, 1, -1 } })) == false);
    }

    {
        {
            math::se3<double> se3_lhs = math::se3<double>::identity();
            math::se3<double> se3_rhs = math::se3<double>::identity();
            math::se3<double> se3 = se3_lhs * se3_rhs;
            math::se3<double> se3_expected = math::se3<double>::identity();
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
            REQUIRE(are_values_approx(se3 * math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3 * math::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, -2.0, 3.0 } }, 3, 1e-4));
        }
        {
            math::se3<double> se3_lhs = math::se3<double>::identity();
            math::se3<double> se3_rhs = { math::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0), { { 1.0, 0.0, 0.0 } } };
            math::se3<double> se3 = se3_lhs * se3_rhs;
            math::se3<double> se3_expected = se3_rhs;
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
            REQUIRE(are_values_approx(se3 * math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3 * math::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 2.0, -3.0, -2.0 } }, 3, 1e-4));
        }
        {
            math::se3<double> se3_lhs = { math::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0), { { 0.0, 1.0, 0.0 } } };
            math::se3<double> se3_rhs = { math::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0), { { 0.0, 0.0, 1.0 } } };
            math::se3<double> se3 = se3_lhs * se3_rhs;
            math::se3<double> se3_expected = { math::so3<double>::rotation(M_PI, 0.0, 0.0), { { 0.0, 0.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
            REQUIRE(are_values_approx(se3 * math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3 * math::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, 2.0, -3.0 } }, 3, 1e-4));
        }
        {
            math::se3<double> se3_lhs = { math::so3<double>::rotation(M_PI, 0.0, 0.0), { { 1.0, 1.0, 0.0 } } };
            math::se3<double> se3_rhs = { math::so3<double>::rotation(M_PI, 0.0, 0.0), { { 1.0, 0.0, 1.0 } } };
            math::se3<double> se3 = (se3_lhs * se3_rhs);
            math::se3<double> se3_expected = { -math::so3<double>::identity(), { { 2.0, 1.0, -1.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
            REQUIRE(are_values_approx(se3 * math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 2.0, 1.0, -1.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3 * math::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 3.0, -1.0, 2.0 } }, 3, 1e-3));
        }
        {
            math::se3<double> se3_lhs = { math::so3<double>::rotation(M_PI, 0.0, 0.0), { { 1.0, 0.0, -1.0 } } };
            math::se3<double> se3_rhs = { math::so3<double>::rotation(0.0, M_PI, 0.0), { { 1.0, 0.0, -1.0 } } };
            math::se3<double> se3 = (se3_lhs * se3_rhs);
            math::se3<double> se3_expected = { math::so3<double>::rotation(0.0, 0.0, M_PI), { { 2.0, 0.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
            REQUIRE(are_values_approx(se3 * math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 2.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3 * math::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, 2.0, 3.0 } }, 3, 1e-4));
        }
    }

    {
        math::matrix<double, 6, 1> tangent = { { 0.5, -0.3, 0.2, 1.0, 2.0, -1.0 } };
        math::matrix<double, 6, 6> jacobian = math::se3<double>::left_jacobian(tangent);
        math::matrix<double, 6, 6> jacobian_inverse = math::se3<double>::left_jacobian_inverse(tangent);
        math::matrix<double, 6, 6> identity_product = jacobian * jacobian_inverse;
        REQUIRE(are_values_approx(identity_product.data(), math::matrix<double, 6, 6>::identity().data(), 36, 1e-4));
        double epsilon = 1e-7;
        for (size_t i = 0; i < 6; ++i) {
            math::matrix<double, 6, 1> tangent_plus = tangent;
            tangent_plus[i] += epsilon;
            math::se3<double> exp_tangent = math::se3<double>::exp(tangent);
            math::se3<double> exp_tangent_plus = math::se3<double>::exp(tangent_plus);
            math::matrix<double, 6, 1> delta_tangent = (exp_tangent_plus * exp_tangent.inverse()).log();
            for (size_t j = 0; j < 6; ++j) {
                REQUIRE(is_value_approx(delta_tangent[j] / epsilon, jacobian[j][i], 1e-3));
            }
        }
        math::matrix<double, 6, 1> tangent_small = { { 1e-8, 0, 0, 1e-8, 0, 0 } };
        math::matrix<double, 6, 6> jacobian_small = math::se3<double>::left_jacobian(tangent_small);
        math::matrix<double, 6, 6> jacobian_inverse_small = math::se3<double>::left_jacobian_inverse(tangent_small);
        math::matrix<double, 6, 6> identity_product_small = jacobian_small * jacobian_inverse_small;
        REQUIRE(are_values_approx(identity_product_small.data(), math::matrix<double, 6, 6>::identity().data(), 36, 1e-4));
    }

    ///////////////////////////////////////////////////////////////////////////////

    {
        {
            math::sim3<double> sim3 = math::sim3<double>::identity();
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { -1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 1.0, 1.0, 1.0 } }, 3, 1e-4) == false);
            sim3.transformation().rotation() = -sim3.transformation().rotation();
            sim3.transformation().translation() = { { 1.0, 1.0, 1.0 } };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { -1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 0.0, 0.0, 0.0 } }, 3, 1e-4) == false);
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 1.0, 1.0, 1.0 } }, 3, 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 0.0, 1.0, 0.0, 0.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 0.0, -1.0, 0.0, 0.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 1.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { -1.0, 0.0, 0.0 } }, 3, 1e-4) == false);
            sim3.transformation().rotation() = -sim3.transformation().rotation();
            sim3.transformation().translation() = { { -1.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 0.0, 1.0, 0.0, 0.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 0.0, -1.0, 0.0, 0.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 1.0, 0.0, 0.0 } }, 3, 1e-4) == false);
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { -1.0, 0.0, 0.0 } }, 3, 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 0.0, 0.0, 1.0, 0.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 0.0, 0.0, -1.0, 0.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 0.0, 1.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 0.0, -1.0, 0.0 } }, 3, 1e-4) == false);
            sim3.transformation().rotation() = -sim3.transformation().rotation();
            sim3.transformation().translation() = { { 0.0, -1.0, 0.0 } };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 0.0, 0.0, 1.0, 0.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 0.0, 0.0, -1.0, 0.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 0.0, 1.0, 0.0 } }, 3, 1e-4) == false);
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 0.0, -1.0, 0.0 } }, 3, 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 0.0, 0.0, 0.0, 1.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 0.0, 0.0, 0.0, -1.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 0.0, 0.0, 1.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 0.0, 0.0, -1.0 } }, 3, 1e-4) == false);
            sim3.transformation().rotation() = -sim3.transformation().rotation();
            sim3.transformation().translation() = { { 0.0, 0.0, -1.0 } };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 0.0, 0.0, 0.0, 1.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 0.0, 0.0, 0.0, -1.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 0.0, 0.0, 1.0 } }, 3, 1e-4) == false);
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 0.0, 0.0, -1.0 } }, 3, 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { -std::sqrt(0.1), std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4) } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 0.5, -0.6, 0.7 } }, 3, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { -0.5, 0.6, -0.7 } }, 3, 1e-4) == false);
            sim3.transformation().rotation() = -sim3.transformation().rotation();
            sim3.transformation().translation() = -sim3.transformation().translation();
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { -std::sqrt(0.1), std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4) } }, 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { 0.5, -0.6, 0.7 } }, 3, 1e-4) == false);
            REQUIRE(are_values_approx(sim3.transformation().translation(), { { -0.5, 0.6, -0.7 } }, 3, 1e-4));
        }
    }

    {
        {
            math::sim3<double> sim3 = math::sim3<double>::identity();
            REQUIRE(is_value_approx(sim3.scale(), 1.0, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), 0.1, 1e-4) == false);
            sim3.scale() *= 0.1;
            REQUIRE(is_value_approx(sim3.scale(), 1.0, 1e-4) == false);
            REQUIRE(is_value_approx(sim3.scale(), 0.1, 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 2.0 };
            REQUIRE(is_value_approx(sim3.scale(), 2.0, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), 0.5, 1e-4) == false);
            sim3.scale() = 0.5;
            REQUIRE(is_value_approx(sim3.scale(), 2.0, 1e-4) == false);
            REQUIRE(is_value_approx(sim3.scale(), 0.5, 1e-4));
        }
    }

    {
        math::sim3<double> sim3 = math::sim3<double>::identity();
        REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4));
        REQUIRE(are_values_approx(sim3.transformation().translation(), { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
        REQUIRE(is_value_approx(sim3.scale(), 1.0, 1e-4));
    }

    {
        {
            math::sim3<double> sim3 = math::sim3<double>::identity();
            math::sim3<double> sim3_inverse = sim3.inverse();
            math::sim3<double> sim3_inverse_expected = math::sim3<double>::identity();
            REQUIRE(are_values_approx(sim3_inverse.transformation().rotation().get_quaternion(), sim3_inverse_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3_inverse.transformation().translation(), sim3_inverse_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_inverse_expected.scale(), 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } }, 2.0 };
            math::sim3<double> sim3_inverse = sim3.inverse();
            math::sim3<double> sim3_inverse_expected = { { { { 0.0, -1.0, 0.0, 0.0 } }, { { -0.5, 0.0, 0.0 } } }, 0.5 };
            REQUIRE(are_values_approx(sim3_inverse.transformation().rotation().get_quaternion(), sim3_inverse_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3_inverse.transformation().translation(), sim3_inverse_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3_inverse.scale(), sim3_inverse_expected.scale(), 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } }, 0.5 };
            math::sim3<double> sim3_inverse = sim3.inverse();
            math::sim3<double> sim3_inverse_expected = { { { { 0.0, 0.0, -1.0, 0.0 } }, { { 0.0, -2.0, 0.0 } } }, 2.0 };
            REQUIRE(are_values_approx(sim3_inverse.transformation().rotation().get_quaternion(), sim3_inverse_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3_inverse.transformation().translation(), sim3_inverse_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3_inverse.scale(), sim3_inverse_expected.scale(), 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } }, 0.1 };
            math::sim3<double> sim3_inverse = sim3.inverse();
            math::sim3<double> sim3_inverse_expected = { { { { 0.0, 0.0, 0.0, -1.0 } }, { { 0.0, 0.0, -10.0 } } }, 10.0 };
            REQUIRE(are_values_approx(sim3_inverse.transformation().rotation().get_quaternion(), sim3_inverse_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3_inverse.transformation().translation(), sim3_inverse_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3_inverse.scale(), sim3_inverse_expected.scale(), 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 1.0 };
            math::sim3<double> sim3_inverse = sim3.inverse();
            math::sim3<double> sim3_inverse_expected = { { { { std::sqrt(0.1), std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4) } }, { { -0.487431, 0.607913, -0.702034 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3_inverse.transformation().rotation().get_quaternion(), sim3_inverse_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3_inverse.transformation().translation(), sim3_inverse_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3_inverse.scale(), sim3_inverse_expected.scale(), 1e-4));
        }
    }

    {
        {
            math::matrix<double, 4, 4> sim3_generator = math::sim3<double>::generator(0);
            REQUIRE(are_values_approx(sim3_generator.data(), math::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, -1.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            math::matrix<double, 4, 4> sim3_generator = math::sim3<double>::generator(1);
            REQUIRE(are_values_approx(sim3_generator.data(), math::matrix<double, 4, 4>{ { { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { -1.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            math::matrix<double, 4, 4> sim3_generator = math::sim3<double>::generator(2);
            REQUIRE(are_values_approx(sim3_generator.data(), math::matrix<double, 4, 4>{ { { 0.0, -1.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            math::matrix<double, 4, 4> sim3_generator = math::sim3<double>::generator(3);
            REQUIRE(are_values_approx(sim3_generator.data(), math::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            math::matrix<double, 4, 4> sim3_generator = math::sim3<double>::generator(4);
            REQUIRE(are_values_approx(sim3_generator.data(), math::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            math::matrix<double, 4, 4> sim3_generator = math::sim3<double>::generator(5);
            REQUIRE(are_values_approx(sim3_generator.data(), math::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            math::matrix<double, 4, 4> sim3_generator = math::sim3<double>::generator(6);
            REQUIRE(are_values_approx(sim3_generator.data(), math::matrix<double, 4, 4>{ { { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 }, { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
    }

    {
        for (unsigned long long int i = 0; i < 7; ++i) {
            math::matrix<double, 4, 4> sim3_generator = math::sim3<double>::generator(i);
            math::matrix<double, 4, 1> point = { { 1, 2, 3, 4 } };
            math::matrix<double, 4, 1> delta = math::sim3<double>::generator_field(i, point);
            math::matrix<double, 4, 1> expected = sim3_generator * point;
            REQUIRE(are_values_approx(delta.data(), expected.data(), 4, 1e-4));
        }
    }

    {
        {
            math::sim3<double> sim3 = math::sim3<double>::identity();
            math::sim3<double> sim3_explog = math::sim3<double>::exp(sim3.log());
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_explog.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_explog.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_explog.scale(), 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { 1.0, 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } } }, 1.0 };
            math::sim3<double> sim3_explog = math::sim3<double>::exp(sim3.log());
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_explog.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_explog.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_explog.scale(), 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } }, 1.0 };
            math::sim3<double> sim3_explog = math::sim3<double>::exp(sim3.log());
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_explog.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_explog.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_explog.scale(), 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } }, 1.0 };
            math::sim3<double> sim3_explog = math::sim3<double>::exp(sim3.log());
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_explog.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_explog.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_explog.scale(), 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } }, 1.0 };
            math::sim3<double> sim3_explog = math::sim3<double>::exp(sim3.log());
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_explog.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_explog.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_explog.scale(), 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 1.0 };
            math::sim3<double> sim3_explog = math::sim3<double>::exp(sim3.log());
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_explog.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_explog.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_explog.scale(), 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 0.8 };
            math::sim3<double> sim3_explog = math::sim3<double>::exp(sim3.log());
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_explog.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_explog.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_explog.scale(), 1e-4));
        }
    }

    {
        {
            math::sim3<double> sim3 = math::sim3<double>::identity();
            math::matrix<double, 7, 1> sim3_log = sim3.log();
            math::matrix<double, 7, 1> sim3_log_expected = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(sim3_log, sim3_log_expected, 7, 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { 1.0, 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } } }, 1.0 };
            math::matrix<double, 7, 1> sim3_log = sim3.log();
            math::matrix<double, 7, 1> sim3_log_expected = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(sim3_log, sim3_log_expected, 7, 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } }, 1.0 };
            math::matrix<double, 7, 1> sim3_log = sim3.log();
            math::matrix<double, 7, 1> sim3_log_expected = { { M_PI, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(sim3_log, sim3_log_expected, 7, 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } }, 1.0 };
            math::matrix<double, 7, 1> sim3_log = sim3.log();
            math::matrix<double, 7, 1> sim3_log_expected = { { 0.0, M_PI, 0.0, 0.0, 1.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(sim3_log, sim3_log_expected, 7, 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } }, 1.0 };
            math::matrix<double, 7, 1> sim3_log = sim3.log();
            math::matrix<double, 7, 1> sim3_log_expected = { { 0.0, 0.0, M_PI, 0.0, 0.0, 1.0, 0.0 } };
            REQUIRE(are_values_approx(sim3_log, sim3_log_expected, 7, 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 1.0 };
            math::matrix<double, 7, 1> sim3_log = sim3.log();
            math::matrix<double, 7, 1> sim3_log_expected = { { -1.177612, 1.442274, -1.665394, 0.491554, -0.599033, 0.706810, 0.0 } };
            REQUIRE(are_values_approx(sim3_log, sim3_log_expected, 7, 1e-4));
        }
        {
            math::sim3<double> sim3 = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 0.8 };
            math::matrix<double, 7, 1> sim3_log = sim3.log();
            math::matrix<double, 7, 1> sim3_log_expected = { { -1.177611, 1.442274, -1.665394, 0.548948, -0.668049, 0.788500, -0.223144 } };
            REQUIRE(are_values_approx(sim3_log, sim3_log_expected, 7, 1e-4));
        }
        {
            const double theta = 9e-4;
            const double translation_y = 1.0;
            const math::so3<double> rotation = math::so3<double>::exp(math::matrix<double, 3, 1>{ { theta, 0.0, 0.0 } });
            const math::se3<double> transformation = { rotation, { { 0.0, translation_y, 0.0 } } };
            const math::sim3<double> sim3 = { transformation, 1.0 };
            const math::matrix<double, 7, 1> sim3_log = sim3.log();
            const double b = 1.0 / 12.0;
            REQUIRE(is_value_approx(sim3_log[6], 0.0, 1e-12));
            REQUIRE(is_value_approx(sim3_log[4], (1.0 - b * theta * theta) * translation_y, 1e-10));
        }
    }

    {
        {
            math::matrix<double, 7, 1> sim3_log = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            math::sim3<double> sim3 = math::sim3<double>::exp(sim3_log);
            math::sim3<double> sim3_expected = math::sim3<double>::identity();
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
        }
        {
            math::matrix<double, 7, 1> sim3_log = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            math::sim3<double> sim3 = math::sim3<double>::exp(sim3_log);
            math::sim3<double> sim3_expected = { { { { 1.0, 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
        }
        {
            math::matrix<double, 7, 1> sim3_log = { { M_PI, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 } };
            math::sim3<double> sim3 = math::sim3<double>::exp(sim3_log);
            math::sim3<double> sim3_expected = { { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
        }
        {
            math::matrix<double, 7, 1> sim3_log = { { 0.0, M_PI, 0.0, 0.0, 1.0, 0.0, 0.0 } };
            math::sim3<double> sim3 = math::sim3<double>::exp(sim3_log);
            math::sim3<double> sim3_expected = { { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
        }
        {
            math::matrix<double, 7, 1> sim3_log = { { 0.0, 0.0, M_PI, 0.0, 0.0, 1.0, 0.0 } };
            math::sim3<double> sim3 = math::sim3<double>::exp(sim3_log);
            math::sim3<double> sim3_expected = { { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
        }
        {
            math::matrix<double, 7, 1> sim3_log = { { -1.177612, 1.442274, -1.665394, 0.491554, -0.599033, 0.706810, 0.0 } };
            math::sim3<double> sim3 = math::sim3<double>::exp(sim3_log);
            math::sim3<double> sim3_expected = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
        }
        {
            math::matrix<double, 7, 1> sim3_log = { { -1.177611, 1.442274, -1.665394, 0.548948, -0.668049, 0.788500, -0.223144 } };
            math::sim3<double> sim3 = math::sim3<double>::exp(sim3_log);
            math::sim3<double> sim3_expected = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 0.8 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
        }
        {
            const double theta = 9.9e-7;
            const double upsilon_y = 1.0;
            const math::matrix<double, 7, 1> sim3_log = { { theta, 0.0, 0.0, 0.0, upsilon_y, 0.0, 0.0 } };
            const math::sim3<double> sim3 = math::sim3<double>::exp(sim3_log);
            const double b = 1.0 / 6.0;
            const math::matrix<double, 3, 1> expected_translation = { { 0.0, (1.0 - b * theta * theta) * upsilon_y, 0.5 * theta * upsilon_y } };
            REQUIRE(are_values_approx(sim3.transformation().translation(), expected_translation, 3, 1e-14));
        }
    }

    {
        const math::sim3<double> similarity = { { math::so3<double>::rotation(0.3, -0.5, 0.7), { { 0.5, -0.6, 0.7 } } }, 0.8 };
        const math::matrix<double, 7, 7> adjoint = math::sim3<double>::adjoint(similarity);
        const math::matrix<double, 7, 1> tangent = { { 0.02, -0.01, 0.03, -0.04, 0.05, 0.01, -0.02 } };
        const math::matrix<double, 7, 1> conjugated = (similarity * math::sim3<double>::exp(tangent) * similarity.inverse()).log();
        REQUIRE(are_values_approx(adjoint * tangent, conjugated, 7, 1e-9));
    }

    {
        REQUIRE((math::sim3<double>({ { { { 1, 0, 0, 0 } }, { { 0, 0, 0 } } }, 1 }) == math::sim3<double>::identity()));
        REQUIRE((math::sim3<double>({ { { { 1, 0, 0, 0 } }, { { 0, 0, 0 } } }, 1 }) != math::sim3<double>::identity()) == false);
        REQUIRE((math::sim3<double>({ { { { 1, 0, 0, 0 } }, { { 0, 0, 0 } } }, 2 }) != math::sim3<double>::identity()));
        REQUIRE((math::sim3<double>({ { { { 1, 0, 0, 0 } }, { { 0, 0, 0 } } }, 2 }) == math::sim3<double>::identity()) == false);
        REQUIRE((math::sim3<double>({ { { { 0, 0, 0, 1 } }, { { 0, 0, 0 } } }, 1 }) != math::sim3<double>({ { { { 0, 1, 0, 0 } }, { { 0, 0, 0 } } }, 1 })));
        REQUIRE((math::sim3<double>({ { { { 0, 0, 0, 1 } }, { { 0, 0, 0 } } }, 1 }) == math::sim3<double>({ { { { 0, 1, 0, 0 } }, { { 0, 0, 0 } } }, 1 })) == false);
        REQUIRE((math::sim3<double>({ { { { 0, 0, 0, 1 } }, { { 0, 0, 0 } } }, 2 }) != math::sim3<double>({ { { { 0, 1, 0, 0 } }, { { 0, 0, 0 } } }, 2 })));
        REQUIRE((math::sim3<double>({ { { { 0, 0, 0, 1 } }, { { 0, 0, 0 } } }, 2 }) == math::sim3<double>({ { { { 0, 1, 0, 0 } }, { { 0, 0, 0 } } }, 2 })) == false);
        REQUIRE((math::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 1, 0, 0 } } }, -1 }) != math::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 0, 1, 0 } } }, -1 })));
        REQUIRE((math::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 1, 0, 0 } } }, -1 }) == math::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 1, 1, 0 } } }, -1 })) == false);
        REQUIRE((math::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 0, 1, -1 } } }, -1 }) == math::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 0, 1, -1 } } }, -1 })));
        REQUIRE((math::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 0, 1, -1 } } }, -1 }) != math::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 0, 1, -1 } } }, -1 })) == false);
    }

    {
        {
            math::sim3<double> sim3_lhs = math::sim3<double>::identity();
            math::sim3<double> sim3_rhs = math::sim3<double>::identity();
            math::sim3<double> sim3 = sim3_lhs * sim3_rhs;
            math::sim3<double> sim3_expected = math::sim3<double>::identity();
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
            REQUIRE(are_values_approx(sim3 * math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(sim3 * math::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, -2.0, 3.0 } }, 3, 1e-4));
        }
        {
            math::sim3<double> sim3_lhs = math::sim3<double>::identity();
            math::sim3<double> sim3_rhs = { { math::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0), { { 1.0, 0.0, 0.0 } } }, 0.1 };
            math::sim3<double> sim3 = sim3_lhs * sim3_rhs;
            math::sim3<double> sim3_expected = { { math::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0), { { 1.0, 0.0, 0.0 } } }, 0.1 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
            REQUIRE(are_values_approx(sim3 * math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(sim3 * math::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.1, -0.3, -0.2 } }, 3, 1e-4));
        }
        {
            math::sim3<double> sim3_lhs = { { math::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0), { { 0.0, 1.0, 0.0 } } }, 0.5 };
            math::sim3<double> sim3_rhs = { { math::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0), { { 0.0, 0.0, 1.0 } } }, 2.0 };
            math::sim3<double> sim3 = sim3_lhs * sim3_rhs;
            math::sim3<double> sim3_expected = { { math::so3<double>::rotation(M_PI, 0.0, 0.0), { { 0.0, 0.5, 0.0 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
            REQUIRE(are_values_approx(sim3 * math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.5, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(sim3 * math::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, 2.5, -3.0 } }, 3, 1e-4));
        }
        {
            math::sim3<double> sim3_lhs = { { math::so3<double>::rotation(M_PI, 0.0, 0.0), { { 1.0, 1.0, 0.0 } } }, 0.5 };
            math::sim3<double> sim3_rhs = { { math::so3<double>::rotation(M_PI, 0.0, 0.0), { { 1.0, 0.0, 1.0 } } }, 0.5 };
            math::sim3<double> sim3 = (sim3_lhs * sim3_rhs);
            math::sim3<double> sim3_expected = { { -math::so3<double>::identity(), { { 1.5, 1.0, -0.5 } } }, 0.25 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
            REQUIRE(are_values_approx(sim3 * math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 1.5, 1.0, -0.5 } }, 3, 1e-4));
            REQUIRE(are_values_approx(sim3 * math::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.75, 0.5, 0.25 } }, 3, 1e-4));
        }
        {
            math::sim3<double> sim3_lhs = { { math::so3<double>::rotation(M_PI, 0.0, 0.0), { { 1.0, 0.0, -1.0 } } }, 2.0 };
            math::sim3<double> sim3_rhs = { { math::so3<double>::rotation(0.0, M_PI, 0.0), { { 1.0, 0.0, -1.0 } } }, 1.0 };
            math::sim3<double> sim3 = (sim3_lhs * sim3_rhs);
            math::sim3<double> sim3_expected = { { math::so3<double>::rotation(0.0, 0.0, M_PI), { { 3.0, 0.0, 1.0 } } }, 2.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
            REQUIRE(are_values_approx(sim3 * math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 3.0, 0.0, 1.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(sim3 * math::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, 4.0, 7.0 } }, 3, 1e-4));
        }
    }

    {
        for (double sigma = -0.4; sigma < 0.4 + 0.01; sigma += 0.4) {
            math::matrix<double, 7, 1> tangent = { { 0.5, -0.3, 0.2, 1.0, 2.0, -1.0, sigma } };
            math::matrix<double, 7, 7> jacobian = math::sim3<double>::left_jacobian(tangent);
            math::matrix<double, 7, 7> jacobian_inverse = math::sim3<double>::left_jacobian_inverse(tangent);
            math::matrix<double, 7, 7> identity_product = jacobian * jacobian_inverse;
            REQUIRE(are_values_approx(identity_product.data(), math::matrix<double, 7, 7>::identity().data(), 49, 1e-4));
            double epsilon = 1e-7;
            for (size_t i = 0; i < 7; ++i) {
                math::matrix<double, 7, 1> tangent_plus = tangent;
                tangent_plus[i] += epsilon;
                math::sim3<double> exp_tangent = math::sim3<double>::exp(tangent);
                math::sim3<double> exp_tangent_plus = math::sim3<double>::exp(tangent_plus);
                math::matrix<double, 7, 1> delta_tangent = (exp_tangent_plus * exp_tangent.inverse()).log();
                for (size_t j = 0; j < 7; ++j) {
                    REQUIRE(is_value_approx(delta_tangent[j] / epsilon, jacobian[j][i], 1e-3));
                }
            }
        }
        math::matrix<double, 7, 1> tangent_small = { { 1e-8, 0, 0, 1e-8, 0, 0, 1e-8 } };
        math::matrix<double, 7, 7> jacobian_small = math::sim3<double>::left_jacobian(tangent_small);
        math::matrix<double, 7, 7> jacobian_inverse_small = math::sim3<double>::left_jacobian_inverse(tangent_small);
        math::matrix<double, 7, 7> identity_product_small = jacobian_small * jacobian_inverse_small;
        REQUIRE(are_values_approx(identity_product_small.data(), math::matrix<double, 7, 7>::identity().data(), 49, 1e-4));
    }

    {
        REQUIRE(math::so3<double>().is_unit());
        REQUIRE(math::se3<double>().rotation().is_unit());
        REQUIRE(math::sim3<double>().transformation().rotation().is_unit());
        REQUIRE(is_value_approx(math::sim3<double>().scale(), 1.0));
        REQUIRE(math::so3<double>::identity().is_unit());
        REQUIRE(!math::so3<double>(0.0, 0.5, 0.0, 0.0).is_unit());

        math::so3<double> accumulated = math::so3<double>::identity();
        math::se3<double> accumulated_pose = math::se3<double>::identity();
        math::sim3<double> accumulated_similarity = math::sim3<double>::identity();
        for (int step = 1; step <= 500; ++step) {
            const double angle = 0.01 * static_cast<double>(step);
            const math::so3<double> increment = math::so3<double>::exp({ { angle, -0.5 * angle, 0.25 * angle } });
            accumulated = increment * accumulated;
            REQUIRE(accumulated.is_unit());
            REQUIRE(accumulated.inverse().is_unit());
            REQUIRE(math::so3<double>(accumulated.get_matrix()).is_unit());
            accumulated_pose = math::se3<double>::exp({ { angle, 0.0, -angle, 0.1, -0.2, 0.3 } }) * accumulated_pose;
            REQUIRE(accumulated_pose.rotation().is_unit());
            REQUIRE(accumulated_pose.inverse().rotation().is_unit());
            accumulated_similarity = math::sim3<double>::exp({ { angle, 0.0, -angle, 0.1, -0.2, 0.3, 0.001 } }) * accumulated_similarity;
            REQUIRE(accumulated_similarity.transformation().rotation().is_unit());
            REQUIRE(accumulated_similarity.inverse().transformation().rotation().is_unit());
        }

        const math::matrix<double, 4, 1> halved = math::so3<double>(0.0, 0.5, 0.0, 0.0).normalised().get_quaternion();
        REQUIRE(is_value_approx(halved[0], 0.0) && is_value_approx(halved[1], 1.0) && is_value_approx(halved[2], 0.0) && is_value_approx(halved[3], 0.0));

        const math::matrix<double, 4, 1> unit = math::so3<double>::exp({ { 0.3, -0.2, 0.1 } }).get_quaternion();
        const double stretch = 1.0 + 1e-7;
        const math::so3<double> stretched(stretch * unit[0], stretch * unit[1], stretch * unit[2], stretch * unit[3]);
        const math::matrix<double, 4, 1> round_trip = math::so3<double>(stretched.get_matrix() * stretch).get_quaternion();
        REQUIRE(std::abs(round_trip.get_length_squared() - 1.0) < 1e-12);
        REQUIRE(std::abs(stretched.normalised().get_quaternion().get_length_squared() - 1.0) < 1e-12);
    }

    // The so3 left jacobian matches a long double reference at every angle, including just above the former series switch.
    {
        core::random_pcg rng;
        const double angles[] = { 0.0, 1e-9, 1e-7, 1e-6, 1.1e-6, 2e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.99, 1.0, 1.01, 2.0, 3.0, 3.14159 };
        for (const double angle : angles) {
            for (int trial = 0; trial < 8; ++trial) {
                const math::matrix<double, 3, 1> omega = random_unit_axis(rng) * angle;
                const math::matrix<double, 3, 3> jacobian = math::so3<double>::left_jacobian(omega);
                long double first = 0.0L;
                long double second = 0.0L;
                reference_rotation_coefficients(std::sqrt(static_cast<long double>(omega.get_length_squared())), first, second);
                const long double omega_hat[3][3] = { { 0.0L, -static_cast<long double>(omega[2]), static_cast<long double>(omega[1]) },
                                                      { static_cast<long double>(omega[2]), 0.0L, -static_cast<long double>(omega[0]) },
                                                      { -static_cast<long double>(omega[1]), static_cast<long double>(omega[0]), 0.0L } };
                for (size_t i = 0; i < 3; ++i) {
                    for (size_t j = 0; j < 3; ++j) {
                        long double omega_hat_squared = 0.0L;
                        for (size_t k = 0; k < 3; ++k) {
                            omega_hat_squared += omega_hat[i][k] * omega_hat[k][j];
                        }
                        const long double identity = (i == j) ? 1.0L : 0.0L;
                        const long double expected = identity + (first * omega_hat[i][j]) + (second * omega_hat_squared);
                        const long double scale = identity + std::abs(first * omega_hat[i][j]) + std::abs(second * omega_hat_squared);
                        REQUIRE(std::abs(static_cast<long double>(jacobian[i][j]) - expected) <= 2e-15L * scale);
                    }
                }
            }
        }
    }

    // The sim3 exponential matches a long double reference, across the scales and angles of the former series switches and either side of the unit circle.
    {
        core::random_pcg rng;
        const long double tolerance = long_double_is_extended ? 4e-15L : 1e-13L;
        const double magnitudes[] = { 0.0, 1e-9, 1e-7, 1e-6, 1.1e-6, 2e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.6, 0.8, 0.8000001, 0.99, 1.0, 1.01, 2.0, 3.0 };
        const double signs[2] = { 1.0, -1.0 };
        for (const double sigma_magnitude : magnitudes) {
            for (const double sigma_sign : signs) {
                for (const double theta : magnitudes) {
                    const double sigma = sigma_sign * sigma_magnitude;
                    const math::matrix<double, 3, 1> omega = random_unit_axis(rng) * theta;
                    const math::matrix<double, 3, 1> upsilon = random_unit_axis(rng) * (0.5 + rng.get_random_exclusive_top());
                    const math::sim3<double> similarity = math::sim3<double>::exp({ { omega[0], omega[1], omega[2], upsilon[0], upsilon[1], upsilon[2], sigma } });
                    REQUIRE(similarity.scale() == math::exp(sigma));
                    REQUIRE(similarity.transformation().rotation() == math::so3<double>::exp(omega));
                    long double a = 0.0L;
                    long double b = 0.0L;
                    long double c = 0.0L;
                    const long double theta_exact = std::sqrt(static_cast<long double>(omega.get_length_squared()));
                    reference_translation_coefficients(static_cast<long double>(sigma), theta_exact, a, b, c);
                    const long double omega_long[3] = { static_cast<long double>(omega[0]), static_cast<long double>(omega[1]), static_cast<long double>(omega[2]) };
                    const long double upsilon_long[3] = { static_cast<long double>(upsilon[0]), static_cast<long double>(upsilon[1]), static_cast<long double>(upsilon[2]) };
                    const long double cross[3] = { (omega_long[1] * upsilon_long[2]) - (omega_long[2] * upsilon_long[1]),
                                                   (omega_long[2] * upsilon_long[0]) - (omega_long[0] * upsilon_long[2]),
                                                   (omega_long[0] * upsilon_long[1]) - (omega_long[1] * upsilon_long[0]) };
                    const long double double_cross[3] = { (omega_long[1] * cross[2]) - (omega_long[2] * cross[1]),
                                                          (omega_long[2] * cross[0]) - (omega_long[0] * cross[2]),
                                                          (omega_long[0] * cross[1]) - (omega_long[1] * cross[0]) };
                    long double upsilon_scale = 0.0L;
                    for (size_t i = 0; i < 3; ++i) {
                        upsilon_scale = std::fmax(upsilon_scale, std::abs(upsilon_long[i]));
                    }
                    const long double scale = (std::abs(c) + (std::abs(a) * theta_exact) + (std::abs(b) * theta_exact * theta_exact)) * upsilon_scale;
                    for (size_t i = 0; i < 3; ++i) {
                        const long double expected = (c * upsilon_long[i]) + (a * cross[i]) + (b * double_cross[i]);
                        REQUIRE(std::abs(static_cast<long double>(similarity.transformation().translation()[i]) - expected) <= tolerance * scale);
                    }
                }
            }
        }
    }

    // The sim3 logarithm inverts the exponential, including in the band between the former series switches.
    {
        core::random_pcg rng;
        const double magnitudes[] = { 0.0, 1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 3.0 };
        const double signs[2] = { 1.0, -1.0 };
        double worst_upsilon = 0.0;
        for (const double sigma_magnitude : magnitudes) {
            for (const double sigma_sign : signs) {
                for (const double theta : magnitudes) {
                    for (int trial = 0; trial < 4; ++trial) {
                        const double sigma = sigma_sign * sigma_magnitude;
                        const math::matrix<double, 3, 1> omega = random_unit_axis(rng) * theta;
                        const math::matrix<double, 3, 1> upsilon = random_unit_axis(rng) * (0.5 + rng.get_random_exclusive_top());
                        const math::matrix<double, 7, 1> tangent = { { omega[0], omega[1], omega[2], upsilon[0], upsilon[1], upsilon[2], sigma } };
                        const math::matrix<double, 7, 1> round_trip = math::sim3<double>::exp(tangent).log();
                        double upsilon_scale = 0.0;
                        for (size_t i = 0; i < 3; ++i) {
                            upsilon_scale = std::fmax(upsilon_scale, std::abs(upsilon[i]));
                        }
                        for (size_t i = 0; i < 3; ++i) {
                            REQUIRE(std::abs(round_trip[i] - tangent[i]) <= 1e-14 * theta);
                            const double upsilon_error = std::abs(round_trip[i + 3] - tangent[i + 3]) / upsilon_scale;
                            worst_upsilon = std::fmax(worst_upsilon, upsilon_error);
                            REQUIRE(upsilon_error <= 1e-14);
                        }
                        // The scale is stored as exp(sigma), which near one resolves sigma only to its rounding.
                        REQUIRE(std::abs(round_trip[6] - sigma) <= (1e-14 * std::abs(sigma)) + 2.3e-16);
                    }
                }
            }
        }
        REQUIRE(worst_upsilon < 1e-14);

        // And the exponential inverts the logarithm.
        for (int trial = 0; trial < 200; ++trial) {
            const math::so3<double> rotation = math::so3<double>::exp(random_unit_axis(rng) * (3.1 * rng.get_random_exclusive_top()));
            const math::matrix<double, 3, 1> translation = random_unit_axis(rng) * (10.0 * rng.get_random_exclusive_top());
            const double scale = std::exp((4.0 * rng.get_random_exclusive_top()) - 2.0);
            const math::sim3<double> similarity(math::se3<double>(rotation, translation), scale);
            const math::sim3<double> round_trip = math::sim3<double>::exp(similarity.log());
            for (size_t i = 0; i < 3; ++i) {
                REQUIRE(std::abs(round_trip.transformation().translation()[i] - translation[i]) <= 1e-14 * 10.0);
            }
            REQUIRE(std::abs(round_trip.scale() - scale) <= 1e-15 * scale);
        }
    }

    // The closed form se3 and sim3 left jacobians and their inverses match the converged series, from zero angle to near pi and for scales up to e^+-1.
    {
        core::random_pcg rng;
        const double angles[] = { 0.0, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.99, 1.0, 1.01, 1.5, 2.0, 2.5, 3.0, 3.1, 3.14 };
        const double sigmas[] = { 0.0, 1e-9, -1e-9, 1e-6, -1e-6, 1e-3, -1e-3, 0.1, -0.1, 0.5, -0.5, 0.9, -0.9, 1.0, -1.0 };
        for (const double angle : angles) {
            for (int trial = 0; trial < 4; ++trial) {
                const math::matrix<double, 3, 1> omega = random_unit_axis(rng) * angle;
                const math::matrix<double, 3, 1> upsilon = random_unit_axis(rng) * (0.2 + (2.0 * rng.get_random_exclusive_top()));
                const math::matrix<double, 6, 1> pose_tangent = { { omega[0], omega[1], omega[2], upsilon[0], upsilon[1], upsilon[2] } };
                long double pose_reference[6][6];
                reference_left_jacobian(pose_tangent, pose_reference);
                const math::matrix<double, 6, 6> pose_jacobian = math::se3<double>::left_jacobian(pose_tangent);
                const math::matrix<double, 6, 6> pose_jacobian_inverse = math::se3<double>::left_jacobian_inverse(pose_tangent);
                for (size_t i = 0; i < 6; ++i) {
                    for (size_t j = 0; j < 6; ++j) {
                        REQUIRE(std::abs(static_cast<long double>(pose_jacobian[i][j]) - pose_reference[i][j]) <= 1e-13L);
                        long double product = 0.0L;
                        for (size_t k = 0; k < 6; ++k) {
                            product += static_cast<long double>(pose_jacobian_inverse[i][k]) * pose_reference[k][j];
                        }
                        REQUIRE(std::abs(product - ((i == j) ? 1.0L : 0.0L)) <= 1e-13L);
                    }
                }
                for (const double sigma : sigmas) {
                    const math::matrix<double, 7, 1> similarity_tangent = { { omega[0], omega[1], omega[2], upsilon[0], upsilon[1], upsilon[2], sigma } };
                    long double similarity_reference[7][7];
                    reference_left_jacobian(similarity_tangent, similarity_reference);
                    const math::matrix<double, 7, 7> similarity_jacobian = math::sim3<double>::left_jacobian(similarity_tangent);
                    const math::matrix<double, 7, 7> similarity_jacobian_inverse = math::sim3<double>::left_jacobian_inverse(similarity_tangent);
                    for (size_t i = 0; i < 7; ++i) {
                        for (size_t j = 0; j < 7; ++j) {
                            REQUIRE(std::abs(static_cast<long double>(similarity_jacobian[i][j]) - similarity_reference[i][j]) <= 1e-13L);
                            long double product = 0.0L;
                            for (size_t k = 0; k < 7; ++k) {
                                product += static_cast<long double>(similarity_jacobian_inverse[i][k]) * similarity_reference[k][j];
                            }
                            REQUIRE(std::abs(product - ((i == j) ? 1.0L : 0.0L)) <= 1e-13L);
                        }
                    }
                    // Without a change of scale the similarity jacobian is the pose one.
                    if (sigma == 0.0) {
                        for (size_t i = 0; i < 6; ++i) {
                            for (size_t j = 0; j < 6; ++j) {
                                REQUIRE(std::abs(similarity_jacobian[i][j] - pose_jacobian[i][j]) <= 1e-14);
                            }
                        }
                    }
                }
            }
        }
    }

    return EXIT_SUCCESS;
}
