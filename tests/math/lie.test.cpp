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

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>

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

    return EXIT_SUCCESS;
}
