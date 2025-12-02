/*
Copyright (C) 2025 Geoffrey Daniels. https://gpdaniels.com/

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

#include "lie.hpp"

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
        lie::so3<double> so3;
        static_cast<void>(so3);
        lie::se3<double> se3;
        static_cast<void>(se3);
        lie::sim3<double> sim3;
        static_cast<void>(sim3);
    }

    {
        lie::so3<double> so3{ -1.0, 2.0, -3.0, 4.0 };
        static_cast<void>(so3);
        lie::se3<double> se3{ { -1.0, 2.0, -3.0, 4.0 }, { -5.0, 6.0, -7.0 } };
        static_cast<void>(se3);
        lie::sim3<double> sim3{ { -1.0, 2.0, -3.0, 4.0 }, { -5.0, 6.0, -7.0 }, 8.0 };
        static_cast<void>(sim3);
    }

    {
        lie::so3<double> so3 = { { -1.0, 2.0, -3.0, 4.0 } };
        REQUIRE(is_value_approx(so3.get_quaternion()[0], -1.0, 1e-4));
        REQUIRE(is_value_approx(so3.get_quaternion()[1], 2.0, 1e-4));
        REQUIRE(is_value_approx(so3.get_quaternion()[2], -3.0, 1e-4));
        REQUIRE(is_value_approx(so3.get_quaternion()[3], 4.0, 1e-4));
    }

    {
        lie::so3<double> so3 = lie::so3<double>::identity();
        matrix::matrix<double, 3, 3> so3_matrix = so3.get_matrix();
        REQUIRE(are_values_approx(so3_matrix.data(), matrix::matrix<double, 3, 3>{ { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } } }.data(), 9, 1e-4));
    }

    {
        lie::so3<double> so3 = lie::so3<double>::identity();
        REQUIRE(is_value_approx(so3.get_quaternion()[0], 1.0, 1e-4));
        REQUIRE(is_value_approx(so3.get_quaternion()[1], 0.0, 1e-4));
        REQUIRE(is_value_approx(so3.get_quaternion()[2], 0.0, 1e-4));
        REQUIRE(is_value_approx(so3.get_quaternion()[3], 0.0, 1e-4));
    }

    {
        lie::so3<double> so3;
        so3 = lie::so3<double>::rotation(0.0, 0.0, 0.0);
        REQUIRE(are_values_approx(so3.get_quaternion(), { { 1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4));
        so3 = lie::so3<double>::rotation(M_PI, 0.0, 0.0);
        REQUIRE(are_values_approx(so3.get_quaternion(), { { 0.0, 1.0, 0.0, 0.0 } }, 4, 1e-4));
        so3 = lie::so3<double>::rotation(0.0, M_PI, 0.0);
        REQUIRE(are_values_approx(so3.get_quaternion(), { { 0.0, 0.0, 1.0, 0.0 } }, 4, 1e-4));
        so3 = lie::so3<double>::rotation(0.0, 0.0, M_PI);
        REQUIRE(are_values_approx(so3.get_quaternion(), { { 0.0, 0.0, 0.0, 1.0 } }, 4, 1e-4));
    }

    {
        {
            lie::so3<double> so3 = lie::so3<double>::identity();
            lie::so3<double> so3_inverse = so3.inverse();
            lie::so3<double> so3_inverse_expected = lie::so3<double>::identity();
            REQUIRE(are_values_approx(so3_inverse.get_quaternion(), so3_inverse_expected.get_quaternion(), 4, 1e-4));
        }

        {
            lie::so3<double> so3 = { { 0.0, 1.0, 0.0, 0.0 } };
            lie::so3<double> so3_inverse = so3.inverse();
            lie::so3<double> so3_inverse_expected = { { 0.0, -1.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(so3_inverse.get_quaternion(), so3_inverse_expected.get_quaternion(), 4, 1e-4));
        }
        {
            lie::so3<double> so3 = { { 0.0, 0.0, 1.0, 0.0 } };
            lie::so3<double> so3_inverse = so3.inverse();
            lie::so3<double> so3_inverse_expected = { { 0.0, 0.0, -1.0, 0.0 } };
            REQUIRE(are_values_approx(so3_inverse.get_quaternion(), so3_inverse_expected.get_quaternion(), 4, 1e-4));
        }
        {
            lie::so3<double> so3 = { { 0.0, 0.0, 0.0, 1.0 } };
            lie::so3<double> so3_inverse = so3.inverse();
            lie::so3<double> so3_inverse_expected = { { 0.0, 0.0, 0.0, -1.0 } };
            REQUIRE(are_values_approx(so3_inverse.get_quaternion(), so3_inverse_expected.get_quaternion(), 4, 1e-4));
        }
        {
            lie::so3<double> so3 = { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } };
            lie::so3<double> so3_inverse = so3.inverse();
            lie::so3<double> so3_inverse_expected = { { std::sqrt(0.1), std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4) } };
            REQUIRE(are_values_approx(so3_inverse.get_quaternion(), so3_inverse_expected.get_quaternion(), 4, 1e-4));
        }
    }

    {
        {
            matrix::matrix<double, 3, 3> so3_generator = lie::so3<double>::generator(0);
            REQUIRE(are_values_approx(so3_generator.data(), matrix::matrix<double, 3, 3>{ { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, -1.0 }, { 0.0, 1.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            matrix::matrix<double, 3, 3> so3_generator = lie::so3<double>::generator(1);
            REQUIRE(are_values_approx(so3_generator.data(), matrix::matrix<double, 3, 3>{ { { 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0 }, { -1.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            matrix::matrix<double, 3, 3> so3_generator = lie::so3<double>::generator(2);
            REQUIRE(are_values_approx(so3_generator.data(), matrix::matrix<double, 3, 3>{ { { 0.0, -1.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
    }

    {
        for (unsigned long long int i = 0; i < 3; ++i) {
            matrix::matrix<double, 3, 3> so3_generator = lie::so3<double>::generator(i);
            matrix::matrix<double, 3, 1> point = { { 1, 2, 3 } };
            matrix::matrix<double, 3, 1> delta = lie::so3<double>::generator_field(i, point);
            matrix::matrix<double, 3, 1> expected = so3_generator * point;
            REQUIRE(are_values_approx(delta.data(), expected.data(), 3, 1e-4));
        }
    }

    {
        {
            lie::so3<double> so3 = lie::so3<double>::identity();
            lie::so3<double> so3_explog = lie::so3<double>::exp(so3.log());
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_explog.get_quaternion(), 4, 1e-4));
        }
        {
            lie::so3<double> so3 = { { 1.0, 0.0, 0.0, 0.0 } };
            lie::so3<double> so3_explog = lie::so3<double>::exp(so3.log());
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_explog.get_quaternion(), 4, 1e-4));
        }
        {
            lie::so3<double> so3 = { { 0.0, 1.0, 0.0, 0.0 } };
            lie::so3<double> so3_explog = lie::so3<double>::exp(so3.log());
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_explog.get_quaternion(), 4, 1e-4));
        }
        {
            lie::so3<double> so3 = { { 0.0, 0.0, 1.0, 0.0 } };
            lie::so3<double> so3_explog = lie::so3<double>::exp(so3.log());
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_explog.get_quaternion(), 4, 1e-4));
        }
        {
            lie::so3<double> so3 = { { 0.0, 0.0, 0.0, 1.0 } };
            lie::so3<double> so3_explog = lie::so3<double>::exp(so3.log());
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_explog.get_quaternion(), 4, 1e-4));
        }
        {
            lie::so3<double> so3 = { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } };
            lie::so3<double> so3_explog = lie::so3<double>::exp(so3.log());
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_explog.get_quaternion(), 4, 1e-4));
        }
    }

    {
        {
            lie::so3<double> so3 = lie::so3<double>::identity();
            matrix::matrix<double, 3, 1> so3_log = so3.log();
            matrix::matrix<double, 3, 1> so3_log_expected = { { 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(so3_log, so3_log_expected, 3, 1e-4));
        }
        {
            lie::so3<double> so3 = { { 1.0, 0.0, 0.0, 0.0 } };
            matrix::matrix<double, 3, 1> so3_log = so3.log();
            matrix::matrix<double, 3, 1> so3_log_expected = { { 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(so3_log, so3_log_expected, 3, 1e-4));
        }
        {
            lie::so3<double> so3 = { { 0.0, 1.0, 0.0, 0.0 } };
            matrix::matrix<double, 3, 1> so3_log = so3.log();
            matrix::matrix<double, 3, 1> so3_log_expected = { { M_PI, 0.0, 0.0 } };
            REQUIRE(are_values_approx(so3_log, so3_log_expected, 3, 1e-4));
        }
        {
            lie::so3<double> so3 = { { 0.0, 0.0, 1.0, 0.0 } };
            matrix::matrix<double, 3, 1> so3_log = so3.log();
            matrix::matrix<double, 3, 1> so3_log_expected = { { 0.0, M_PI, 0.0 } };
            REQUIRE(are_values_approx(so3_log, so3_log_expected, 3, 1e-4));
        }
        {
            lie::so3<double> so3 = { { 0.0, 0.0, 0.0, 1.0 } };
            matrix::matrix<double, 3, 1> so3_log = so3.log();
            matrix::matrix<double, 3, 1> so3_log_expected = { { 0.0, 0.0, M_PI } };
            REQUIRE(are_values_approx(so3_log, so3_log_expected, 3, 1e-4));
        }
        {
            lie::so3<double> so3 = { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } };
            matrix::matrix<double, 3, 1> so3_log = so3.log();
            matrix::matrix<double, 3, 1> so3_log_expected = { { -1.177612, 1.442274, -1.665394 } };
            REQUIRE(are_values_approx(so3_log, so3_log_expected, 3, 1e-4));
        }
    }

    {
        {
            matrix::matrix<double, 3, 1> so3_log = { { 0.0, 0.0, 0.0 } };
            lie::so3<double> so3 = lie::so3<double>::exp(so3_log);
            lie::so3<double> so3_expected = lie::so3<double>::identity();
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
        }
        {
            matrix::matrix<double, 3, 1> so3_log = { { 0.0, 0.0, 0.0 } };
            lie::so3<double> so3 = lie::so3<double>::exp(so3_log);
            lie::so3<double> so3_expected = { { 1.0, 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
        }
        {
            matrix::matrix<double, 3, 1> so3_log = { { M_PI, 0.0, 0.0 } };
            lie::so3<double> so3 = lie::so3<double>::exp(so3_log);
            lie::so3<double> so3_expected = { { 0.0, 1.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
        }
        {
            matrix::matrix<double, 3, 1> so3_log = { { 0.0, M_PI, 0.0 } };
            lie::so3<double> so3 = lie::so3<double>::exp(so3_log);
            lie::so3<double> so3_expected = { { 0.0, 0.0, 1.0, 0.0 } };
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
        }
        {
            matrix::matrix<double, 3, 1> so3_log = { { 0.0, 0.0, M_PI } };
            lie::so3<double> so3 = lie::so3<double>::exp(so3_log);
            lie::so3<double> so3_expected = { { 0.0, 0.0, 0.0, 1.0 } };
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
        }
        {
            matrix::matrix<double, 3, 1> so3_log = { { -1.177612, 1.442274, -1.665394 } };
            lie::so3<double> so3 = lie::so3<double>::exp(so3_log);
            lie::so3<double> so3_expected = { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } };
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
        }
    }

    {
        REQUIRE((lie::so3<double>(1, 0, 0, 0) == lie::so3<double>::identity()));
        REQUIRE((lie::so3<double>(1, 0, 0, 0) != lie::so3<double>::identity()) == false);
        REQUIRE((lie::so3<double>(0, 0, 0, 1) != lie::so3<double>({ 0, 1, 0, 0 })));
        REQUIRE((lie::so3<double>(0, 0, 0, 1) == lie::so3<double>({ 0, 1, 0, 0 })) == false);

        REQUIRE((lie::so3<double>({ 1, 0, 0, 0 }) == lie::so3<double>::identity()));
        REQUIRE((lie::so3<double>({ 1, 0, 0, 0 }) != lie::so3<double>::identity()) == false);
        REQUIRE((lie::so3<double>({ 0, 0, 0, 1 }) != lie::so3<double>({ 0, 1, 0, 0 })));
        REQUIRE((lie::so3<double>({ 0, 0, 0, 1 }) == lie::so3<double>({ 0, 1, 0, 0 })) == false);
    }

    {
        {
            lie::so3<double> so3_lhs = lie::so3<double>::identity();
            lie::so3<double> so3_rhs = lie::so3<double>::identity();
            lie::so3<double> so3 = so3_lhs * so3_rhs;
            lie::so3<double> so3_expected = lie::so3<double>::identity();
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(so3 * matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(so3 * matrix::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, -2.0, 3.0 } }, 3, 1e-4));
        }
        {
            lie::so3<double> so3_lhs = lie::so3<double>::identity();
            lie::so3<double> so3_rhs = lie::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0);
            lie::so3<double> so3 = so3_lhs * so3_rhs;
            lie::so3<double> so3_expected = so3_rhs;
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(so3 * matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(so3 * matrix::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, -3.0, -2.0 } }, 3, 1e-4));
        }
        {
            lie::so3<double> so3_lhs = lie::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0);
            lie::so3<double> so3_rhs = lie::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0);
            lie::so3<double> so3 = so3_lhs * so3_rhs;
            lie::so3<double> so3_expected = lie::so3<double>::rotation(M_PI, 0.0, 0.0);
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(so3 * matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(so3 * matrix::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, 2.0, -3.0 } }, 3, 1e-4));
        }
        {
            lie::so3<double> so3_lhs = lie::so3<double>::rotation(M_PI, 0.0, 0.0);
            lie::so3<double> so3_rhs = lie::so3<double>::rotation(M_PI, 0.0, 0.0);
            lie::so3<double> so3 = (so3_lhs * so3_rhs);
            lie::so3<double> so3_expected = -lie::so3<double>::identity();
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(so3 * matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(so3 * matrix::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, -2.0, 3.0 } }, 3, 1e-4));
        }
        {
            lie::so3<double> so3_lhs = lie::so3<double>::rotation(M_PI, 0.0, 0.0);
            lie::so3<double> so3_rhs = lie::so3<double>::rotation(0.0, M_PI, 0.0);
            lie::so3<double> so3 = (so3_lhs * so3_rhs);
            lie::so3<double> so3_expected = lie::so3<double>::rotation(0.0, 0.0, M_PI);
            REQUIRE(are_values_approx(so3.get_quaternion(), so3_expected.get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(so3 * matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(so3 * matrix::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { -1.0, 2.0, 3.0 } }, 3, 1e-4));
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    {
        {
            lie::se3<double> se3 = lie::se3<double>::identity();
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { -1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4) == false);
            se3.rotation() = { { -1.0, 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { -1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 1.0, 0.0, 0.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, -1.0, 0.0, 0.0 } }, 4, 1e-4) == false);
            se3.rotation() = { { 0.0, -1.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 1.0, 0.0, 0.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, -1.0, 0.0, 0.0 } }, 4, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, 1.0, 0.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, -1.0, 0.0 } }, 4, 1e-4) == false);
            se3.rotation() = { { 0.0, 0.0, -1.0, 0.0 } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, 1.0, 0.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, -1.0, 0.0 } }, 4, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, 0.0, 1.0 } }, 4, 1e-4));
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, 0.0, -1.0 } }, 4, 1e-4) == false);
            se3.rotation() = { { 0.0, 0.0, 0.0, -1.0 } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, 0.0, 1.0 } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 0.0, 0.0, 0.0, -1.0 } }, 4, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, 4, 1e-4));
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { -std::sqrt(0.1), std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4) } }, 4, 1e-4) == false);
            se3.rotation() = -se3.rotation();
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, 4, 1e-4) == false);
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { -std::sqrt(0.1), std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4) } }, 4, 1e-4));
        }
    }

    {
        {
            lie::se3<double> se3 = lie::se3<double>::identity();
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), { { 1.0, 1.0, 1.0 } }, 3, 1e-4) == false);
            se3.translation() = { { 1.0, 1.0, 1.0 } };
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 0.0, 0.0 } }, 3, 1e-4) == false);
            REQUIRE(are_values_approx(se3.translation(), { { 1.0, 1.0, 1.0 } }, 3, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.translation(), { { 1.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), { { -1.0, 0.0, 0.0 } }, 3, 1e-4) == false);
            se3.translation() = { { -1.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(se3.translation(), { { 1.0, 0.0, 0.0 } }, 3, 1e-4) == false);
            REQUIRE(are_values_approx(se3.translation(), { { -1.0, 0.0, 0.0 } }, 3, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 1.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, -1.0, 0.0 } }, 3, 1e-4) == false);
            se3.translation() = { { 0.0, -1.0, 0.0 } };
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 1.0, 0.0 } }, 3, 1e-4) == false);
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, -1.0, 0.0 } }, 3, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } };
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 0.0, 1.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 0.0, -1.0 } }, 3, 1e-4) == false);
            se3.translation() = { { 0.0, 0.0, -1.0 } };
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 0.0, 1.0 } }, 3, 1e-4) == false);
            REQUIRE(are_values_approx(se3.translation(), { { 0.0, 0.0, -1.0 } }, 3, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } };
            REQUIRE(are_values_approx(se3.translation(), { { 0.5, -0.6, 0.7 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), { { -0.5, 0.6, -0.7 } }, 3, 1e-4) == false);
            se3.translation() = -se3.translation();
            REQUIRE(are_values_approx(se3.translation(), { { 0.5, -0.6, 0.7 } }, 3, 1e-4) == false);
            REQUIRE(are_values_approx(se3.translation(), { { -0.5, 0.6, -0.7 } }, 3, 1e-4));
        }
    }

    {
        lie::se3<double> se3 = lie::se3<double>::identity();
        REQUIRE(are_values_approx(se3.rotation().get_quaternion(), { { 1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4));
        REQUIRE(are_values_approx(se3.translation(), { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
    }

    {
        {
            lie::se3<double> se3 = lie::se3<double>::identity();
            lie::se3<double> se3_inverse = se3.inverse();
            lie::se3<double> se3_inverse_expected = lie::se3<double>::identity();
            REQUIRE(are_values_approx(se3_inverse.rotation().get_quaternion(), se3_inverse_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3_inverse.translation(), se3_inverse_expected.translation(), 3, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } };
            lie::se3<double> se3_inverse = se3.inverse();
            lie::se3<double> se3_inverse_expected = { { { 0.0, -1.0, 0.0, 0.0 } }, { { -1.0, 0.0, 0.0 } } };
            REQUIRE(are_values_approx(se3_inverse.rotation().get_quaternion(), se3_inverse_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3_inverse.translation(), se3_inverse_expected.translation(), 3, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } };
            lie::se3<double> se3_inverse = se3.inverse();
            lie::se3<double> se3_inverse_expected = { { { 0.0, .0f, -1.0, 0.0 } }, { { 0.0, -1.0, 0.0 } } };
            REQUIRE(are_values_approx(se3_inverse.rotation().get_quaternion(), se3_inverse_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3_inverse.translation(), se3_inverse_expected.translation(), 3, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } };
            lie::se3<double> se3_inverse = se3.inverse();
            lie::se3<double> se3_inverse_expected = { { { 0.0, 0.0, 0.0, -1.0 } }, { { 0.0, 0.0, -1.0 } } };
            REQUIRE(are_values_approx(se3_inverse.rotation().get_quaternion(), se3_inverse_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3_inverse.translation(), se3_inverse_expected.translation(), 3, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } };
            lie::se3<double> se3_inverse = se3.inverse();
            lie::se3<double> se3_inverse_expected = { { { std::sqrt(0.1), std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4) } }, { { -0.487431, 0.607913, -0.702034 } } };
            REQUIRE(are_values_approx(se3_inverse.rotation().get_quaternion(), se3_inverse_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3_inverse.translation(), se3_inverse_expected.translation(), 3, 1e-4));
        }
    }

    {
        {
            matrix::matrix<double, 4, 4> se3_generator = lie::se3<double>::generator(0);
            REQUIRE(are_values_approx(se3_generator.data(), matrix::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, -1.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            matrix::matrix<double, 4, 4> se3_generator = lie::se3<double>::generator(1);
            REQUIRE(are_values_approx(se3_generator.data(), matrix::matrix<double, 4, 4>{ { { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { -1.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            matrix::matrix<double, 4, 4> se3_generator = lie::se3<double>::generator(2);
            REQUIRE(are_values_approx(se3_generator.data(), matrix::matrix<double, 4, 4>{ { { 0.0, -1.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            matrix::matrix<double, 4, 4> se3_generator = lie::se3<double>::generator(3);
            REQUIRE(are_values_approx(se3_generator.data(), matrix::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            matrix::matrix<double, 4, 4> se3_generator = lie::se3<double>::generator(4);
            REQUIRE(are_values_approx(se3_generator.data(), matrix::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            matrix::matrix<double, 4, 4> se3_generator = lie::se3<double>::generator(5);
            REQUIRE(are_values_approx(se3_generator.data(), matrix::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
    }

    {
        for (unsigned long long int i = 0; i < 6; ++i) {
            matrix::matrix<double, 4, 4> se3_generator = lie::se3<double>::generator(i);
            matrix::matrix<double, 4, 1> point = { { 1, 2, 3, 4 } };
            matrix::matrix<double, 4, 1> delta = lie::se3<double>::generator_field(i, point);
            matrix::matrix<double, 4, 1> expected = se3_generator * point;
            REQUIRE(are_values_approx(delta.data(), expected.data(), 4, 1e-4));
        }
    }

    {
        {
            lie::se3<double> se3 = lie::se3<double>::identity();
            lie::se3<double> se3_explog = lie::se3<double>::exp(se3.log());
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_explog.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_explog.translation(), 3, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 1.0, 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } } };
            lie::se3<double> se3_explog = lie::se3<double>::exp(se3.log());
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_explog.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_explog.translation(), 3, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } };
            lie::se3<double> se3_explog = lie::se3<double>::exp(se3.log());
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_explog.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_explog.translation(), 3, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } };
            lie::se3<double> se3_explog = lie::se3<double>::exp(se3.log());
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_explog.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_explog.translation(), 3, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } };
            lie::se3<double> se3_explog = lie::se3<double>::exp(se3.log());
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_explog.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_explog.translation(), 3, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } };
            lie::se3<double> se3_explog = lie::se3<double>::exp(se3.log());
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_explog.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_explog.translation(), 3, 1e-4));
        }
    }

    {
        {
            lie::se3<double> se3 = lie::se3<double>::identity();
            matrix::matrix<double, 6, 1> se3_log = se3.log();
            matrix::matrix<double, 6, 1> se3_log_expected = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(se3_log, se3_log_expected, 6, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 1.0, 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } } };
            matrix::matrix<double, 6, 1> se3_log = se3.log();
            matrix::matrix<double, 6, 1> se3_log_expected = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(se3_log, se3_log_expected, 6, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } };
            matrix::matrix<double, 6, 1> se3_log = se3.log();
            matrix::matrix<double, 6, 1> se3_log_expected = { { M_PI, 0.0, 0.0, 1.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(se3_log, se3_log_expected, 6, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } };
            matrix::matrix<double, 6, 1> se3_log = se3.log();
            matrix::matrix<double, 6, 1> se3_log_expected = { { 0.0, M_PI, 0.0, 0.0, 1.0, 0.0 } };
            REQUIRE(are_values_approx(se3_log, se3_log_expected, 6, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } };
            matrix::matrix<double, 6, 1> se3_log = se3.log();
            matrix::matrix<double, 6, 1> se3_log_expected = { { 0.0, 0.0, M_PI, 0.0, 0.0, 1.0 } };
            REQUIRE(are_values_approx(se3_log, se3_log_expected, 6, 1e-4));
        }
        {
            lie::se3<double> se3 = { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } };
            matrix::matrix<double, 6, 1> se3_log = se3.log();
            matrix::matrix<double, 6, 1> se3_log_expected = { { -1.177612, 1.442274, -1.665394, 0.491554, -0.599033, 0.706810 } };
            REQUIRE(are_values_approx(se3_log, se3_log_expected, 6, 1e-4));
        }
    }

    {
        {
            matrix::matrix<double, 6, 1> se3_log = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            lie::se3<double> se3 = lie::se3<double>::exp(se3_log);
            lie::se3<double> se3_expected = lie::se3<double>::identity();
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
        }
        {
            matrix::matrix<double, 6, 1> se3_log = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            lie::se3<double> se3 = lie::se3<double>::exp(se3_log);
            lie::se3<double> se3_expected = { { { 1.0, 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
        }
        {
            matrix::matrix<double, 6, 1> se3_log = { { M_PI, 0.0, 0.0, 1.0, 0.0, 0.0 } };
            lie::se3<double> se3 = lie::se3<double>::exp(se3_log);
            lie::se3<double> se3_expected = { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
        }
        {
            matrix::matrix<double, 6, 1> se3_log = { { 0.0, M_PI, 0.0, 0.0, 1.0, 0.0 } };
            lie::se3<double> se3 = lie::se3<double>::exp(se3_log);
            lie::se3<double> se3_expected = { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
        }
        {
            matrix::matrix<double, 6, 1> se3_log = { { 0.0, 0.0, M_PI, 0.0, 0.0, 1.0 } };
            lie::se3<double> se3 = lie::se3<double>::exp(se3_log);
            lie::se3<double> se3_expected = { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
        }
        {
            matrix::matrix<double, 6, 1> se3_log = { { -1.177612, 1.442274, -1.665394, 0.491554, -0.599033, 0.706810 } };
            lie::se3<double> se3 = lie::se3<double>::exp(se3_log);
            lie::se3<double> se3_expected = { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
        }
    }

    {
        REQUIRE((lie::se3<double>({ { 1, 0, 0, 0 } }, { { 0, 0, 0 } }) == lie::se3<double>::identity()));
        REQUIRE((lie::se3<double>({ { 1, 0, 0, 0 } }, { { 0, 0, 0 } }) != lie::se3<double>::identity()) == false);
        REQUIRE((lie::se3<double>({ { 0, 0, 0, 1 } }, { { 0, 0, 0 } }) != lie::se3<double>({ { 0, 1, 0, 0 } }, { { 0, 0, 0 } })));
        REQUIRE((lie::se3<double>({ { 0, 0, 0, 1 } }, { { 0, 0, 0 } }) == lie::se3<double>({ { 0, 1, 0, 0 } }, { { 0, 0, 0 } })) == false);
        REQUIRE((lie::se3<double>({ { 0, 0, 1, 0 } }, { { 1, 0, 0 } }) != lie::se3<double>({ { 0, 0, 1, 0 } }, { { 0, 1, 0 } })));
        REQUIRE((lie::se3<double>({ { 0, 0, 1, 0 } }, { { 1, 0, 0 } }) == lie::se3<double>({ { 0, 0, 1, 0 } }, { { 1, 1, 0 } })) == false);
        REQUIRE((lie::se3<double>({ { 0, 0, 1, 0 } }, { { 0, 1, -1 } }) == lie::se3<double>({ { 0, 0, 1, 0 } }, { { 0, 1, -1 } })));
        REQUIRE((lie::se3<double>({ { 0, 0, 1, 0 } }, { { 0, 1, -1 } }) != lie::se3<double>({ { 0, 0, 1, 0 } }, { { 0, 1, -1 } })) == false);
    }

    {
        {
            lie::se3<double> se3_lhs = lie::se3<double>::identity();
            lie::se3<double> se3_rhs = lie::se3<double>::identity();
            lie::se3<double> se3 = se3_lhs * se3_rhs;
            lie::se3<double> se3_expected = lie::se3<double>::identity();
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
            REQUIRE(are_values_approx(se3 * matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3 * matrix::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, -2.0, 3.0 } }, 3, 1e-4));
        }
        {
            lie::se3<double> se3_lhs = lie::se3<double>::identity();
            lie::se3<double> se3_rhs = { lie::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0), { { 1.0, 0.0, 0.0 } } };
            lie::se3<double> se3 = se3_lhs * se3_rhs;
            lie::se3<double> se3_expected = se3_rhs;
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
            REQUIRE(are_values_approx(se3 * matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3 * matrix::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 2.0, -3.0, -2.0 } }, 3, 1e-4));
        }
        {
            lie::se3<double> se3_lhs = { lie::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0), { { 0.0, 1.0, 0.0 } } };
            lie::se3<double> se3_rhs = { lie::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0), { { 0.0, 0.0, 1.0 } } };
            lie::se3<double> se3 = se3_lhs * se3_rhs;
            lie::se3<double> se3_expected = { lie::so3<double>::rotation(M_PI, 0.0, 0.0), { { 0.0, 0.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
            REQUIRE(are_values_approx(se3 * matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3 * matrix::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, 2.0, -3.0 } }, 3, 1e-4));
        }
        {
            lie::se3<double> se3_lhs = { lie::so3<double>::rotation(M_PI, 0.0, 0.0), { { 1.0, 1.0, 0.0 } } };
            lie::se3<double> se3_rhs = { lie::so3<double>::rotation(M_PI, 0.0, 0.0), { { 1.0, 0.0, 1.0 } } };
            lie::se3<double> se3 = (se3_lhs * se3_rhs);
            lie::se3<double> se3_expected = { -lie::so3<double>::identity(), { { 2.0, 1.0, -1.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
            REQUIRE(are_values_approx(se3 * matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 2.0, 1.0, -1.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3 * matrix::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 3.0, -1.0, 2.0 } }, 3, 1e-3));
        }
        {
            lie::se3<double> se3_lhs = { lie::so3<double>::rotation(M_PI, 0.0, 0.0), { { 1.0, 0.0, -1.0 } } };
            lie::se3<double> se3_rhs = { lie::so3<double>::rotation(0.0, M_PI, 0.0), { { 1.0, 0.0, -1.0 } } };
            lie::se3<double> se3 = (se3_lhs * se3_rhs);
            lie::se3<double> se3_expected = { lie::so3<double>::rotation(0.0, 0.0, M_PI), { { 2.0, 0.0, 0.0 } } };
            REQUIRE(are_values_approx(se3.rotation().get_quaternion(), se3_expected.rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(se3.translation(), se3_expected.translation(), 3, 1e-4));
            REQUIRE(are_values_approx(se3 * matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 2.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(se3 * matrix::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, 2.0, 3.0 } }, 3, 1e-4));
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    {
        {
            lie::sim3<double> sim3 = lie::sim3<double>::identity();
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
            lie::sim3<double> sim3 = { { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } }, 1.0 };
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
            lie::sim3<double> sim3 = { { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } }, 1.0 };
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
            lie::sim3<double> sim3 = { { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } }, 1.0 };
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
            lie::sim3<double> sim3 = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 1.0 };
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
            lie::sim3<double> sim3 = lie::sim3<double>::identity();
            REQUIRE(is_value_approx(sim3.scale(), 1.0, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), 0.1, 1e-4) == false);
            sim3.scale() *= 0.1;
            REQUIRE(is_value_approx(sim3.scale(), 1.0, 1e-4) == false);
            REQUIRE(is_value_approx(sim3.scale(), 0.1, 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 2.0 };
            REQUIRE(is_value_approx(sim3.scale(), 2.0, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), 0.5, 1e-4) == false);
            sim3.scale() = 0.5;
            REQUIRE(is_value_approx(sim3.scale(), 2.0, 1e-4) == false);
            REQUIRE(is_value_approx(sim3.scale(), 0.5, 1e-4));
        }
    }

    {
        lie::sim3<double> sim3 = lie::sim3<double>::identity();
        REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), { { 1.0, 0.0, 0.0, 0.0 } }, 4, 1e-4));
        REQUIRE(are_values_approx(sim3.transformation().translation(), { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
        REQUIRE(is_value_approx(sim3.scale(), 1.0, 1e-4));
    }

    {
        {
            lie::sim3<double> sim3 = lie::sim3<double>::identity();
            lie::sim3<double> sim3_inverse = sim3.inverse();
            lie::sim3<double> sim3_inverse_expected = lie::sim3<double>::identity();
            REQUIRE(are_values_approx(sim3_inverse.transformation().rotation().get_quaternion(), sim3_inverse_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3_inverse.transformation().translation(), sim3_inverse_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_inverse_expected.scale(), 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } }, 2.0 };
            lie::sim3<double> sim3_inverse = sim3.inverse();
            lie::sim3<double> sim3_inverse_expected = { { { { 0.0, -1.0, 0.0, 0.0 } }, { { -0.5, 0.0, 0.0 } } }, 0.5 };
            REQUIRE(are_values_approx(sim3_inverse.transformation().rotation().get_quaternion(), sim3_inverse_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3_inverse.transformation().translation(), sim3_inverse_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3_inverse.scale(), sim3_inverse_expected.scale(), 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } }, 0.5 };
            lie::sim3<double> sim3_inverse = sim3.inverse();
            lie::sim3<double> sim3_inverse_expected = { { { { 0.0, .0f, -1.0, 0.0 } }, { { 0.0, -2.0, 0.0 } } }, 2.0 };
            REQUIRE(are_values_approx(sim3_inverse.transformation().rotation().get_quaternion(), sim3_inverse_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3_inverse.transformation().translation(), sim3_inverse_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3_inverse.scale(), sim3_inverse_expected.scale(), 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } }, 0.1 };
            lie::sim3<double> sim3_inverse = sim3.inverse();
            lie::sim3<double> sim3_inverse_expected = { { { { 0.0, 0.0, 0.0, -1.0 } }, { { 0.0, 0.0, -10.0 } } }, 10.0 };
            REQUIRE(are_values_approx(sim3_inverse.transformation().rotation().get_quaternion(), sim3_inverse_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3_inverse.transformation().translation(), sim3_inverse_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3_inverse.scale(), sim3_inverse_expected.scale(), 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 1.0 };
            lie::sim3<double> sim3_inverse = sim3.inverse();
            lie::sim3<double> sim3_inverse_expected = { { { { std::sqrt(0.1), std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4) } }, { { -0.487431, 0.607913, -0.702034 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3_inverse.transformation().rotation().get_quaternion(), sim3_inverse_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3_inverse.transformation().translation(), sim3_inverse_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3_inverse.scale(), sim3_inverse_expected.scale(), 1e-4));
        }
    }

    {
        {
            matrix::matrix<double, 4, 4> sim3_generator = lie::sim3<double>::generator(0);
            REQUIRE(are_values_approx(sim3_generator.data(), matrix::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, -1.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            matrix::matrix<double, 4, 4> sim3_generator = lie::sim3<double>::generator(1);
            REQUIRE(are_values_approx(sim3_generator.data(), matrix::matrix<double, 4, 4>{ { { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { -1.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            matrix::matrix<double, 4, 4> sim3_generator = lie::sim3<double>::generator(2);
            REQUIRE(are_values_approx(sim3_generator.data(), matrix::matrix<double, 4, 4>{ { { 0.0, -1.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            matrix::matrix<double, 4, 4> sim3_generator = lie::sim3<double>::generator(3);
            REQUIRE(are_values_approx(sim3_generator.data(), matrix::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            matrix::matrix<double, 4, 4> sim3_generator = lie::sim3<double>::generator(4);
            REQUIRE(are_values_approx(sim3_generator.data(), matrix::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            matrix::matrix<double, 4, 4> sim3_generator = lie::sim3<double>::generator(5);
            REQUIRE(are_values_approx(sim3_generator.data(), matrix::matrix<double, 4, 4>{ { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
        {
            matrix::matrix<double, 4, 4> sim3_generator = lie::sim3<double>::generator(6);
            REQUIRE(are_values_approx(sim3_generator.data(), matrix::matrix<double, 4, 4>{ { { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 }, { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } } }.data(), 9, 1e-4));
        }
    }

    {
        for (unsigned long long int i = 0; i < 7; ++i) {
            matrix::matrix<double, 4, 4> sim3_generator = lie::sim3<double>::generator(i);
            matrix::matrix<double, 4, 1> point = { { 1, 2, 3, 4 } };
            matrix::matrix<double, 4, 1> delta = lie::sim3<double>::generator_field(i, point);
            matrix::matrix<double, 4, 1> expected = sim3_generator * point;
            REQUIRE(are_values_approx(delta.data(), expected.data(), 4, 1e-4));
        }
    }

    {
        {
            lie::sim3<double> sim3 = lie::sim3<double>::identity();
            lie::sim3<double> sim3_explog = lie::sim3<double>::exp(sim3.log());
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_explog.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_explog.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_explog.scale(), 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { 1.0, 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } } }, 1.0 };
            lie::sim3<double> sim3_explog = lie::sim3<double>::exp(sim3.log());
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_explog.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_explog.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_explog.scale(), 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } }, 1.0 };
            lie::sim3<double> sim3_explog = lie::sim3<double>::exp(sim3.log());
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_explog.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_explog.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_explog.scale(), 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } }, 1.0 };
            lie::sim3<double> sim3_explog = lie::sim3<double>::exp(sim3.log());
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_explog.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_explog.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_explog.scale(), 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } }, 1.0 };
            lie::sim3<double> sim3_explog = lie::sim3<double>::exp(sim3.log());
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_explog.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_explog.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_explog.scale(), 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 1.0 };
            lie::sim3<double> sim3_explog = lie::sim3<double>::exp(sim3.log());
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_explog.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_explog.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_explog.scale(), 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 0.8 };
            lie::sim3<double> sim3_explog = lie::sim3<double>::exp(sim3.log());
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_explog.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_explog.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_explog.scale(), 1e-4));
        }
    }

    {
        {
            lie::sim3<double> sim3 = lie::sim3<double>::identity();
            matrix::matrix<double, 7, 1> sim3_log = sim3.log();
            matrix::matrix<double, 7, 1> sim3_log_expected = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(sim3_log, sim3_log_expected, 7, 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { 1.0, 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } } }, 1.0 };
            matrix::matrix<double, 7, 1> sim3_log = sim3.log();
            matrix::matrix<double, 7, 1> sim3_log_expected = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(sim3_log, sim3_log_expected, 7, 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } }, 1.0 };
            matrix::matrix<double, 7, 1> sim3_log = sim3.log();
            matrix::matrix<double, 7, 1> sim3_log_expected = { { M_PI, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(sim3_log, sim3_log_expected, 7, 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } }, 1.0 };
            matrix::matrix<double, 7, 1> sim3_log = sim3.log();
            matrix::matrix<double, 7, 1> sim3_log_expected = { { 0.0, M_PI, 0.0, 0.0, 1.0, 0.0, 0.0 } };
            REQUIRE(are_values_approx(sim3_log, sim3_log_expected, 7, 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } }, 1.0 };
            matrix::matrix<double, 7, 1> sim3_log = sim3.log();
            matrix::matrix<double, 7, 1> sim3_log_expected = { { 0.0, 0.0, M_PI, 0.0, 0.0, 1.0, 0.0 } };
            REQUIRE(are_values_approx(sim3_log, sim3_log_expected, 7, 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 1.0 };
            matrix::matrix<double, 7, 1> sim3_log = sim3.log();
            matrix::matrix<double, 7, 1> sim3_log_expected = { { -1.177612, 1.442274, -1.665394, 0.491554, -0.599033, 0.706810, 0.0 } };
            REQUIRE(are_values_approx(sim3_log, sim3_log_expected, 7, 1e-4));
        }
        {
            lie::sim3<double> sim3 = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 0.8 };
            matrix::matrix<double, 7, 1> sim3_log = sim3.log();
            matrix::matrix<double, 7, 1> sim3_log_expected = { { -1.177611, 1.442274, -1.665394, 0.548948, -0.668049, 0.788500, -0.223144 } };
            REQUIRE(are_values_approx(sim3_log, sim3_log_expected, 7, 1e-4));
        }
    }

    {
        {
            matrix::matrix<double, 7, 1> sim3_log = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            lie::sim3<double> sim3 = lie::sim3<double>::exp(sim3_log);
            lie::sim3<double> sim3_expected = lie::sim3<double>::identity();
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
        }
        {
            matrix::matrix<double, 7, 1> sim3_log = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
            lie::sim3<double> sim3 = lie::sim3<double>::exp(sim3_log);
            lie::sim3<double> sim3_expected = { { { { 1.0, 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
        }
        {
            matrix::matrix<double, 7, 1> sim3_log = { { M_PI, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 } };
            lie::sim3<double> sim3 = lie::sim3<double>::exp(sim3_log);
            lie::sim3<double> sim3_expected = { { { { 0.0, 1.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
        }
        {
            matrix::matrix<double, 7, 1> sim3_log = { { 0.0, M_PI, 0.0, 0.0, 1.0, 0.0, 0.0 } };
            lie::sim3<double> sim3 = lie::sim3<double>::exp(sim3_log);
            lie::sim3<double> sim3_expected = { { { { 0.0, 0.0, 1.0, 0.0 } }, { { 0.0, 1.0, 0.0 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
        }
        {
            matrix::matrix<double, 7, 1> sim3_log = { { 0.0, 0.0, M_PI, 0.0, 0.0, 1.0, 0.0 } };
            lie::sim3<double> sim3 = lie::sim3<double>::exp(sim3_log);
            lie::sim3<double> sim3_expected = { { { { 0.0, 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 1.0 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
        }
        {
            matrix::matrix<double, 7, 1> sim3_log = { { -1.177612, 1.442274, -1.665394, 0.491554, -0.599033, 0.706810, 0.0 } };
            lie::sim3<double> sim3 = lie::sim3<double>::exp(sim3_log);
            lie::sim3<double> sim3_expected = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
        }
        {
            matrix::matrix<double, 7, 1> sim3_log = { { -1.177611, 1.442274, -1.665394, 0.548948, -0.668049, 0.788500, -0.223144 } };
            lie::sim3<double> sim3 = lie::sim3<double>::exp(sim3_log);
            lie::sim3<double> sim3_expected = { { { { std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3), -std::sqrt(0.4) } }, { { 0.5, -0.6, 0.7 } } }, 0.8 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
        }
    }

    {
        REQUIRE((lie::sim3<double>({ { { { 1, 0, 0, 0 } }, { { 0, 0, 0 } } }, 1 }) == lie::sim3<double>::identity()));
        REQUIRE((lie::sim3<double>({ { { { 1, 0, 0, 0 } }, { { 0, 0, 0 } } }, 1 }) != lie::sim3<double>::identity()) == false);
        REQUIRE((lie::sim3<double>({ { { { 1, 0, 0, 0 } }, { { 0, 0, 0 } } }, 2 }) != lie::sim3<double>::identity()));
        REQUIRE((lie::sim3<double>({ { { { 1, 0, 0, 0 } }, { { 0, 0, 0 } } }, 2 }) == lie::sim3<double>::identity()) == false);
        REQUIRE((lie::sim3<double>({ { { { 0, 0, 0, 1 } }, { { 0, 0, 0 } } }, 1 }) != lie::sim3<double>({ { { { 0, 1, 0, 0 } }, { { 0, 0, 0 } } }, 1 })));
        REQUIRE((lie::sim3<double>({ { { { 0, 0, 0, 1 } }, { { 0, 0, 0 } } }, 1 }) == lie::sim3<double>({ { { { 0, 1, 0, 0 } }, { { 0, 0, 0 } } }, 1 })) == false);
        REQUIRE((lie::sim3<double>({ { { { 0, 0, 0, 1 } }, { { 0, 0, 0 } } }, 2 }) != lie::sim3<double>({ { { { 0, 1, 0, 0 } }, { { 0, 0, 0 } } }, 2 })));
        REQUIRE((lie::sim3<double>({ { { { 0, 0, 0, 1 } }, { { 0, 0, 0 } } }, 2 }) == lie::sim3<double>({ { { { 0, 1, 0, 0 } }, { { 0, 0, 0 } } }, 2 })) == false);
        REQUIRE((lie::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 1, 0, 0 } } }, -1 }) != lie::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 0, 1, 0 } } }, -1 })));
        REQUIRE((lie::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 1, 0, 0 } } }, -1 }) == lie::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 1, 1, 0 } } }, -1 })) == false);
        REQUIRE((lie::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 0, 1, -1 } } }, -1 }) == lie::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 0, 1, -1 } } }, -1 })));
        REQUIRE((lie::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 0, 1, -1 } } }, -1 }) != lie::sim3<double>({ { { { 0, 0, 1, 0 } }, { { 0, 1, -1 } } }, -1 })) == false);
    }

    {
        {
            lie::sim3<double> sim3_lhs = lie::sim3<double>::identity();
            lie::sim3<double> sim3_rhs = lie::sim3<double>::identity();
            lie::sim3<double> sim3 = sim3_lhs * sim3_rhs;
            lie::sim3<double> sim3_expected = lie::sim3<double>::identity();
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
            REQUIRE(are_values_approx(sim3 * matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(sim3 * matrix::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, -2.0, 3.0 } }, 3, 1e-4));
        }
        {
            lie::sim3<double> sim3_lhs = lie::sim3<double>::identity();
            lie::sim3<double> sim3_rhs = { { lie::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0), { { 1.0, 0.0, 0.0 } } }, 0.1 };
            lie::sim3<double> sim3 = sim3_lhs * sim3_rhs;
            lie::sim3<double> sim3_expected = { { lie::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0), { { 1.0, 0.0, 0.0 } } }, 0.1 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
            REQUIRE(are_values_approx(sim3 * matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(sim3 * matrix::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.1, -0.3, -0.2 } }, 3, 1e-4));
        }
        {
            lie::sim3<double> sim3_lhs = { { lie::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0), { { 0.0, 1.0, 0.0 } } }, 0.5 };
            lie::sim3<double> sim3_rhs = { { lie::so3<double>::rotation(M_PI / 2.0, 0.0, 0.0), { { 0.0, 0.0, 1.0 } } }, 2.0 };
            lie::sim3<double> sim3 = sim3_lhs * sim3_rhs;
            lie::sim3<double> sim3_expected = { { lie::so3<double>::rotation(M_PI, 0.0, 0.0), { { 0.0, 0.5, 0.0 } } }, 1.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
            REQUIRE(are_values_approx(sim3 * matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.5, 0.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(sim3 * matrix::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, 2.5, -3.0 } }, 3, 1e-4));
        }
        {
            lie::sim3<double> sim3_lhs = { { lie::so3<double>::rotation(M_PI, 0.0, 0.0), { { 1.0, 1.0, 0.0 } } }, 0.5 };
            lie::sim3<double> sim3_rhs = { { lie::so3<double>::rotation(M_PI, 0.0, 0.0), { { 1.0, 0.0, 1.0 } } }, 0.5 };
            lie::sim3<double> sim3 = (sim3_lhs * sim3_rhs);
            lie::sim3<double> sim3_expected = { { -lie::so3<double>::identity(), { { 1.5, 1.0, -0.5 } } }, 0.25 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
            REQUIRE(are_values_approx(sim3 * matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 1.5, 1.0, -0.5 } }, 3, 1e-4));
            REQUIRE(are_values_approx(sim3 * matrix::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.75, 0.5, 0.25 } }, 3, 1e-4));
        }
        {
            lie::sim3<double> sim3_lhs = { { lie::so3<double>::rotation(M_PI, 0.0, 0.0), { { 1.0, 0.0, -1.0 } } }, 2.0 };
            lie::sim3<double> sim3_rhs = { { lie::so3<double>::rotation(0.0, M_PI, 0.0), { { 1.0, 0.0, -1.0 } } }, 1.0 };
            lie::sim3<double> sim3 = (sim3_lhs * sim3_rhs);
            lie::sim3<double> sim3_expected = { { lie::so3<double>::rotation(0.0, 0.0, M_PI), { { 3.0, 0.0, 1.0 } } }, 2.0 };
            REQUIRE(are_values_approx(sim3.transformation().rotation().get_quaternion(), sim3_expected.transformation().rotation().get_quaternion(), 4, 1e-4));
            REQUIRE(are_values_approx(sim3.transformation().translation(), sim3_expected.transformation().translation(), 3, 1e-4));
            REQUIRE(is_value_approx(sim3.scale(), sim3_expected.scale(), 1e-4));
            REQUIRE(are_values_approx(sim3 * matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.0 } }, { { 3.0, 0.0, 1.0 } }, 3, 1e-4));
            REQUIRE(are_values_approx(sim3 * matrix::matrix<double, 3, 1>{ { 1.0, -2.0, 3.0 } }, { { 1.0, 4.0, 7.0 } }, 3, 1e-4));
        }
    }

    return EXIT_SUCCESS;
}
