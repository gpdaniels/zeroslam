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

#include "matrix.hpp"

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
static inline bool are_values_approx(const array_type& lhs, const array_type& rhs, size_t length, double epsilon = 1e-8) {
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
        {
            matrix::matrix<double, 0, 0> m;
            static_cast<void>(m);
        }
        {
            matrix::matrix<double, 1, 1> m;
            static_cast<void>(m);
        }
        {
            matrix::matrix<double, 1, 2> m;
            static_cast<void>(m);
        }
        {
            matrix::matrix<double, 2, 1> m;
            static_cast<void>(m);
        }
        {
            matrix::matrix<double, 2, 2> m;
            static_cast<void>(m);
        }
    }

    {
        matrix::matrix<double, 3, 2> m1 = { { {} } };
        m1[2][1] = 1234567890;
        matrix::matrix<double, 3, 2> m2(m1);
        REQUIRE(m2[2][1] == 1234567890);
    }

    {
        {
            matrix::matrix<double, 1, 1> m = { { 101.0 } };
            REQUIRE(m[0] == 101.0);
            REQUIRE(m(0, 0) == 101.0);
        }

        {
            matrix::matrix<double, 5, 1> m = { { 101.0, 202.0, 303.0, 404.0, 505.0 } };
            REQUIRE(m[0] == 101.0);
            REQUIRE(m[1] == 202.0);
            REQUIRE(m[2] == 303.0);
            REQUIRE(m[3] == 404.0);
            REQUIRE(m[4] == 505.0);
            REQUIRE(m(0, 0) == 101.0);
            REQUIRE(m(1, 0) == 202.0);
            REQUIRE(m(2, 0) == 303.0);
            REQUIRE(m(3, 0) == 404.0);
            REQUIRE(m(4, 0) == 505.0);
        }
        {
            matrix::matrix<double, 1, 5> m = { { 101.0, 202.0, 303.0, 404.0, 505.0 } };
            REQUIRE(m[0] == 101.0);
            REQUIRE(m[1] == 202.0);
            REQUIRE(m[2] == 303.0);
            REQUIRE(m[3] == 404.0);
            REQUIRE(m[4] == 505.0);
            REQUIRE(m(0, 0) == 101.0);
            REQUIRE(m(0, 1) == 202.0);
            REQUIRE(m(0, 2) == 303.0);
            REQUIRE(m(0, 3) == 404.0);
            REQUIRE(m(0, 4) == 505.0);
        }
        {
            matrix::matrix<double, 3, 2> m = { { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } } };
            REQUIRE(m[0][0] == 1.0);
            REQUIRE(m[0][1] == 2.0);
            REQUIRE(m[1][0] == 3.0);
            REQUIRE(m[1][1] == 4.0);
            REQUIRE(m[2][0] == 5.0);
            REQUIRE(m[2][1] == 6.0);
            REQUIRE(m(0, 0) == 1.0);
            REQUIRE(m(0, 1) == 2.0);
            REQUIRE(m(1, 0) == 3.0);
            REQUIRE(m(1, 1) == 4.0);
            REQUIRE(m(2, 0) == 5.0);
            REQUIRE(m(2, 1) == 6.0);
        }
    }

    {
        double data[3][2] = { { 1, 2 }, { 3, 4 }, { 5, 6 } };
        const double* data_pointer = &data[0][0];
        matrix::matrix<double, 3, 2> m(data_pointer);
        REQUIRE(m[0][0] == 1);
        REQUIRE(m[0][1] == 2);
        REQUIRE(m[1][0] == 3);
        REQUIRE(m[1][1] == 4);
        REQUIRE(m[2][0] == 5);
        REQUIRE(m[2][1] == 6);
    }

    {
        matrix::matrix<double, 3, 2> m = matrix::matrix<double, 3, 2>::zero();
        REQUIRE(m[0][0] == 0);
        REQUIRE(m[0][1] == 0);
        REQUIRE(m[1][0] == 0);
        REQUIRE(m[1][1] == 0);
        REQUIRE(m[2][0] == 0);
        REQUIRE(m[2][1] == 0);
    }

    {
        matrix::matrix<double, 3, 2> m = matrix::matrix<double, 3, 2>::identity();
        REQUIRE(m[0][0] == 1);
        REQUIRE(m[0][1] == 0);
        REQUIRE(m[1][0] == 0);
        REQUIRE(m[1][1] == 1);
        REQUIRE(m[2][0] == 0);
        REQUIRE(m[2][1] == 0);
    }

    {
        matrix::matrix<double, 3, 2> m;
        double* data_pointer_in = m.data();
        data_pointer_in[0] = 1;
        data_pointer_in[1] = 2;
        data_pointer_in[2] = 3;
        data_pointer_in[3] = 4;
        data_pointer_in[4] = 5;
        data_pointer_in[5] = 6;
        const double* data_pointer_out = m.data();
        REQUIRE(data_pointer_out[0] == 1);
        REQUIRE(data_pointer_out[1] == 2);
        REQUIRE(data_pointer_out[2] == 3);
        REQUIRE(data_pointer_out[3] == 4);
        REQUIRE(data_pointer_out[4] == 5);
        REQUIRE(data_pointer_out[5] == 6);
    }

    {
        matrix::matrix<double, 3, 2> m;
        REQUIRE(m.size() == (3 * 2));
    }

    {
        matrix::matrix<double, 3, 2> m;
        REQUIRE(m.rows() == 3);
    }

    {
        matrix::matrix<double, 3, 2> m;
        REQUIRE(m.cols() == 2);
    }

    {
        matrix::matrix<double, 0, 0> m(3, 2);
        m[0][0] = 1;
        m[0][1] = 2;
        m[1][0] = 3;
        m[1][1] = 4;
        m[2][0] = 5;
        m[2][1] = 6;
        matrix::matrix<double, 0, 0> block = get_block(m, 1, 0, 2, 1);
        REQUIRE(block[0][0] == 3);
        REQUIRE(block[1][0] == 5);
        block[0][0] = 20;
        block[1][0] = 40;
        set_block(m, 0, 1, block);
        REQUIRE(m[0][0] == 1);
        REQUIRE(m[0][1] == 20);
        REQUIRE(m[1][0] == 3);
        REQUIRE(m[1][1] == 40);
        REQUIRE(m[2][0] == 5);
        REQUIRE(m[2][1] == 6);
    }

    {
        {
            matrix::matrix<double, 2, 2> m;
            m[0][0] = 1;
            m[0][1] = 2;
            m[1][0] = 3;
            m[1][1] = 4;
            m = transpose(m);
            REQUIRE(m[0][0] == 1);
            REQUIRE(m[0][1] == 3);
            REQUIRE(m[1][0] == 2);
            REQUIRE(m[1][1] == 4);
        }
        {
            matrix::matrix<double, 3, 2> m1;
            m1[0][0] = 1;
            m1[0][1] = 2;
            m1[1][0] = 3;
            m1[1][1] = 4;
            m1[2][0] = 5;
            m1[2][1] = 6;
            matrix::matrix<double, 2, 3> m2 = transpose(m1);
            REQUIRE(m2[0][0] == 1);
            REQUIRE(m2[1][0] == 2);
            REQUIRE(m2[0][1] == 3);
            REQUIRE(m2[1][1] == 4);
            REQUIRE(m2[0][2] == 5);
            REQUIRE(m2[1][2] == 6);
        }
        {
            matrix::matrix<double, 0, 0> m1(3, 2);
            m1[0][0] = 1;
            m1[0][1] = 2;
            m1[1][0] = 3;
            m1[1][1] = 4;
            m1[2][0] = 5;
            m1[2][1] = 6;
            matrix::matrix<double, 0, 0> m2 = transpose(m1);
            REQUIRE(m2[0][0] == 1);
            REQUIRE(m2[1][0] == 2);
            REQUIRE(m2[0][1] == 3);
            REQUIRE(m2[1][1] == 4);
            REQUIRE(m2[0][2] == 5);
            REQUIRE(m2[1][2] == 6);
        }
    }

    {
        {
            matrix::matrix<double, 1, 1> m = { { 0.1 } };
            matrix::matrix<double, 1, 1> inverse = { { 10.0 } };
            matrix::matrix<double, 1, 1> result = invert(m);
            REQUIRE(are_values_approx(result.data(), inverse.data(), 1 * 1, 1e-6));
        }
        {
            matrix::matrix<double, 2, 2> m = { { { 1.0, 2.0 }, { 3.0, 4.0 } } };
            matrix::matrix<double, 2, 2> inverse = { { { -2.0, +1.0 }, { +1.5, -0.5 } } };
            matrix::matrix<double, 2, 2> result = invert(m);
            REQUIRE(are_values_approx(result.data(), inverse.data(), 2 * 2, 1e-6));
        }
        {
            matrix::matrix<double, 3, 3> m = {
                { { 1.0, 2.0, 0.0 }, { 0.0, 1.0, 2.0 }, { 2.0, 0.0, 1.0 } }
            };
            matrix::matrix<double, 3, 3> inverse = { { { +1.0 / 9.0, -2.0 / 9.0, +4.0 / 9.0 }, { +4.0 / 9.0, +1.0 / 9.0, -2.0 / 9.0 }, { -2.0 / 9.0, +4.0 / 9.0, +1.0 / 9.0 } } };
            matrix::matrix<double, 3, 3> result = invert(m);
            REQUIRE(are_values_approx(result.data(), inverse.data(), 3 * 3, 1e-6));
        }
        {
            matrix::matrix<double, 4, 4> m = { { { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0, 1.0 } } };
            matrix::matrix<double, 4, 4> inverse = { { { -1.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0, 0.0 } } };
            matrix::matrix<double, 4, 4> result = invert(m);
            REQUIRE(are_values_approx(result.data(), inverse.data(), 4 * 4, 1e-6));
        }
        {
            matrix::matrix<double, 5, 5> m = { { { 2.0, 12.0, 5.0, 12.0, 14.0 }, { 16.0, 0.0, 12.0, 16.0, 16.0 }, { 10.0, 14.0, 15.0, 10.0, 11.0 }, { 14.0, 1.0, 1.0, 18.0, 0.0 }, { 9.0, 13.0, 1.0, 6.0, 6.0 } } };
            matrix::matrix<double, 5, 5> inverse = {
                { { -0.0717574493, 0.0417861176, -0.0144209886, -0.0087742159, 0.0824428807 },
                  { 0.0102365852, -0.0419779631, 0.0274809602, 0.0026641183, 0.0376741088 },
                  { -0.0234847826, -0.0129586192, 0.0897112906, 0.0023745402, -0.0751165551 },
                  { 0.0565473604, -0.0294482813, 0.0047056438, 0.0621000202, -0.0620421046 },
                  { 0.0328236759, 0.0598811281, -0.0575681232, -0.0551067095, 0.0359366403 } }
            };
            matrix::matrix<double, 5, 5> result = invert(m);
            REQUIRE(are_values_approx(result.data(), inverse.data(), 5 * 5, 1e-6));
        }
        // Un-invertible
        {
            matrix::matrix<double, 1, 1> m = { { 0.0 } };
            matrix::matrix<double, 1, 1> result = invert(m);
            REQUIRE(are_values_approx(result.data(), matrix::matrix<double, 1, 1>::zero().data(), 1 * 1, 1e-6));
        }
        {
            matrix::matrix<double, 2, 2> m = { { { 0.0, 0.0 }, { 0.0, 1.0 } } };
            matrix::matrix<double, 2, 2> result = invert(m);
            REQUIRE(are_values_approx(result.data(), matrix::matrix<double, 2, 2>::zero().data(), 2 * 2, 1e-6));
        }
        {
            matrix::matrix<double, 3, 3> m = {
                { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } }
            };
            matrix::matrix<double, 3, 3> result = invert(m);
            REQUIRE(are_values_approx(result.data(), matrix::matrix<double, 3, 3>::zero().data(), 3 * 3, 1e-6));
        }
        {
            matrix::matrix<double, 4, 4> m = { { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 }, { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 } } };
            matrix::matrix<double, 4, 4> result = invert(m);
            REQUIRE(are_values_approx(result.data(), matrix::matrix<double, 4, 4>::zero().data(), 4 * 4, 1e-6));
        }
    }

    {
        {
            matrix::matrix<double, 3, 1> m{ { 0, 2, 4 } };
            REQUIRE(m[0] == 0);
            REQUIRE(m[1] == 2);
            REQUIRE(m[2] == 4);
            m = { { { 0 }, { 2 }, { 4 } } };
            REQUIRE(m[0] == 0);
            REQUIRE(m[1] == 2);
            REQUIRE(m[2] == 4);
            m[1] = 1234567890;
            REQUIRE(m[0] == 0);
            REQUIRE(m[1] == 1234567890);
            REQUIRE(m[2] == 4);
        }
        {
            matrix::matrix<double, 1, 3> m = { { 0, 1, 2 } };
            REQUIRE(m[0] == 0);
            REQUIRE(m[1] == 1);
            REQUIRE(m[2] == 2);
            m = { { { 0, 1, 2 } } };
            REQUIRE(m[0] == 0);
            REQUIRE(m[1] == 1);
            REQUIRE(m[2] == 2);
            m[1] = 1234567890;
            REQUIRE(m[0] == 0);
            REQUIRE(m[1] == 1234567890);
            REQUIRE(m[2] == 2);
        }
        {
            matrix::matrix<double, 3, 2> m{ { { 0, 1 }, { 2, 3 }, { 4, 5 } } };
            REQUIRE(m[0][0] == 0);
            REQUIRE(m[0][1] == 1);
            REQUIRE(m[1][0] == 2);
            REQUIRE(m[1][1] == 3);
            REQUIRE(m[2][0] == 4);
            REQUIRE(m[2][1] == 5);
            m[2][1] = 1234567890;
            REQUIRE(m[0][0] == 0);
            REQUIRE(m[0][1] == 1);
            REQUIRE(m[1][0] == 2);
            REQUIRE(m[1][1] == 3);
            REQUIRE(m[2][0] == 4);
            REQUIRE(m[2][1] == 1234567890);
        }
    }

    {
        matrix::matrix<double, 2, 2> m1 = { { { 1, 2 }, { 3, 4 } } };
        matrix::matrix<double, 2, 2> m2 = { { { 1, 2 }, { 3, 4 } } };
        matrix::matrix<double, 2, 2> m3 = { { { 1, 2 }, { 3, 5 } } };
        REQUIRE((m1 == m2) == true);
        REQUIRE((m1 == m3) == false);
        REQUIRE((m1 != m2) == false);
        REQUIRE((m1 != m3) == true);
    }

    {
        matrix::matrix<double, 2, 2> m1 = { { { 1, 2 }, { 3, 4 } } };
        matrix::matrix<double, 2, 2> m2 = { { { -1, -2 }, { -3, -4 } } };
        REQUIRE(m1 == -m2);
        REQUIRE(m1 != +m2);
        REQUIRE(m1 == +m1);
        REQUIRE(m2 != -m2);
    }

    {
        matrix::matrix<double, 2, 2> m1 = { { { 1.0, 2.0 }, { 3.0, 4.0 } } };
        matrix::matrix<double, 2, 2> m2 = { { { -1.0, -2.0 }, { -3.0, -40.0 } } };
        matrix::matrix<double, 2, 2> result1 = m1 + m2;
        REQUIRE(result1[0][0] == 0.0);
        REQUIRE(result1[0][1] == 0.0);
        REQUIRE(result1[1][0] == 0.0);
        REQUIRE(result1[1][1] == -36.0);
        matrix::matrix<double, 2, 2> result2 = m2 + 7.0;
        REQUIRE(result2[0][0] == 6.0);
        REQUIRE(result2[0][1] == 5.0);
        REQUIRE(result2[1][0] == 4.0);
        REQUIRE(result2[1][1] == -33.0);
    }

    {
        matrix::matrix<double, 2, 2> m1 = { { { 1.0, 2.0 }, { 3.0, 4.0 } } };
        matrix::matrix<double, 2, 2> m2 = { { { -1.0, -2.0 }, { -3.0, -40.0 } } };
        matrix::matrix<double, 2, 2> result1 = m1 - m2;
        REQUIRE(result1[0][0] == 2.0);
        REQUIRE(result1[0][1] == 4.0);
        REQUIRE(result1[1][0] == 6.0);
        REQUIRE(result1[1][1] == 44.0);
        matrix::matrix<double, 2, 2> result2 = m2 - 7.0;
        REQUIRE(result2[0][0] == -8.0);
        REQUIRE(result2[0][1] == -9.0);
        REQUIRE(result2[1][0] == -10.0);
        REQUIRE(result2[1][1] == -47.0);
    }

    {
        {
            matrix::matrix<double, 2, 2> m1 = { { { 1.0, 2.0 }, { 3.0, 4.0 } } };
            matrix::matrix<double, 2, 2> m2 = { { { -1.0, -2.0 }, { -3.0, -40.0 } } };
            matrix::matrix<double, 2, 2> result1 = m1 * m2;
            REQUIRE(result1[0][0] == -7.0);
            REQUIRE(result1[0][1] == -82.0);
            REQUIRE(result1[1][0] == -15.0);
            REQUIRE(result1[1][1] == -166.0);
            matrix::matrix<double, 2, 2> result2 = m2 * 7.0;
            REQUIRE(result2[0][0] == -7.0);
            REQUIRE(result2[0][1] == -14.0);
            REQUIRE(result2[1][0] == -21.0);
            REQUIRE(result2[1][1] == -280.0);
        }
        {
            matrix::matrix<double, 3, 2> m;
            m[0][0] = 1;
            m[0][1] = 2;
            m[1][0] = 3;
            m[1][1] = 4;
            m[2][0] = 5;
            m[2][1] = 6;
            matrix::matrix<double, 2, 2> result = transpose(m) * m;
            REQUIRE(result[0][0] == 35);
            REQUIRE(result[0][1] == 44);
            REQUIRE(result[1][0] == 44);
            REQUIRE(result[1][1] == 56);
        }
    }

    return EXIT_SUCCESS;
}