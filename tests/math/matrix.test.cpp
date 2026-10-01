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

#include "math/matrix.hpp"

#include "core/random_pcg.hpp"

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

static inline double random_signed(core::random_pcg& rng) {
    return (static_cast<double>(rng.get_random_raw() % 2000000u) / 1000000.0) - 1.0;
}

// Inverts D1 * M * D2 for a well conditioned M and diagonal scalings D1 and D2 spanning 10^-range to 10^range, which must
// succeed and match D2^-1 * M^-1 * D1^-1, and rejects the same scalings of a rank deficient matrix.
template <size_t size>
static void check_scaled_inverse(core::random_pcg& rng, const double range) {
    for (int trial = 0; trial < 200; ++trial) {
        math::matrix<double, size, size> base;
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                base[i][j] = random_signed(rng) + ((i == j) ? static_cast<double>(size) : 0.0);
            }
        }
        double row_scale[size];
        double column_scale[size];
        for (size_t i = 0; i < size; ++i) {
            row_scale[i] = std::pow(10.0, range * random_signed(rng));
            column_scale[i] = std::pow(10.0, range * random_signed(rng));
        }
        math::matrix<double, size, size> scaled;
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                scaled[i][j] = row_scale[i] * base[i][j] * column_scale[j];
            }
        }
        math::matrix<double, size, size> base_inverse;
        REQUIRE(invert(base, base_inverse));
        double base_inverse_scale = 0.0;
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                base_inverse_scale = std::fmax(base_inverse_scale, std::abs(base_inverse[i][j]));
            }
        }
        math::matrix<double, size, size> scaled_inverse;
        REQUIRE(invert(scaled, scaled_inverse));
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                const double unscaled = scaled_inverse[i][j] * column_scale[i] * row_scale[j];
                REQUIRE(std::abs(unscaled - base_inverse[i][j]) <= 1e-12 * base_inverse_scale);
            }
        }

        // The last row is the sum of the others, which the scalings only perturb at the rounding level.
        math::matrix<double, size, size> deficient = base;
        for (size_t j = 0; j < size; ++j) {
            deficient[size - 1][j] = 0.0;
            for (size_t i = 0; i + 1 < size; ++i) {
                deficient[size - 1][j] += base[i][j];
            }
        }
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                deficient[i][j] = row_scale[i] * deficient[i][j] * column_scale[j];
            }
        }
        math::matrix<double, size, size> deficient_inverse;
        REQUIRE(!invert(deficient, deficient_inverse));
        REQUIRE(deficient_inverse == (math::matrix<double, size, size>::zero()));
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);
    {
        {
            math::matrix<double, 0, 0> m;
            static_cast<void>(m);
        }
        {
            math::matrix<double, 1, 1> m;
            static_cast<void>(m);
        }
        {
            math::matrix<double, 1, 2> m;
            static_cast<void>(m);
        }
        {
            math::matrix<double, 2, 1> m;
            static_cast<void>(m);
        }
        {
            math::matrix<double, 2, 2> m;
            static_cast<void>(m);
        }
    }

    {
        math::matrix<double, 3, 2> m1 = { { {} } };
        m1[2][1] = 1234567890;
        math::matrix<double, 3, 2> m2(m1);
        REQUIRE(m2[2][1] == 1234567890);
    }

    {
        {
            math::matrix<double, 1, 1> m = { { 101.0 } };
            REQUIRE(m[0] == 101.0);
            REQUIRE(m(0, 0) == 101.0);
        }

        {
            math::matrix<double, 5, 1> m = { { 101.0, 202.0, 303.0, 404.0, 505.0 } };
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
            math::matrix<double, 1, 5> m = { { 101.0, 202.0, 303.0, 404.0, 505.0 } };
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
            math::matrix<double, 3, 2> m = { { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } } };
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
        math::matrix<double, 3, 2> m(data_pointer);
        REQUIRE(m[0][0] == 1);
        REQUIRE(m[0][1] == 2);
        REQUIRE(m[1][0] == 3);
        REQUIRE(m[1][1] == 4);
        REQUIRE(m[2][0] == 5);
        REQUIRE(m[2][1] == 6);
    }

    {
        math::matrix<double, 3, 2> m = math::matrix<double, 3, 2>::zero();
        REQUIRE(m[0][0] == 0);
        REQUIRE(m[0][1] == 0);
        REQUIRE(m[1][0] == 0);
        REQUIRE(m[1][1] == 0);
        REQUIRE(m[2][0] == 0);
        REQUIRE(m[2][1] == 0);
    }

    {
        math::matrix<double, 3, 2> m1;
        REQUIRE(m1[0][0] == 0);
        REQUIRE(m1[0][1] == 0);
        REQUIRE(m1[1][0] == 0);
        REQUIRE(m1[1][1] == 0);
        REQUIRE(m1[2][0] == 0);
        REQUIRE(m1[2][1] == 0);
        math::matrix<double, 3, 2> m2(m1);
        math::matrix<double, 3, 2> zero = math::matrix<double, 3, 2>::zero();
        REQUIRE(m2 == zero);
    }

    {
        math::matrix<double, 3, 2> m = math::matrix<double, 3, 2>::identity();
        REQUIRE(m[0][0] == 1);
        REQUIRE(m[0][1] == 0);
        REQUIRE(m[1][0] == 0);
        REQUIRE(m[1][1] == 1);
        REQUIRE(m[2][0] == 0);
        REQUIRE(m[2][1] == 0);
    }

    {
        math::matrix<double, 3, 2> m;
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
        math::matrix<double, 3, 2> m;
        REQUIRE(m.size() == (3 * 2));
    }

    {
        math::matrix<double, 3, 2> m;
        REQUIRE(m.rows() == 3);
    }

    {
        math::matrix<double, 3, 2> m;
        REQUIRE(m.cols() == 2);
    }

    {
        math::matrix<double, 0, 0> m(3, 2);
        m[0][0] = 1;
        m[0][1] = 2;
        m[1][0] = 3;
        m[1][1] = 4;
        m[2][0] = 5;
        m[2][1] = 6;
        math::matrix<double, 0, 0> block = get_block(m, 1, 0, 2, 1);
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
            math::matrix<double, 2, 2> m;
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
            math::matrix<double, 3, 2> m1;
            m1[0][0] = 1;
            m1[0][1] = 2;
            m1[1][0] = 3;
            m1[1][1] = 4;
            m1[2][0] = 5;
            m1[2][1] = 6;
            math::matrix<double, 2, 3> m2 = transpose(m1);
            REQUIRE(m2[0][0] == 1);
            REQUIRE(m2[1][0] == 2);
            REQUIRE(m2[0][1] == 3);
            REQUIRE(m2[1][1] == 4);
            REQUIRE(m2[0][2] == 5);
            REQUIRE(m2[1][2] == 6);
        }
        {
            math::matrix<double, 0, 0> m1(3, 2);
            m1[0][0] = 1;
            m1[0][1] = 2;
            m1[1][0] = 3;
            m1[1][1] = 4;
            m1[2][0] = 5;
            m1[2][1] = 6;
            math::matrix<double, 0, 0> m2 = transpose(m1);
            REQUIRE(m2[0][0] == 1);
            REQUIRE(m2[1][0] == 2);
            REQUIRE(m2[0][1] == 3);
            REQUIRE(m2[1][1] == 4);
            REQUIRE(m2[0][2] == 5);
            REQUIRE(m2[1][2] == 6);
        }
        // Row and column vectors, where operator[] returns an element rather than a row.
        {
            const math::matrix<double, 3, 1> column = { { 1.0, 2.0, 3.0 } };
            const math::matrix<double, 1, 3> row = transpose(column);
            REQUIRE(row.rows() == 1);
            REQUIRE(row.cols() == 3);
            REQUIRE(row[0] == 1.0);
            REQUIRE(row[1] == 2.0);
            REQUIRE(row[2] == 3.0);
            REQUIRE(row(0, 2) == 3.0);
            const math::matrix<double, 3, 1> column_again = transpose(row);
            REQUIRE(column_again == column);
            const math::matrix<double, 1, 1> product = row * column;
            REQUIRE(product[0] == 14.0);
            const math::matrix<double, 3, 3> outer = column * row;
            REQUIRE(outer[2][1] == 6.0);
            const math::matrix<float, 1, 1> single = { { 5.0f } };
            REQUIRE(transpose(single)[0] == 5.0f);
            const math::matrix<int, 1, 2> integer_row = transpose(math::matrix<int, 2, 1>({ { 7, 8 } }));
            REQUIRE(integer_row(0, 1) == 8);
            math::matrix<double, 0, 0> dynamic_column(3, 1);
            dynamic_column(0, 0) = 1.0;
            dynamic_column(1, 0) = 2.0;
            dynamic_column(2, 0) = 3.0;
            const math::matrix<double, 0, 0> dynamic_row = transpose(dynamic_column);
            REQUIRE(dynamic_row.rows() == 1);
            REQUIRE(dynamic_row.cols() == 3);
            REQUIRE(dynamic_row(0, 1) == 2.0);
        }
    }

    {
        {
            math::matrix<double, 1, 1> m = { { 0.1 } };
            math::matrix<double, 1, 1> inverse = { { 10.0 } };
            math::matrix<double, 1, 1> result;
            REQUIRE(invert(m, result));
            REQUIRE(are_values_approx(result.data(), inverse.data(), 1 * 1, 1e-6));
        }
        {
            math::matrix<double, 2, 2> m = { { { 1.0, 2.0 }, { 3.0, 4.0 } } };
            math::matrix<double, 2, 2> inverse = { { { -2.0, +1.0 }, { +1.5, -0.5 } } };
            math::matrix<double, 2, 2> result;
            REQUIRE(invert(m, result));
            REQUIRE(are_values_approx(result.data(), inverse.data(), 2 * 2, 1e-6));
        }
        {
            math::matrix<double, 3, 3> m = {
                { { 1.0, 2.0, 0.0 }, { 0.0, 1.0, 2.0 }, { 2.0, 0.0, 1.0 } }
            };
            math::matrix<double, 3, 3> inverse = { { { +1.0 / 9.0, -2.0 / 9.0, +4.0 / 9.0 }, { +4.0 / 9.0, +1.0 / 9.0, -2.0 / 9.0 }, { -2.0 / 9.0, +4.0 / 9.0, +1.0 / 9.0 } } };
            math::matrix<double, 3, 3> result;
            REQUIRE(invert(m, result));
            REQUIRE(are_values_approx(result.data(), inverse.data(), 3 * 3, 1e-6));
        }
        {
            math::matrix<double, 4, 4> m = { { { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0, 1.0 } } };
            math::matrix<double, 4, 4> inverse = { { { -1.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0, 0.0 } } };
            math::matrix<double, 4, 4> result;
            REQUIRE(invert(m, result));
            REQUIRE(are_values_approx(result.data(), inverse.data(), 4 * 4, 1e-6));
        }
        {
            math::matrix<double, 5, 5> m = { { { 2.0, 12.0, 5.0, 12.0, 14.0 }, { 16.0, 0.0, 12.0, 16.0, 16.0 }, { 10.0, 14.0, 15.0, 10.0, 11.0 }, { 14.0, 1.0, 1.0, 18.0, 0.0 }, { 9.0, 13.0, 1.0, 6.0, 6.0 } } };
            math::matrix<double, 5, 5> inverse = {
                { { -0.0717574493, 0.0417861176, -0.0144209886, -0.0087742159, 0.0824428807 },
                  { 0.0102365852, -0.0419779631, 0.0274809602, 0.0026641183, 0.0376741088 },
                  { -0.0234847826, -0.0129586192, 0.0897112906, 0.0023745402, -0.0751165551 },
                  { 0.0565473604, -0.0294482813, 0.0047056438, 0.0621000202, -0.0620421046 },
                  { 0.0328236759, 0.0598811281, -0.0575681232, -0.0551067095, 0.0359366403 } }
            };
            math::matrix<double, 5, 5> result;
            REQUIRE(invert(m, result));
            REQUIRE(are_values_approx(result.data(), inverse.data(), 5 * 5, 1e-6));
        }
        // Un-invertible
        {
            math::matrix<double, 1, 1> m = { { 0.0 } };
            math::matrix<double, 1, 1> result;
            REQUIRE(!invert(m, result));
            REQUIRE(are_values_approx(result.data(), math::matrix<double, 1, 1>::zero().data(), 1 * 1, 1e-6));
        }
        {
            math::matrix<double, 2, 2> m = { { { 0.0, 0.0 }, { 0.0, 1.0 } } };
            math::matrix<double, 2, 2> result;
            REQUIRE(!invert(m, result));
            REQUIRE(are_values_approx(result.data(), math::matrix<double, 2, 2>::zero().data(), 2 * 2, 1e-6));
        }
        {
            math::matrix<double, 3, 3> m = {
                { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } }
            };
            math::matrix<double, 3, 3> result;
            REQUIRE(!invert(m, result));
            REQUIRE(are_values_approx(result.data(), math::matrix<double, 3, 3>::zero().data(), 3 * 3, 1e-6));
        }
        {
            math::matrix<double, 4, 4> m = { { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 }, { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 } } };
            math::matrix<double, 4, 4> result;
            REQUIRE(!invert(m, result));
            REQUIRE(are_values_approx(result.data(), math::matrix<double, 4, 4>::zero().data(), 4 * 4, 1e-6));
        }
        // Small-scale but well-conditioned matrices must still be invertible.
        {
            math::matrix<double, 3, 3> m = { { { 1e-3, 0.0, 0.0 }, { 0.0, 1e-3, 0.0 }, { 0.0, 0.0, 1e-3 } } };
            math::matrix<double, 3, 3> inverse = { { { 1e3, 0.0, 0.0 }, { 0.0, 1e3, 0.0 }, { 0.0, 0.0, 1e3 } } };
            math::matrix<double, 3, 3> result;
            REQUIRE(invert(m, result));
            REQUIRE(are_values_approx(result.data(), inverse.data(), 3 * 3, 1e-6));
        }
        {
            math::matrix<double, 4, 4> m = { { { 1e-3, 0.0, 0.0, 0.0 }, { 0.0, 1e-3, 0.0, 0.0 }, { 0.0, 0.0, 1e-3, 0.0 }, { 0.0, 0.0, 0.0, 1e-3 } } };
            math::matrix<double, 4, 4> inverse = { { { 1e3, 0.0, 0.0, 0.0 }, { 0.0, 1e3, 0.0, 0.0 }, { 0.0, 0.0, 1e3, 0.0 }, { 0.0, 0.0, 0.0, 1e3 } } };
            math::matrix<double, 4, 4> result;
            REQUIRE(invert(m, result));
            REQUIRE(are_values_approx(result.data(), inverse.data(), 4 * 4, 1e-6));
        }
        // A rank deficiency that only appears at the final pivot must still return the zero matrix, the last row is the sum of the first three.
        {
            math::matrix<double, 4, 4> m = { { { 2.0, 0.0, 0.0, 1.0 }, { 0.0, 2.0, 0.0, 1.0 }, { 0.0, 0.0, 2.0, 1.0 }, { 2.0, 2.0, 2.0, 3.0 } } };
            math::matrix<double, 4, 4> result;
            REQUIRE(!invert(m, result));
            REQUIRE(are_values_approx(result.data(), math::matrix<double, 4, 4>::zero().data(), 4 * 4, 1e-6));
        }
        // The singularity test is independent of the scale of the rows and the columns.
        {
            core::random_pcg rng;
            check_scaled_inverse<2>(rng, 8.0);
            check_scaled_inverse<3>(rng, 8.0);
            check_scaled_inverse<4>(rng, 8.0);
            check_scaled_inverse<6>(rng, 8.0);
        }
        // Graded diagonal matrices are exactly invertible.
        {
            const math::matrix<double, 2, 2> m2 = { { { 1e-10, 0.0 }, { 0.0, 1e10 } } };
            const math::matrix<double, 3, 3> m3 = { { { 1e-10, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1e10 } } };
            const math::matrix<double, 4, 4> m4 = { { { 1e-10, 0.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 }, { 0.0, 0.0, 1e10, 0.0 }, { 0.0, 0.0, 0.0, 1e-20 } } };
            math::matrix<double, 2, 2> r2;
            math::matrix<double, 3, 3> r3;
            math::matrix<double, 4, 4> r4;
            REQUIRE(invert(m2, r2));
            REQUIRE(invert(m3, r3));
            REQUIRE(invert(m4, r4));
            REQUIRE(r2[0][0] == 1e10);
            REQUIRE(r2[1][1] == 1e-10);
            REQUIRE(r3[0][0] == 1e10);
            REQUIRE(r3[1][1] == 1.0);
            REQUIRE(r3[2][2] == 1e-10);
            REQUIRE(r4[0][0] == 1e10);
            REQUIRE(r4[1][1] == 1.0);
            REQUIRE(r4[2][2] == 1e-10);
            REQUIRE(r4[3][3] == 1e20);
        }
        // The normal matrix of an affine fit to uncentred pixel positions, 50 points in a 40 pixel window at (620, 460).
        {
            core::random_pcg rng;
            math::matrix<double, 3, 3> normal = math::matrix<double, 3, 3>::zero();
            math::matrix<double, 3, 1> rhs = math::matrix<double, 3, 1>::zero();
            const double parameters[3] = { 1.01, 0.02, 3.5 };
            for (int i = 0; i < 50; ++i) {
                const double x = 600.0 + (40.0 * rng.get_random_exclusive_top());
                const double y = 440.0 + (40.0 * rng.get_random_exclusive_top());
                const double weight = 0.5 + (0.5 * rng.get_random_exclusive_top());
                const double phi[3] = { x, y, 1.0 };
                const double target = (parameters[0] * x) + (parameters[1] * y) + parameters[2];
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t column = 0; column < 3; ++column) {
                        normal[row][column] += weight * phi[row] * phi[column];
                    }
                    rhs[row] += weight * phi[row] * target;
                }
            }
            math::matrix<double, 3, 3> normal_inverse;
            REQUIRE(invert(normal, normal_inverse));
            const math::matrix<double, 3, 1> solution = normal_inverse * rhs;
            REQUIRE(std::abs(solution[0] - parameters[0]) < 1e-6);
            REQUIRE(std::abs(solution[1] - parameters[1]) < 1e-6);
            REQUIRE(std::abs(solution[2] - parameters[2]) < 1e-3);
        }
        // Non-finite entries are never invertible.
        {
            const double non_finite_values[3] = { HUGE_VAL, -HUGE_VAL, std::nan("") };
            for (const double non_finite_value : non_finite_values) {
                for (size_t index = 0; index < 16; ++index) {
                    math::matrix<double, 2, 2> m2 = { { { 2.0, 1.0 }, { 1.0, 3.0 } } };
                    math::matrix<double, 3, 3> m3 = { { { 2.0, 1.0, 0.0 }, { 1.0, 3.0, 1.0 }, { 0.0, 1.0, 4.0 } } };
                    math::matrix<double, 4, 4> m4 = math::matrix<double, 4, 4>::identity();
                    m2.data()[index % 4] = non_finite_value;
                    m3.data()[index % 9] = non_finite_value;
                    m4.data()[index] = non_finite_value;
                    math::matrix<double, 2, 2> r2;
                    math::matrix<double, 3, 3> r3;
                    math::matrix<double, 4, 4> r4;
                    REQUIRE(!invert(m2, r2));
                    REQUIRE(!invert(m3, r3));
                    REQUIRE(!invert(m4, r4));
                    REQUIRE(r4 == (math::matrix<double, 4, 4>::zero()));
                }
            }
        }
    }

    {
        {
            math::matrix<double, 3, 1> m{ { 0, 2, 4 } };
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
            math::matrix<double, 1, 3> m = { { 0, 1, 2 } };
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
            math::matrix<double, 3, 2> m{ { { 0, 1 }, { 2, 3 }, { 4, 5 } } };
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
        math::matrix<double, 2, 2> m1 = { { { 1, 2 }, { 3, 4 } } };
        math::matrix<double, 2, 2> m2 = { { { 1, 2 }, { 3, 4 } } };
        math::matrix<double, 2, 2> m3 = { { { 1, 2 }, { 3, 5 } } };
        REQUIRE((m1 == m2) == true);
        REQUIRE((m1 == m3) == false);
        REQUIRE((m1 != m2) == false);
        REQUIRE((m1 != m3) == true);
    }

    {
        math::matrix<double, 2, 2> m1 = { { { 1, 2 }, { 3, 4 } } };
        math::matrix<double, 2, 2> m2 = { { { -1, -2 }, { -3, -4 } } };
        REQUIRE(m1 == -m2);
        REQUIRE(m1 != +m2);
        REQUIRE(m1 == +m1);
        REQUIRE(m2 != -m2);
    }

    {
        math::matrix<double, 2, 2> m1 = { { { 1.0, 2.0 }, { 3.0, 4.0 } } };
        math::matrix<double, 2, 2> m2 = { { { -1.0, -2.0 }, { -3.0, -40.0 } } };
        math::matrix<double, 2, 2> result1 = m1 + m2;
        REQUIRE(result1[0][0] == 0.0);
        REQUIRE(result1[0][1] == 0.0);
        REQUIRE(result1[1][0] == 0.0);
        REQUIRE(result1[1][1] == -36.0);
        math::matrix<double, 2, 2> result2 = m2 + 7.0;
        REQUIRE(result2[0][0] == 6.0);
        REQUIRE(result2[0][1] == 5.0);
        REQUIRE(result2[1][0] == 4.0);
        REQUIRE(result2[1][1] == -33.0);
        math::matrix<double, 2, 2> result3 = 7.0 + m2;
        REQUIRE(result3[0][0] == 6.0);
        REQUIRE(result3[0][1] == 5.0);
        REQUIRE(result3[1][0] == 4.0);
        REQUIRE(result3[1][1] == -33.0);
    }

    {
        math::matrix<double, 2, 2> m1 = { { { 1.0, 2.0 }, { 3.0, 4.0 } } };
        math::matrix<double, 2, 2> m2 = { { { -1.0, -2.0 }, { -3.0, -40.0 } } };
        math::matrix<double, 2, 2> result1 = m1 - m2;
        REQUIRE(result1[0][0] == 2.0);
        REQUIRE(result1[0][1] == 4.0);
        REQUIRE(result1[1][0] == 6.0);
        REQUIRE(result1[1][1] == 44.0);
        math::matrix<double, 2, 2> result2 = m2 - 7.0;
        REQUIRE(result2[0][0] == -8.0);
        REQUIRE(result2[0][1] == -9.0);
        REQUIRE(result2[1][0] == -10.0);
        REQUIRE(result2[1][1] == -47.0);
        math::matrix<double, 2, 2> m3 = { { { 1.0, 2.0 }, { 3.0, 4.0 } } };
        math::matrix<double, 2, 2> result3 = 10.0 - m3;
        REQUIRE(result3[0][0] == 9.0);
        REQUIRE(result3[0][1] == 8.0);
        REQUIRE(result3[1][0] == 7.0);
        REQUIRE(result3[1][1] == 6.0);
    }

    {
        {
            math::matrix<double, 2, 2> m1 = { { { 1.0, 2.0 }, { 3.0, 4.0 } } };
            math::matrix<double, 2, 2> m2 = { { { -1.0, -2.0 }, { -3.0, -40.0 } } };
            math::matrix<double, 2, 2> result1 = m1 * m2;
            REQUIRE(result1[0][0] == -7.0);
            REQUIRE(result1[0][1] == -82.0);
            REQUIRE(result1[1][0] == -15.0);
            REQUIRE(result1[1][1] == -166.0);
            math::matrix<double, 2, 2> result2 = m2 * 7.0;
            REQUIRE(result2[0][0] == -7.0);
            REQUIRE(result2[0][1] == -14.0);
            REQUIRE(result2[1][0] == -21.0);
            REQUIRE(result2[1][1] == -280.0);
            math::matrix<double, 2, 2> result3 = 7.0 * m2;
            REQUIRE(result3[0][0] == -7.0);
            REQUIRE(result3[0][1] == -14.0);
            REQUIRE(result3[1][0] == -21.0);
            REQUIRE(result3[1][1] == -280.0);
        }
        {
            math::matrix<double, 3, 2> m;
            m[0][0] = 1;
            m[0][1] = 2;
            m[1][0] = 3;
            m[1][1] = 4;
            m[2][0] = 5;
            m[2][1] = 6;
            math::matrix<double, 2, 2> result = transpose(m) * m;
            REQUIRE(result[0][0] == 35);
            REQUIRE(result[0][1] == 44);
            REQUIRE(result[1][0] == 44);
            REQUIRE(result[1][1] == 56);
        }
    }

    {
        core::random_pcg rng;

        for (int trial = 0; trial < 200; ++trial) {
            const double b00 = random_signed(rng);
            const double b01 = random_signed(rng);
            const double b10 = random_signed(rng);
            const double b11 = ((trial % 5) == 0) ? 0.0 : random_signed(rng);
            const math::matrix<double, 2, 2> b = { { { b00, b01 }, { b10, b11 } } };
            const math::matrix<double, 2, 2> w = transpose(b) * b;
            math::matrix<double, 2, 2> s;
            REQUIRE(math::sqrt_symmetric_2x2(w, s));
            REQUIRE(s[0][1] == s[1][0]);
            const math::matrix<double, 2, 2> reconstructed = s * s;
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    REQUIRE(is_value_approx(reconstructed[i][j], w[i][j], 1e-13));
                }
            }
        }

        {
            const math::matrix<double, 2, 2> w = { { { 0.25, 0.0 }, { 0.0, 0.25 } } };
            math::matrix<double, 2, 2> s;
            REQUIRE(math::sqrt_symmetric_2x2(w, s));
            REQUIRE(is_value_approx(s[0][0], 0.5, 1e-15));
            REQUIRE(is_value_approx(s[1][1], 0.5, 1e-15));
            REQUIRE(s[0][1] == 0.0);
            REQUIRE(s[1][0] == 0.0);
        }

        {
            const math::matrix<double, 2, 2> w = math::matrix<double, 2, 2>::zero();
            math::matrix<double, 2, 2> s = math::matrix<double, 2, 2>::identity();
            REQUIRE(math::sqrt_symmetric_2x2(w, s));
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    REQUIRE(s[i][j] == 0.0);
                }
            }
        }

        {
            const math::matrix<double, 2, 2> w = { { { 4.0, 2.0 }, { 2.0, 1.0 } } };
            math::matrix<double, 2, 2> s;
            REQUIRE(math::sqrt_symmetric_2x2(w, s));
            const math::matrix<double, 2, 2> reconstructed = s * s;
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    REQUIRE(is_value_approx(reconstructed[i][j], w[i][j], 1e-13));
                }
            }
        }

        {
            const double rejected[][4] = {
                { 1.0, 0.0, 0.0, -1.0 },
                { -1.0, 0.0, 0.0, -1.0 },
                { 1.0, 2.0, 2.0, 1.0 },
                { 1.0, 1.0, -1.0, 1.0 },
                { 0.0, 1.0, 1.0, 0.0 }
            };
            for (const auto& entries : rejected) {
                const math::matrix<double, 2, 2> w = { { { entries[0], entries[1] }, { entries[2], entries[3] } } };
                math::matrix<double, 2, 2> s = math::matrix<double, 2, 2>::identity();
                REQUIRE(!math::sqrt_symmetric_2x2(w, s));
                for (size_t i = 0; i < 2; ++i) {
                    for (size_t j = 0; j < 2; ++j) {
                        REQUIRE(s[i][j] == 0.0);
                    }
                }
            }
        }

        {
            const math::matrix<float, 2, 2> w = { { { 5.0f, 2.0f }, { 2.0f, 3.0f } } };
            math::matrix<float, 2, 2> s;
            REQUIRE(math::sqrt_symmetric_2x2(w, s));
            const math::matrix<float, 2, 2> reconstructed = s * s;
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    REQUIRE(is_value_approx(static_cast<double>(reconstructed[i][j]), static_cast<double>(w[i][j]), 1e-6));
                }
            }
        }
    }

    return EXIT_SUCCESS;
}
