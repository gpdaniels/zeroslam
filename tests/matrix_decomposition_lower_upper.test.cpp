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

#include "matrix_decomposition_lower_upper.hpp"

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

static inline bool is_value_equal(double lhs, double rhs) {
    return lhs == rhs;
}

void matrix_multiply(const double* lhs, int lhs_width, int lhs_height, const double* rhs, int rhs_width, int rhs_height, double* result);

void matrix_multiply(const double* lhs, int lhs_width, int lhs_height, const double* rhs, int rhs_width, int rhs_height, double* result) {
    REQUIRE(lhs_width == rhs_height);
    for (int lhs_y = 0; lhs_y < lhs_height; ++lhs_y) {
        for (int rhs_x = 0; rhs_x < rhs_width; ++rhs_x) {
            double sum = 0;
            for (int lhs_x_rhs_y = 0; lhs_x_rhs_y < lhs_width; ++lhs_x_rhs_y) {
                sum += lhs[lhs_y * lhs_width + lhs_x_rhs_y] * rhs[lhs_x_rhs_y * rhs_width + rhs_x];
            }
            result[lhs_y * rhs_width + rhs_x] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        using test_type = double;
        constexpr static const int width = 2;
        constexpr static const int height = 2;

        const test_type matrix[height][width] = {
            { 4.0, 3.0 },
            { 6.0, 3.0 }
        };

        test_type L1[height][height];
        test_type U1[height][width];
        test_type P1[height][height];
        int swaps;
        REQUIRE(matrix::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps));

        test_type PA[height][width];
        matrix_multiply(&P1[0][0], height, height, &matrix[0][0], width, height, &PA[0][0]);

        test_type LU[height][width];
        matrix_multiply(&L1[0][0], height, height, &U1[0][0], width, height, &LU[0][0]);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_equal(PA[i][j], LU[i][j]));
            }
        }
    }

    {
        using test_type = double;
        constexpr static const int width = 2;
        constexpr static const int height = 3;

        const test_type matrix[height][width] = {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        };

        test_type L1[height][height];
        test_type U1[height][width];
        test_type P1[height][height];
        int swaps;
        REQUIRE(matrix::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps));

        test_type PA[height][width];
        matrix_multiply(&P1[0][0], height, height, &matrix[0][0], width, height, &PA[0][0]);

        test_type LU[height][width];
        matrix_multiply(&L1[0][0], height, height, &U1[0][0], width, height, &LU[0][0]);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_equal(PA[i][j], LU[i][j]));
            }
        }
    }

    {
        using test_type = double;
        constexpr static const int width = 4;
        constexpr static const int height = 2;

        const test_type matrix[height][width] = {
            { 1.0, 2.0, 3.0, 4.0 },
            { 4.0, 5.0, 6.0, 7.0 }
        };

        test_type L1[height][height];
        test_type U1[height][width];
        test_type P1[height][height];
        int swaps;
        REQUIRE(matrix::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps));

        test_type PA[height][width];
        matrix_multiply(&P1[0][0], height, height, &matrix[0][0], width, height, &PA[0][0]);

        test_type LU[height][width];
        matrix_multiply(&L1[0][0], height, height, &U1[0][0], width, height, &LU[0][0]);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_equal(PA[i][j], LU[i][j]));
            }
        }
    }

    {
        using test_type = double;
        constexpr static const int width = 5;
        constexpr static const int height = 3;

        const test_type matrix[height][width] = {
            { 1.0, 2.0, 3.0, 4.0, 5.0 },
            { 6.0, 7.0, 8.0, 9.0, 10.0 },
            { 2.0, 1.0, 4.0, 3.0, 6.0 }
        };

        test_type L1[height][height];
        test_type U1[height][width];
        test_type P1[height][height];
        int swaps;
        REQUIRE(matrix::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps));

        test_type PA[height][width];
        matrix_multiply(&P1[0][0], height, height, &matrix[0][0], width, height, &PA[0][0]);

        test_type LU[height][width];
        matrix_multiply(&L1[0][0], height, height, &U1[0][0], width, height, &LU[0][0]);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_approx(PA[i][j], LU[i][j], 1e-9));
            }
        }
    }

    {
        const float A[4][4] = {
            { -0.0f, 1.0f, 1.0f, -0.0f },
            { 2.0f, 2.0f, -2.0f, 3.0f },
            { 1.0f, 2.0f, -3.0f, 4.0f },
            { 1.0f, -1.0f, 1.0f, -2.0f }
        };

        const float B[4] = {
            -1.0f,
            10.0f,
            12.0f,
            -4.0f
        };

        float result[4];
        REQUIRE(matrix::solve_lower_upper<float>(&A[0][0], &B[0], 4, 4, &result[0]));

        REQUIRE(is_value_approx(result[0], 1.0f, 1e-6f));
        REQUIRE(is_value_approx(result[1], 0.0f, 1e-6f));
        REQUIRE(is_value_approx(result[2], -1.0f, 1e-6f));
        REQUIRE(is_value_approx(result[3], 2.0f, 1e-6f));

        float A2[2][2] = {
            { 0.0f, 2.0f },
            { 0.0f, 1.0f },
        };

        float B2[2] = {
            8.0f,
            4.0f,
        };

        float result2[2];
        REQUIRE(matrix::solve_lower_upper<float>(&A2[0][0], &B2[0], 2, 2, &result2[0]) == false);

        float A3[2][2] = {
            { 1.0f, 2.0f },
            { 3.0f, 1.0f },
        };

        float B3[2] = {
            8.0f,
            4.0f,
        };

        float result3[2];
        REQUIRE(matrix::solve_lower_upper<float>(&A3[0][0], &B3[0], 2, 2, &result3[0]));

        REQUIRE(is_value_approx(result3[0], 0.0f, 1e-6f));
        REQUIRE(is_value_approx(result3[1], 4.0f, 1e-6f));
    }

    {
        constexpr static const auto permute_via_multiply = [](const float* p, const float* a, int width, int height, float* result) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    float sum = 0;
                    for (int k = 0; k < height; ++k) {
                        sum += p[y * height + k] * a[k * width + x];
                    }
                    result[y * width + x] = sum;
                }
            }
        };
        constexpr static const auto permute_via_row_swap = [](const float* p, const float* a, int width, int height, float* result) {
            for (int y = 0; y < height; ++y) {
                // Each row of a permutation matrix has exactly one element equal to one.
                for (int x = 0; x < height; ++x) {
                    if (p[y * height + x] == 1.0f) {
                        for (int c = 0; c < width; ++c) {
                            result[y * width + c] = a[x * width + c];
                        }
                        break;
                    }
                }
            }
        };

        // A permutation matrix built by swapping rows 0 and 2 of the identity.
        const float p[3][3] = {
            { 0.0f, 0.0f, 1.0f },
            { 0.0f, 1.0f, 0.0f },
            { 1.0f, 0.0f, 0.0f }
        };

        // Case 1: an ordinary matrix with no exact zero elements.
        {
            const float a[3][3] = {
                { 1.5f, -2.25f, 3.125f },
                { -4.0f, 5.0f, -6.5f },
                { 7.0f, -8.0f, 9.0f }
            };
            float via_multiply[3][3];
            float via_row_swap[3][3];
            permute_via_multiply(&p[0][0], &a[0][0], 3, 3, &via_multiply[0][0]);
            permute_via_row_swap(&p[0][0], &a[0][0], 3, 3, &via_row_swap[0][0]);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    REQUIRE(is_value_equal(via_multiply[i][j], via_row_swap[i][j]));
                }
            }
        }

        // Case 2: a matrix containing a signed negative zero being permuted into a new row.
        {
            const float a[3][3] = {
                { -0.0f, 1.0f, 2.0f },
                { 3.0f, 4.0f, 5.0f },
                { 6.0f, 7.0f, 8.0f }
            };
            float via_multiply[3][3];
            float via_row_swap[3][3];
            permute_via_multiply(&p[0][0], &a[0][0], 3, 3, &via_multiply[0][0]);
            permute_via_row_swap(&p[0][0], &a[0][0], 3, 3, &via_row_swap[0][0]);
            // Row 2 of the permuted result comes from row 0 of a, i.e. the row containing -0.0.
            REQUIRE(std::signbit(via_row_swap[2][0]));         // Row swap preserves the sign of -0.0.
            REQUIRE(!std::signbit(via_multiply[2][0]));        // Dense multiply loses it: (+0) + 1*(-0.0) == +0.
            REQUIRE(via_row_swap[2][0] == via_multiply[2][0]); // Numerically equal despite the sign difference.
            // All other (non-zero) elements are unaffected and remain bit-identical.
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    if (i == 2 && j == 0) {
                        continue;
                    }
                    REQUIRE(is_value_equal(via_multiply[i][j], via_row_swap[i][j]));
                }
            }
        }
    }

    // This particular matrix is invertible but the previously buggy pivot incorrectly report failure.
    {
        using test_type = double;
        constexpr static const int width = 3;
        constexpr static const int height = 3;

        const test_type matrix[height][width] = {
            { 1.0, 1.0, 1.0 },
            { 2.0, 2.0, 1.0 },
            { 1.0, 2.0, 1.0 }
        };

        test_type L1[height][height];
        test_type U1[height][width];
        test_type P1[height][height];
        int swaps;
        REQUIRE(matrix::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps));

        test_type PA[height][width];
        matrix_multiply(&P1[0][0], height, height, &matrix[0][0], width, height, &PA[0][0]);

        test_type LU[height][width];
        matrix_multiply(&L1[0][0], height, height, &U1[0][0], width, height, &LU[0][0]);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_approx(PA[i][j], LU[i][j], 1e-9));
            }
        }

        // Each row and column of the permutation matrix must have exactly one entry equal to one
        // (and the rest zero): otherwise it is not a valid permutation.
        for (int i = 0; i < height; ++i) {
            int row_ones = 0;
            int col_ones = 0;
            for (int j = 0; j < height; ++j) {
                row_ones += (P1[i][j] == test_type(1)) ? 1 : 0;
                col_ones += (P1[j][i] == test_type(1)) ? 1 : 0;
            }
            REQUIRE(row_ones == 1);
            REQUIRE(col_ones == 1);
        }

        const test_type rhs[height] = { 1.0, 2.0, 3.0 };
        test_type solution[height];
        REQUIRE(matrix::solve_lower_upper<test_type>(&matrix[0][0], &rhs[0], width, height, &solution[0]));

        // Verify matrix * solution == rhs.
        for (int i = 0; i < height; ++i) {
            test_type sum = 0;
            for (int j = 0; j < width; ++j) {
                sum += matrix[i][j] * solution[j];
            }
            REQUIRE(is_value_approx(sum, rhs[i], 1e-9));
        }
    }

    {
        class random_pcg final {
        private:
            unsigned long long int state = 0x853C49E6748FEA9Bull;
            unsigned long long int increment = 0xDA3E39CB94B95BDBull;

        private:
            unsigned int get_random_raw() {
                unsigned long long int state_previous = this->state;
                this->state = state_previous * 0x5851F42D4C957F2Dull + this->increment;
                unsigned int state_shift_xor_shift = static_cast<unsigned int>(((state_previous >> 18u) ^ state_previous) >> 27u);
                int rotation = state_previous >> 59u;
                return (state_shift_xor_shift >> rotation) | (state_shift_xor_shift << ((-rotation) & 31));
            }

        public:
            double get_random_exclusive_top() {
                return static_cast<double>(this->get_random_raw()) * (1.0 / static_cast<double>(1ull << 32));
            }
        };

        random_pcg rng;

        constexpr static const int trial_count = 5000;
        constexpr static const int max_size = 8;

        for (int trial = 0; trial < trial_count; ++trial) {
            const int n = 1 + static_cast<int>(rng.get_random_exclusive_top() * max_size);

            double L[max_size * max_size] = {};
            double U[max_size * max_size] = {};
            int permutation[max_size];

            for (int i = 0; i < n; ++i) {
                permutation[i] = i;
                L[i * n + i] = 1.0;
                for (int j = 0; j < i; ++j) {
                    L[i * n + j] = (4.0 * rng.get_random_exclusive_top()) - 2.0;
                }
                // Diagonal magnitude bounded away from zero, so det(U) (and hence det(A)) is known to
                // be non-zero regardless of what the random off-diagonal entries happen to be.
                const double sign = (rng.get_random_exclusive_top() < 0.5) ? -1.0 : 1.0;
                U[i * n + i] = sign * (0.5 + (2.5 * rng.get_random_exclusive_top()));
                for (int j = i + 1; j < n; ++j) {
                    U[i * n + j] = (4.0 * rng.get_random_exclusive_top()) - 2.0;
                }
            }

            // Fisher-Yates shuffle of the row permutation.
            for (int i = n - 1; i > 0; --i) {
                const int j = static_cast<int>(rng.get_random_exclusive_top() * (i + 1));
                const int temp = permutation[i];
                permutation[i] = permutation[j];
                permutation[j] = temp;
            }

            double LU_temp[max_size * max_size];
            matrix_multiply(&L[0], n, n, &U[0], n, n, &LU_temp[0]);

            double A[max_size * max_size];
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    A[i * n + j] = LU_temp[permutation[i] * n + j];
                }
            }

            double matrix_l[max_size * max_size];
            double matrix_u[max_size * max_size];
            double matrix_p[max_size * max_size];
            int swaps;
            REQUIRE(matrix::decompose_lower_upper<double>(&A[0], n, n, &matrix_l[0], &matrix_u[0], &matrix_p[0], &swaps));

            // Verify the returned permutation is a valid permutation matrix: exactly one entry equal
            // to one in every row and every column.
            for (int i = 0; i < n; ++i) {
                int row_ones = 0;
                int col_ones = 0;
                for (int j = 0; j < n; ++j) {
                    row_ones += (matrix_p[i * n + j] == 1.0) ? 1 : 0;
                    col_ones += (matrix_p[j * n + i] == 1.0) ? 1 : 0;
                }
                REQUIRE(row_ones == 1);
                REQUIRE(col_ones == 1);
            }

            // Verify P * A == L * U to tight tolerance.
            double PA[max_size * max_size];
            double LU[max_size * max_size];
            matrix_multiply(&matrix_p[0], n, n, &A[0], n, n, &PA[0]);
            matrix_multiply(&matrix_l[0], n, n, &matrix_u[0], n, n, &LU[0]);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    REQUIRE(is_value_approx(PA[i * n + j], LU[i * n + j], 1e-9));
                }
            }

            // Verify solve_lower_upper succeeds and actually solves the system.
            double rhs[max_size];
            for (int i = 0; i < n; ++i) {
                rhs[i] = (4.0 * rng.get_random_exclusive_top()) - 2.0;
            }
            double solution[max_size];
            REQUIRE(matrix::solve_lower_upper<double>(&matrix_l[0], &matrix_u[0], &matrix_p[0], &rhs[0], n, n, &solution[0]));

            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j) {
                    sum += A[i * n + j] * solution[j];
                }
                REQUIRE(is_value_approx(sum, rhs[i], 1e-8));
            }
        }
    }

    {
        using test_type = double;
        constexpr static const int width = 3;
        constexpr static const int height = 3;

        const test_type matrix[height][width] = {
            { 2.0, 1.0, 1.0 },
            { 2.0, 1.0, 3.0 },
            { 1.0, 3.0, 2.0 }
        };

        test_type L1[height][height];
        test_type U1[height][width];
        test_type P1[height][height];
        int swaps;
        REQUIRE(matrix::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps));

        // A pivot swap that only elimination fill-in reveals must actually have happened.
        REQUIRE(swaps >= 1);

        // The returned permutation must be a valid permutation matrix.
        for (int i = 0; i < height; ++i) {
            int row_ones = 0;
            int col_ones = 0;
            for (int j = 0; j < height; ++j) {
                row_ones += (P1[i][j] == test_type(1)) ? 1 : 0;
                col_ones += (P1[j][i] == test_type(1)) ? 1 : 0;
            }
            REQUIRE(row_ones == 1);
            REQUIRE(col_ones == 1);
        }

        // Reconstruct P * A == L * U.
        test_type PA[height][width];
        matrix_multiply(&P1[0][0], height, height, &matrix[0][0], width, height, &PA[0][0]);
        test_type LU[height][width];
        matrix_multiply(&L1[0][0], height, height, &U1[0][0], width, height, &LU[0][0]);
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_approx(PA[i][j], LU[i][j], 1e-9));
            }
        }

        // Solve A * x == b for a known x = { 1, 2, 3 }, so b = { 7, 13, 13 }, and recover x.
        const test_type rhs[height] = { 7.0, 13.0, 13.0 };
        test_type solution[height];
        REQUIRE(matrix::solve_lower_upper<test_type>(&matrix[0][0], &rhs[0], width, height, &solution[0]));
        REQUIRE(is_value_approx(solution[0], 1.0, 1e-9));
        REQUIRE(is_value_approx(solution[1], 2.0, 1e-9));
        REQUIRE(is_value_approx(solution[2], 3.0, 1e-9));

        // And, independently, verify A * solution == rhs.
        for (int i = 0; i < height; ++i) {
            test_type sum = 0;
            for (int j = 0; j < width; ++j) {
                sum += matrix[i][j] * solution[j];
            }
            REQUIRE(is_value_approx(sum, rhs[i], 1e-9));
        }
    }

    {
        using test_type = double;
        constexpr static const int width = 3;
        constexpr static const int height = 3;

        const test_type matrix[height][width] = {
            { 1.0, 2.0, 3.0 },
            { 2.0, 4.0, 6.0 },
            { 1.0, 1.0, 1.0 }
        };

        test_type L1[height][height];
        test_type U1[height][width];
        test_type P1[height][height];
        int swaps;
        REQUIRE(matrix::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps) == false);

        const test_type rhs[height] = { 1.0, 2.0, 3.0 };
        test_type solution[height];
        REQUIRE(matrix::solve_lower_upper<test_type>(&matrix[0][0], &rhs[0], width, height, &solution[0]) == false);
    }

    return EXIT_SUCCESS;
}