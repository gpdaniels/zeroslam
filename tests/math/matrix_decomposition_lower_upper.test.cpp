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

#include "math/matrix_decomposition_lower_upper.hpp"

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

static inline bool is_value_approx(float lhs, float rhs, double epsilon = 1e-8) {
    return is_value_approx(static_cast<double>(lhs), static_cast<double>(rhs), epsilon);
}

static inline bool is_value_equal(double lhs, double rhs) {
    return lhs == rhs;
}

static inline bool is_value_equal(float lhs, float rhs) {
    return is_value_equal(static_cast<double>(lhs), static_cast<double>(rhs));
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
        REQUIRE(math::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps));

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
        REQUIRE(math::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps));

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
        REQUIRE(math::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps));

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
        REQUIRE(math::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps));

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
        REQUIRE(math::solve_lower_upper<float>(&A[0][0], &B[0], 4, 4, &result[0]));

        REQUIRE(is_value_approx(result[0], 1.0f, 1e-6));
        REQUIRE(is_value_approx(result[1], 0.0f, 1e-6));
        REQUIRE(is_value_approx(result[2], -1.0f, 1e-6));
        REQUIRE(is_value_approx(result[3], 2.0f, 1e-6));

        float A2[2][2] = {
            { 0.0f, 2.0f },
            { 0.0f, 1.0f },
        };

        float B2[2] = {
            8.0f,
            4.0f,
        };

        float result2[2];
        REQUIRE(math::solve_lower_upper<float>(&A2[0][0], &B2[0], 2, 2, &result2[0]) == false);

        float A3[2][2] = {
            { 1.0f, 2.0f },
            { 3.0f, 1.0f },
        };

        float B3[2] = {
            8.0f,
            4.0f,
        };

        float result3[2];
        REQUIRE(math::solve_lower_upper<float>(&A3[0][0], &B3[0], 2, 2, &result3[0]));

        REQUIRE(is_value_approx(result3[0], 0.0f, 1e-6));
        REQUIRE(is_value_approx(result3[1], 4.0f, 1e-6));
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
        REQUIRE(math::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps));

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
        REQUIRE(math::solve_lower_upper<test_type>(&matrix[0][0], &rhs[0], width, height, &solution[0]));

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
        core::random_pcg rng;

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
            REQUIRE(math::decompose_lower_upper<double>(&A[0], n, n, &matrix_l[0], &matrix_u[0], &matrix_p[0], &swaps));

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
            REQUIRE(math::solve_lower_upper<double>(&matrix_l[0], &matrix_u[0], &matrix_p[0], &rhs[0], n, n, &solution[0]));

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
        REQUIRE(math::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps));

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
        REQUIRE(math::solve_lower_upper<test_type>(&matrix[0][0], &rhs[0], width, height, &solution[0]));
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
        REQUIRE(math::decompose_lower_upper<test_type>(&matrix[0][0], width, height, &L1[0][0], &U1[0][0], &P1[0][0], &swaps) == false);

        const test_type rhs[height] = { 1.0, 2.0, 3.0 };
        test_type solution[height];
        REQUIRE(math::solve_lower_upper<test_type>(&matrix[0][0], &rhs[0], width, height, &solution[0]) == false);
    }

    // The pivot tolerance is relative, so well conditioned matrices factorise and solve at any scale.
    {
        core::random_pcg rng;
        constexpr static const int n = 10;
        const double scales[] = { 0x1p-900, 1e-300, 1e-12, 1e-9, 1e-6, 1e-4, 1.0, 1e6, 1e12, 1e300 };
        for (const double scale : scales) {
            for (int trial = 0; trial < 200; ++trial) {
                double matrix[n * n];
                double scale_max = 0.0;
                for (int i = 0; i < n * n; ++i) {
                    matrix[i] = scale * ((2.0 * rng.get_random_exclusive_top()) - 1.0);
                    scale_max = std::fmax(scale_max, std::abs(matrix[i]));
                }
                double matrix_l[n * n];
                double matrix_u[n * n];
                double matrix_p[n * n];
                REQUIRE(math::decompose_lower_upper<double>(&matrix[0], n, n, &matrix_l[0], &matrix_u[0], &matrix_p[0]));
                double rhs[n];
                for (int i = 0; i < n; ++i) {
                    rhs[i] = (2.0 * rng.get_random_exclusive_top()) - 1.0;
                }
                double solution[n];
                REQUIRE(math::solve_lower_upper<double>(&matrix_l[0], &matrix_u[0], &matrix_p[0], &rhs[0], n, n, &solution[0]));
                double solution_wrapped[n];
                REQUIRE(math::solve_lower_upper<double>(&matrix[0], &rhs[0], n, n, &solution_wrapped[0]));
                // The backward error, relative to the scale of the matrix and the solution, is at the rounding level.
                double solution_max = 0.0;
                for (int i = 0; i < n; ++i) {
                    REQUIRE(solution_wrapped[i] == solution[i]);
                    solution_max = std::fmax(solution_max, std::abs(solution[i]));
                }
                for (int i = 0; i < n; ++i) {
                    double residual = -rhs[i];
                    for (int j = 0; j < n; ++j) {
                        residual += (matrix[i * n + j] / scale_max) * (solution[j] * scale_max);
                    }
                    REQUIRE(std::abs(residual) <= 1e-13 * std::fmax(1.0, solution_max * scale_max));
                }
            }
        }
    }

    // Scaling by a power of two is exact, so it scales the upper matrix and leaves the lower and permutation matrices, and the decision, bit identical.
    {
        core::random_pcg rng;
        constexpr static const int n = 8;
        for (int trial = 0; trial < 200; ++trial) {
            double matrix[n * n];
            double matrix_scaled_down[n * n];
            double matrix_scaled_up[n * n];
            for (int i = 0; i < n * n; ++i) {
                matrix[i] = (2.0 * rng.get_random_exclusive_top()) - 1.0;
                matrix_scaled_down[i] = matrix[i] * 0x1p-60;
                matrix_scaled_up[i] = matrix[i] * 0x1p+60;
            }
            double matrix_l[n * n];
            double matrix_u[n * n];
            double matrix_p[n * n];
            REQUIRE(math::decompose_lower_upper<double>(&matrix[0], n, n, &matrix_l[0], &matrix_u[0], &matrix_p[0]));
            const double* const scaled_matrices[2] = { &matrix_scaled_down[0], &matrix_scaled_up[0] };
            const double scaled_factors[2] = { 0x1p-60, 0x1p+60 };
            for (int s = 0; s < 2; ++s) {
                double scaled_l[n * n];
                double scaled_u[n * n];
                double scaled_p[n * n];
                REQUIRE(math::decompose_lower_upper<double>(scaled_matrices[s], n, n, &scaled_l[0], &scaled_u[0], &scaled_p[0]));
                for (int i = 0; i < n * n; ++i) {
                    REQUIRE(scaled_l[i] == matrix_l[i]);
                    REQUIRE(scaled_p[i] == matrix_p[i]);
                    REQUIRE(scaled_u[i] == matrix_u[i] * scaled_factors[s]);
                }
            }
        }
    }

    // A rank deficient matrix is still rejected at any scale.
    {
        const double scales[] = { 1e-200, 1e-10, 1.0, 1e10, 1e200 };
        for (const double scale : scales) {
            const double matrix[3][3] = {
                { 1.0 * scale, 2.0 * scale, 3.0 * scale },
                { 2.0 * scale, 4.0 * scale, 6.0 * scale },
                { 1.0 * scale, 1.0 * scale, 1.0 * scale }
            };
            double matrix_l[3][3];
            double matrix_u[3][3];
            double matrix_p[3][3];
            REQUIRE(math::decompose_lower_upper<double>(&matrix[0][0], 3, 3, &matrix_l[0][0], &matrix_u[0][0], &matrix_p[0][0]) == false);
            const double rhs[3] = { 1.0, 2.0, 3.0 };
            double solution[3];
            REQUIRE(math::solve_lower_upper<double>(&matrix[0][0], &rhs[0], 3, 3, &solution[0]) == false);
        }
    }

    // A pivot at the rounding level of the largest entry is numerically zero, one just above it is not.
    {
        const double singular[2][2] = { { 1.0, 1.0 }, { 1.0, 1.0 + 0x1p-52 } };
        const double regular[2][2] = { { 1.0, 1.0 }, { 1.0, 1.0 + 0x1p-40 } };
        double matrix_l[2][2];
        double matrix_u[2][2];
        double matrix_p[2][2];
        REQUIRE(math::decompose_lower_upper<double>(&singular[0][0], 2, 2, &matrix_l[0][0], &matrix_u[0][0], &matrix_p[0][0]) == false);
        REQUIRE(math::decompose_lower_upper<double>(&regular[0][0], 2, 2, &matrix_l[0][0], &matrix_u[0][0], &matrix_p[0][0]));
        const double rhs[2] = { 2.0, 2.0 + 0x1p-40 };
        double solution[2];
        REQUIRE(math::solve_lower_upper<double>(&matrix_l[0][0], &matrix_u[0][0], &matrix_p[0][0], &rhs[0], 2, 2, &solution[0]));
        REQUIRE(is_value_equal(solution[0], 1.0));
        REQUIRE(is_value_equal(solution[1], 1.0));

        const float singular_float[2][2] = { { 1.0f, 1.0f }, { 1.0f, 1.0f + 0x1p-23f } };
        const float regular_float[2][2] = { { 1.0f, 1.0f }, { 1.0f, 1.0f + 0x1p-16f } };
        float matrix_l_float[2][2];
        float matrix_u_float[2][2];
        float matrix_p_float[2][2];
        REQUIRE(math::decompose_lower_upper<float>(&singular_float[0][0], 2, 2, &matrix_l_float[0][0], &matrix_u_float[0][0], &matrix_p_float[0][0]) == false);
        REQUIRE(math::decompose_lower_upper<float>(&regular_float[0][0], 2, 2, &matrix_l_float[0][0], &matrix_u_float[0][0], &matrix_p_float[0][0]));
    }

    // The solve tests the diagonals of the triangles it is given relative to their own scale.
    {
        const double matrix_l[2][2] = { { 1.0, 0.0 }, { 0.5, 1.0 } };
        const double matrix_p[2][2] = { { 1.0, 0.0 }, { 0.0, 1.0 } };
        const double small_u[2][2] = { { 2e-9, 1e-9 }, { 0.0, 3e-9 } };
        const double tiny_diagonal_u[2][2] = { { 2.0, 1.0 }, { 0.0, 1e-17 } };
        const double zero_diagonal_u[2][2] = { { 0.0, 1.0 }, { 0.0, 1.0 } };
        const double rhs[2] = { 3e-9, 4e-9 };
        double solution[2];
        REQUIRE(math::solve_lower_upper<double>(&matrix_l[0][0], &small_u[0][0], &matrix_p[0][0], &rhs[0], 2, 2, &solution[0]));
        REQUIRE(is_value_approx(solution[1], (4e-9 - (0.5 * 3e-9)) / 3e-9, 1e-12));
        REQUIRE(is_value_approx(solution[0], (3e-9 - (1e-9 * solution[1])) / 2e-9, 1e-12));
        REQUIRE(math::solve_lower_upper<double>(&matrix_l[0][0], &tiny_diagonal_u[0][0], &matrix_p[0][0], &rhs[0], 2, 2, &solution[0]) == false);
        REQUIRE(math::solve_lower_upper<double>(&matrix_l[0][0], &zero_diagonal_u[0][0], &matrix_p[0][0], &rhs[0], 2, 2, &solution[0]) == false);
        const double nan_l[2][2] = { { 1.0, 0.0 }, { 0.5, std::nan("") } };
        REQUIRE(math::solve_lower_upper<double>(&nan_l[0][0], &small_u[0][0], &matrix_p[0][0], &rhs[0], 2, 2, &solution[0]) == false);
        const double infinite_u[2][2] = { { 2.0, 1.0 }, { 0.0, HUGE_VAL } };
        REQUIRE(math::solve_lower_upper<double>(&matrix_l[0][0], &infinite_u[0][0], &matrix_p[0][0], &rhs[0], 2, 2, &solution[0]) == false);
    }

    // A non-finite entry anywhere is rejected, including in the columns beyond the square part of a wide matrix.
    {
        const double non_finite_values[3] = { std::nan(""), HUGE_VAL, -HUGE_VAL };
        for (const double non_finite_value : non_finite_values) {
            for (int index = 0; index < 3 * 4; ++index) {
                double matrix[3 * 4] = { 2.0, 1.0, 1.0, 5.0, 2.0, 1.0, 3.0, 6.0, 1.0, 3.0, 2.0, 7.0 };
                matrix[index] = non_finite_value;
                double matrix_l[3 * 3];
                double matrix_u[3 * 4];
                double matrix_p[3 * 3];
                REQUIRE(math::decompose_lower_upper<double>(&matrix[0], 4, 3, &matrix_l[0], &matrix_u[0], &matrix_p[0]) == false);
            }
            for (int index = 0; index < 3 * 3; ++index) {
                double matrix[3 * 3] = { 2.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 3.0, 2.0 };
                matrix[index] = non_finite_value;
                const double rhs[3] = { 7.0, 13.0, 13.0 };
                double solution[3];
                REQUIRE(math::solve_lower_upper<double>(&matrix[0], &rhs[0], 3, 3, &solution[0]) == false);
            }
        }
    }

    // Systems too large for the wrapper's stack storage are solved the same way.
    {
        core::random_pcg rng;
        const int sizes[] = { 16, 17, 24 };
        for (const int n : sizes) {
            double matrix[24 * 24];
            double rhs[24];
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    matrix[i * n + j] = (2.0 * rng.get_random_exclusive_top()) - 1.0;
                }
                rhs[i] = (2.0 * rng.get_random_exclusive_top()) - 1.0;
            }
            double matrix_l[24 * 24];
            double matrix_u[24 * 24];
            double matrix_p[24 * 24];
            REQUIRE(math::decompose_lower_upper<double>(&matrix[0], n, n, &matrix_l[0], &matrix_u[0], &matrix_p[0]));
            double solution[24];
            REQUIRE(math::solve_lower_upper<double>(&matrix_l[0], &matrix_u[0], &matrix_p[0], &rhs[0], n, n, &solution[0]));
            double solution_wrapped[24];
            REQUIRE(math::solve_lower_upper<double>(&matrix[0], &rhs[0], n, n, &solution_wrapped[0]));
            for (int i = 0; i < n; ++i) {
                REQUIRE(solution_wrapped[i] == solution[i]);
                double sum = 0.0;
                for (int j = 0; j < n; ++j) {
                    sum += matrix[i * n + j] * solution[j];
                }
                REQUIRE(is_value_approx(sum, rhs[i], 1e-9));
            }
        }
    }

    return EXIT_SUCCESS;
}
