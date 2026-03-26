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

#include "matrix_eigen_solver.hpp"

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

static inline bool is_value_approx(float lhs, float rhs, float epsilon = 1e-5f) {
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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        // Test:
        // - symmetric 4x4
        // - doubles
        // - with eigen vectors
        // - sort results

        const double matrix[4][4] = {
            { -2.0000, -0.6714, 0.8698, 0.5792 },
            { -0.6714, -1.1242, -0.0365, -0.5731 },
            { 0.8698, -0.0365, -0.4660, -0.8542 },
            { 0.5792, -0.5731, -0.8542, 0.1188 }
        };

        const double vectors[4][4] = {
            { -0.846873, 0.054867, 0.522012, 0.085436 },
            { -0.252527, 0.814103, -0.452288, -0.262490 },
            { 0.402834, 0.356538, 0.694297, -0.478076 },
            { 0.238246, 0.455089, 0.202213, 0.833812 }
        };

        const double values[8] = {
            -2.776886,
            0.0,
            -1.505801,
            0.0,
            -0.037042,
            0.0,
            0.848329,
            0.0
        };

        double values2[8] = {};
        double vectors2[4][4] = {};
        REQUIRE((matrix::eigen_solver<double, true, true>(&matrix[0][0], 4, &values2[0], &vectors2[0][0])));

        // Normalise eigen vectors.
        for (int j = 0; j < 4; ++j) {
            const double scale = vectors[0][j] / vectors2[0][j];
            for (int i = 0; i < 4; ++i) {
                vectors2[i][j] = scale * vectors2[i][j];
            }
        }

        for (int i = 0; i < 4; ++i) {
            REQUIRE(is_value_approx(values[i * 2 + 0], values2[i * 2 + 0], 1e-5));
            REQUIRE(is_value_approx(values[i * 2 + 1], values2[i * 2 + 1], 1e-5));
            for (int j = 0; j < 4; ++j) {
                REQUIRE(is_value_approx(vectors[i][j], vectors2[i][j], 1e-5));
            }
        }
    }

    {
        // Test:
        // - symmetric 4x4
        // - floats
        // - with eigen vectors
        // - sort results

        const float matrix[4][4] = {
            { -2.0000f, -0.6714f, 0.8698f, 0.5792f },
            { -0.6714f, -1.1242f, -0.0365f, -0.5731f },
            { 0.8698f, -0.0365f, -0.4660f, -0.8542f },
            { 0.5792f, -0.5731f, -0.8542f, 0.1188f }
        };

        const float vectors[4][4] = {
            { -0.846873f, 0.054867f, 0.522012f, 0.085436f },
            { -0.252527f, 0.814103f, -0.452288f, -0.262490f },
            { 0.402834f, 0.356538f, 0.694297f, -0.478076f },
            { 0.238246f, 0.455089f, 0.202213f, 0.833812f }
        };

        const float values[8] = {
            -2.776886f,
            0.0f,
            -1.505801f,
            0.0f,
            -0.037042f,
            0.0f,
            0.848329f,
            0.0f
        };

        float values2[8] = {};
        float vectors2[4][4] = {};
        REQUIRE((matrix::eigen_solver<float, true, true>(&matrix[0][0], 4, &values2[0], &vectors2[0][0])));

        // Normalise eigen vectors.
        for (int j = 0; j < 4; ++j) {
            const float scale = vectors[0][j] / vectors2[0][j];
            for (int i = 0; i < 4; ++i) {
                vectors2[i][j] = scale * vectors2[i][j];
            }
        }

        for (int i = 0; i < 4; ++i) {
            REQUIRE(is_value_approx(values[i * 2 + 0], values2[i * 2 + 0], 1e-4f));
            REQUIRE(is_value_approx(values[i * 2 + 1], values2[i * 2 + 1], 1e-4f));
            for (int j = 0; j < 4; ++j) {
                REQUIRE(is_value_approx(vectors[i][j], vectors2[i][j], 1e-4f));
            }
        }
    }

    {
        // Test:
        // - symmetric 4x4
        // - doubles
        // - without eigen vectors, values only
        // - sort results

        const double matrix[4][4] = {
            { -2.0000, -0.6714, 0.8698, 0.5792 },
            { -0.6714, -1.1242, -0.0365, -0.5731 },
            { 0.8698, -0.0365, -0.4660, -0.8542 },
            { 0.5792, -0.5731, -0.8542, 0.1188 }
        };

        const double values[8] = {
            -2.776886,
            0,
            -1.505801,
            0,
            -0.037042,
            0,
            0.848329,
            0
        };

        double values2[8] = {};
        REQUIRE((matrix::eigen_solver<double, false, true>(&matrix[0][0], 4, &values2[0], nullptr)));

        for (int i = 0; i < 4; ++i) {
            REQUIRE(is_value_approx(values[i * 2 + 0], values2[i * 2 + 0], 1e-5));
            REQUIRE(is_value_approx(values[i * 2 + 1], values2[i * 2 + 1], 1e-5));
        }
    }

    {
        // Test:
        // - symmetric 4x4
        // - doubles
        // - without eigen vectors, values only
        // - do not sort results

        const double matrix[4][4] = {
            { -2.0000, -0.6714, 0.8698, 0.5792 },
            { -0.6714, -1.1242, -0.0365, -0.5731 },
            { 0.8698, -0.0365, -0.4660, -0.8542 },
            { 0.5792, -0.5731, -0.8542, 0.1188 }
        };

        // Expected values (may be in any order when unsorted).
        const double expected_values[4] = { -2.776886, -1.505801, -0.037042, 0.848329 };

        double values2[8] = {};
        REQUIRE((matrix::eigen_solver<double, false, false>(&matrix[0][0], 4, &values2[0], nullptr)));

        // Check that all expected eigenvalues are present (order doesn't matter).
        bool found[4] = { false, false, false, false };
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (!found[j] && is_value_approx(expected_values[j], values2[i * 2 + 0], 1e-5)) {
                    found[j] = true;
                    REQUIRE(is_value_approx(0.0, values2[i * 2 + 1], 1e-5)); // Imaginary part should be 0
                    break;
                }
            }
        }

        for (int i = 0; i < 4; ++i) {
            REQUIRE(found[i]);
        }
    }

    {
        // Test:
        // - symmetric 4x4
        // - doubles
        // - with eigen vectors
        // - sort results

        const double matrix[4][4] = {
            { -2.0000, -0.6714, 0.8698, 0.5792 },
            { -0.6714, -1.1242, -0.0365, -0.5731 },
            { 0.8698, -0.0365, -0.4660, -0.8542 },
            { 0.5792, -0.5731, -0.8542, 0.1188 }
        };

        double values2[8] = {};
        double vectors2[4][4] = {};
        REQUIRE((matrix::eigen_solver<double, true, false>(&matrix[0][0], 4, &values2[0], &vectors2[0][0])));

        // Verify (A * v) = (lambda * v) for each eigenpair.
        for (int j = 0; j < 4; ++j) {
            double lambda = values2[j * 2 + 0];
            double result[4] = {};

            // Compute (A * v).
            for (int i = 0; i < 4; ++i) {
                result[i] = 0.0;
                for (int k = 0; k < 4; ++k) {
                    result[i] += matrix[i][k] * vectors2[k][j];
                }
            }

            // Verify result is approx (lambda * v).
            for (int i = 0; i < 4; ++i) {
                REQUIRE(is_value_approx(result[i], lambda * vectors2[i][j], 1e-5));
            }
        }
    }

    {
        // Test:
        // - identity 3xx3
        // - doubles
        // - with eigen vectors
        // - sort results

        const double matrix[3][3] = {
            { 1.0, 0.0, 0.0 },
            { 0.0, 1.0, 0.0 },
            { 0.0, 0.0, 1.0 }
        };

        double values[6] = {};
        double vectors[3][3] = {};
        REQUIRE((matrix::eigen_solver<double, true, true>(&matrix[0][0], 3, &values[0], &vectors[0][0])));

        // All eigenvalues should be 1.0
        for (int i = 0; i < 3; ++i) {
            REQUIRE(is_value_approx(1.0, values[i * 2 + 0], 1e-8));
            REQUIRE(is_value_approx(0.0, values[i * 2 + 1], 1e-8));
        }
    }

    {
        // Test:
        // - diagonal 3x3
        // - doubles
        // - with eigen vectors
        // - sort results

        const double matrix[3][3] = {
            { 5.0, 0.0, 0.0 },
            { 0.0, -2.0, 0.0 },
            { 0.0, 0.0, 3.0 }
        };

        double values[6] = {};
        double vectors[3][3] = {};
        REQUIRE((matrix::eigen_solver<double, true, true>(&matrix[0][0], 3, &values[0], &vectors[0][0])));

        // Expected eigenvalues (sorted): -2, 3, 5
        REQUIRE(is_value_approx(-2.0, values[0], 1e-8));
        REQUIRE(is_value_approx(3.0, values[2], 1e-8));
        REQUIRE(is_value_approx(5.0, values[4], 1e-8));

        for (int i = 0; i < 3; ++i) {
            REQUIRE(is_value_approx(0.0, values[i * 2 + 1], 1e-8));
        }
    }

    {
        // Test:
        // - symmetric 2x2
        // - doubles
        // - with eigen vectors
        // - sort results

        const double matrix[2][2] = {
            { 4.0, 1.0 },
            { 1.0, 3.0 }
        };

        double values[4] = {};
        double vectors[2][2] = {};
        REQUIRE((matrix::eigen_solver<double, true, true>(&matrix[0][0], 2, &values[0], &vectors[0][0])));

        // Expected eigenvalues: (7 - sqrt(5))/2 = 2.382, (7 + sqrt(5))/2 = 4.618
        REQUIRE(is_value_approx(2.38196601125, values[0], 1e-5));
        REQUIRE(is_value_approx(4.61803398875, values[2], 1e-5));

        for (int i = 0; i < 2; ++i) {
            REQUIRE(is_value_approx(0.0, values[i * 2 + 1], 1e-8));
        }
    }

    {
        // Test:
        // - Non-symmetric 2x2
        // - doubles
        // - with eigen vectors
        // - sort results

        // Rotation matrix (90 degrees) - has complex eigenvalues
        const double matrix[2][2] = {
            { 0.0, -1.0 },
            { 1.0, 0.0 }
        };

        double values[4] = {};
        double vectors[2][2] = {};
        REQUIRE((matrix::eigen_solver<double, true, true>(&matrix[0][0], 2, &values[0], &vectors[0][0])));

        // Expected eigenvalues: (+/-)i (0 (+/-) i)
        // The first eigenvalue might be -i (0 - i) or +i (0 + i)
        bool has_pos_i = false;
        bool has_neg_i = false;

        for (int i = 0; i < 2; ++i) {
            // Real part should be 0
            REQUIRE(is_value_approx(0.0, values[i * 2 + 0], 1e-8));
            if (is_value_approx(1.0, values[i * 2 + 1], 1e-8)) {
                has_pos_i = true;
            }
            else if (is_value_approx(-1.0, values[i * 2 + 1], 1e-8)) {
                has_neg_i = true;
            }
        }

        REQUIRE(has_pos_i && has_neg_i);
    }

    {
        // Test:
        // - identity 3x3
        // - floats
        // - with eigen vectors
        // - sort results

        const float matrix[3][3] = {
            { 1.0f, 0.0f, 0.0f },
            { 0.0f, 1.0f, 0.0f },
            { 0.0f, 0.0f, 1.0f }
        };

        float values[6] = {};
        float vectors[3][3] = {};
        REQUIRE((matrix::eigen_solver<float, true, true>(&matrix[0][0], 3, &values[0], &vectors[0][0])));

        for (int i = 0; i < 3; ++i) {
            REQUIRE(is_value_approx(1.0f, values[i * 2 + 0], 1e-5f));
            REQUIRE(is_value_approx(0.0f, values[i * 2 + 1], 1e-5f));
        }
    }

    {
        // Test:
        // - zero 3x3
        // - doubles
        // - with eigen vectors
        // - sort results

        const double matrix[3][3] = {
            { 0.0, 0.0, 0.0 },
            { 0.0, 0.0, 0.0 },
            { 0.0, 0.0, 0.0 }
        };

        double values[6] = {};
        double vectors[3][3] = {};
        REQUIRE((matrix::eigen_solver<double, true, true>(&matrix[0][0], 3, &values[0], &vectors[0][0])));

        // All eigenvalues should be 0.0
        for (int i = 0; i < 3; ++i) {
            REQUIRE(is_value_approx(0.0, values[i * 2 + 0], 1e-8));
            REQUIRE(is_value_approx(0.0, values[i * 2 + 1], 1e-8));
        }
    }

    {
        // Test:
        // - symmetric 3x3
        // - doubles
        // - with eigen vectors
        // - sort results

        const double matrix[3][3] = {
            { 2.0, 0.0, 0.0 },
            { 0.0, 2.0, 0.0 },
            { 0.0, 0.0, 5.0 }
        };

        double values[6] = {};
        double vectors[3][3] = {};
        REQUIRE((matrix::eigen_solver<double, true, true>(&matrix[0][0], 3, &values[0], &vectors[0][0])));

        // Expected eigenvalues (sorted): 2, 2, 5
        REQUIRE(is_value_approx(2.0, values[0], 1e-8));
        REQUIRE(is_value_approx(2.0, values[2], 1e-8));
        REQUIRE(is_value_approx(5.0, values[4], 1e-8));
    }

    {
        // Test:
        // - symmetric 3x3
        // - doubles
        // - with eigen vectors
        // - sort results

        const double matrix[3][3] = {
            { 1.0, 1.0, 10000.0 },
            { 1.0, 1.0, 10000.0 },
            { 0.01, 0.01, 1.0 }
        };

        double values[6] = {};
        double vectors[3][3] = {};
        REQUIRE((matrix::eigen_solver<double, true, true>(&matrix[0][0], 3, &values[0], &vectors[0][0])));

        // Expected eigenvalues (sorted):
        REQUIRE(is_value_approx(-(3.0 / 2.0) * (std::sqrt(89) - 1), values[0], 1e-8));
        REQUIRE(is_value_approx(0.0, values[1], 1e-8));
        REQUIRE(is_value_approx(0.0, values[2], 1e-8));
        REQUIRE(is_value_approx(0.0, values[3], 1e-8));
        REQUIRE(is_value_approx(+(3.0 / 2.0) * (std::sqrt(89) + 1), values[4], 1e-8));
        REQUIRE(is_value_approx(0.0, values[5], 1e-8));
    }

    {
        // Test:
        // - random 6x6
        // - doubles
        // - with eigen vectors
        // - sort results

        const double matrix[6][6] = {
            { 0.68, -0.33, -0.27, -0.717, -0.687, 0.0259 },
            { -0.211, 0.536, 0.0268, 0.214, -0.198, 0.678 },
            { 0.566, -0.444, 0.904, -0.967, -0.74, 0.225 },
            { 0.597, 0.108, 0.832, -0.514, -0.782, -0.408 },
            { 0.823, -0.0452, 0.271, -0.726, 0.998, 0.275 },
            { -0.605, 0.258, 0.435, 0.608, -0.563, 0.0486 }
        };

        // Expected eigenvalues.
        const double expected_eigenvalues[][2] = {
            { 0.049, -1.06 },
            { 0.049, 1.06 },
            { 0.353, 0.0 },
            { 0.618, -0.129 },
            { 0.618, 0.129 },
            { 0.967, 0.0 }
        };

        double values[12] = {};
        double vectors[6][6] = {};
        REQUIRE((matrix::eigen_solver<double, true, true>(&matrix[0][0], 6, &values[0], &vectors[0][0])));

        // Relaxed tolerance for 6x6 matrices.
        const double tol = 1e-2;

        // Check eigenvalues are sorted.
        for (int i = 0; i < 5; ++i) {
            REQUIRE(values[i * 2 + 0] <= values[(i + 1) * 2 + 0] + tol);
        }

        // Verify each eigenvalue matches the expected value.
        for (int i = 0; i < 6; ++i) {
            REQUIRE(is_value_approx(values[i * 2 + 0], expected_eigenvalues[i][0], tol));
            REQUIRE(is_value_approx(values[i * 2 + 1], expected_eigenvalues[i][1], tol));
        }

        // Verify eigenvector relationships, check: (A * v) = (lambda * v).
        for (int col = 0; col < 6; ++col) {
            double lambda_real = values[col * 2 + 0];
            double lambda_imag = values[col * 2 + 1];

            if (std::abs(lambda_imag) < tol) {
                // Real eigenvalue, check: (A * v) = (lambda * v).
                double result[6] = {};
                for (int row = 0; row < 6; ++row) {
                    result[row] = 0.0;
                    for (int k = 0; k < 6; ++k) {
                        result[row] += matrix[row][k] * vectors[k][col];
                    }
                }

                for (int row = 0; row < 6; ++row) {
                    REQUIRE(is_value_approx(result[row], lambda_real * vectors[row][col], tol));
                }
            }
            else if (lambda_imag > 0) {
                // Complex eigenvalue with positive imaginary part, find the conjugate pair.
                // Skip negative imaginary eigenvalues as they're conjugates already checked.
                // Conjugate will be one before this one.
                const int conj_col = col - 1;

                // Verify A * (v_real + i * v_imag) = (a + i * b) * (v_real + i * v_imag)
                // This gives: A * v_real = a * v_real - b * v_imag
                //             A * v_imag = b * v_real + a * v_imag

                double Av_real[6] = {};
                double Av_imag[6] = {};

                for (int row = 0; row < 6; ++row) {
                    Av_real[row] = 0.0;
                    Av_imag[row] = 0.0;
                    for (int k = 0; k < 6; ++k) {
                        Av_real[row] += matrix[row][k] * vectors[k][col];
                        Av_imag[row] += matrix[row][k] * vectors[k][conj_col];
                    }
                }

                for (int row = 0; row < 6; ++row) {
                    double expected_real = lambda_real * vectors[row][col] - lambda_imag * vectors[row][conj_col];
                    double expected_imag = lambda_imag * vectors[row][col] + lambda_real * vectors[row][conj_col];
                    REQUIRE(is_value_approx(Av_real[row], expected_real, tol));
                    REQUIRE(is_value_approx(Av_imag[row], expected_imag, tol));
                }
            }
        }
    }
}
