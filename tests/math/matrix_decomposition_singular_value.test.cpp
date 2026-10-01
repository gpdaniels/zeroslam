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

#include "math/matrix_decomposition_singular_value.hpp"

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

void check_decomposition(const double* A, int width, int height, double tolerance);

void check_decomposition(const double* A, int width, int height, double tolerance) {
    double* u = new double[static_cast<unsigned int>(height * height)];
    double* s = new double[static_cast<unsigned int>(height * width)];
    double* vt = new double[static_cast<unsigned int>(width * width)];

    REQUIRE(math::decompose_singular_value(A, static_cast<size_t>(width), static_cast<size_t>(height), u, s, vt));

    double scale = 0.0;
    for (int i = 0; i < (width * height); ++i) {
        if (scale < std::abs(A[i])) {
            scale = std::abs(A[i]);
        }
    }
    const double scaled_tolerance = tolerance * ((scale > 1.0e-300) ? scale : 1.0e-300);

    // matrix_u is orthogonal, including the columns belonging to zero singular values.
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < height; ++j) {
            double dot = 0.0;
            for (int k = 0; k < height; ++k) {
                dot += u[i * height + k] * u[j * height + k];
            }
            REQUIRE(std::abs(dot - ((i == j) ? 1.0 : 0.0)) < tolerance);
        }
    }

    // matrix_vt is orthogonal.
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            double dot = 0.0;
            for (int k = 0; k < width; ++k) {
                dot += vt[i * width + k] * vt[j * width + k];
            }
            REQUIRE(std::abs(dot - ((i == j) ? 1.0 : 0.0)) < tolerance);
        }
    }

    // matrix_s is exactly zero away from the leading diagonal.
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (i != j) {
                REQUIRE(s[i * width + j] == 0.0);
            }
        }
    }

    // The singular values are non-negative and sorted from largest to smallest.
    const int minimum_dimension = (width < height) ? width : height;
    for (int i = 0; i < minimum_dimension; ++i) {
        REQUIRE(s[i * width + i] >= 0.0);
    }
    for (int i = 0; (i + 1) < minimum_dimension; ++i) {
        REQUIRE(s[i * width + i] >= s[(i + 1) * width + (i + 1)]);
    }

    // The product reproduces the input.
    double* t = new double[static_cast<unsigned int>(height * width)];
    matrix_multiply(u, height, height, s, width, height, t);
    double* a = new double[static_cast<unsigned int>(height * width)];
    matrix_multiply(t, width, height, vt, width, width, a);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            REQUIRE(std::abs(a[i * width + j] - A[i * width + j]) < scaled_tolerance);
        }
    }

    delete[] a;
    delete[] t;
    delete[] vt;
    delete[] s;
    delete[] u;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        constexpr static const int width = 2;
        constexpr static const int height = 2;

        const double A[height][width] = {
            { 3, 0 },
            { 4, 5 }
        };

        const double U[height][height] = {
            { 1.0 / std::sqrt(10.0), -3.0 / std::sqrt(10.0) },
            { 3.0 / std::sqrt(10.0), 1.0 / std::sqrt(10.0) }
        };

        const double S[height][width] = {
            { 3.0 * std::sqrt(5.0), 0.0 },
            { 0.0, std::sqrt(5.0) }
        };

        const double Vt[width][width] = {
            { 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0) },
            { -1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0) }
        };

        double u[height][height];
        double s[height][width];
        double vt[width][width];

        REQUIRE(math::decompose_singular_value(&A[0][0], width, height, &u[0][0], &s[0][0], &vt[0][0]));

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < height; ++j) {
                REQUIRE(is_value_approx(U[i][j], u[i][j], 1e-10));
            }
        }

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_approx(S[i][j], s[i][j], 1e-10));
            }
        }

        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_approx(Vt[i][j], vt[i][j], 1e-10));
            }
        }
    }

    {
        constexpr static const int width = 2;
        constexpr static const int height = 3;

        const double A[height][width] = {
            { 3, 2 },
            { 2, 3 },
            { 2, -2 }
        };

        const double U[height][height] = {
            { std::sqrt(2.0) / 2.0, std::sqrt(2.0) / 6.0, -2.0 / 3.0 },
            { std::sqrt(2.0) / 2.0, -std::sqrt(2.0) / 6.0, 2.0 / 3.0 },
            { 0.0, 2.0 * std::sqrt(2.0) / 3.0, 1.0 / 3.0 }
        };

        const double S[height][width] = {
            { 5, 0 },
            { 0, 3 },
            { 0, 0 }
        };

        const double Vt[width][width] = {
            { 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0) },
            { 1.0 / std::sqrt(2.0), -1.0 / std::sqrt(2.0) }
        };

        double u[height][height];
        double s[height][width];
        double vt[width][width];

        REQUIRE(math::decompose_singular_value(&A[0][0], width, height, &u[0][0], &s[0][0], &vt[0][0]));

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < height; ++j) {
                REQUIRE(is_value_approx(U[i][j], u[i][j], 1e-10));
            }
        }

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_approx(S[i][j], s[i][j], 1e-10));
            }
        }

        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_approx(Vt[i][j], vt[i][j], 1e-10));
            }
        }
    }

    {
        constexpr static const int width = 1;
        constexpr static const int height = 5;

        const double A[height][width] = {
            { 1 },
            { 2 },
            { 3 },
            { 4 },
            { 5 }
        };

        double u[height][height];
        double s[height][width];
        double vt[width][width];

        REQUIRE(math::decompose_singular_value(&A[0][0], width, height, &u[0][0], &s[0][0], &vt[0][0]));

        double t[height][width];
        matrix_multiply(&u[0][0], height, height, &s[0][0], width, height, &t[0][0]);
        double a[height][width];
        matrix_multiply(&t[0][0], width, height, &vt[0][0], width, width, &a[0][0]);

        double error_sum = 0.0;
        double error_max = 0.0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_approx(A[i][j], a[i][j], 1e-10));
                const double error = std::abs(A[i][j] - a[i][j]);
                error_sum += error;
                if (error_max < error) {
                    error_max = error;
                }
            }
        }
        REQUIRE(is_value_approx(error_sum, 0.0, 1e-10));
        REQUIRE(is_value_approx(error_max, 0.0, 1e-10));
    }

    {
        constexpr static const int width = 3;
        constexpr static const int height = 2;

        double A[height][width] = {
            { 3, 2, 2 },
            { 2, 3, -2 }
        };

        double U[height][height] = {
            { 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0) },
            { 1.0 / std::sqrt(2.0), -1.0 / std::sqrt(2.0) }
        };

        double S[height][width] = {
            { 5, 0, 0 },
            { 0, 3, 0 }
        };

        double Vt[width][width] = {
            { 1.0 / (1.0 * std::sqrt(2.0)), 1.0 / (1.0 * std::sqrt(2.0)), 0.0 },
            { 1.0 / (3.0 * std::sqrt(2.0)), -1.0 / (3.0 * std::sqrt(2.0)), 4.0 / (3.0 * std::sqrt(2.0)) },
            { -2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0 }
        };

        double u[height][height];
        double s[height][width];
        double vt[width][width];

        REQUIRE(math::decompose_singular_value(&A[0][0], width, height, &u[0][0], &s[0][0], &vt[0][0]));

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < height; ++j) {
                REQUIRE(is_value_approx(U[i][j], u[i][j], 1e-10));
            }
        }

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_approx(S[i][j], s[i][j], 1e-10));
            }
        }

        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_approx(Vt[i][j], vt[i][j], 1e-10));
            }
        }
    }

    {
        constexpr static const int width = 4;
        constexpr static const int height = 3;

        double A[height][width] = {
            { 0, 1, 0, 0 },
            { 0, 0, 2, 0 },
            { 0, 0, 0, 3 },
        };

        double U[height][height] = {
            { 0, 0, 1 },
            { 0, 1, 0 },
            { 1, 0, 0 }
        };

        double S[height][width] = {
            { 3, 0, 0, 0 },
            { 0, 2, 0, 0 },
            { 0, 0, 1, 0 }
        };

        double Vt[width][width] = {
            { 0, 0, 0, 1 },
            { 0, 0, 1, 0 },
            { 0, 1, 0, 0 },
            { 1, 0, 0, 0 }
        };

        double u[height][height];
        double s[height][width];
        double vt[width][width];

        REQUIRE(math::decompose_singular_value(&A[0][0], width, height, &u[0][0], &s[0][0], &vt[0][0]));

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < height; ++j) {
                REQUIRE(is_value_approx(U[i][j], u[i][j], 1e-10));
            }
        }

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_approx(S[i][j], s[i][j], 1e-10));
            }
        }

        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(is_value_approx(Vt[i][j], vt[i][j], 1e-10));
            }
        }
    }

    {
        core::random_pcg rng;

        for (int width = 1; width < 20; ++width) {
            for (int height = 1; height < 20; ++height) {
                double* A = new double[static_cast<size_t>(height * width)];
                for (int i = 0; i < height; ++i) {
                    for (int j = 0; j < width; ++j) {
                        A[i * width + j] = (20.0 * rng.get_random_exclusive_top()) - 10.0;
                    }
                }

                double* u = new double[static_cast<size_t>(height * height)];
                double* s = new double[static_cast<size_t>(height * width)];
                double* vt = new double[static_cast<size_t>(width * width)];

                REQUIRE(math::decompose_singular_value(A, static_cast<size_t>(width), static_cast<size_t>(height), u, s, vt));

                double* t = new double[static_cast<size_t>(height * width)];
                matrix_multiply(u, height, height, s, width, height, t);
                double* a = new double[static_cast<size_t>(height * width)];
                matrix_multiply(t, width, height, vt, width, width, a);
                delete[] t;

                delete[] vt;
                delete[] s;
                delete[] u;

                double error_sum = 0.0;
                double error_max = 0.0;
                for (int i = 0; i < height; ++i) {
                    for (int j = 0; j < width; ++j) {
                        REQUIRE(is_value_approx(A[i * width + j], a[i * width + j], 1e-10));
                        const double error = std::abs(A[i * width + j] - a[i * width + j]);
                        error_sum += error;
                        if (error_max < error) {
                            error_max = error;
                        }
                    }
                }
                REQUIRE(is_value_approx(error_sum, 0.0, 1e-10));
                REQUIRE(is_value_approx(error_max, 0.0, 1e-10));

                delete[] a;
                delete[] A;
            }
        }
    }

    {
        const double A[3][3] = {
            { 1, 2, 3 },
            { 2, 4, 6 },
            { 3, 6, 9 }
        };
        check_decomposition(&A[0][0], 3, 3, 1e-12);

        double u[3][3];
        double s[3][3];
        double vt[3][3];
        REQUIRE(math::decompose_singular_value(&A[0][0], 3, 3, &u[0][0], &s[0][0], &vt[0][0]));
        REQUIRE(is_value_approx(s[0][0], 14.0, 1e-12));
        REQUIRE(is_value_approx(s[1][1], 0.0, 1e-12));
        REQUIRE(is_value_approx(s[2][2], 0.0, 1e-12));
    }
    {
        const double A[2][2] = {
            { 1, 0 },
            { 0, 0 }
        };
        check_decomposition(&A[0][0], 2, 2, 1e-12);

        double u[2][2];
        double s[2][2];
        double vt[2][2];
        REQUIRE(math::decompose_singular_value(&A[0][0], 2, 2, &u[0][0], &s[0][0], &vt[0][0]));
        REQUIRE(is_value_approx(s[0][0], 1.0, 1e-12));
        REQUIRE(is_value_approx(s[1][1], 0.0, 1e-12));
    }
    {
        const double A[3][3] = {
            { 0, 0, 0 },
            { 0, 0, 0 },
            { 0, 0, 0 }
        };
        check_decomposition(&A[0][0], 3, 3, 1e-12);
    }
    {
        const double A[3][3] = {
            { 1, 0, 2 },
            { 0, 0, 0 },
            { 3, 0, 4 }
        };
        check_decomposition(&A[0][0], 3, 3, 1e-12);
    }
    {
        const double A[3][3] = {
            { 2, 0, 0 },
            { 0, 2, 0 },
            { 0, 0, 2 }
        };
        check_decomposition(&A[0][0], 3, 3, 1e-12);

        double u[3][3];
        double s[3][3];
        double vt[3][3];
        REQUIRE(math::decompose_singular_value(&A[0][0], 3, 3, &u[0][0], &s[0][0], &vt[0][0]));
        for (int i = 0; i < 3; ++i) {
            REQUIRE(is_value_approx(s[i][i], 2.0, 1e-12));
        }
    }

    {
        core::random_pcg rng;

        for (int width = 1; width < 14; ++width) {
            for (int height = 1; height < 14; ++height) {
                double* A = new double[static_cast<unsigned int>(height * width)];
                for (int i = 0; i < height; ++i) {
                    for (int j = 0; j < width; ++j) {
                        A[i * width + j] = (20.0 * rng.get_random_exclusive_top()) - 10.0;
                    }
                }
                if (height > 1) {
                    for (int j = 0; j < width; ++j) {
                        A[(height - 1) * width + j] = A[j];
                    }
                }
                check_decomposition(A, width, height, 1e-10);
                delete[] A;
            }
        }
    }

    {
        const double B[3][3] = {
            { 3, 1, 4 },
            { 1, 5, 9 },
            { 2, 6, 5 }
        };

        double u[3][3];
        double s[3][3];
        double vt[3][3];
        REQUIRE(math::decompose_singular_value(&B[0][0], 3, 3, &u[0][0], &s[0][0], &vt[0][0]));
        const double reference[3] = { s[0][0], s[1][1], s[2][2] };

        const double scales[6] = { 1.0e-20, 1.0e-15, 1.0e-8, 1.0e8, 1.0e15, 1.0e20 };
        for (int scale_index = 0; scale_index < 6; ++scale_index) {
            const double scale = scales[scale_index];
            double A[3][3];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    A[i][j] = B[i][j] * scale;
                }
            }
            check_decomposition(&A[0][0], 3, 3, 1e-10);

            double scaled_u[3][3];
            double scaled_s[3][3];
            double scaled_vt[3][3];
            REQUIRE(math::decompose_singular_value(&A[0][0], 3, 3, &scaled_u[0][0], &scaled_s[0][0], &scaled_vt[0][0]));
            for (int i = 0; i < 3; ++i) {
                REQUIRE(std::abs((scaled_s[i][i] / scale) - reference[i]) < (1e-12 * reference[i]));
            }
        }
    }

    {
        const double A[2][2] = {
            { 1, 2 },
            { 3, 4 }
        };
        double u[4];
        double s[4];
        double vt[4];
        REQUIRE(!math::decompose_singular_value(&A[0][0], 0, 2, &u[0], &s[0], &vt[0]));
        REQUIRE(!math::decompose_singular_value(&A[0][0], 2, 0, &u[0], &s[0], &vt[0]));
    }

    {
        double u[9];
        double s[9];
        double vt[9];
        double A[3][3] = {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        };
        A[1][1] = std::nan("");
        REQUIRE(!math::decompose_singular_value(&A[0][0], 3, 3, &u[0], &s[0], &vt[0]));
        A[1][1] = HUGE_VAL;
        REQUIRE(!math::decompose_singular_value(&A[0][0], 3, 3, &u[0], &s[0], &vt[0]));
        A[1][1] = -HUGE_VAL;
        REQUIRE(!math::decompose_singular_value(&A[0][0], 3, 3, &u[0], &s[0], &vt[0]));
    }

    {
        core::random_pcg rng;

        for (int width = 1; width < 10; ++width) {
            for (int height = 1; height < 10; ++height) {
                for (int deficient = 0; deficient < 2; ++deficient) {
                    float* A = new float[static_cast<unsigned int>(height * width)];
                    for (int i = 0; i < height; ++i) {
                        for (int j = 0; j < width; ++j) {
                            A[i * width + j] = static_cast<float>((20.0 * rng.get_random_exclusive_top()) - 10.0);
                        }
                    }
                    if ((deficient != 0) && (height > 1)) {
                        for (int j = 0; j < width; ++j) {
                            A[(height - 1) * width + j] = A[j];
                        }
                    }

                    float* u = new float[static_cast<unsigned int>(height * height)];
                    float* s = new float[static_cast<unsigned int>(height * width)];
                    float* vt = new float[static_cast<unsigned int>(width * width)];

                    REQUIRE(math::decompose_singular_value(A, static_cast<size_t>(width), static_cast<size_t>(height), u, s, vt));

                    for (int i = 0; i < height; ++i) {
                        for (int j = 0; j < height; ++j) {
                            float dot = 0.0f;
                            for (int k = 0; k < height; ++k) {
                                dot += u[i * height + k] * u[j * height + k];
                            }
                            REQUIRE(std::abs(dot - ((i == j) ? 1.0f : 0.0f)) < 1e-4f);
                        }
                    }
                    for (int i = 0; i < height; ++i) {
                        for (int j = 0; j < width; ++j) {
                            if (i != j) {
                                REQUIRE(s[i * width + j] == 0.0f);
                            }
                        }
                    }
                    const int minimum_dimension = (width < height) ? width : height;
                    for (int i = 0; (i + 1) < minimum_dimension; ++i) {
                        REQUIRE(s[i * width + i] >= s[(i + 1) * width + (i + 1)]);
                    }

                    delete[] vt;
                    delete[] s;
                    delete[] u;
                    delete[] A;
                }
            }
        }
    }

    // Matrices whose squares overflow or underflow, where the norms used to be unscaled sums of squares.
    {
        core::random_pcg rng;
        double base[9];
        for (int i = 0; i < 9; ++i) {
            base[i] = (2.0 * rng.get_random_exclusive_top()) - 1.0;
        }
        double base_u[9];
        double base_s[9];
        double base_vt[9];
        REQUIRE(math::decompose_singular_value(&base[0], 3, 3, &base_u[0], &base_s[0], &base_vt[0]));
        const double scales[] = { 1e155, 1e-155, 1e160, 1e-160, 1e300, 1e-300 };
        for (const double scale : scales) {
            double scaled[9];
            for (int i = 0; i < 9; ++i) {
                scaled[i] = base[i] * scale;
            }
            check_decomposition(&scaled[0], 3, 3, 1e-13);
            double u[9];
            double s[9];
            double vt[9];
            REQUIRE(math::decompose_singular_value(&scaled[0], 3, 3, &u[0], &s[0], &vt[0]));
            for (int i = 0; i < 3; ++i) {
                REQUIRE(is_value_approx(s[i * 3 + i] / scale, base_s[i * 3 + i], 1e-13));
            }
        }
        // Subnormal entries are only as precise as they are stored, but still decompose.
        double subnormal[9];
        for (int i = 0; i < 9; ++i) {
            subnormal[i] = base[i] * 1e-310;
        }
        check_decomposition(&subnormal[0], 3, 3, 1e-11);

        // Scaling by a power of two outside the range is exact, so the factors are bit identical and the singular values scale exactly.
        const double power_scales[] = { 0x1p+600, 0x1p-600, 0x1p+1000, 0x1p-1000 };
        for (const double scale : power_scales) {
            double scaled[9];
            for (int i = 0; i < 9; ++i) {
                scaled[i] = base[i] * scale;
            }
            double u[9];
            double s[9];
            double vt[9];
            REQUIRE(math::decompose_singular_value(&scaled[0], 3, 3, &u[0], &s[0], &vt[0]));
            for (int i = 0; i < 9; ++i) {
                REQUIRE(u[i] == base_u[i]);
                REQUIRE(vt[i] == base_vt[i]);
                REQUIRE(s[i] == base_s[i] * scale);
            }
        }
    }

    // Graded matrices, with rows and columns of very different scales, keep orthogonal factors and reproduce the input.
    {
        core::random_pcg rng;
        const int sizes[] = { 3, 9, 65 };
        const double ranges[] = { 50.0, 100.0, 150.0 };
        for (const int n : sizes) {
            for (const double range : ranges) {
                double graded[65 * 65];
                double row_scale[65];
                double column_scale[65];
                for (int i = 0; i < n; ++i) {
                    row_scale[i] = std::pow(10.0, range * ((2.0 * rng.get_random_exclusive_top()) - 1.0));
                    column_scale[i] = std::pow(10.0, range * ((2.0 * rng.get_random_exclusive_top()) - 1.0));
                }
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        graded[i * n + j] = row_scale[i] * column_scale[j] * ((2.0 * rng.get_random_exclusive_top()) - 1.0);
                    }
                }
                check_decomposition(&graded[0], n, n, 1e-12);
            }
        }
        const int n = 65;
        double graded_rows[65 * 65];
        double graded_columns[65 * 65];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                graded_rows[i * n + j] = std::pow(10.0, -300.0 + ((600.0 * i) / (n - 1))) * ((2.0 * rng.get_random_exclusive_top()) - 1.0);
                graded_columns[i * n + j] = std::pow(10.0, -300.0 + ((600.0 * j) / (n - 1))) * ((2.0 * rng.get_random_exclusive_top()) - 1.0);
            }
        }
        check_decomposition(&graded_rows[0], n, n, 1e-12);
        check_decomposition(&graded_columns[0], n, n, 1e-12);
        double wide[5 * 70];
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 70; ++j) {
                wide[i * 70 + j] = std::pow(10.0, -200.0 + ((400.0 * j) / 69.0)) * ((2.0 * rng.get_random_exclusive_top()) - 1.0);
            }
        }
        check_decomposition(&wide[0], 70, 5, 1e-12);
        check_decomposition(&wide[0], 5, 70, 1e-12);
    }

    // Single precision has the same protection against its narrower range.
    {
        core::random_pcg rng;
        const float scales[] = { 1e30f, 1e-30f, 1e20f, 1e-20f };
        for (const float scale : scales) {
            float matrix[9];
            for (int i = 0; i < 9; ++i) {
                matrix[i] = scale * static_cast<float>((2.0 * rng.get_random_exclusive_top()) - 1.0);
            }
            float u[9];
            float s[9];
            float vt[9];
            REQUIRE(math::decompose_singular_value(&matrix[0], 3, 3, &u[0], &s[0], &vt[0]));
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    double dot_u = 0.0;
                    double dot_v = 0.0;
                    double product = 0.0;
                    for (int k = 0; k < 3; ++k) {
                        dot_u += static_cast<double>(u[i * 3 + k]) * static_cast<double>(u[j * 3 + k]);
                        dot_v += static_cast<double>(vt[i * 3 + k]) * static_cast<double>(vt[j * 3 + k]);
                        product += static_cast<double>(u[i * 3 + k]) * static_cast<double>(s[k * 3 + k]) * static_cast<double>(vt[k * 3 + j]);
                    }
                    REQUIRE(std::abs(dot_u - ((i == j) ? 1.0 : 0.0)) < 1e-5);
                    REQUIRE(std::abs(dot_v - ((i == j) ? 1.0 : 0.0)) < 1e-5);
                    REQUIRE(std::abs(product - static_cast<double>(matrix[i * 3 + j])) < 1e-5 * static_cast<double>(scale));
                }
            }
        }
    }
}
