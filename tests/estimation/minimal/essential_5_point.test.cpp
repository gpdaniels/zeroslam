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

#include "estimation/minimal/essential_5_point.hpp"

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
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

static inline bool are_values_approx(const double* lhs, const double* rhs, unsigned long long int length, double epsilon = 1e-8) {
    for (size_t index = 0; index < length; ++index) {
        if (!is_value_approx(lhs[index], rhs[index], epsilon)) {
            return false;
        }
    }
    return true;
}

static inline void matrix_multiply(const double* lhs, const double* rhs, double* result) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += lhs[r * 3 + k] * rhs[k * 3 + c];
            }
            result[r * 3 + c] = sum;
        }
    }
}

static inline void matrix_vector_multiply(const double* matrix, const double* vector, double* result) {
    for (int r = 0; r < 3; ++r) {
        result[r] = matrix[r * 3 + 0] * vector[0] + matrix[r * 3 + 1] * vector[1] + matrix[r * 3 + 2] * vector[2];
    }
}

static inline void cross_matrix(const double* vector, double* matrix) {
    matrix[0] = 0;
    matrix[1] = -vector[2];
    matrix[2] = vector[1];
    matrix[3] = vector[2];
    matrix[4] = 0;
    matrix[5] = -vector[0];
    matrix[6] = -vector[1];
    matrix[7] = vector[0];
    matrix[8] = 0;
}

static inline double frobenius_norm(const double* matrix) {
    double sum = 0.0;
    for (int i = 0; i < 9; ++i) {
        sum += matrix[i] * matrix[i];
    }
    return std::sqrt(sum);
}

static inline void normalize_matrix(double* matrix) {
    const double norm = frobenius_norm(matrix);
    if (norm > 0) {
        for (int i = 0; i < 9; ++i) {
            matrix[i] /= norm;
        }
    }
}

static inline void project_point(const double* rotation, const double* translation, const double* point_xyz, double* point_xy) {
    double point[3];
    matrix_vector_multiply(rotation, point_xyz, point);
    point[0] += translation[0];
    point[1] += translation[1];
    point[2] += translation[2];
    point_xy[0] = point[0] / point[2];
    point_xy[1] = point[1] / point[2];
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // Known transformation.
    {
        const double alpha = 0.25;
        const double beta = -0.17;
        const double gamma = 0.1;
        const double rotation_x[3][3] = {
            { 1, 0, 0 },
            { 0, std::cos(alpha), -std::sin(alpha) },
            { 0, std::sin(alpha), std::cos(alpha) }
        };
        const double rotation_y[3][3] = {
            { std::cos(beta), 0, std::sin(beta) },
            { 0, 1, 0 },
            { -std::sin(beta), 0, std::cos(beta) }
        };
        const double rotation_z[3][3] = {
            { std::cos(gamma), -std::sin(gamma), 0 },
            { std::sin(gamma), std::cos(gamma), 0 },
            { 0, 0, 1 }
        };
        double temp[9];
        matrix_multiply(&rotation_z[0][0], &rotation_y[0][0], temp);
        double rotation[9];
        matrix_multiply(temp, &rotation_x[0][0], rotation);
        const double translation[3] = { 0.5, -0.3, 1.0 };
        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double zero[3] = { 0, 0, 0 };
        const int point_count = 5;
        double world_points[5][3] = {
            { 0.1, 0.2, 3.0 },
            { -0.5, 0.4, 4.2 },
            { 0.7, -0.3, 5.1 },
            { -0.2, -0.1, 2.7 },
            { 0.0, 0.0, 6.0 }
        };

        double lhs_points[10];
        double rhs_points[10];
        for (int i = 0; i < point_count; ++i) {
            project_point(identity, zero, &world_points[i][0], &lhs_points[2 * i]);
            project_point(rotation, translation, &world_points[i][0], &rhs_points[2 * i]);
        }

        double translation_matrix[9];
        cross_matrix(translation, translation_matrix);
        double expected[9];
        matrix_multiply(translation_matrix, rotation, expected);
        normalize_matrix(expected);

        double essentials[10 * 9] = {};
        int solutions = estimation::minimal::essential_5_point<double>::solve(lhs_points, rhs_points, essentials);
        REQUIRE(solutions >= 1);
        REQUIRE(solutions <= 10);

        bool found_match = false;
        for (int s = 0; s < solutions; ++s) {
            double* current_essential = &essentials[s * 9];
            normalize_matrix(current_essential);
            if (are_values_approx(current_essential, &expected[0], 9, 1e-5)) {
                found_match = true;
                break;
            }
            for (int k = 0; k < 9; ++k) {
                current_essential[k] = -current_essential[k];
            }
            if (are_values_approx(current_essential, &expected[0], 9, 1e-5)) {
                found_match = true;
                break;
            }
        }
        REQUIRE(found_match);
    }

    // Epipolar constraint and repeatability.
    {
        constexpr static const int num_samples = 8;
        for (int sample = 0; sample < num_samples; ++sample) {
            const double alpha = 0.1 + 0.05 * sample;
            const double beta = -0.05 - 0.02 * sample;
            const double gamma = 0.01 * sample;
            const double rotation_x[3][3] = {
                { 1, 0, 0 },
                { 0, std::cos(alpha), -std::sin(alpha) },
                { 0, std::sin(alpha), std::cos(alpha) }
            };
            const double rotation_y[3][3] = {
                { std::cos(beta), 0, std::sin(beta) },
                { 0, 1, 0 },
                { -std::sin(beta), 0, std::cos(beta) }
            };
            const double rotation_z[3][3] = {
                { std::cos(gamma), -std::sin(gamma), 0 },
                { std::sin(gamma), std::cos(gamma), 0 },
                { 0, 0, 1 }
            };
            double temp[9];
            matrix_multiply(&rotation_z[0][0], &rotation_y[0][0], temp);
            double rotation[9];
            matrix_multiply(temp, &rotation_x[0][0], rotation);
            const double translation[3] = { 0.4 + 0.05 * sample, -0.2 - 0.03 * sample, 1.0 + 0.1 * sample };
            const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
            const double zero[3] = { 0, 0, 0 };
            const double world_points[5][3] = {
                { 0.1 + 0.02 * sample, 0.2, 3.0 },
                { -0.5, 0.4 - 0.01 * sample, 4.2 },
                { 0.7, -0.3 + 0.02 * sample, 5.1 },
                { -0.2 - 0.01 * sample, -0.1, 2.7 },
                { 0.05 * sample, 0.0, 6.0 }
            };

            double lhs_points[10];
            double rhs_points[10];
            for (int i = 0; i < 5; ++i) {
                project_point(identity, zero, &world_points[i][0], &lhs_points[2 * i]);
                project_point(rotation, translation, &world_points[i][0], &rhs_points[2 * i]);
            }

            double essentials[10 * 9] = {};
            const int solutions = estimation::minimal::essential_5_point<double>::solve(lhs_points, rhs_points, essentials);
            REQUIRE(solutions >= 1);
            REQUIRE(solutions <= 10);

            for (int s = 0; s < solutions; ++s) {
                const double* e = &essentials[s * 9];
                for (int i = 0; i < 5; ++i) {
                    const double lx = lhs_points[2 * i + 0];
                    const double ly = lhs_points[2 * i + 1];
                    const double rx = rhs_points[2 * i + 0];
                    const double ry = rhs_points[2 * i + 1];
                    const double ex0 = e[0] * lx + e[1] * ly + e[2];
                    const double ex1 = e[3] * lx + e[4] * ly + e[5];
                    const double ex2 = e[6] * lx + e[7] * ly + e[8];
                    const double constraint = rx * ex0 + ry * ex1 + ex2;
                    REQUIRE(std::abs(constraint) < 1e-8);
                }
            }

            double essentials_repeat[10 * 9] = {};
            const int solutions_repeat = estimation::minimal::essential_5_point<double>::solve(lhs_points, rhs_points, essentials_repeat);
            REQUIRE(solutions_repeat == solutions);
            for (int i = 0; i < solutions * 9; ++i) {
                REQUIRE(essentials[i] == essentials_repeat[i]);
            }
        }
    }

    // Duplicated and degenerate correspondences.
    {
        const double alpha = 0.25;
        const double beta = -0.17;
        const double gamma = 0.1;
        const double rotation_x[3][3] = {
            { 1, 0, 0 },
            { 0, std::cos(alpha), -std::sin(alpha) },
            { 0, std::sin(alpha), std::cos(alpha) }
        };
        const double rotation_y[3][3] = {
            { std::cos(beta), 0, std::sin(beta) },
            { 0, 1, 0 },
            { -std::sin(beta), 0, std::cos(beta) }
        };
        const double rotation_z[3][3] = {
            { std::cos(gamma), -std::sin(gamma), 0 },
            { std::sin(gamma), std::cos(gamma), 0 },
            { 0, 0, 1 }
        };
        double temp[9];
        matrix_multiply(&rotation_z[0][0], &rotation_y[0][0], temp);
        double rotation[9];
        matrix_multiply(temp, &rotation_x[0][0], rotation);
        const double translation[3] = { 0.5, -0.3, 1.0 };
        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double zero[3] = { 0, 0, 0 };
        const int point_count = 5;
        double world_points[5][3] = {
            { 0.1, 0.2, 3.0 },
            { 0.1, 0.2, 3.0 },
            { 0.7, -0.3, 5.1 },
            { -0.2, -0.1, 2.7 },
            { 0.0, 0.0, 6.0 }
        };

        double lhs_points[10];
        double rhs_points[10];
        for (int i = 0; i < point_count; ++i) {
            project_point(identity, zero, &world_points[i][0], &lhs_points[2 * i]);
            project_point(rotation, translation, &world_points[i][0], &rhs_points[2 * i]);
        }

        double essentials[10 * 9] = {};
        int solutions = estimation::minimal::essential_5_point<double>::solve(lhs_points, rhs_points, essentials);
        REQUIRE(solutions >= 0);
        REQUIRE(solutions <= 10);

        for (int s = 0; s < solutions; ++s) {
            double* current_essential = &essentials[s * 9];
            for (int k = 0; k < 9; ++k) {
                REQUIRE(std::isfinite(current_essential[k]));
            }
            normalize_matrix(current_essential);
            for (int i = 0; i < point_count; ++i) {
                const double lhs_homogeneous[3] = { lhs_points[2 * i + 0], lhs_points[2 * i + 1], 1.0 };
                const double rhs_homogeneous[3] = { rhs_points[2 * i + 0], rhs_points[2 * i + 1], 1.0 };
                double essential_lhs[3];
                matrix_vector_multiply(current_essential, lhs_homogeneous, essential_lhs);
                const double residual = rhs_homogeneous[0] * essential_lhs[0] + rhs_homogeneous[1] * essential_lhs[1] + rhs_homogeneous[2] * essential_lhs[2];
                REQUIRE(std::abs(residual) < 1e-6);
            }
        }

        double degenerate_lhs_points[10] = {};
        double degenerate_rhs_points[10] = {};
        double degenerate_essentials[10 * 9] = {};
        solutions = estimation::minimal::essential_5_point<double>::solve(degenerate_lhs_points, degenerate_rhs_points, degenerate_essentials);
        REQUIRE(solutions == 0);
    }

    // Coplanar points.
    {
        const double alpha = 0.2;
        const double beta = -0.15;
        const double gamma = 0.08;
        const double rotation_x[3][3] = { { 1, 0, 0 }, { 0, std::cos(alpha), -std::sin(alpha) }, { 0, std::sin(alpha), std::cos(alpha) } };
        const double rotation_y[3][3] = { { std::cos(beta), 0, std::sin(beta) }, { 0, 1, 0 }, { -std::sin(beta), 0, std::cos(beta) } };
        const double rotation_z[3][3] = { { std::cos(gamma), -std::sin(gamma), 0 }, { std::sin(gamma), std::cos(gamma), 0 }, { 0, 0, 1 } };
        double temp[9];
        matrix_multiply(&rotation_z[0][0], &rotation_y[0][0], temp);
        double rotation[9];
        matrix_multiply(temp, &rotation_x[0][0], rotation);
        const double translation[3] = { 0.4, -0.25, 0.6 };
        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double zero[3] = { 0, 0, 0 };

        const int point_count = 5;
        const double plane_xy[5][2] = { { -0.4, -0.3 }, { 0.5, -0.2 }, { 0.3, 0.6 }, { -0.5, 0.4 }, { 0.1, 0.1 } };
        double lhs_points[10];
        double rhs_points[10];
        for (int i = 0; i < point_count; ++i) {
            const double world_point[3] = { plane_xy[i][0], plane_xy[i][1], 4.0 };
            project_point(identity, zero, world_point, &lhs_points[2 * i]);
            project_point(rotation, translation, world_point, &rhs_points[2 * i]);
        }

        double essentials[10 * 9] = {};
        const int solutions = estimation::minimal::essential_5_point<double>::solve(lhs_points, rhs_points, essentials);
        REQUIRE(solutions >= 0);
        REQUIRE(solutions <= 10);
        for (int s = 0; s < solutions; ++s) {
            double* current_essential = &essentials[s * 9];
            for (int k = 0; k < 9; ++k) {
                REQUIRE(std::isfinite(current_essential[k]));
            }
            normalize_matrix(current_essential);
            for (int i = 0; i < point_count; ++i) {
                const double lhs_homogeneous[3] = { lhs_points[2 * i + 0], lhs_points[2 * i + 1], 1.0 };
                const double rhs_homogeneous[3] = { rhs_points[2 * i + 0], rhs_points[2 * i + 1], 1.0 };
                double essential_lhs[3];
                matrix_vector_multiply(current_essential, lhs_homogeneous, essential_lhs);
                const double residual = rhs_homogeneous[0] * essential_lhs[0] + rhs_homogeneous[1] * essential_lhs[1] + rhs_homogeneous[2] * essential_lhs[2];
                REQUIRE(std::abs(residual) < 1e-6);
            }
        }
    }

    // Identical correspondences.
    {
        double lhs_points[10];
        double rhs_points[10];
        for (int i = 0; i < 5; ++i) {
            lhs_points[2 * i + 0] = 0.12;
            lhs_points[2 * i + 1] = -0.07;
            rhs_points[2 * i + 0] = 0.20;
            rhs_points[2 * i + 1] = 0.05;
        }

        double essentials[10 * 9] = {};
        const int solutions = estimation::minimal::essential_5_point<double>::solve(lhs_points, rhs_points, essentials);
        REQUIRE(solutions >= 0);
        REQUIRE(solutions <= 10);
        for (int s = 0; s < solutions; ++s) {
            double* current_essential = &essentials[s * 9];
            for (int k = 0; k < 9; ++k) {
                REQUIRE(std::isfinite(current_essential[k]));
            }
            const double lhs_homogeneous[3] = { lhs_points[0], lhs_points[1], 1.0 };
            const double rhs_homogeneous[3] = { rhs_points[0], rhs_points[1], 1.0 };
            double essential_lhs[3];
            matrix_vector_multiply(current_essential, lhs_homogeneous, essential_lhs);
            const double residual = rhs_homogeneous[0] * essential_lhs[0] + rhs_homogeneous[1] * essential_lhs[1] + rhs_homogeneous[2] * essential_lhs[2];
            REQUIRE(std::abs(residual) < 1e-6);
        }
    }

    return EXIT_SUCCESS;
}
