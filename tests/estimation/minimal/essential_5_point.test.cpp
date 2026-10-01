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

static inline void rotation_from_axis_angle(const double* axis_angle, double* rotation) {
    const double angle = std::sqrt(axis_angle[0] * axis_angle[0] + axis_angle[1] * axis_angle[1] + axis_angle[2] * axis_angle[2]);
    if (!(angle > 0.0)) {
        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        for (int i = 0; i < 9; ++i) {
            rotation[i] = identity[i];
        }
        return;
    }
    const double axis[3] = { axis_angle[0] / angle, axis_angle[1] / angle, axis_angle[2] / angle };
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double v = 1.0 - c;
    rotation[0] = c + axis[0] * axis[0] * v;
    rotation[1] = axis[0] * axis[1] * v - axis[2] * s;
    rotation[2] = axis[0] * axis[2] * v + axis[1] * s;
    rotation[3] = axis[1] * axis[0] * v + axis[2] * s;
    rotation[4] = c + axis[1] * axis[1] * v;
    rotation[5] = axis[1] * axis[2] * v - axis[0] * s;
    rotation[6] = axis[2] * axis[0] * v - axis[1] * s;
    rotation[7] = axis[2] * axis[1] * v + axis[0] * s;
    rotation[8] = c + axis[2] * axis[2] * v;
}

// The smallest distance between a solution and the expected unit matrix, of either sign.
static inline double closest_solution(const double* essentials, int solutions, const double* expected) {
    double closest = 1e9;
    for (int s = 0; s < solutions; ++s) {
        double current[9];
        for (int k = 0; k < 9; ++k) {
            current[k] = essentials[s * 9 + k];
        }
        normalize_matrix(current);
        double plus = 0.0;
        double minus = 0.0;
        for (int k = 0; k < 9; ++k) {
            plus += (current[k] - expected[k]) * (current[k] - expected[k]);
            minus += (current[k] + expected[k]) * (current[k] + expected[k]);
        }
        closest = std::fmin(closest, std::sqrt(std::fmin(plus, minus)));
    }
    return closest;
}

// Five correspondences of points at depths 2 to 6 under a random rotation of up to 0.15 radians, with the baseline scaled so the mean parallax is the given angle.
static inline bool make_parallax_sample(core::random_pcg& random, double parallax_degrees, double* lhs_points, double* rhs_points, double* expected) {
    const double axis_angle[3] = { 0.087 * random.get_random(-1.0, 1.0), 0.087 * random.get_random(-1.0, 1.0), 0.087 * random.get_random(-1.0, 1.0) };
    double rotation[9];
    rotation_from_axis_angle(axis_angle, rotation);
    double translation[3] = { random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0) };
    const double translation_norm = std::sqrt(translation[0] * translation[0] + translation[1] * translation[1] + translation[2] * translation[2]);
    for (int k = 0; k < 3; ++k) {
        translation[k] /= translation_norm;
    }
    double points[5][3];
    for (int i = 0; i < 5; ++i) {
        const double depth = random.get_random(2.0, 6.0);
        points[i][0] = 0.6 * random.get_random(-1.0, 1.0) * depth;
        points[i][1] = 0.45 * random.get_random(-1.0, 1.0) * depth;
        points[i][2] = depth;
    }
    const double centre[3] = {
        -(rotation[0] * translation[0] + rotation[3] * translation[1] + rotation[6] * translation[2]),
        -(rotation[1] * translation[0] + rotation[4] * translation[1] + rotation[7] * translation[2]),
        -(rotation[2] * translation[0] + rotation[5] * translation[1] + rotation[8] * translation[2])
    };
    double parallax_mean = 0.0;
    for (int i = 0; i < 5; ++i) {
        const double from_centre[3] = { points[i][0] - centre[0], points[i][1] - centre[1], points[i][2] - centre[2] };
        const double dot = points[i][0] * from_centre[0] + points[i][1] * from_centre[1] + points[i][2] * from_centre[2];
        const double cross[3] = {
            points[i][1] * from_centre[2] - points[i][2] * from_centre[1],
            points[i][2] * from_centre[0] - points[i][0] * from_centre[2],
            points[i][0] * from_centre[1] - points[i][1] * from_centre[0]
        };
        parallax_mean += std::atan2(std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]), dot) / 5.0;
    }
    const double scale = (parallax_degrees * 3.14159265358979323846 / 180.0) / parallax_mean;
    for (int k = 0; k < 3; ++k) {
        translation[k] *= scale;
    }
    for (int i = 0; i < 5; ++i) {
        double rhs_camera[3];
        matrix_vector_multiply(rotation, points[i], rhs_camera);
        for (int k = 0; k < 3; ++k) {
            rhs_camera[k] += translation[k];
        }
        if (rhs_camera[2] < 0.1) {
            return false;
        }
        lhs_points[2 * i + 0] = points[i][0] / points[i][2];
        lhs_points[2 * i + 1] = points[i][1] / points[i][2];
        rhs_points[2 * i + 0] = rhs_camera[0] / rhs_camera[2];
        rhs_points[2 * i + 1] = rhs_camera[1] / rhs_camera[2];
    }
    double translation_matrix[9];
    cross_matrix(translation, translation_matrix);
    matrix_multiply(translation_matrix, rotation, expected);
    normalize_matrix(expected);
    return true;
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

    // Translation along an axis without rotation makes y' = y, x' = x or x' y = y' x hold at every point, which ties two columns of the constraint matrix; every sample keeps the true matrix, and again with a small rotation.
    {
        const double translations[3][3] = { { 0.5, 0.0, 0.0 }, { 0.0, 0.5, 0.0 }, { 0.0, 0.0, 0.5 } };
        for (int axis = 0; axis < 3; ++axis) {
            for (int rotated = 0; rotated < 2; ++rotated) {
                const double axis_angle[3] = { 0.0, (rotated == 1) ? 0.05 : 0.0, 0.0 };
                double rotation[9];
                rotation_from_axis_angle(axis_angle, rotation);
                double translation_matrix[9];
                cross_matrix(translations[axis], translation_matrix);
                double expected[9];
                matrix_multiply(translation_matrix, rotation, expected);
                normalize_matrix(expected);

                core::random_pcg random;
                double lhs_points[200 * 2];
                double rhs_points[200 * 2];
                for (int i = 0; i < 200; ++i) {
                    const double point_xyz[3] = {
                        (static_cast<double>(random.get_random_raw() % 6000) / 1000.0) - 3.0,
                        (static_cast<double>(random.get_random_raw() % 6000) / 1000.0) - 3.0,
                        4.0 + (static_cast<double>(random.get_random_raw() % 10000) / 1000.0)
                    };
                    lhs_points[2 * i + 0] = point_xyz[0] / point_xyz[2];
                    lhs_points[2 * i + 1] = point_xyz[1] / point_xyz[2];
                    project_point(rotation, translations[axis], point_xyz, &rhs_points[2 * i]);
                }
                for (int sample = 0; sample < 40; ++sample) {
                    double essentials[10 * 9] = {};
                    const int solutions = estimation::minimal::essential_5_point<double>::solve(&lhs_points[sample * 10], &rhs_points[sample * 10], essentials);
                    REQUIRE(solutions >= 1);
                    REQUIRE(closest_solution(essentials, solutions, expected) < 1e-6);
                }
            }
        }
    }

    // Half a degree of parallax: samples keep their solutions, which have unit norm.
    {
        core::random_pcg random(0x5eed0101ull);
        int samples = 0;
        int without_solutions = 0;
        int missed = 0;
        while (samples < 1000) {
            double lhs_points[10];
            double rhs_points[10];
            double expected[9];
            if (!make_parallax_sample(random, 0.5, lhs_points, rhs_points, expected)) {
                continue;
            }
            ++samples;
            double essentials[10 * 9] = {};
            const int solutions = estimation::minimal::essential_5_point<double>::solve(lhs_points, rhs_points, essentials);
            if (solutions == 0) {
                ++without_solutions;
            }
            for (int s = 0; s < solutions; ++s) {
                REQUIRE(is_value_approx(frobenius_norm(&essentials[s * 9]), 1.0, 1e-12));
            }
            if (!(closest_solution(essentials, solutions, expected) < 1e-3)) {
                ++missed;
            }
        }
        REQUIRE(without_solutions <= 5);
        REQUIRE(missed <= 5);
    }

    // Single precision is solved in double precision, so the true matrix is found to single precision.
    {
        core::random_pcg random(0x5eed0102ull);
        int samples = 0;
        int missed = 0;
        while (samples < 500) {
            double lhs_points[10];
            double rhs_points[10];
            double expected[9];
            if (!make_parallax_sample(random, 4.0, lhs_points, rhs_points, expected)) {
                continue;
            }
            ++samples;
            float lhs_points_float[10];
            float rhs_points_float[10];
            for (int i = 0; i < 10; ++i) {
                lhs_points_float[i] = static_cast<float>(lhs_points[i]);
                rhs_points_float[i] = static_cast<float>(rhs_points[i]);
            }
            float essentials_float[10 * 9] = {};
            const int solutions = estimation::minimal::essential_5_point<float>::solve(lhs_points_float, rhs_points_float, essentials_float);
            double essentials[10 * 9] = {};
            for (int i = 0; i < solutions * 9; ++i) {
                essentials[i] = static_cast<double>(essentials_float[i]);
            }
            if (!(closest_solution(essentials, solutions, expected) < 1e-3)) {
                ++missed;
            }
        }
        REQUIRE(missed <= 3);
    }

    return EXIT_SUCCESS;
}
