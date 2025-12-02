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

#include "pose_estimation.hpp"

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
static inline bool are_values_approx(const array_type& lhs, const array_type& rhs, unsigned long long int length, double epsilon = 1e-8) {
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

    constexpr static const auto matrix_multiply = [](const double* lhs, const double* rhs, double* result) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                double sum = 0.0;
                for (int k = 0; k < 3; ++k) {
                    sum += lhs[r * 3 + k] * rhs[k * 3 + c];
                }
                result[r * 3 + c] = sum;
            }
        }
    };
    constexpr static const auto matrix_vector_multiply = [](const double* matrix, const double* vector, double* result) {
        for (int r = 0; r < 3; ++r) {
            result[r] = matrix[r * 3 + 0] * vector[0] + matrix[r * 3 + 1] * vector[1] + matrix[r * 3 + 2] * vector[2];
        }
    };
    constexpr static const auto cross_matrix = [](const double* vector, double* matrix) {
        matrix[0] = 0;
        matrix[1] = -vector[2];
        matrix[2] = vector[1];
        matrix[3] = vector[2];
        matrix[4] = 0;
        matrix[5] = -vector[0];
        matrix[6] = -vector[1];
        matrix[7] = vector[0];
        matrix[8] = 0;
    };
    constexpr static const auto frobenius_norm = [](const double* matrix) -> double {
        double sum = 0.0;
        for (int i = 0; i < 9; ++i) {
            sum += matrix[i] * matrix[i];
        }
        return std::sqrt(sum);
    };
    constexpr static const auto normalize_matrix = [](double* matrix) {
        const double norm = frobenius_norm(matrix);
        if (norm > 0) {
            for (int i = 0; i < 9; ++i) {
                matrix[i] /= norm;
            }
        }
    };
    constexpr static const auto normalize_vector = [](double* vector) {
        const double norm = std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
        if (norm > 0) {
            for (int i = 0; i < 3; ++i) {
                vector[i] /= norm;
            }
        }
    };
    constexpr static const auto matrix_determinant = [](const double* matrix) -> double {
        return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
               matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
               matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    };
    constexpr static const auto matrix_transpose = [](const double* matrix, double* transposed) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                transposed[c * 3 + r] = matrix[r * 3 + c];
            }
        }
    };
    constexpr static const auto is_rotation_matrix = [](const double* matrix, double epsilon = 1e-6) -> bool {
        // Compute transpose(matrix) * matrix.
        double transposed[9];
        matrix_transpose(matrix, transposed);
        double identity[9];
        matrix_multiply(transposed, matrix, identity);
        // Compare to identity
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                const double expect = (r == c) ? 1.0 : 0.0;
                if (!is_value_approx(identity[r * 3 + c], expect, epsilon)) {
                    return false;
                }
            }
        }
        // Check determinant is one.
        const double determinant = matrix_determinant(matrix);
        if (!is_value_approx(determinant, 1.0, 1e-6)) {
            return false;
        }
        return true;
    };

    constexpr static const auto project_point = [](const double* rotation, const double* translation, const double* point_xyz, double* point_xy) {
        double point[3];
        matrix_vector_multiply(rotation, point_xyz, point);
        point[0] += translation[0];
        point[1] += translation[1];
        point[2] += translation[2];
        point_xy[0] = point[0] / point[2];
        point_xy[1] = point[1] / point[2];
    };
    constexpr static const auto reprojection_error = [](const double* observation_xy, const double* rotation, const double* translation, const double* point_xyz) -> double {
        double xy[2];
        project_point(rotation, translation, point_xyz, xy);
        double dx = observation_xy[0] - xy[0];
        double dy = observation_xy[1] - xy[1];
        return (dx * dx) + (dy * dy);
    };

    // 1) Test composing essential matrices matches (up to scale).
    {
        const double rotation[3][3] = {
            { +0.36, -0.48, +0.8 },
            { +0.8, +0.6, 0.0 },
            { -0.48, +0.64, +0.6 }
        };
        const double translation[3] = { 0.5, -1.2, 2.3 };
        double translation_matrix[9];
        cross_matrix(translation, translation_matrix);
        double expected[9];
        matrix_multiply(translation_matrix, &rotation[0][0], expected);
        normalize_matrix(expected);
        double essential[9];
        pose_estimation::essential_matrix<double>::compose(&rotation[0][0], translation, essential);
        normalize_matrix(essential);
        REQUIRE(are_values_approx(essential, expected, 9, 1e-10));
    }

    // 2) Test decompose: compose -> decompose -> recomposition matches (up to scale).
    {
        const double theta = 0.37;
        const double rotation[3][3] = {
            { std::cos(theta), -std::sin(theta), 0.0 },
            { std::sin(theta), std::cos(theta), 0.0 },
            { 0.0, 0.0, 1.0 }
        };
        const double translation[3] = { 0.3, 0.7, 1.2 };
        double translation_matrix[9];
        cross_matrix(translation, translation_matrix);
        double expected[9];
        matrix_multiply(translation_matrix, &rotation[0][0], expected);
        normalize_matrix(expected);

        double rotation_0[9];
        double rotation_1[9];
        double translation_0[3];
        double translation_1[3];
        pose_estimation::essential_matrix<double>::decompose(expected, rotation_0, rotation_1, translation_0, translation_1);

        // Rotations should be orthonormal.
        REQUIRE(is_rotation_matrix(rotation_0));
        REQUIRE(is_rotation_matrix(rotation_1));

        // Translations should be non-zero vectors.
        const double norm_translation_0 = std::sqrt(translation_0[0] * translation_0[0] + translation_0[1] * translation_0[1] + translation_0[2] * translation_0[2]);
        const double norm_translation_1 = std::sqrt(translation_1[0] * translation_1[0] + translation_1[1] * translation_1[1] + translation_1[2] * translation_1[2]);
        REQUIRE(norm_translation_0 > 1e-9);
        REQUIRE(norm_translation_1 > 1e-9);

        // Check that one combination recreates correct essential matrix (up to scale).
        bool found_match = false;
        for (int i = 0; i < 2 && !found_match; ++i) {
            const double* current_rotation = (i == 0) ? rotation_0 : rotation_1;
            for (int j = 0; j < 2 && !found_match; ++j) {
                const double* current_translation = (j == 0) ? translation_0 : translation_1;
                double current_translation_matrix[9];
                cross_matrix(current_translation, current_translation_matrix);
                double current_essential[9];
                matrix_multiply(current_translation_matrix, current_rotation, current_essential);
                normalize_matrix(current_essential);
                if (are_values_approx(current_essential, expected, 9, 1e-6)) {
                    found_match = true;
                }
                for (int k = 0; k < 9; ++k) {
                    current_essential[k] = -current_essential[k];
                }
                if (are_values_approx(current_essential, expected, 9, 1e-5)) {
                    found_match = true;
                }
            }
        }
        REQUIRE(found_match);
    }

    // 3) Test essential_5_point: synthetic scene with known transformation and 5 normalised correspondences.
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
        // R = Rz * Ry * Rx
        matrix_multiply(temp, &rotation_x[0][0], rotation);
        const double translation[3] = { 0.5, -0.3, 1.0 };
        // For other camera use the origin.
        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double zero[3] = { 0, 0, 0 };
        // Create five world points (not coplanar).
        const int point_count = 5;
        double world_points[5][3] = {
            { 0.1, 0.2, 3.0 },
            { -0.5, 0.4, 4.2 },
            { 0.7, -0.3, 5.1 },
            { -0.2, -0.1, 2.7 },
            { 0.0, 0.0, 6.0 }
        };

        // Project to normalized image coords for two cameras:
        // Left camera is identity pose (R=I, t=0), Right camera has (R=rotation, t=translation).
        double lhs_points[10];
        double rhs_points[10];
        for (int i = 0; i < point_count; ++i) {
            project_point(identity, zero, &world_points[i][0], &lhs_points[2 * i]);
            project_point(rotation, translation, &world_points[i][0], &rhs_points[2 * i]);
        }

        // Ground truth essential.
        double translation_matrix[9];
        cross_matrix(translation, translation_matrix);
        double expected[9];
        matrix_multiply(translation_matrix, rotation, expected);
        normalize_matrix(expected);

        // Call essential_5_point
        double essentials[10 * 9] = {};
        int solutions = pose_estimation::essential_5_point<double>(lhs_points, rhs_points, essentials);
        REQUIRE(solutions >= 1);
        REQUIRE(solutions <= 10);

        // Check that among returned essentials one matches (up to scale).
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

    // 4) Test recover_pose: use the true essential (from previous construction), call recover_pose and check reprojection of triangulated points
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
        // R = Rz * Ry * Rx
        matrix_multiply(temp, &rotation_x[0][0], rotation);
        double translation[3] = { 0.5, -0.3, 1.0 };
        // For other camera use the origin.
        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double zero[3] = { 0, 0, 0 };
        // Create more than five world points (not coplanar).
        const int point_count = 6;
        double world_points[6][3] = {
            { 0.1, 0.2, 3.0 },
            { -0.5, 0.4, 4.2 },
            { 0.7, -0.3, 5.1 },
            { -0.2, -0.1, 2.7 },
            { 0.0, 0.0, 6.0 },
            { 0.3, -0.25, 4.0 }
        };

        double lhs_points[12];
        double rhs_points[12];
        for (int i = 0; i < point_count; ++i) {
            project_point(identity, zero, &world_points[i][0], &lhs_points[2 * i]);
            project_point(rotation, translation, &world_points[i][0], &rhs_points[2 * i]);
        }

        // Compute the true essential.
        double translation_matrix[9];
        cross_matrix(translation, translation_matrix);
        double essential[9];
        matrix_multiply(translation_matrix, rotation, essential);
        normalize_matrix(essential);

        // Call recover_pose with the essential.
        double recovered_rotation[9];
        double recovered_translation[3];
        double points_xyz[point_count * 3] = {};
        pose_estimation::essential_matrix<double>::recover_pose(essential, lhs_points, rhs_points, static_cast<size_t>(point_count), recovered_rotation, recovered_translation, points_xyz);

        // Calculate reprojection error.
        double total_error = 0.0;
        for (int i = 0; i < point_count; ++i) {
            const double Xtri[3] = { points_xyz[3 * i + 0], points_xyz[3 * i + 1], points_xyz[3 * i + 2] };
            const double e1 = reprojection_error(lhs_points + 2 * i, identity, zero, Xtri);
            const double e2 = reprojection_error(rhs_points + 2 * i, recovered_rotation, recovered_translation, Xtri);
            total_error += (e1 + e2);
        }
        const double mean_error = total_error / double(point_count);
        REQUIRE(mean_error < 1e-5);

        // Rotation should be approximately equal to true rotation (or one of the two valid rotations).
        // Compare normalized rotation matrices (they might differ by sign conventions).
        normalize_matrix(rotation);
        normalize_matrix(recovered_rotation);
        bool rotation_correct = are_values_approx(rotation, recovered_rotation, 9, 1e-3);
        if (!rotation_correct) {
            for (int i = 0; i < 9; ++i) {
                recovered_rotation[i] = -recovered_rotation[i];
            }
            rotation_correct = are_values_approx(rotation, recovered_rotation, 9, 1e-3);
        }
        REQUIRE(rotation_correct);

        // Translation direction should be parallel to true translation.
        normalize_vector(translation);
        normalize_vector(recovered_translation);
        // The dot product should be close to +/-1.
        double dot = std::abs(translation[0] * recovered_translation[0] + translation[1] * recovered_translation[1] + translation[2] * recovered_translation[2]);
        REQUIRE(dot > 0.98);
    }

    return EXIT_SUCCESS;
}