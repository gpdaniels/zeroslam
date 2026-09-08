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

#include "estimation/pose/homography.hpp"

#include "estimation/minimal/homography_4_point.hpp"

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

static inline void normalize_vector(double* vector) {
    const double norm = std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
    if (norm > 0) {
        for (int i = 0; i < 3; ++i) {
            vector[i] /= norm;
        }
    }
}

static inline double matrix_determinant(const double* matrix) {
    return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
           matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
           matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

static inline void matrix_transpose(const double* matrix, double* transposed) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            transposed[c * 3 + r] = matrix[r * 3 + c];
        }
    }
}

static inline bool is_rotation_matrix(const double* matrix, double epsilon = 1e-6) {
    double transposed[9];
    matrix_transpose(matrix, transposed);
    double identity[9];
    matrix_multiply(transposed, matrix, identity);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (!is_value_approx(identity[r * 3 + c], (r == c) ? 1.0 : 0.0, epsilon)) {
                return false;
            }
        }
    }
    return is_value_approx(matrix_determinant(matrix), 1.0, 1e-6);
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

static inline double reprojection_error(const double* observation_xy, const double* rotation, const double* translation, const double* point_xyz) {
    double xy[2];
    project_point(rotation, translation, point_xyz, xy);
    const double dx = observation_xy[0] - xy[0];
    const double dy = observation_xy[1] - xy[1];
    return (dx * dx) + (dy * dy);
}

static inline void build_planar_scene(double* rotation_gt, double* translation_gt, double* lhs_points, double* rhs_points, int point_count) {
    const double alpha = 0.15;
    const double beta = -0.22;
    const double gamma = 0.1;
    const double rotation_x[3][3] = { { 1, 0, 0 }, { 0, std::cos(alpha), -std::sin(alpha) }, { 0, std::sin(alpha), std::cos(alpha) } };
    const double rotation_y[3][3] = { { std::cos(beta), 0, std::sin(beta) }, { 0, 1, 0 }, { -std::sin(beta), 0, std::cos(beta) } };
    const double rotation_z[3][3] = { { std::cos(gamma), -std::sin(gamma), 0 }, { std::sin(gamma), std::cos(gamma), 0 }, { 0, 0, 1 } };
    double temp[9];
    matrix_multiply(&rotation_z[0][0], &rotation_y[0][0], temp);
    matrix_multiply(temp, &rotation_x[0][0], rotation_gt);
    translation_gt[0] = 0.8;
    translation_gt[1] = 0.6;
    translation_gt[2] = 0.2;
    normalize_vector(translation_gt);
    const double plane_xy[12][2] = { { -0.4, -0.3 }, { 0.5, -0.2 }, { 0.3, 0.6 }, { -0.5, 0.4 }, { 0.1, 0.1 }, { 0.6, 0.5 }, { -0.2, 0.5 }, { 0.0, -0.5 }, { 0.4, 0.2 }, { -0.3, -0.1 }, { 0.55, -0.45 }, { -0.45, 0.15 } };
    const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const double zero[3] = { 0, 0, 0 };
    for (int i = 0; i < point_count; ++i) {
        const double world_point[3] = { plane_xy[i][0], plane_xy[i][1], 4.0 + 0.3 * plane_xy[i][0] - 0.2 * plane_xy[i][1] };
        project_point(identity, zero, world_point, &lhs_points[2 * i]);
        project_point(rotation_gt, translation_gt, world_point, &rhs_points[2 * i]);
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // Known planar motion.
    {
        const int point_count = 12;
        double rotation_gt[9];
        double translation_gt[3];
        double lhs_points[24];
        double rhs_points[24];
        build_planar_scene(rotation_gt, translation_gt, lhs_points, rhs_points, point_count);

        double homography[9];
        const bool homography_ok = estimation::minimal::homography_4_point<double>::solve(lhs_points, rhs_points, homography);
        REQUIRE(homography_ok);

        {
            const double px = homography[0] * rhs_points[2 * 5 + 0] + homography[1] * rhs_points[2 * 5 + 1] + homography[2];
            const double py = homography[3] * rhs_points[2 * 5 + 0] + homography[4] * rhs_points[2 * 5 + 1] + homography[5];
            const double pw = homography[6] * rhs_points[2 * 5 + 0] + homography[7] * rhs_points[2 * 5 + 1] + homography[8];
            REQUIRE(is_value_approx(px / pw, lhs_points[2 * 5 + 0], 1e-9));
            REQUIRE(is_value_approx(py / pw, lhs_points[2 * 5 + 1], 1e-9));
        }

        double recovered_rotation[9];
        double recovered_translation[3];
        double points_xyz[12 * 3] = {};
        size_t support_count = 0;
        const bool recovered = estimation::pose::homography<double>::recover(homography, lhs_points, rhs_points, static_cast<size_t>(point_count), recovered_rotation, recovered_translation, points_xyz, &support_count);
        REQUIRE(recovered);
        REQUIRE(support_count == static_cast<size_t>(point_count));

        REQUIRE(is_rotation_matrix(recovered_rotation, 1e-6));
        REQUIRE(are_values_approx(recovered_rotation, rotation_gt, 9, 1e-6));

        double translation_direction[3] = { recovered_translation[0], recovered_translation[1], recovered_translation[2] };
        normalize_vector(translation_direction);
        const double translation_dot = translation_direction[0] * translation_gt[0] + translation_direction[1] * translation_gt[1] + translation_direction[2] * translation_gt[2];
        REQUIRE(translation_dot > 0.999);

        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double zero[3] = { 0, 0, 0 };
        double total_error = 0.0;
        for (int i = 0; i < point_count; ++i) {
            const double point_xyz[3] = { points_xyz[3 * i + 0], points_xyz[3 * i + 1], points_xyz[3 * i + 2] };
            total_error += reprojection_error(lhs_points + 2 * i, identity, zero, point_xyz);
            total_error += reprojection_error(rhs_points + 2 * i, recovered_rotation, recovered_translation, point_xyz);
        }
        REQUIRE((total_error / double(point_count)) < 1e-5);
    }

    // Camera to world convention.
    {
        const int point_count = 12;
        double rotation_gt[9];
        double translation_gt[3];
        double lhs_points[24];
        double rhs_points[24];
        build_planar_scene(rotation_gt, translation_gt, lhs_points, rhs_points, point_count);

        double homography[9];
        REQUIRE(estimation::minimal::homography_4_point<double>::solve(lhs_points, rhs_points, homography));

        double recovered_rotation[9];
        double recovered_translation[3];
        double points_xyz[12 * 3] = {};
        const bool recovered = estimation::pose::homography<double>::recover(homography, lhs_points, rhs_points, static_cast<size_t>(point_count), recovered_rotation, recovered_translation, points_xyz, nullptr);
        REQUIRE(recovered);

        double inverse_rotation[9];
        matrix_transpose(recovered_rotation, inverse_rotation);
        double inverse_translation[3];
        matrix_vector_multiply(inverse_rotation, recovered_translation, inverse_translation);
        inverse_translation[0] = -inverse_translation[0];
        inverse_translation[1] = -inverse_translation[1];
        inverse_translation[2] = -inverse_translation[2];

        double absolute_rotation[9];
        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        matrix_multiply(inverse_rotation, identity, absolute_rotation);
        const double absolute_translation[3] = { inverse_translation[0], inverse_translation[1], inverse_translation[2] };

        double ground_truth_absolute_rotation[9];
        matrix_transpose(rotation_gt, ground_truth_absolute_rotation);
        double ground_truth_absolute_translation[3];
        matrix_vector_multiply(ground_truth_absolute_rotation, translation_gt, ground_truth_absolute_translation);
        ground_truth_absolute_translation[0] = -ground_truth_absolute_translation[0];
        ground_truth_absolute_translation[1] = -ground_truth_absolute_translation[1];
        ground_truth_absolute_translation[2] = -ground_truth_absolute_translation[2];

        REQUIRE(are_values_approx(absolute_rotation, ground_truth_absolute_rotation, 9, 1e-6));
        REQUIRE(are_values_approx(absolute_translation, ground_truth_absolute_translation, 3, 1e-6));
    }

    // Degenerate homographies are rejected.
    {
        const int point_count = 12;
        double rotation_gt[9];
        double translation_gt[3];
        double lhs_points[24];
        double rhs_points[24];
        build_planar_scene(rotation_gt, translation_gt, lhs_points, rhs_points, point_count);

        double recovered_rotation[9];
        double recovered_translation[3];
        double points_xyz[12 * 3] = {};
        size_t support_count = 0;

        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double zero[3] = { 0, 0, 0 };
        const double plane_xy[12][2] = {
            { -0.4, -0.3 },
            { 0.5, -0.2 },
            { 0.3, 0.6 },
            { -0.5, 0.4 },
            { 0.1, 0.1 },
            { 0.6, 0.5 },
            { -0.2, 0.5 },
            { 0.0, -0.5 },
            { 0.4, 0.2 },
            { -0.3, -0.1 },
            { 0.55, -0.45 },
            { -0.45, 0.15 }
        };
        double rotation_only_lhs[24];
        double rotation_only_rhs[24];
        for (int i = 0; i < point_count; ++i) {
            const double world_point[3] = {
                plane_xy[i][0],
                plane_xy[i][1],
                4.0 + 0.3 * plane_xy[i][0] - 0.2 * plane_xy[i][1]
            };
            project_point(identity, zero, world_point, &rotation_only_lhs[2 * i]);
            project_point(rotation_gt, zero, world_point, &rotation_only_rhs[2 * i]);
        }
        double rotation_only_homography[9];
        REQUIRE(estimation::minimal::homography_4_point<double>::solve(rotation_only_lhs, rotation_only_rhs, rotation_only_homography));
        const bool recovered_rotation_only = estimation::pose::homography<double>::recover(rotation_only_homography, rotation_only_lhs, rotation_only_rhs, static_cast<size_t>(point_count), recovered_rotation, recovered_translation, points_xyz, &support_count);
        REQUIRE(!recovered_rotation_only);

        const double singular_homography[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 0 };
        const bool recovered_singular = estimation::pose::homography<double>::recover(singular_homography, lhs_points, rhs_points, static_cast<size_t>(point_count), recovered_rotation, recovered_translation, points_xyz, &support_count);
        REQUIRE(!recovered_singular);
    }

    return EXIT_SUCCESS;
}
