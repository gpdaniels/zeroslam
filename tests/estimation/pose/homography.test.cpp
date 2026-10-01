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

#include "core/random_pcg.hpp"
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

static inline void rotation_from_axis_angle(const double* axis_angle, double* rotation) {
    const double angle = std::sqrt(axis_angle[0] * axis_angle[0] + axis_angle[1] * axis_angle[1] + axis_angle[2] * axis_angle[2]);
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

static inline bool matrix_invert(const double* matrix, double* inverse) {
    const double determinant = matrix_determinant(matrix);
    if (std::abs(determinant) < 1e-15) {
        return false;
    }
    inverse[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) / determinant;
    inverse[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) / determinant;
    inverse[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) / determinant;
    inverse[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) / determinant;
    inverse[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) / determinant;
    inverse[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) / determinant;
    inverse[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) / determinant;
    inverse[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) / determinant;
    inverse[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) / determinant;
    return true;
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

        double homography[9];
        REQUIRE(estimation::minimal::homography_4_point<double>::solve(lhs_points, rhs_points, homography));
        REQUIRE(!estimation::pose::homography<double>::recover(homography, lhs_points, rhs_points, 0, recovered_rotation, recovered_translation, points_xyz, &support_count));
        REQUIRE(support_count == 0);
    }

    // Oblique planes, a floor seen by a camera pitched up so the normal has n_z < 0, and planes facing the camera: every accepted pose is the true one.
    {
        core::random_pcg random(0x5eed0070ull);
        size_t trials[2] = { 0, 0 };
        size_t accepted[2] = { 0, 0 };
        for (int trial = 0; trial < 400; ++trial) {
            const size_t oblique = ((trial % 2) == 0) ? size_t(1) : size_t(0);
            double normal[3] = { 0.3 * random.get_random(-1.0, 1.0), 0.3 * random.get_random(-1.0, 1.0), 1.0 };
            if (oblique == 1) {
                const double pitch = random.get_random(0.05, 0.35);
                normal[0] = 0.2 * random.get_random(-1.0, 1.0);
                normal[1] = std::cos(pitch);
                normal[2] = -std::sin(pitch);
            }
            normalize_vector(normal);
            const double distance = 1.5;
            const double axis_angle[3] = { 0.05 * random.get_random(-1.0, 1.0), 0.05 * random.get_random(-1.0, 1.0), 0.05 * random.get_random(-1.0, 1.0) };
            double rotation_gt[9];
            rotation_from_axis_angle(axis_angle, rotation_gt);
            double translation_gt[3] = { 0.3 * random.get_random(-1.0, 1.0), 0.1 * random.get_random(-1.0, 1.0), 0.3 * random.get_random(-1.0, 1.0) };

            double lhs_points[400];
            double rhs_points[400];
            int point_count = 0;
            for (int attempt = 0; (attempt < 5000) && (point_count < 200); ++attempt) {
                const double ray[3] = { 0.6 * random.get_random(-1.0, 1.0), 0.45 * random.get_random(-1.0, 1.0), 1.0 };
                const double denominator = normal[0] * ray[0] + normal[1] * ray[1] + normal[2] * ray[2];
                if (denominator <= 1e-6) {
                    continue;
                }
                const double depth = distance / denominator;
                if (depth > 30.0) {
                    continue;
                }
                const double point_xyz[3] = { ray[0] * depth, ray[1] * depth, ray[2] * depth };
                double rhs_camera[3];
                matrix_vector_multiply(rotation_gt, point_xyz, rhs_camera);
                rhs_camera[0] += translation_gt[0];
                rhs_camera[1] += translation_gt[1];
                rhs_camera[2] += translation_gt[2];
                if (rhs_camera[2] <= 0.1) {
                    continue;
                }
                const double rhs_x = rhs_camera[0] / rhs_camera[2];
                const double rhs_y = rhs_camera[1] / rhs_camera[2];
                if ((std::abs(rhs_x) > 0.6) || (std::abs(rhs_y) > 0.45)) {
                    continue;
                }
                lhs_points[2 * point_count + 0] = ray[0];
                lhs_points[2 * point_count + 1] = ray[1];
                rhs_points[2 * point_count + 0] = rhs_x;
                rhs_points[2 * point_count + 1] = rhs_y;
                ++point_count;
            }
            if (point_count < 100) {
                continue;
            }

            // The homography maps rhs to lhs, its inverse is R + t n^T / d.
            double homography_inverse[9];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    homography_inverse[r * 3 + c] = rotation_gt[r * 3 + c] + translation_gt[r] * normal[c] / distance;
                }
            }
            double homography[9];
            REQUIRE(matrix_invert(homography_inverse, homography));

            double recovered_rotation[9];
            double recovered_translation[3];
            double points_xyz[200 * 3];
            size_t support_count = 0;
            ++trials[oblique];
            if (!estimation::pose::homography<double>::recover(homography, lhs_points, rhs_points, static_cast<size_t>(point_count), recovered_rotation, recovered_translation, points_xyz, &support_count)) {
                continue;
            }
            ++accepted[oblique];
            REQUIRE(support_count > 0);
            REQUIRE(support_count <= static_cast<size_t>(point_count));
            REQUIRE(is_rotation_matrix(recovered_rotation, 1e-9));
            REQUIRE(are_values_approx(recovered_rotation, rotation_gt, 9, 1e-6));
            normalize_vector(translation_gt);
            const double translation_dot = recovered_translation[0] * translation_gt[0] + recovered_translation[1] * translation_gt[1] + recovered_translation[2] * translation_gt[2];
            REQUIRE(translation_dot > 1.0 - 1e-9);
        }
        REQUIRE(trials[0] > 150);
        REQUIRE(trials[1] > 150);
        REQUIRE(accepted[0] > 10);
        REQUIRE(accepted[1] > 10);
    }

    return EXIT_SUCCESS;
}
