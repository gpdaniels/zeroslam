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

#include "estimation/minimal/perspective_3_point.hpp"

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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // Pose recovery.
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
        double temp_r[9];
        matrix_multiply(&rotation_z[0][0], &rotation_y[0][0], temp_r);
        double gt_rotation[9];
        matrix_multiply(temp_r, &rotation_x[0][0], gt_rotation);
        const double gt_translation[3] = { 0.5, -0.3, 1.0 };

        const double world_points[3][3] = {
            { 0.1, 0.2, 3.0 },
            { -0.5, 0.4, 4.2 },
            { 0.7, -0.3, 5.1 }
        };

        double bearing_vectors[3][3];
        for (int i = 0; i < 3; ++i) {
            double p[3];
            matrix_vector_multiply(gt_rotation, &world_points[i][0], p);
            p[0] += gt_translation[0];
            p[1] += gt_translation[1];
            p[2] += gt_translation[2];
            const double norm = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            bearing_vectors[i][0] = p[0] / norm;
            bearing_vectors[i][1] = p[1] / norm;
            bearing_vectors[i][2] = p[2] / norm;
        }

        double p3p_rotations[4][9];
        double p3p_translations[4][3];
        int n_solutions = estimation::minimal::perspective_3_point<double>::solve(bearing_vectors, world_points, p3p_rotations, p3p_translations);

        REQUIRE(n_solutions >= 1);
        REQUIRE(n_solutions <= 4);

        bool found_match = false;
        for (int s = 0; s < n_solutions && !found_match; ++s) {
            bool r_match = true;
            for (int k = 0; k < 9; ++k) {
                if (!is_value_approx(gt_rotation[k], p3p_rotations[s][k], 1e-4)) {
                    r_match = false;
                    break;
                }
            }

            bool t_match = true;
            for (int k = 0; k < 3; ++k) {
                if (!is_value_approx(gt_translation[k], p3p_translations[s][k], 1e-4)) {
                    t_match = false;
                    break;
                }
            }

            if (r_match && t_match) {
                double total_error = 0.0;
                for (int i = 0; i < 3; ++i) {
                    double proj[3];
                    matrix_vector_multiply(&p3p_rotations[s][0], &world_points[i][0], proj);
                    proj[0] += p3p_translations[s][0];
                    proj[1] += p3p_translations[s][1];
                    proj[2] += p3p_translations[s][2];
                    const double proj_norm = std::sqrt(proj[0] * proj[0] + proj[1] * proj[1] + proj[2] * proj[2]);
                    proj[0] /= proj_norm;
                    proj[1] /= proj_norm;
                    proj[2] /= proj_norm;
                    double dx = proj[0] - bearing_vectors[i][0];
                    double dy = proj[1] - bearing_vectors[i][1];
                    double dz = proj[2] - bearing_vectors[i][2];
                    total_error += dx * dx + dy * dy + dz * dz;
                }
                REQUIRE(total_error < 1e-10);
                REQUIRE(is_rotation_matrix(&p3p_rotations[s][0], 1e-4));
                found_match = true;
            }
        }
        REQUIRE(found_match);
    }

    // Distance consistency.
    {
        const double gt_rotation[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double gt_translation[3] = { 0.0, 0.0, 0.5 };

        const double world_points[3][3] = {
            { -1.0, 0.0, 5.0 },
            { 1.0, 0.0, 5.0 },
            { 0.0, 1.0, 5.0 }
        };

        double bearing_vectors[3][3];
        for (int i = 0; i < 3; ++i) {
            double p[3];
            matrix_vector_multiply(gt_rotation, &world_points[i][0], p);
            p[0] += gt_translation[0];
            p[1] += gt_translation[1];
            p[2] += gt_translation[2];
            const double norm = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            bearing_vectors[i][0] = p[0] / norm;
            bearing_vectors[i][1] = p[1] / norm;
            bearing_vectors[i][2] = p[2] / norm;
        }

        double p3p_rotations[4][9];
        double p3p_translations[4][3];
        int n_solutions = estimation::minimal::perspective_3_point<double>::solve(bearing_vectors, world_points, p3p_rotations, p3p_translations);

        REQUIRE(n_solutions >= 1);

        for (int s = 0; s < n_solutions; ++s) {
            double cam_pts[3][3];
            for (int i = 0; i < 3; ++i) {
                matrix_vector_multiply(&p3p_rotations[s][0], &world_points[i][0], &cam_pts[i][0]);
                cam_pts[i][0] += p3p_translations[s][0];
                cam_pts[i][1] += p3p_translations[s][1];
                cam_pts[i][2] += p3p_translations[s][2];
            }
            for (int i = 0; i < 3; ++i) {
                for (int j = i + 1; j < 3; ++j) {
                    double d_world = 0.0, d_cam = 0.0;
                    for (int k = 0; k < 3; ++k) {
                        double dw = world_points[i][k] - world_points[j][k];
                        double dc = cam_pts[i][k] - cam_pts[j][k];
                        d_world += dw * dw;
                        d_cam += dc * dc;
                    }
                    REQUIRE(is_value_approx(d_world, d_cam, 1e-6));
                }
            }
        }
    }

    // Larger rotation.
    {
        const double angle = 0.7;
        const double gt_rotation[9] = {
            std::cos(angle),
            0.0,
            std::sin(angle),
            0.0,
            1.0,
            0.0,
            -std::sin(angle),
            0.0,
            std::cos(angle)
        };
        const double gt_translation[3] = { 1.0, -0.5, 2.0 };

        const double world_points[3][3] = {
            { 0.3, 0.5, 4.0 },
            { -0.8, -0.2, 3.5 },
            { 0.0, 0.7, 6.0 }
        };

        double bearing_vectors[3][3];
        for (int i = 0; i < 3; ++i) {
            double p[3];
            matrix_vector_multiply(gt_rotation, &world_points[i][0], p);
            p[0] += gt_translation[0];
            p[1] += gt_translation[1];
            p[2] += gt_translation[2];
            const double norm = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            bearing_vectors[i][0] = p[0] / norm;
            bearing_vectors[i][1] = p[1] / norm;
            bearing_vectors[i][2] = p[2] / norm;
        }

        double p3p_rotations[4][9];
        double p3p_translations[4][3];
        int n_solutions = estimation::minimal::perspective_3_point<double>::solve(bearing_vectors, world_points, p3p_rotations, p3p_translations);

        REQUIRE(n_solutions >= 1);
        REQUIRE(n_solutions <= 4);

        bool found_match = false;
        for (int s = 0; s < n_solutions && !found_match; ++s) {
            bool r_match = true;
            for (int k = 0; k < 9; ++k) {
                if (!is_value_approx(gt_rotation[k], p3p_rotations[s][k], 1e-4)) {
                    r_match = false;
                    break;
                }
            }

            bool t_match = true;
            for (int k = 0; k < 3; ++k) {
                if (!is_value_approx(gt_translation[k], p3p_translations[s][k], 1e-4)) {
                    t_match = false;
                    break;
                }
            }

            if (r_match && t_match) {
                double total_error = 0.0;
                for (int i = 0; i < 3; ++i) {
                    double proj[3];
                    matrix_vector_multiply(&p3p_rotations[s][0], &world_points[i][0], proj);
                    proj[0] += p3p_translations[s][0];
                    proj[1] += p3p_translations[s][1];
                    proj[2] += p3p_translations[s][2];
                    const double proj_norm = std::sqrt(proj[0] * proj[0] + proj[1] * proj[1] + proj[2] * proj[2]);
                    proj[0] /= proj_norm;
                    proj[1] /= proj_norm;
                    proj[2] /= proj_norm;
                    double dx = proj[0] - bearing_vectors[i][0];
                    double dy = proj[1] - bearing_vectors[i][1];
                    double dz = proj[2] - bearing_vectors[i][2];
                    total_error += dx * dx + dy * dy + dz * dz;
                }
                REQUIRE(total_error < 1e-10);
                found_match = true;
            }
        }
        REQUIRE(found_match);
    }

    // Collinear points.
    {
        const double gt_rotation[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double gt_translation[3] = { 0.0, 0.0, 0.5 };

        const double world_points[3][3] = {
            { 0.0, 0.0, 5.0 },
            { 0.5, 0.0, 5.0 },
            { 1.0, 0.0, 5.0 }
        };

        double bearing_vectors[3][3];
        for (int i = 0; i < 3; ++i) {
            double p[3];
            matrix_vector_multiply(gt_rotation, &world_points[i][0], p);
            p[0] += gt_translation[0];
            p[1] += gt_translation[1];
            p[2] += gt_translation[2];
            const double norm = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            bearing_vectors[i][0] = p[0] / norm;
            bearing_vectors[i][1] = p[1] / norm;
            bearing_vectors[i][2] = p[2] / norm;
        }

        double p3p_rotations[4][9];
        double p3p_translations[4][3];
        const int n_solutions = estimation::minimal::perspective_3_point<double>::solve(bearing_vectors, world_points, p3p_rotations, p3p_translations);
        REQUIRE(n_solutions == 0);
    }

    return EXIT_SUCCESS;
}
