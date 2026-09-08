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

#include "estimation/pose/essential.hpp"

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

static inline void compose_essential(const double* rotation, const double* translation, double* essential) {
    double unit[3] = { translation[0], translation[1], translation[2] };
    normalize_vector(unit);
    double skew[9];
    cross_matrix(unit, skew);
    matrix_multiply(skew, rotation, essential);
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const double degenerate_world_points[6][3] = {
        { 0.1, 0.2, 3.0 },
        { -0.5, 0.4, 4.2 },
        { 0.7, -0.3, 5.1 },
        { -0.2, -0.1, 2.7 },
        { 0.0, 0.0, 6.0 },
        { 0.3, -0.25, 4.0 }
    };

    // Reprojection of the recovered pose.
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
        double translation[3] = { 0.5, -0.3, 1.0 };
        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double zero[3] = { 0, 0, 0 };
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

        double translation_matrix[9];
        cross_matrix(translation, translation_matrix);
        double essential[9];
        matrix_multiply(translation_matrix, rotation, essential);
        normalize_matrix(essential);

        double recovered_rotation[9];
        double recovered_translation[3];
        double points_xyz[point_count * 3] = {};
        size_t support_count = 0;
        const bool recovered = estimation::pose::essential<double>::recover(essential, lhs_points, rhs_points, static_cast<size_t>(point_count), recovered_rotation, recovered_translation, points_xyz, &support_count);
        REQUIRE(recovered);
        REQUIRE(support_count == static_cast<size_t>(point_count));

        double total_error = 0.0;
        for (int i = 0; i < point_count; ++i) {
            const double Xtri[3] = { points_xyz[3 * i + 0], points_xyz[3 * i + 1], points_xyz[3 * i + 2] };
            const double e1 = reprojection_error(lhs_points + 2 * i, identity, zero, Xtri);
            const double e2 = reprojection_error(rhs_points + 2 * i, recovered_rotation, recovered_translation, Xtri);
            total_error += (e1 + e2);
        }
        const double mean_error = total_error / double(point_count);
        REQUIRE(mean_error < 1e-5);

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

        normalize_vector(translation);
        normalize_vector(recovered_translation);
        double dot = std::abs(translation[0] * recovered_translation[0] + translation[1] * recovered_translation[1] + translation[2] * recovered_translation[2]);
        REQUIRE(dot > 0.98);
    }

    // Distant low parallax scene.
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
        double translation[3] = { 0.5, -0.3, 1.0 };
        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double zero[3] = { 0, 0, 0 };
        const int point_count = 6;
        double world_points[6][3] = {
            { 5.0, 8.0, 55.0 },
            { -12.0, 6.0, 60.0 },
            { 9.0, -7.0, 62.0 },
            { -4.0, -3.0, 58.0 },
            { 0.0, 1.0, 65.0 },
            { 7.0, -5.0, 57.0 }
        };

        double lhs_points[12];
        double rhs_points[12];
        for (int i = 0; i < point_count; ++i) {
            project_point(identity, zero, &world_points[i][0], &lhs_points[2 * i]);
            project_point(rotation, translation, &world_points[i][0], &rhs_points[2 * i]);
        }

        double translation_matrix[9];
        cross_matrix(translation, translation_matrix);
        double essential[9];
        matrix_multiply(translation_matrix, rotation, essential);
        normalize_matrix(essential);

        double recovered_rotation[9];
        double recovered_translation[3];
        double points_xyz[point_count * 3] = {};
        size_t support_count = 0;
        const bool recovered = estimation::pose::essential<double>::recover(essential, lhs_points, rhs_points, static_cast<size_t>(point_count), recovered_rotation, recovered_translation, points_xyz, &support_count);
        REQUIRE(recovered);
        REQUIRE(support_count * 2 >= static_cast<size_t>(point_count));

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

        normalize_vector(translation);
        normalize_vector(recovered_translation);
        double dot = std::abs(translation[0] * recovered_translation[0] + translation[1] * recovered_translation[1] + translation[2] * recovered_translation[2]);
        REQUIRE(dot > 0.98);
    }

    // Pure lateral translations.
    {
        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double zero[3] = { 0, 0, 0 };
        const int point_count = 6;

        const double lateral_translations[2][3] = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 } };
        for (int axis = 0; axis < 2; ++axis) {
            double translation[3] = { lateral_translations[axis][0], lateral_translations[axis][1], lateral_translations[axis][2] };
            double essential[9];
            compose_essential(identity, translation, essential);
            if (axis == 0) {
                REQUIRE(is_value_approx(essential[0], 0.0));
                REQUIRE(is_value_approx(essential[1], 0.0));
                REQUIRE(is_value_approx(essential[2], 0.0));
            }

            double lhs_points[12];
            double rhs_points[12];
            for (int i = 0; i < point_count; ++i) {
                project_point(identity, zero, &degenerate_world_points[i][0], &lhs_points[2 * i]);
                project_point(identity, translation, &degenerate_world_points[i][0], &rhs_points[2 * i]);
            }

            double recovered_rotation[9];
            double recovered_translation[3];
            double points_xyz[6 * 3] = {};
            size_t support_count = 0;
            const bool recovered = estimation::pose::essential<double>::recover(essential, lhs_points, rhs_points, static_cast<size_t>(point_count), recovered_rotation, recovered_translation, points_xyz, &support_count);
            REQUIRE(recovered);
            REQUIRE(support_count == static_cast<size_t>(point_count));

            for (int i = 0; i < 9; ++i) {
                REQUIRE(std::isfinite(recovered_rotation[i]));
            }
            for (int i = 0; i < 3; ++i) {
                REQUIRE(std::isfinite(recovered_translation[i]));
            }
            for (int i = 0; i < point_count * 3; ++i) {
                REQUIRE(std::isfinite(points_xyz[i]));
            }

            REQUIRE(is_rotation_matrix(recovered_rotation, 1e-6));
            REQUIRE(are_values_approx(recovered_rotation, identity, 9, 1e-6));

            double recovered_direction[3] = { recovered_translation[0], recovered_translation[1], recovered_translation[2] };
            normalize_vector(recovered_direction);
            const double dot = std::abs(recovered_direction[0] * translation[0] + recovered_direction[1] * translation[1] + recovered_direction[2] * translation[2]);
            REQUIRE(dot > 0.999);
        }

        {
            const double alpha = 0.25;
            const double beta = -0.17;
            const double gamma = 0.1;
            const double rotation_x[3][3] = { { 1, 0, 0 }, { 0, std::cos(alpha), -std::sin(alpha) }, { 0, std::sin(alpha), std::cos(alpha) } };
            const double rotation_y[3][3] = { { std::cos(beta), 0, std::sin(beta) }, { 0, 1, 0 }, { -std::sin(beta), 0, std::cos(beta) } };
            const double rotation_z[3][3] = { { std::cos(gamma), -std::sin(gamma), 0 }, { std::sin(gamma), std::cos(gamma), 0 }, { 0, 0, 1 } };
            double temp[9];
            matrix_multiply(&rotation_z[0][0], &rotation_y[0][0], temp);
            double rotation[9];
            matrix_multiply(temp, &rotation_x[0][0], rotation);
            double translation[3] = { 1.0, 0.0, 0.0 };
            double essential[9];
            compose_essential(rotation, translation, essential);

            double lhs_points[12];
            double rhs_points[12];
            for (int i = 0; i < point_count; ++i) {
                project_point(identity, zero, &degenerate_world_points[i][0], &lhs_points[2 * i]);
                project_point(rotation, translation, &degenerate_world_points[i][0], &rhs_points[2 * i]);
            }

            double recovered_rotation[9];
            double recovered_translation[3];
            double points_xyz[6 * 3] = {};
            size_t support_count = 0;
            const bool recovered = estimation::pose::essential<double>::recover(essential, lhs_points, rhs_points, static_cast<size_t>(point_count), recovered_rotation, recovered_translation, points_xyz, &support_count);
            REQUIRE(recovered);
            for (int i = 0; i < 9; ++i) {
                REQUIRE(std::isfinite(recovered_rotation[i]));
            }
            for (int i = 0; i < 3; ++i) {
                REQUIRE(std::isfinite(recovered_translation[i]));
            }

            double rotation_normalized[9];
            double recovered_normalized[9];
            for (int i = 0; i < 9; ++i) {
                rotation_normalized[i] = rotation[i];
                recovered_normalized[i] = recovered_rotation[i];
            }
            normalize_matrix(rotation_normalized);
            normalize_matrix(recovered_normalized);
            bool rotation_correct = are_values_approx(rotation_normalized, recovered_normalized, 9, 1e-3);
            if (!rotation_correct) {
                for (int i = 0; i < 9; ++i) {
                    recovered_normalized[i] = -recovered_normalized[i];
                }
                rotation_correct = are_values_approx(rotation_normalized, recovered_normalized, 9, 1e-3);
            }
            REQUIRE(rotation_correct);

            double recovered_direction[3] = { recovered_translation[0], recovered_translation[1], recovered_translation[2] };
            normalize_vector(recovered_direction);
            const double dot = std::abs(recovered_direction[0] * 1.0 + recovered_direction[1] * 0.0 + recovered_direction[2] * 0.0);
            REQUIRE(dot > 0.999);
        }
    }

    // Zero parallax is rejected.
    {
        const double alpha = 0.25;
        const double beta = -0.17;
        const double gamma = 0.1;
        const double rotation_x[3][3] = { { 1, 0, 0 }, { 0, std::cos(alpha), -std::sin(alpha) }, { 0, std::sin(alpha), std::cos(alpha) } };
        const double rotation_y[3][3] = { { std::cos(beta), 0, std::sin(beta) }, { 0, 1, 0 }, { -std::sin(beta), 0, std::cos(beta) } };
        const double rotation_z[3][3] = { { std::cos(gamma), -std::sin(gamma), 0 }, { std::sin(gamma), std::cos(gamma), 0 }, { 0, 0, 1 } };
        double temp[9];
        matrix_multiply(&rotation_z[0][0], &rotation_y[0][0], temp);
        double rotation[9];
        matrix_multiply(temp, &rotation_x[0][0], rotation);

        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double zero[3] = { 0, 0, 0 };
        const int point_count = 6;

        double lhs_points[12];
        double rhs_points[12];
        for (int i = 0; i < point_count; ++i) {
            project_point(identity, zero, &degenerate_world_points[i][0], &lhs_points[2 * i]);
            project_point(rotation, zero, &degenerate_world_points[i][0], &rhs_points[2 * i]);
        }

        double translation[3] = { 0.5, -0.3, 1.0 };
        double essential[9];
        compose_essential(rotation, translation, essential);

        for (int i = 0; i < point_count; ++i) {
            const double lhs_homogeneous[3] = { lhs_points[2 * i + 0], lhs_points[2 * i + 1], 1.0 };
            const double rhs_homogeneous[3] = { rhs_points[2 * i + 0], rhs_points[2 * i + 1], 1.0 };
            double essential_lhs[3];
            matrix_vector_multiply(essential, lhs_homogeneous, essential_lhs);
            const double residual = rhs_homogeneous[0] * essential_lhs[0] + rhs_homogeneous[1] * essential_lhs[1] + rhs_homogeneous[2] * essential_lhs[2];
            REQUIRE(std::abs(residual) < 1e-9);
        }

        double recovered_rotation[9];
        double recovered_translation[3];
        double points_xyz[6 * 3] = {};
        size_t support_count = 0;
        const bool recovered = estimation::pose::essential<double>::recover(essential, lhs_points, rhs_points, static_cast<size_t>(point_count), recovered_rotation, recovered_translation, points_xyz, &support_count);

        REQUIRE(!recovered);
        REQUIRE(support_count < static_cast<size_t>(point_count));
        for (int i = 0; i < 9; ++i) {
            REQUIRE(std::isfinite(recovered_rotation[i]));
        }
        for (int i = 0; i < 3; ++i) {
            REQUIRE(std::isfinite(recovered_translation[i]));
        }
        for (int i = 0; i < point_count * 3; ++i) {
            REQUIRE(std::isfinite(points_xyz[i]));
        }
    }

    return EXIT_SUCCESS;
}
