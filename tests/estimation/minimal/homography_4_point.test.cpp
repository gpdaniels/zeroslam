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

static inline void project_point(const double* rotation, const double* translation, const double* point_xyz, double* point_xy) {
    double point[3];
    matrix_vector_multiply(rotation, point_xyz, point);
    point[0] += translation[0];
    point[1] += translation[1];
    point[2] += translation[2];
    point_xy[0] = point[0] / point[2];
    point_xy[1] = point[1] / point[2];
}

static inline void apply_homography(const double* h, double x, double y, double* out_xy) {
    const double px = h[0] * x + h[1] * y + h[2];
    const double py = h[3] * x + h[4] * y + h[5];
    const double pw = h[6] * x + h[7] * y + h[8];
    out_xy[0] = px / pw;
    out_xy[1] = py / pw;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // Planar scene transfer.
    {
        const double alpha = 0.2;
        const double beta = -0.15;
        const double gamma = 0.08;
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
        const double translation[3] = { 0.4, -0.25, 0.6 };
        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double zero[3] = { 0, 0, 0 };

        const int point_count = 8;
        const double plane_xy[8][2] = {
            { -0.4, -0.3 },
            { 0.5, -0.2 },
            { 0.3, 0.6 },
            { -0.5, 0.4 },
            { 0.1, 0.1 },
            { 0.6, 0.5 },
            { -0.2, 0.5 },
            { 0.0, -0.5 }
        };
        double world_points[8][3];
        for (int i = 0; i < point_count; ++i) {
            world_points[i][0] = plane_xy[i][0];
            world_points[i][1] = plane_xy[i][1];
            world_points[i][2] = 4.0 + 0.3 * plane_xy[i][0] - 0.2 * plane_xy[i][1];
        }

        double lhs_points[16];
        double rhs_points[16];
        for (int i = 0; i < point_count; ++i) {
            project_point(identity, zero, &world_points[i][0], &lhs_points[2 * i]);
            project_point(rotation, translation, &world_points[i][0], &rhs_points[2 * i]);
        }

        double homography[9];
        const bool ok = estimation::minimal::homography_4_point<double>::solve(lhs_points, rhs_points, homography);
        REQUIRE(ok);

        const double determinant =
            homography[0] * (homography[4] * homography[8] - homography[5] * homography[7]) -
            homography[1] * (homography[3] * homography[8] - homography[5] * homography[6]) +
            homography[2] * (homography[3] * homography[7] - homography[4] * homography[6]);
        REQUIRE(std::abs(determinant) > 1e-12);
        const double determinant_inverse = 1.0 / determinant;
        const double homography_inverse[9] = {
            (homography[4] * homography[8] - homography[5] * homography[7]) * determinant_inverse,
            (homography[2] * homography[7] - homography[1] * homography[8]) * determinant_inverse,
            (homography[1] * homography[5] - homography[2] * homography[4]) * determinant_inverse,
            (homography[5] * homography[6] - homography[3] * homography[8]) * determinant_inverse,
            (homography[0] * homography[8] - homography[2] * homography[6]) * determinant_inverse,
            (homography[2] * homography[3] - homography[0] * homography[5]) * determinant_inverse,
            (homography[3] * homography[7] - homography[4] * homography[6]) * determinant_inverse,
            (homography[1] * homography[6] - homography[0] * homography[7]) * determinant_inverse,
            (homography[0] * homography[4] - homography[1] * homography[3]) * determinant_inverse
        };

        for (int i = 0; i < point_count; ++i) {
            double predicted_lhs[2];
            apply_homography(homography, rhs_points[2 * i + 0], rhs_points[2 * i + 1], predicted_lhs);
            REQUIRE(is_value_approx(predicted_lhs[0], lhs_points[2 * i + 0], 1e-9));
            REQUIRE(is_value_approx(predicted_lhs[1], lhs_points[2 * i + 1], 1e-9));

            double predicted_rhs[2];
            apply_homography(homography_inverse, lhs_points[2 * i + 0], lhs_points[2 * i + 1], predicted_rhs);
            REQUIRE(is_value_approx(predicted_rhs[0], rhs_points[2 * i + 0], 1e-9));
            REQUIRE(is_value_approx(predicted_rhs[1], rhs_points[2 * i + 1], 1e-9));
        }
    }

    // Collinear and coincident points.
    {
        const double collinear_lhs[8] = {
            0.0,
            0.0,
            0.1,
            0.1,
            0.2,
            0.2,
            0.3,
            0.3
        };
        const double collinear_rhs[8] = {
            0.0,
            0.0,
            0.2,
            0.2,
            0.4,
            0.4,
            0.6,
            0.6
        };
        double homography[9];
        const bool ok = estimation::minimal::homography_4_point<double>::solve(collinear_lhs, collinear_rhs, homography);
        REQUIRE(!ok);

        const double coincident_lhs[8] = { 0.2, -0.1, 0.2, -0.1, 0.2, -0.1, 0.2, -0.1 };
        const double coincident_rhs[8] = { -0.3, 0.4, -0.3, 0.4, -0.3, 0.4, -0.3, 0.4 };
        const bool ok_coincident = estimation::minimal::homography_4_point<double>::solve(coincident_lhs, coincident_rhs, homography);
        REQUIRE(!ok_coincident);
    }

    return EXIT_SUCCESS;
}
