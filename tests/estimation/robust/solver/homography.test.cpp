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

#include "estimation/robust/solver/homography.hpp"

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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // Planar scene with gross outliers.
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

        constexpr static const int inlier_count = 20;
        constexpr static const int outlier_count = 8;
        constexpr static const int correspondence_count = inlier_count + outlier_count;

        const double plane_xy[inlier_count][2] = {
            { -0.40, -0.30 },
            { 0.50, -0.20 },
            { 0.30, 0.60 },
            { -0.50, 0.40 },
            { 0.10, 0.10 },
            { 0.60, 0.50 },
            { -0.20, 0.50 },
            { 0.00, -0.50 },
            { 0.35, 0.15 },
            { -0.30, -0.10 },
            { 0.45, -0.40 },
            { -0.15, 0.30 },
            { 0.20, -0.25 },
            { -0.45, -0.35 },
            { 0.55, 0.25 },
            { -0.05, 0.45 },
            { 0.25, -0.55 },
            { -0.35, 0.20 },
            { 0.15, 0.35 },
            { 0.40, 0.05 }
        };

        const double outlier_offsets[outlier_count][2] = {
            { 0.9, -1.4 },
            { -1.2, 1.1 },
            { 1.6, 0.7 },
            { -0.9, -1.8 },
            { 1.3, 1.5 },
            { -1.7, 0.4 },
            { 0.5, 1.9 },
            { -1.4, -0.6 }
        };

        estimation::correspondence_2d_2d<double> data[correspondence_count];
        for (int i = 0; i < inlier_count; ++i) {
            const double world_point[3] = { plane_xy[i][0], plane_xy[i][1], 4.0 + 0.3 * plane_xy[i][0] - 0.2 * plane_xy[i][1] };
            double point[2];
            project_point(identity, zero, world_point, point);
            data[i].lhs[0] = point[0];
            data[i].lhs[1] = point[1];
            project_point(rotation, translation, world_point, point);
            data[i].rhs[0] = point[0];
            data[i].rhs[1] = point[1];
        }
        for (int i = 0; i < outlier_count; ++i) {
            const int base = i % inlier_count;
            const double world_point[3] = { plane_xy[base][0], plane_xy[base][1], 4.0 + 0.3 * plane_xy[base][0] - 0.2 * plane_xy[base][1] };
            double point[2];
            project_point(identity, zero, world_point, point);
            data[inlier_count + i].lhs[0] = point[0] + outlier_offsets[i][0];
            data[inlier_count + i].lhs[1] = point[1] + outlier_offsets[i][1];
            project_point(rotation, translation, world_point, point);
            data[inlier_count + i].rhs[0] = point[0];
            data[inlier_count + i].rhs[1] = point[1];
        }

        float residuals[correspondence_count];
        size_t inliers[correspondence_count];
        size_t inliers_size = 0;
        estimation::robust::estimate::homography<double>::model model{};

        const bool ok = estimation::robust::solver::homography<double>::solve(
            data,
            correspondence_count,
            residuals,
            inliers,
            inliers_size,
            model
        );
        REQUIRE(ok);

        REQUIRE(inliers_size == static_cast<size_t>(inlier_count));
        for (size_t i = 0; i < inliers_size; ++i) {
            REQUIRE(inliers[i] < static_cast<size_t>(inlier_count));
        }

        for (int i = 0; i < inlier_count; ++i) {
            REQUIRE(std::isfinite(residuals[i]));
            REQUIRE(residuals[i] < 2.0e-5f);
        }
        for (int i = inlier_count; i < correspondence_count; ++i) {
            REQUIRE(residuals[i] >= 2.0e-5f);
        }

        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                REQUIRE(std::isfinite(model.homography[y][x]));
            }
        }
    }

    return EXIT_SUCCESS;
}
