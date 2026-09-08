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

#include "estimation/robust/solver/p3p.hpp"

#include "estimation/robust/consensus.hpp"
#include "estimation/robust/evaluate/maximum_likelihood.hpp"
#include "estimation/robust/sample/random.hpp"

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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // Five outliers.
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
        double gt_rotation[9];
        matrix_multiply(temp, &rotation_x[0][0], gt_rotation);
        double gt_translation[3] = { 0.5, -0.3, 1.0 };

        constexpr static const int inlier_count = 15;
        constexpr static const int outlier_count = 5;
        constexpr static const int correspondence_count = inlier_count + outlier_count;

        double world_points[correspondence_count][3] = {
            { 0.1, 0.2, 3.0 },
            { -0.5, 0.4, 4.2 },
            { 0.7, -0.3, 5.1 },
            { -0.2, -0.1, 2.7 },
            { 0.0, 0.0, 6.0 },
            { 0.3, -0.25, 4.0 },
            { 0.4, 0.1, 7.5 },
            { -0.6, -0.2, 3.8 },
            { 0.2, 0.6, 5.4 },
            { -0.1, 0.3, 8.2 },
            { 0.55, -0.45, 4.7 },
            { -0.35, 0.25, 6.3 },
            { 0.15, -0.55, 9.1 },
            { -0.25, -0.35, 7.0 },
            { 0.6, 0.4, 3.3 },
            { 0.1, 0.2, 3.0 },
            { -0.5, 0.4, 4.2 },
            { 0.7, -0.3, 5.1 },
            { -0.2, -0.1, 2.7 },
            { 0.0, 0.0, 6.0 },
        };

        estimation::correspondence_2d_3d<double> data[correspondence_count];
        for (int i = 0; i < inlier_count; ++i) {
            double point[2];
            project_point(gt_rotation, gt_translation, &world_points[i][0], point);
            data[i].lhs[0] = point[0];
            data[i].lhs[1] = point[1];
            data[i].rhs[0] = world_points[i][0];
            data[i].rhs[1] = world_points[i][1];
            data[i].rhs[2] = world_points[i][2];
        }
        for (int i = 0; i < outlier_count; ++i) {
            data[inlier_count + i].lhs[0] = 1.0;
            data[inlier_count + i].lhs[1] = -1.0;
            data[inlier_count + i].rhs[0] = world_points[inlier_count + i][0];
            data[inlier_count + i].rhs[1] = world_points[inlier_count + i][1];
            data[inlier_count + i].rhs[2] = world_points[inlier_count + i][2];
        }

        float residuals[correspondence_count];
        size_t inliers[correspondence_count];
        size_t inliers_size = 0;
        estimation::robust::estimate::p3p<double>::model model{};

        bool ok = estimation::robust::solver::p3p<double>::solve(
            data,
            correspondence_count,
            residuals,
            inliers,
            inliers_size,
            model
        );
        REQUIRE(ok);
        REQUIRE(inliers_size == inlier_count);

        for (size_t i = 0; i < correspondence_count; ++i) {
            REQUIRE(std::isfinite(residuals[i]));
        }

        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                REQUIRE(std::isfinite(model.rotation[y][x]));
            }
            REQUIRE(std::isfinite(model.translation[y]));
        }

        for (int k = 0; k < 9; ++k) {
            REQUIRE(is_value_approx(gt_rotation[k], model.rotation[k / 3][k % 3], 1e-4));
        }
        for (int k = 0; k < 3; ++k) {
            REQUIRE(is_value_approx(gt_translation[k], model.translation[k], 1e-4));
        }
    }

    // Half outliers.
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
        double gt_rotation[9];
        matrix_multiply(temp, &rotation_x[0][0], gt_rotation);
        double gt_translation[3] = { 0.5, -0.3, 1.0 };

        constexpr static const int inlier_count = 15;
        constexpr static const int outlier_count = 15;
        constexpr static const int correspondence_count = inlier_count + outlier_count;

        double world_points[inlier_count][3] = {
            { 0.1, 0.2, 3.0 },
            { -0.5, 0.4, 4.2 },
            { 0.7, -0.3, 5.1 },
            { -0.2, -0.1, 2.7 },
            { 0.0, 0.0, 6.0 },
            { 0.3, -0.25, 4.0 },
            { 0.4, 0.1, 7.5 },
            { -0.6, -0.2, 3.8 },
            { 0.2, 0.6, 5.4 },
            { -0.1, 0.3, 8.2 },
            { 0.55, -0.45, 4.7 },
            { -0.35, 0.25, 6.3 },
            { 0.15, -0.55, 9.1 },
            { -0.25, -0.35, 7.0 },
            { 0.6, 0.4, 3.3 },
        };

        const double outlier_offsets[outlier_count][2] = {
            { 0.7, -1.3 },
            { -1.1, 0.9 },
            { 1.5, 0.6 },
            { -0.8, -1.7 },
            { 1.2, 1.4 },
            { -1.6, 0.3 },
            { 0.4, 1.8 },
            { -1.3, -0.5 },
            { 0.9, -1.9 },
            { -0.6, 1.1 },
            { 1.8, -0.2 },
            { -1.9, -1.2 },
            { 0.3, 0.8 },
            { -0.4, -0.9 },
            { 1.0, 1.6 },
        };

        estimation::correspondence_2d_3d<double> data[correspondence_count];
        for (int i = 0; i < inlier_count; ++i) {
            double point[2];
            project_point(gt_rotation, gt_translation, &world_points[i][0], point);
            data[i].lhs[0] = point[0];
            data[i].lhs[1] = point[1];
            data[i].rhs[0] = world_points[i][0];
            data[i].rhs[1] = world_points[i][1];
            data[i].rhs[2] = world_points[i][2];
        }
        for (int i = 0; i < outlier_count; ++i) {
            double point[2];
            project_point(gt_rotation, gt_translation, &world_points[i % inlier_count][0], point);
            data[inlier_count + i].lhs[0] = point[0] + outlier_offsets[i][0];
            data[inlier_count + i].lhs[1] = point[1] + outlier_offsets[i][1];
            data[inlier_count + i].rhs[0] = world_points[i % inlier_count][0];
            data[inlier_count + i].rhs[1] = world_points[i % inlier_count][1];
            data[inlier_count + i].rhs[2] = world_points[i % inlier_count][2];
        }

        float residuals[correspondence_count];
        size_t inliers[correspondence_count];
        size_t inliers_size = 0;
        estimation::robust::estimate::p3p<double>::model model{};

        bool ok = estimation::robust::solver::p3p<double>::solve(
            data,
            correspondence_count,
            residuals,
            inliers,
            inliers_size,
            model
        );
        REQUIRE(ok);
        REQUIRE(inliers_size == inlier_count);

        for (int k = 0; k < 9; ++k) {
            REQUIRE(is_value_approx(gt_rotation[k], model.rotation[k / 3][k % 3], 1e-4));
        }
        for (int k = 0; k < 3; ++k) {
            REQUIRE(is_value_approx(gt_translation[k], model.translation[k], 1e-4));
        }
    }

    // Noisy inliers.
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
        double gt_rotation[9];
        matrix_multiply(temp, &rotation_x[0][0], gt_rotation);
        const double gt_translation[3] = { 0.5, -0.3, 1.0 };

        constexpr static const int inlier_count = 15;
        constexpr static const int outlier_count = 5;
        constexpr static const int correspondence_count = inlier_count + outlier_count;
        const double world_points[inlier_count][3] = {
            { 0.1, 0.2, 3.0 },
            { -0.5, 0.4, 4.2 },
            { 0.7, -0.3, 5.1 },
            { -0.2, -0.1, 2.7 },
            { 0.0, 0.0, 6.0 },
            { 0.3, -0.25, 4.0 },
            { 0.4, 0.1, 7.5 },
            { -0.6, -0.2, 3.8 },
            { 0.2, 0.6, 5.4 },
            { -0.1, 0.3, 8.2 },
            { 0.55, -0.45, 4.7 },
            { -0.35, 0.25, 6.3 },
            { 0.15, -0.55, 9.1 },
            { -0.25, -0.35, 7.0 },
            { 0.6, 0.4, 3.3 }
        };

        constexpr static const double noise_amplitude = 1e-3;
        unsigned long long int noise_state = 0x9E3779B97F4A7C15ull;
        const auto next_noise = [&noise_state](double amplitude) -> double {
            noise_state = noise_state * 6364136223846793005ull + 1442695040888963407ull;
            const double unit = static_cast<double>((noise_state >> 11) & 0x1FFFFF) / static_cast<double>(0x200000);
            return (2.0 * unit - 1.0) * amplitude;
        };

        estimation::correspondence_2d_3d<double> data[correspondence_count];
        for (int i = 0; i < inlier_count; ++i) {
            double point[2];
            project_point(gt_rotation, gt_translation, &world_points[i][0], point);
            data[i].lhs[0] = point[0] + next_noise(noise_amplitude);
            data[i].lhs[1] = point[1] + next_noise(noise_amplitude);
            data[i].rhs[0] = world_points[i][0];
            data[i].rhs[1] = world_points[i][1];
            data[i].rhs[2] = world_points[i][2];
        }
        for (int i = 0; i < outlier_count; ++i) {
            data[inlier_count + i].lhs[0] = 1.0;
            data[inlier_count + i].lhs[1] = -1.0;
            data[inlier_count + i].rhs[0] = world_points[i][0];
            data[inlier_count + i].rhs[1] = world_points[i][1];
            data[inlier_count + i].rhs[2] = world_points[i][2];
        }

        const float probability_failure = 0.01f;
        const size_t iterations_minimum = 5;
        const size_t iterations_maximum = 300;
        const float threshold_angle = 5.0e-3f;
        const float residual_threshold = static_cast<float>(1.0 - std::cos(static_cast<double>(threshold_angle)));

        estimation::robust::sample::random<3> random;
        estimation::robust::estimate::p3p<double> estimator;
        estimation::robust::evaluate::maximum_likelihood inlier_support(residual_threshold);
        estimation::robust::consensus<estimation::robust::sample::random<3>, estimation::robust::estimate::p3p<double>, estimation::robust::evaluate::maximum_likelihood> consensus(
            random,
            estimator,
            inlier_support,
            probability_failure,
            iterations_minimum,
            iterations_maximum
        );

        float residuals[correspondence_count];
        size_t inliers[correspondence_count];
        size_t inliers_size = 0;
        estimation::robust::estimate::p3p<double>::model model{};
        REQUIRE(consensus.estimate(data, correspondence_count, residuals, inliers, inliers_size, model));

        REQUIRE(inliers_size == static_cast<size_t>(inlier_count));
        for (size_t i = 0; i < inliers_size; ++i) {
            REQUIRE(inliers[i] < static_cast<size_t>(inlier_count));
        }

        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                REQUIRE(std::isfinite(model.rotation[y][x]));
            }
            REQUIRE(std::isfinite(model.translation[y]));
        }

        double trace = 0.0;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                trace += gt_rotation[r * 3 + c] * model.rotation[r][c];
            }
        }
        double cosine = (trace - 1.0) / 2.0;
        cosine = (cosine > 1.0) ? 1.0 : ((cosine < -1.0) ? -1.0 : cosine);
        const double rotation_angle_error = std::acos(cosine);
        REQUIRE(rotation_angle_error < 0.02);

        const double translation_error = std::sqrt(
            (gt_translation[0] - model.translation[0]) * (gt_translation[0] - model.translation[0]) +
            (gt_translation[1] - model.translation[1]) * (gt_translation[1] - model.translation[1]) +
            (gt_translation[2] - model.translation[2]) * (gt_translation[2] - model.translation[2])
        );
        REQUIRE(translation_error < 0.05);
        const double gt_translation_norm = std::sqrt(gt_translation[0] * gt_translation[0] + gt_translation[1] * gt_translation[1] + gt_translation[2] * gt_translation[2]);
        const double recovered_translation_norm = std::sqrt(model.translation[0] * model.translation[0] + model.translation[1] * model.translation[1] + model.translation[2] * model.translation[2]);
        const double translation_direction_dot =
            (gt_translation[0] * model.translation[0] + gt_translation[1] * model.translation[1] + gt_translation[2] * model.translation[2]) /
            (gt_translation_norm * recovered_translation_norm);
        REQUIRE(translation_direction_dot > 0.999);
    }

    return EXIT_SUCCESS;
}
