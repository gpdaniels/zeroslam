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

#include "estimation/robust/solver/essential.hpp"

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

    // One outlier.
    {
        const double alpha = 0.025;
        const double beta = -0.017;
        const double gamma = 0.01;
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
        double translation[3] = { 0.5, -0.3, 0.7 };
        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double zero[3] = { 0, 0, 0 };
        constexpr static const int inlier_count = 15;
        constexpr static const int outlier_count = 1;
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
            { 0.6, 0.4, 3.3 }
        };
        estimation::correspondence_2d_2d<double> data[correspondence_count];
        for (int i = 0; i < inlier_count; ++i) {
            double point[2];
            project_point(identity, zero, &world_points[i][0], point);
            data[i].lhs[0] = point[0];
            data[i].lhs[1] = point[1];
            project_point(rotation, translation, &world_points[i][0], point);
            data[i].rhs[0] = point[0];
            data[i].rhs[1] = point[1];
        }
        for (int i = 0; i < outlier_count; ++i) {
            double point[2];
            project_point(identity, zero, &world_points[i][0], point);
            data[inlier_count + i].lhs[0] = point[0] - 50;
            data[inlier_count + i].lhs[1] = point[1] + 13;
            project_point(identity, zero, &world_points[i][0], point);
            data[inlier_count + i].rhs[0] = point[0] + 20;
            data[inlier_count + i].rhs[1] = point[1] - 80;
        }

        float residuals[correspondence_count];
        size_t inliers[correspondence_count];
        size_t inliers_size = 0;
        estimation::robust::estimate::essential<double>::model model{};

        bool ok = estimation::robust::solver::essential<double>::solve(
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
                REQUIRE(std::isfinite(model.essential[y][x]));
            }
        }
    }

    // Half outliers against a known pose, repeated on the same object.
    {
        constexpr static const auto essential_matches = [](const double* recovered, const double* expected, double epsilon) -> bool {
            double a[9];
            double b[9];
            for (int i = 0; i < 9; ++i) {
                a[i] = recovered[i];
                b[i] = expected[i];
            }
            normalize_matrix(&a[0]);
            normalize_matrix(&b[0]);
            bool positive = true;
            bool negative = true;
            for (int i = 0; i < 9; ++i) {
                if (std::abs(a[i] - b[i]) > epsilon) {
                    positive = false;
                }
                if (std::abs(a[i] + b[i]) > epsilon) {
                    negative = false;
                }
            }
            return positive || negative;
        };

        const double alpha = 0.025;
        const double beta = -0.017;
        const double gamma = 0.01;
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
        double gt_translation[3] = { 0.5, -0.3, 0.7 };

        double gt_translation_matrix[9];
        cross_matrix(gt_translation, gt_translation_matrix);
        double gt_essential[9];
        matrix_multiply(gt_translation_matrix, gt_rotation, gt_essential);

        constexpr static const int inlier_count = 15;
        constexpr static const int outlier_count = 15;
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
            { 1.0, 1.6 }
        };

        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double zero[3] = { 0, 0, 0 };

        estimation::correspondence_2d_2d<double> data[correspondence_count];
        for (int i = 0; i < inlier_count; ++i) {
            double point[2];
            project_point(identity, zero, &world_points[i][0], point);
            data[i].lhs[0] = point[0];
            data[i].lhs[1] = point[1];
            project_point(gt_rotation, gt_translation, &world_points[i][0], point);
            data[i].rhs[0] = point[0];
            data[i].rhs[1] = point[1];
        }
        for (int i = 0; i < outlier_count; ++i) {
            const int base = i % inlier_count;
            double point[2];
            project_point(identity, zero, &world_points[base][0], point);
            data[inlier_count + i].lhs[0] = point[0];
            data[inlier_count + i].lhs[1] = point[1];
            project_point(gt_rotation, gt_translation, &world_points[base][0], point);
            data[inlier_count + i].rhs[0] = point[0] + outlier_offsets[i][0];
            data[inlier_count + i].rhs[1] = point[1] + outlier_offsets[i][1];
        }

        const float probability_failure = 0.01f;
        const size_t iterations_minimum = 5;
        const size_t iterations_maximum = 300;
        const float residual_threshold = 1.0e-5f;

        estimation::robust::sample::random<5> random;
        estimation::robust::estimate::essential<double> estimator;
        estimation::robust::evaluate::maximum_likelihood inlier_support(residual_threshold);

        estimation::robust::consensus<estimation::robust::sample::random<5>, estimation::robust::estimate::essential<double>, estimation::robust::evaluate::maximum_likelihood> consensus(
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
        estimation::robust::estimate::essential<double>::model model{};
        REQUIRE(consensus.estimate(data, correspondence_count, residuals, inliers, inliers_size, model));

        REQUIRE(inliers_size == static_cast<size_t>(inlier_count));
        for (size_t i = 0; i < inliers_size; ++i) {
            REQUIRE(inliers[i] < static_cast<size_t>(inlier_count));
        }
        for (int i = 0; i < inlier_count; ++i) {
            REQUIRE(std::isfinite(residuals[i]));
            REQUIRE(residuals[i] < residual_threshold);
        }
        for (int i = inlier_count; i < correspondence_count; ++i) {
            REQUIRE(residuals[i] >= residual_threshold);
        }
        REQUIRE(essential_matches(&model.essential[0][0], &gt_essential[0], 1e-6));

        float residuals_repeat[correspondence_count];
        size_t inliers_repeat[correspondence_count];
        size_t inliers_size_repeat = 0;
        estimation::robust::estimate::essential<double>::model model_repeat{};
        REQUIRE(consensus.estimate(data, correspondence_count, residuals_repeat, inliers_repeat, inliers_size_repeat, model_repeat));

        REQUIRE(inliers_size_repeat == static_cast<size_t>(inlier_count));
        for (size_t i = 0; i < inliers_size_repeat; ++i) {
            REQUIRE(inliers_repeat[i] < static_cast<size_t>(inlier_count));
        }
        REQUIRE(essential_matches(&model_repeat.essential[0][0], &gt_essential[0], 1e-6));

        decltype(consensus) consensus_fresh(
            random,
            estimator,
            inlier_support,
            probability_failure,
            iterations_minimum,
            iterations_maximum
        );
        float residuals_fresh[correspondence_count];
        size_t inliers_fresh[correspondence_count];
        size_t inliers_size_fresh = 0;
        estimation::robust::estimate::essential<double>::model model_fresh{};
        REQUIRE(consensus_fresh.estimate(data, correspondence_count, residuals_fresh, inliers_fresh, inliers_size_fresh, model_fresh));
        REQUIRE(inliers_size_fresh == inliers_size);
        for (int i = 0; i < 9; ++i) {
            REQUIRE(model_fresh.essential[i / 3][i % 3] == model.essential[i / 3][i % 3]);
        }
    }

    return EXIT_SUCCESS;
}
