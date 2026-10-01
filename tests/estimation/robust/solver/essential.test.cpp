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

#include "core/random_pcg.hpp"
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

static inline double gaussian(core::random_pcg& random) {
    const double radius = std::sqrt(-2.0 * std::log(random.get_random_exclusive()));
    return radius * std::cos(6.283185307179586 * random.get_random_exclusive());
}

// Points at depths 2 to 6, or on a plane, seen before and after a random motion; the first inlier_count with noise of sigma on every coordinate, the rest random; the true matrix is written with unit norm.
static inline void make_scene(core::random_pcg& random, size_t inlier_count, size_t outlier_count, double sigma, bool planar, estimation::correspondence_2d_2d<double>* data, double* essential) {
    const double alpha = random.get_random(-0.05, 0.05);
    const double beta = random.get_random(-0.05, 0.05);
    const double rotation_x[9] = { 1, 0, 0, 0, std::cos(alpha), -std::sin(alpha), 0, std::sin(alpha), std::cos(alpha) };
    const double rotation_y[9] = { std::cos(beta), 0, std::sin(beta), 0, 1, 0, -std::sin(beta), 0, std::cos(beta) };
    double rotation[9];
    matrix_multiply(rotation_y, rotation_x, rotation);
    const double translation[3] = { random.get_random(-0.4, 0.4), random.get_random(-0.2, 0.2), random.get_random(-0.2, 0.2) };
    const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const double zero[3] = { 0, 0, 0 };
    for (size_t i = 0; i < inlier_count + outlier_count; ++i) {
        const double x = random.get_random(-0.6, 0.6);
        const double y = random.get_random(-0.45, 0.45);
        const double depth = planar ? (4.0 / (1.0 - 0.2 * x + 0.1 * y)) : random.get_random(2.0, 6.0);
        const double point_xyz[3] = { x * depth, y * depth, depth };
        double lhs[2];
        double rhs[2];
        project_point(identity, zero, point_xyz, lhs);
        project_point(rotation, translation, point_xyz, rhs);
        if (i >= inlier_count) {
            rhs[0] = random.get_random(-0.6, 0.6);
            rhs[1] = random.get_random(-0.45, 0.45);
        }
        data[i].lhs[0] = lhs[0] + sigma * gaussian(random);
        data[i].lhs[1] = lhs[1] + sigma * gaussian(random);
        data[i].rhs[0] = rhs[0] + sigma * gaussian(random);
        data[i].rhs[1] = rhs[1] + sigma * gaussian(random);
    }
    double translation_matrix[9];
    cross_matrix(translation, translation_matrix);
    matrix_multiply(translation_matrix, rotation, essential);
    normalize_matrix(essential);
}

static inline float scene_cost(const estimation::correspondence_2d_2d<double>* data, const size_t data_size, const double* essential, const float residual_threshold) {
    estimation::robust::estimate::essential<double>::model model;
    for (int i = 0; i < 9; ++i) {
        model.essential[i / 3][i % 3] = essential[i];
    }
    float residuals[400];
    size_t inliers[400];
    size_t inliers_size = 0;
    estimation::robust::estimate::essential<double>().compute_residuals(data, data_size, model, residuals);
    return estimation::robust::evaluate::maximum_likelihood(residual_threshold).evaluate(residuals, data_size, inliers, inliers_size);
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // Noise free planar correspondences leave the linear refit a null space of three dimensions, so it is skipped and the five point model, exact on every correspondence, is kept.
    {
        core::random_pcg random(0x5eed0300ull);
        for (int trial = 0; trial < 8; ++trial) {
            constexpr static const size_t correspondence_count = 60;
            estimation::correspondence_2d_2d<double> data[correspondence_count];
            double essential[9];
            make_scene(random, correspondence_count, 0, 0.0, true, data, essential);
            float residuals[correspondence_count];
            size_t inliers[correspondence_count];
            size_t inliers_size = 0;
            estimation::robust::estimate::essential<double>::model model{};
            REQUIRE(estimation::robust::solver::essential<double>::solve(data, correspondence_count, residuals, inliers, inliers_size, model));
            REQUIRE(inliers_size == correspondence_count);
            for (size_t i = 0; i < correspondence_count; ++i) {
                REQUIRE(residuals[i] < 1e-12f);
            }
        }
    }

    // With noise, in general and planar scenes, the refit replaces the consensus model only when it lowers the cost, and the result is never far worse than the true matrix.
    {
        core::random_pcg random(0x5eed0301ull);
        size_t improved = 0;
        for (int trial = 0; trial < 12; ++trial) {
            constexpr static const size_t inlier_count = 160;
            constexpr static const size_t correspondence_count = 200;
            const float residual_threshold = 1.0e-5f;
            estimation::correspondence_2d_2d<double> data[correspondence_count];
            double essential[9];
            make_scene(random, inlier_count, correspondence_count - inlier_count, 1.0e-3, (trial % 3) == 0, data, essential);

            // The solver's own consensus: the same seed, evaluator and iteration budget, without the refit.
            estimation::robust::sample::random<5> sampler(estimation::robust::sample::random<5>::seed_from(data, correspondence_count));
            estimation::robust::estimate::essential<double> estimator;
            estimation::robust::evaluate::maximum_likelihood support(residual_threshold);
            estimation::robust::consensus<estimation::robust::sample::random<5>, estimation::robust::estimate::essential<double>, estimation::robust::evaluate::maximum_likelihood> consensus(sampler, estimator, support, 0.01f, 100, 300);
            float consensus_residuals[correspondence_count];
            size_t consensus_inliers[correspondence_count];
            size_t consensus_inliers_size = 0;
            estimation::robust::estimate::essential<double>::model consensus_model{};
            REQUIRE(consensus.estimate(data, correspondence_count, consensus_residuals, consensus_inliers, consensus_inliers_size, consensus_model));
            const float consensus_cost = support.evaluate(consensus_residuals, correspondence_count, consensus_inliers, consensus_inliers_size);

            float residuals[correspondence_count];
            size_t inliers[correspondence_count];
            size_t inliers_size = 0;
            estimation::robust::estimate::essential<double>::model model{};
            REQUIRE(estimation::robust::solver::essential<double>::solve(data, correspondence_count, residuals, inliers, inliers_size, model));
            const float cost = support.evaluate(residuals, correspondence_count, inliers, inliers_size);
            REQUIRE(cost <= consensus_cost);
            if (cost < consensus_cost) {
                ++improved;
            }
            REQUIRE(cost <= 1.5f * scene_cost(data, correspondence_count, essential, residual_threshold));
        }
        REQUIRE(improved > 0);
    }

    // The threshold: passing the default is the same as leaving it out, a tighter one keeps fewer inliers and a looser one more.
    {
        core::random_pcg random(0x5eed0302ull);
        constexpr static const size_t correspondence_count = 200;
        estimation::correspondence_2d_2d<double> data[correspondence_count];
        double essential[9];
        make_scene(random, 160, correspondence_count - 160, 1.0e-3, false, data, essential);
        float residuals[correspondence_count];
        size_t inliers[correspondence_count];
        size_t inliers_default = 0;
        estimation::robust::estimate::essential<double>::model model_default{};
        REQUIRE(estimation::robust::solver::essential<double>::solve(data, correspondence_count, residuals, inliers, inliers_default, model_default));
        size_t inliers_explicit = 0;
        estimation::robust::estimate::essential<double>::model model_explicit{};
        REQUIRE(estimation::robust::solver::essential<double>::solve(data, correspondence_count, residuals, inliers, inliers_explicit, model_explicit, 1.0e-5f));
        REQUIRE(inliers_explicit == inliers_default);
        for (int i = 0; i < 9; ++i) {
            REQUIRE(model_explicit.essential[i / 3][i % 3] == model_default.essential[i / 3][i % 3]);
        }
        size_t inliers_tight = 0;
        estimation::robust::estimate::essential<double>::model model_tight{};
        REQUIRE(estimation::robust::solver::essential<double>::solve(data, correspondence_count, residuals, inliers, inliers_tight, model_tight, 1.0e-7f));
        size_t inliers_loose = 0;
        estimation::robust::estimate::essential<double>::model model_loose{};
        REQUIRE(estimation::robust::solver::essential<double>::solve(data, correspondence_count, residuals, inliers, inliers_loose, model_loose, 1.0e-3f));
        REQUIRE(inliers_tight < inliers_default);
        REQUIRE(inliers_loose > inliers_default);
    }

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

    // A sideways translation without rotation, where y' = y at every point, and the same with a rotation: every correspondence is an inlier of the true matrix.
    {
        for (int rotated = 0; rotated < 2; ++rotated) {
            const double angle = (rotated == 1) ? 0.05 : 0.0;
            const double rotation[9] = { std::cos(angle), 0.0, std::sin(angle), 0.0, 1.0, 0.0, -std::sin(angle), 0.0, std::cos(angle) };
            const double translation[3] = { 0.5, 0.0, 0.0 };
            core::random_pcg random;
            constexpr static const size_t correspondence_count = 200;
            estimation::correspondence_2d_2d<double> data[correspondence_count];
            for (size_t i = 0; i < correspondence_count; ++i) {
                const double point_xyz[3] = {
                    (static_cast<double>(random.get_random_raw() % 6000) / 1000.0) - 3.0,
                    (static_cast<double>(random.get_random_raw() % 6000) / 1000.0) - 3.0,
                    4.0 + (static_cast<double>(random.get_random_raw() % 10000) / 1000.0)
                };
                data[i].lhs[0] = point_xyz[0] / point_xyz[2];
                data[i].lhs[1] = point_xyz[1] / point_xyz[2];
                double point[2];
                project_point(rotation, translation, point_xyz, point);
                data[i].rhs[0] = point[0];
                data[i].rhs[1] = point[1];
            }
            double translation_matrix[9];
            cross_matrix(translation, translation_matrix);
            double expected[9];
            matrix_multiply(translation_matrix, rotation, expected);

            float residuals[correspondence_count];
            size_t inliers[correspondence_count];
            size_t inliers_size = 0;
            estimation::robust::estimate::essential<double>::model model{};
            REQUIRE(estimation::robust::solver::essential<double>::solve(data, correspondence_count, residuals, inliers, inliers_size, model));
            REQUIRE(inliers_size == correspondence_count);
            double recovered[9];
            for (int i = 0; i < 9; ++i) {
                recovered[i] = model.essential[i / 3][i % 3];
            }
            normalize_matrix(recovered);
            normalize_matrix(expected);
            double plus = 0.0;
            double minus = 0.0;
            for (int i = 0; i < 9; ++i) {
                plus += (recovered[i] - expected[i]) * (recovered[i] - expected[i]);
                minus += (recovered[i] + expected[i]) * (recovered[i] + expected[i]);
            }
            REQUIRE(std::sqrt(std::fmin(plus, minus)) < 1e-6);
        }
    }

    return EXIT_SUCCESS;
}
