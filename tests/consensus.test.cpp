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

#include "consensus.hpp"

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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // Line fitting test:
    if (0) {
        class xy final {
        public:
            float x, y;
        };

        class line final {
        public:
            float gradient;
            float intercept;
        };

        class line_estimator final
            : public consensus::estimator_base<xy, 2, line, 1> {
        public:
            virtual std::size_t generate_models(const data_type* const __restrict data, std::size_t data_size, model_type* const __restrict models) override final {
                REQUIRE(data_size == 2);
                models[0].gradient = (data[1].y - data[0].y) / (data[1].x - data[0].x);
                models[0].intercept = data[1].y - models[0].gradient * data[1].x;
                return 1;
            }

            virtual void compute_residuals(const data_type* const __restrict data, std::size_t data_size, const model_type& model, float* const __restrict residuals) override final {
                const float denominator = std::sqrt(model.gradient * model.gradient + 1.0f);
                for (std::size_t i = 0; i < data_size; ++i) {
                    residuals[i] = std::abs(-model.gradient * data[i].x + data[i].y - model.intercept) / denominator;
                }
            }
        };

        class random_pcg final {
        private:
            unsigned long long int state = 0x853C49E6748FEA9Bull;
            unsigned long long int increment = 0xDA3E39CB94B95BDBull;

        private:
            unsigned int get_random_raw() {
                unsigned long long int state_previous = this->state;
                this->state = state_previous * 0x5851F42D4C957F2Dull + this->increment;
                unsigned int state_shift_xor_shift = static_cast<unsigned int>(((state_previous >> 18u) ^ state_previous) >> 27u);
                int rotation = state_previous >> 59u;
                return (state_shift_xor_shift >> rotation) | (state_shift_xor_shift << ((-rotation) & 31));
            }

        public:
            float get_random_exclusive_top() {
                return static_cast<float>(this->get_random_raw()) * (1.0f / static_cast<float>(1ull << 32));
            }
        };

        constexpr static const std::size_t data_size = 2000;
        random_pcg rng;

        const float gradient = 1.234f;
        const float intercept = 5.6789f;

        // Create a dataset with half the points fitting the line (y = 1.234x + 5.6789) and half noise.
        xy data[data_size];
        for (std::size_t i = 0; i < data_size; ++i) {
            if (i % 2 == 0) {
                const float line_noise_x = rng.get_random_exclusive_top() * 0.1f;
                const float line_noise_y = rng.get_random_exclusive_top() * 0.1f;
                data[i] = xy{ static_cast<float>(i) + line_noise_x, static_cast<float>(i) * gradient + intercept + line_noise_y };
            }
            else {
                const float random_noise_x = rng.get_random_exclusive_top() * data_size;
                const float random_noise_y = rng.get_random_exclusive_top() * data_size;
                data[i] = xy{ random_noise_x, random_noise_y };
            }
        }

        const float probability_failure = 0.01f;
        const float inlier_ratio = 0.4f;
        const std::size_t iterations_minimum = 0;
        const std::size_t iterations_maximum = 100;
        const float residual_threshold = 0.1f;

        consensus::random<2> random;
        line_estimator estimator;
        consensus::inlier_support inlier_support(residual_threshold);

        consensus::consensus<consensus::random<2>, line_estimator, consensus::inlier_support> consensus(
            random,
            estimator,
            inlier_support,
            inlier_ratio,
            probability_failure,
            iterations_minimum,
            iterations_maximum
        );

        float residuals[data_size];
        std::size_t inliers[data_size];
        std::size_t inliers_size = 0;
        line best_model;
        REQUIRE(consensus.estimate(data, data_size, residuals, inliers, inliers_size, best_model));

        REQUIRE(is_value_approx(best_model.gradient, gradient, 0.05f));
        REQUIRE(is_value_approx(best_model.intercept, intercept, 0.05f));
        REQUIRE(is_value_approx(inliers_size, data_size / 2, static_cast<std::size_t>(data_size * 0.01f)));
    }

    {
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
        constexpr static const auto project_point = [](const double* rotation, const double* translation, const double* point_xyz, double* point_xy) {
            double point[3];
            matrix_vector_multiply(rotation, point_xyz, point);
            point[0] += translation[0];
            point[1] += translation[1];
            point[2] += translation[2];
            point_xy[0] = point[0] / point[2];
            point_xy[1] = point[1] / point[2];
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
        double rotation[9];
        // R = Rz * Ry * Rx
        matrix_multiply(temp, &rotation_x[0][0], rotation);
        double translation[3] = { 0.5, -0.3, 0.7 };
        // For other camera use the origin.
        const double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const double zero[3] = { 0, 0, 0 };
        // Create more than five world points (not coplanar).
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
        consensus::correspondence_2d_2d<double> data[correspondence_count];
        for (int i = 0; i < inlier_count; ++i) {
            double point[2];
            project_point(identity, zero, &world_points[i][0], point);
            data[i].lhs.x = point[0];
            data[i].lhs.y = point[1];
            project_point(rotation, translation, &world_points[i][0], point);
            data[i].rhs.x = point[0];
            data[i].rhs.y = point[1];
        }
        for (int i = 0; i < outlier_count; ++i) {
            double point[2];
            project_point(identity, zero, &world_points[i][0], point);
            data[inlier_count + i].lhs.x = point[0] - 50;
            data[inlier_count + i].lhs.y = point[1] + 13;
            project_point(identity, zero, &world_points[i][0], point);
            data[inlier_count + i].rhs.x = point[0] + 20;
            data[inlier_count + i].rhs.y = point[1] - 80;
        }

        float residuals[correspondence_count];
        size_t inliers[correspondence_count];
        size_t inliers_size = 0;
        consensus::model_essential<double> model{};

        bool ok = consensus::solve_ransac_essential(
            data,
            correspondence_count,
            residuals,
            inliers,
            inliers_size,
            model
        );
        REQUIRE(ok);
        REQUIRE(inliers_size == inlier_count);

        // Residuals should not be NaN
        for (size_t i = 0; i < correspondence_count; ++i) {
            REQUIRE(std::isfinite(residuals[i]));
        }

        // Model should contain finite numbers
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                REQUIRE(std::isfinite(model.essential[y][x]));
            }
        }
    }

    return EXIT_SUCCESS;
}