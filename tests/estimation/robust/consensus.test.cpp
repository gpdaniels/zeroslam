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

#include "estimation/robust/consensus.hpp"

#include "core/random_pcg.hpp"
#include "estimation/robust/evaluate/inlier_support.hpp"
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

static inline bool is_value_approx(float lhs, float rhs, float epsilon = 1e-5f) {
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
    : public estimation::robust::estimator<xy, 2, line, 1> {
public:
    virtual size_t generate_models(const xy* const __restrict data, const size_t data_size, line* const __restrict models) const override final {
        REQUIRE(data_size == 2);
        models[0].gradient = (data[1].y - data[0].y) / (data[1].x - data[0].x);
        models[0].intercept = data[1].y - models[0].gradient * data[1].x;
        return 1;
    }

    virtual void compute_residuals(const xy* const __restrict data, const size_t data_size, const line& model, float* const __restrict residuals) const override final {
        const float denominator = std::sqrt(model.gradient * model.gradient + 1.0f);
        for (size_t i = 0; i < data_size; ++i) {
            residuals[i] = std::abs(-model.gradient * data[i].x + data[i].y - model.intercept) / denominator;
        }
    }
};

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // Line fitting with half the points as noise, repeated on the same object and on a fresh one.
    {
        constexpr static const size_t data_size = 2000;
        core::random_pcg rng;

        const float gradient = 1.234f;
        const float intercept = 5.6789f;

        xy data[data_size];
        for (size_t i = 0; i < data_size; ++i) {
            if (i % 2 == 0) {
                const float line_noise_x = static_cast<float>(rng.get_random_exclusive_top()) * 0.1f;
                const float line_noise_y = static_cast<float>(rng.get_random_exclusive_top()) * 0.1f;
                data[i] = xy{ static_cast<float>(i) + line_noise_x, static_cast<float>(i) * gradient + intercept + line_noise_y };
            }
            else {
                const float random_noise_x = static_cast<float>(rng.get_random_exclusive_top()) * data_size;
                const float random_noise_y = static_cast<float>(rng.get_random_exclusive_top()) * data_size;
                data[i] = xy{ random_noise_x, random_noise_y };
            }
        }

        const float probability_failure = 0.01f;
        const size_t iterations_minimum = 0;
        const size_t iterations_maximum = 100;
        const float residual_threshold = 0.1f;

        estimation::robust::sample::random<2> random;
        line_estimator estimator;
        estimation::robust::evaluate::inlier_support inlier_support(residual_threshold);

        estimation::robust::consensus<estimation::robust::sample::random<2>, line_estimator, estimation::robust::evaluate::inlier_support> consensus(
            random,
            estimator,
            inlier_support,
            probability_failure,
            iterations_minimum,
            iterations_maximum
        );

        float residuals[data_size];
        size_t inliers[data_size];
        size_t inliers_size = 0;
        line best_model;
        REQUIRE(consensus.estimate(data, data_size, residuals, inliers, inliers_size, best_model));

        REQUIRE(is_value_approx(best_model.gradient, gradient, 0.05f));
        REQUIRE(is_value_approx(best_model.intercept, intercept, 0.05f));
        REQUIRE(is_value_approx(static_cast<float>(inliers_size), static_cast<float>(data_size / 2), static_cast<float>(data_size) * 0.01f));

        float residuals_repeat[data_size];
        size_t inliers_repeat[data_size];
        size_t inliers_size_repeat = 0;
        line best_model_repeat;
        REQUIRE(consensus.estimate(data, data_size, residuals_repeat, inliers_repeat, inliers_size_repeat, best_model_repeat));

        REQUIRE(is_value_approx(best_model_repeat.gradient, gradient, 0.05f));
        REQUIRE(is_value_approx(best_model_repeat.intercept, intercept, 0.05f));
        REQUIRE(is_value_approx(static_cast<float>(inliers_size_repeat), static_cast<float>(data_size / 2), static_cast<float>(data_size) * 0.01f));

        decltype(consensus) consensus_fresh(
            random,
            estimator,
            inlier_support,
            probability_failure,
            iterations_minimum,
            iterations_maximum
        );

        float residuals_fresh[data_size];
        size_t inliers_fresh[data_size];
        size_t inliers_size_fresh = 0;
        line best_model_fresh;
        REQUIRE(consensus_fresh.estimate(data, data_size, residuals_fresh, inliers_fresh, inliers_size_fresh, best_model_fresh));

        REQUIRE(best_model_fresh.gradient == best_model.gradient);
        REQUIRE(best_model_fresh.intercept == best_model.intercept);
        REQUIRE(inliers_size_fresh == inliers_size);

        REQUIRE(is_value_approx(best_model_repeat.gradient, best_model_fresh.gradient, 0.05f));
        REQUIRE(is_value_approx(best_model_repeat.intercept, best_model_fresh.intercept, 0.05f));
        REQUIRE(is_value_approx(static_cast<float>(inliers_size_repeat), static_cast<float>(inliers_size_fresh), static_cast<float>(data_size) * 0.01f));
    }

    return EXIT_SUCCESS;
}
