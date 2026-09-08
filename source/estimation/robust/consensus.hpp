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

#pragma once
#ifndef ZEROSLAM_ESTIMATION_ROBUST_CONSENSUS_HPP
#define ZEROSLAM_ESTIMATION_ROBUST_CONSENSUS_HPP

#include "estimation/robust/estimator.hpp"
#include "estimation/robust/evaluator.hpp"
#include "estimation/robust/sampler.hpp"
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <type_traits>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::robust {
    // Sample consensus over a sampler, an estimator and an evaluator; the iteration budget adapts to the best inlier ratio found so far.
    template <
        typename template_sampler_type,
        typename template_estimator_type,
        typename template_evaluator_type
    >
    class consensus final {
    private:
        static_assert(template_sampler_type::sample_size == template_estimator_type::sample_size, "Sampler sample size must match estimator sample size.");
        static_assert(std::is_base_of<sampler<template_sampler_type::sample_size>, template_sampler_type>::value, "Sampler must implement the sampler interface.");
        static_assert(std::is_base_of<estimator<typename template_estimator_type::data_type, template_estimator_type::sample_size, typename template_estimator_type::model_type, template_estimator_type::models_size>, template_estimator_type>::value, "Estimator must implement the estimator interface.");
        static_assert(std::is_base_of<evaluator, template_evaluator_type>::value, "Evaluator must implement the evaluator interface.");

    public:
        constexpr static const size_t sample_size = template_estimator_type::sample_size;
        constexpr static const size_t models_size = template_estimator_type::models_size;

        using sampler_type = template_sampler_type;
        using estimator_type = template_estimator_type;
        using evaluator_type = template_evaluator_type;

        using data_type = typename template_estimator_type::data_type;
        using model_type = typename template_estimator_type::model_type;

    private:
        sampler_type consensus_sampler;
        estimator_type consensus_estimator;
        evaluator_type consensus_evaluator;

        float probability_failure;

        size_t iterations_minimum;

        size_t iterations_maximum;

    public:
        ~consensus() = default;

        consensus(
            const sampler_type& initial_sampler,
            const estimator_type& initial_estimator,
            const evaluator_type& initial_evaluator,
            const float initial_probability_failure,
            const size_t initial_iterations_minimum,
            const size_t initial_iterations_maximum
        )
            : consensus_sampler(initial_sampler)
            , consensus_estimator(initial_estimator)
            , consensus_evaluator(initial_evaluator)
            , probability_failure(initial_probability_failure)
            , iterations_minimum(initial_iterations_minimum)
            , iterations_maximum(initial_iterations_maximum) {
        }

    private:
        template <size_t raised_to_value>
        static constexpr double power(const double value) {
            if constexpr (raised_to_value == 0) {
                return 1.0;
            }
            else if constexpr (raised_to_value == 1) {
                return value;
            }
            else {
                const double temp = consensus::power<raised_to_value / 2>(value);
                if constexpr ((raised_to_value % 2) == 0) {
                    return temp * temp;
                }
                else {
                    return value * temp * temp;
                }
            }
        }

        static size_t constrain_max_iterations(
            const float inlier_ratio,
            const float probability_failure,
            const size_t iterations_minimum,
            const size_t iterations_maximum
        ) {
            if (inlier_ratio <= 0) {
                return iterations_maximum;
            }
            if (inlier_ratio >= 1) {
                return iterations_minimum;
            }
            if (probability_failure <= 0) {
                return iterations_maximum;
            }
            if (probability_failure >= 1) {
                return iterations_minimum;
            }

            const double probability_bad_sample = 1.0 - consensus::power<sample_size>(static_cast<double>(inlier_ratio));
            const double log_probability = math::log(probability_bad_sample);
            if (math::abs(log_probability) < 1e-12) {
                return iterations_maximum;
            }

            const double log_probability_failure = math::log(static_cast<double>(probability_failure));
            const double iterations_estimate = math::ceil(log_probability_failure / log_probability);
            const size_t iterations = static_cast<size_t>(iterations_estimate);

            if (iterations >= iterations_maximum) {
                return iterations_maximum;
            }
            if (iterations < iterations_minimum) {
                return iterations_minimum;
            }
            return iterations;
        }

    public:
        bool estimate(
            const data_type* const __restrict data,
            const size_t data_size,
            float* const __restrict residuals,
            size_t* const __restrict inliers,
            size_t& inliers_size,
            model_type& best_model
        ) {
            if (data_size < template_sampler_type::sample_size) {
                return false;
            }
            best_model = {};
            bool solution_found = false;
            float best_cost = 0;
            this->consensus_sampler.prepare(data_size);
            size_t iterations_adaptive = this->iterations_maximum;
            for (size_t i = 0; i < iterations_adaptive; ++i) {
                size_t subset_indices[sample_size];
                this->consensus_sampler.sample(&subset_indices[0]);
                data_type subset_data[sample_size];
                for (size_t j = 0; j < sample_size; ++j) {
                    subset_data[j] = data[subset_indices[j]];
                }
                model_type models[models_size];
                size_t model_count = this->consensus_estimator.generate_models(subset_data, sample_size, models);
                for (size_t j = 0; j < model_count; ++j) {
                    const model_type& model = models[j];
                    this->consensus_estimator.compute_residuals(data, data_size, model, residuals);
                    const float cost = this->consensus_evaluator.evaluate(residuals, data_size, inliers, inliers_size);
                    if ((cost < best_cost) || (solution_found == false)) {
                        solution_found = true;
                        best_model = model;
                        best_cost = cost;
                        iterations_adaptive = consensus::constrain_max_iterations(
                            static_cast<float>(inliers_size) / static_cast<float>(data_size),
                            this->probability_failure,
                            this->iterations_minimum,
                            iterations_adaptive
                        );
                    }
                }
            }
            if (!solution_found) {
                return false;
            }
            this->consensus_estimator.compute_residuals(data, data_size, best_model, residuals);
            this->consensus_evaluator.evaluate(residuals, data_size, inliers, inliers_size);
            return true;
        }
    };
}

#endif // ZEROSLAM_ESTIMATION_ROBUST_CONSENSUS_HPP
