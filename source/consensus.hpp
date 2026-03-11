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

#pragma once
#ifndef CONSENSUS_HPP
#define CONSENSUS_HPP

#include "assert.hpp"
#include "pose_estimation.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace consensus {
    template <size_t template_sample_size>
    class sampler_base {
    private:
        static_assert(template_sample_size > 0, "Invalid number of samples, must be greater than zero.");

    public:
        constexpr static const size_t sample_size = template_sample_size;

    protected:
        virtual ~sampler_base() = default;
        sampler_base() = default;
        sampler_base(const sampler_base&) = default;
        sampler_base(sampler_base&&) = default;
        sampler_base& operator=(const sampler_base&) = default;
        sampler_base& operator=(sampler_base&&) = default;

        virtual void prepare(const size_t data_size) = 0;

        virtual void sample(size_t* const __restrict indices) = 0;
    };

    template <
        typename template_data_type,
        size_t template_required_sample_size,
        typename template_model_type,
        size_t template_potential_models_size
    >
    class estimator_base {
    public:
        using data_type = template_data_type;
        constexpr static const size_t required_sample_size = template_required_sample_size;
        using model_type = template_model_type;
        constexpr static const size_t potential_models_size = template_potential_models_size;

    protected:
        virtual ~estimator_base() = default;
        estimator_base() = default;
        estimator_base(const estimator_base&) = default;
        estimator_base(estimator_base&&) = default;
        estimator_base& operator=(const estimator_base&) = default;
        estimator_base& operator=(estimator_base&&) = default;

    public:
        virtual size_t generate_models(
            const data_type* const __restrict data,
            size_t data_size,
            model_type* const __restrict models
        ) = 0;

        virtual void compute_residuals(
            const data_type* const __restrict data,
            size_t data_size,
            const model_type& model,
            float* const __restrict residuals
        ) = 0;
    };

    class evaluator_base {
    protected:
        virtual ~evaluator_base() = default;
        evaluator_base() = default;
        evaluator_base(const evaluator_base&) = default;
        evaluator_base(evaluator_base&&) = default;
        evaluator_base& operator=(const evaluator_base&) = default;
        evaluator_base& operator=(evaluator_base&&) = default;

    public:
        virtual float evaluate(
            const float* const __restrict residuals,
            const size_t residuals_size,
            size_t* const __restrict inliers,
            size_t& inliers_size
        ) = 0;
    };

    template <
        typename template_sampler_type,
        typename template_estimator_type,
        typename template_evaluator_type
    >
    class consensus final {
    private:
        static_assert(template_sampler_type::sample_size == template_estimator_type::required_sample_size, "Sampler sample size must match estimator sample size.");

    public:
        constexpr static const size_t sample_size = template_estimator_type::required_sample_size;
        constexpr static const size_t models_size = template_estimator_type::potential_models_size;

        using sampler_type = template_sampler_type;
        using estimator_type = template_estimator_type;
        using evaluator_type = template_evaluator_type;

        using data_type = typename template_estimator_type::data_type;
        using model_type = typename template_estimator_type::model_type;

    private:
        sampler_type consensus_sampler;
        estimator_type consensus_estimator;
        evaluator_type consensus_evaluator;

        // Criteria for early exit: Reached a minimal model inlier ratio.
        float criteria_inlier_ratio;

        // Criteria for early exit: The failure probability a given model, e.g. 0.01 means a 1% chance of incorrect result.
        float criteria_probability_failure;

        // The minimum number of iterations before exiting if other criteria are met.
        size_t criteria_iterations_minimum;

        // The maximum number of iterations before exiting.
        size_t criteria_iterations_maximum;

    public:
        ~consensus() = default;

        consensus(
            const sampler_type& sampler,
            const estimator_type& estimator,
            const evaluator_type& evaluator,
            const float inlier_ratio,
            const float probability_failure,
            const size_t iterations_minimum,
            const size_t iterations_maximum
        )
            : consensus_sampler(sampler)
            , consensus_estimator(estimator)
            , consensus_evaluator(evaluator)
            , criteria_inlier_ratio(inlier_ratio)
            , criteria_probability_failure(probability_failure)
            , criteria_iterations_minimum(iterations_minimum)
            , criteria_iterations_maximum(constrain_max_iterations(inlier_ratio, probability_failure, iterations_minimum, iterations_maximum)) {
        }

    private:
        template <size_t raised_to_value>
        static float constexpr power(const float value) {
            if constexpr (raised_to_value == 0) {
                return 1;
            }
            else if constexpr (raised_to_value == 1) {
                return value;
            }
            else {
                const float temp = power<raised_to_value / 2>(value);
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

            constexpr static const auto log_approx = [](float value) -> float {
                int punned;
                const unsigned char* value_pointer = reinterpret_cast<const unsigned char*>(&value);
                unsigned char* punned_pointer = reinterpret_cast<unsigned char*>(&punned);
                for (int i = 0; i < 4; ++i) {
                    punned_pointer[i] = value_pointer[i];
                }
                const float magic = static_cast<float>(static_cast<double>(static_cast<float>((punned & 8388607) - 4074142)) * 5.828231702537851e-8);
                return static_cast<float>(static_cast<double>(static_cast<float>(punned)) * 8.262958294867817e-8 - static_cast<double>(magic * magic) - 87.96988524938206);
            };

            constexpr static const auto abs = [](float value) -> float {
                if ((value + 0.0f) < 0) {
                    return -value;
                }
                return value;
            };

            const float log_probability = log_approx(1.0f - power<sample_size>(inlier_ratio));
            if (abs(log_probability) < 1e-5f) {
                return iterations_maximum;
            }

            const float log_probability_failure = log_approx(probability_failure);
            const size_t iterations = static_cast<size_t>(log_probability_failure / log_probability);

            // Constrain the iterations estimate by the max and min parameters.
            if (iterations > iterations_maximum) {
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
            for (size_t i = 0; i < this->criteria_iterations_maximum; ++i) {
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
                        this->criteria_iterations_maximum = constrain_max_iterations(
                            static_cast<float>(inliers_size) / static_cast<float>(data_size),
                            this->criteria_probability_failure,
                            this->criteria_iterations_minimum,
                            this->criteria_iterations_maximum
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

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace consensus {
    template <size_t sample_size>
    class random final
        : public sampler_base<sample_size> {
    private:
        class random_pcg final {
        private:
            unsigned long long int state = 0x853C49E6748FEA9Bull;
            unsigned long long int increment = 0xDA3E39CB94B95BDBull;

        public:
            unsigned int get_random() {
                unsigned long long int state_previous = this->state;
                this->state = state_previous * 0x5851F42D4C957F2Dull + this->increment;
                unsigned int state_shift_xor_shift = static_cast<unsigned int>(((state_previous >> 18u) ^ state_previous) >> 27u);
                int rotation = state_previous >> 59u;
                return (state_shift_xor_shift >> rotation) | (state_shift_xor_shift << ((-rotation) & 31));
            }
        };

    private:
        random_pcg rng;
        size_t size = 0;

    public:
        virtual void prepare(
            const size_t data_size
        ) override final {
            ASSERT(data_size > 0, "Data size must be greater than zero.");
            ASSERT(data_size >= sample_size, "Data size must be greater or equal to the sample size.");
            this->size = data_size;
        }

    public:
        virtual void sample(
            size_t* const __restrict indices
        ) override final {
            ASSERT(this->size > 0, "Data size must be greater than zero.");
            ASSERT(this->size >= sample_size, "Data size must be greater or equal to the sample size.");
            for (size_t i = 0; i < sample_size; ++i) {
                // Get a random index.
                indices[i] = static_cast<size_t>(this->rng.get_random()) % this->size;
                // Ensure it is unique.
                for (size_t j = 0; j < i; ++j) {
                    if (indices[j] == indices[i]) {
                        indices[i] = (indices[i] + 1) % this->size;
                        j = static_cast<size_t>(-1);
                    }
                }
            }
        }
    };
}

namespace consensus {
    template <typename scalar_type>
    class xy {
    public:
        scalar_type x, y;
    };

    template <typename scalar_type>
    class correspondence_2d_2d {
    public:
        xy<scalar_type> lhs, rhs;
    };

    template <typename scalar_type>
    struct model_essential {
        scalar_type essential[3][3];
    };

    template <typename scalar_type>
    class essential final
        : public estimator_base<correspondence_2d_2d<scalar_type>, 5, model_essential<scalar_type>, 10> {
    public:
        virtual size_t generate_models(
            const typename essential::data_type* const __restrict data,
            size_t data_size,
            typename essential::model_type* const __restrict models
        ) override final {
            if (data_size < 5) {
                return 0;
            }
            const scalar_type lhs[5 * 2]{
                data[0].lhs.x,
                data[0].lhs.y,
                data[1].lhs.x,
                data[1].lhs.y,
                data[2].lhs.x,
                data[2].lhs.y,
                data[3].lhs.x,
                data[3].lhs.y,
                data[4].lhs.x,
                data[4].lhs.y,
            };
            const scalar_type rhs[5 * 2]{
                data[0].rhs.x,
                data[0].rhs.y,
                data[1].rhs.x,
                data[1].rhs.y,
                data[2].rhs.x,
                data[2].rhs.y,
                data[3].rhs.x,
                data[3].rhs.y,
                data[4].rhs.x,
                data[4].rhs.y,
            };
            scalar_type essentials[10][3][3];
            size_t generated_models_count = pose_estimation::essential_5_point(&lhs[0], &rhs[0], &essentials[0][0][0]);
            for (size_t i = 0; i < generated_models_count; ++i) {
                for (size_t y = 0; y < 3; ++y) {
                    for (size_t x = 0; x < 3; ++x) {
                        models[i].essential[y][x] = essentials[i][y][x];
                    }
                }
            }
            return generated_models_count;
        }

        virtual void compute_residuals(
            const typename essential::data_type* const __restrict data,
            size_t data_size,
            const typename essential::model_type& model,
            float* const __restrict residuals
        ) override final {
            constexpr static const auto matrix_transpose = [](const scalar_type* matrix, int width, int height, scalar_type* result) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        result[x * height + y] = matrix[y * width + x];
                    }
                }
            };

            constexpr static const auto matrix_multiply = [](const scalar_type* lhs, int lhs_width, int lhs_height, const scalar_type* rhs, int rhs_width, int rhs_height, scalar_type* result) {
                static_cast<void>(rhs_height);
                for (int lhs_y = 0; lhs_y < lhs_height; ++lhs_y) {
                    for (int rhs_x = 0; rhs_x < rhs_width; ++rhs_x) {
                        scalar_type sum = 0;
                        for (int lhs_x_rhs_y = 0; lhs_x_rhs_y < lhs_width; ++lhs_x_rhs_y) {
                            sum += lhs[lhs_y * lhs_width + lhs_x_rhs_y] * rhs[lhs_x_rhs_y * rhs_width + rhs_x];
                        }
                        result[lhs_y * rhs_width + rhs_x] = sum;
                    }
                }
            };

            scalar_type essential_transpose[3][3];
            matrix_transpose(&model.essential[0][0], 3, 3, &essential_transpose[0][0]);

            // Sampson Distance.
            for (size_t i = 0; i < data_size; ++i) {
                const scalar_type lhs_point[3] = {
                    data[i].lhs.x,
                    data[i].lhs.y,
                    1
                };
                const scalar_type rhs_point[3] = {
                    data[i].rhs.x,
                    data[i].rhs.y,
                    1
                };

                scalar_type essential_lhs[3];
                scalar_type essential_rhs[3];
                matrix_multiply(&model.essential[0][0], 3, 3, &lhs_point[0], 1, 3, &essential_lhs[0]);
                matrix_multiply(&essential_transpose[0][0], 3, 3, &rhs_point[0], 1, 3, &essential_rhs[0]);

                const scalar_type prxelx_pryely_przelz = rhs_point[0] * essential_lhs[0] + rhs_point[1] * essential_lhs[1] + rhs_point[2] * essential_lhs[2];
                const scalar_type elx2 = (essential_lhs[0] * essential_lhs[0]);
                const scalar_type ely2 = (essential_lhs[1] * essential_lhs[1]);
                const scalar_type erx2 = (essential_rhs[0] * essential_rhs[0]);
                const scalar_type ery2 = (essential_rhs[1] * essential_rhs[1]);
                residuals[i] = (prxelx_pryely_przelz * prxelx_pryely_przelz) / (elx2 + ely2 + erx2 + ery2);
            }
        }
    };
}

namespace consensus {
    class inlier_support final
        : public evaluator_base {
    private:
        const float threshold;

    public:
        inlier_support(const float residual_threshold)
            : threshold(residual_threshold) {
        }

    public:
        virtual float evaluate(
            const float* const __restrict residuals,
            const size_t residuals_size,
            size_t* const __restrict inliers,
            size_t& inliers_size
        ) override final {
            inliers_size = 0;
            for (size_t i = 0; i < residuals_size; ++i) {
                if (residuals[i] < this->threshold) {
                    inliers[inliers_size++] = i;
                }
            }
            return static_cast<float>(residuals_size - inliers_size);
        }
    };
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace consensus {
    static inline bool solve_ransac_essential(const correspondence_2d_2d<double>* correspondences, size_t data_size, float* residuals, size_t* inliers, size_t& inliers_size, model_essential<double>& model) {
        const float probability_failure = 0.99f;
        const float inlier_ratio = 0.8f;
        const size_t iterations_minimum = 5;
        const size_t iterations_maximum = 300;
        const float residual_threshold = 1.0e-5f;

        random<5> random;
        essential<double> estimator;
        inlier_support inlier_support(residual_threshold);

        consensus<decltype(random), decltype(estimator), decltype(inlier_support)> consensus(
            random,
            estimator,
            inlier_support,
            inlier_ratio,
            probability_failure,
            iterations_minimum,
            iterations_maximum
        );

        return consensus.estimate(correspondences, data_size, residuals, inliers, inliers_size, model);
    }
}

#endif // CONSENSUS_HPP
