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
#ifndef ZEROSLAM_ESTIMATION_ROBUST_ESTIMATOR_HPP
#define ZEROSLAM_ESTIMATION_ROBUST_ESTIMATOR_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::robust {
    // Interface of a model estimator: generate up to models_size models from sample_size data, and score every datum against a model.
    template <
        typename template_data_type,
        size_t template_sample_size,
        typename template_model_type,
        size_t template_models_size
    >
    class estimator {
    public:
        using data_type = template_data_type;
        using model_type = template_model_type;
        constexpr static const size_t sample_size = template_sample_size;
        constexpr static const size_t models_size = template_models_size;

    protected:
        virtual ~estimator() = default;
        estimator() = default;
        estimator(const estimator&) = default;
        estimator(estimator&&) = default;
        estimator& operator=(const estimator&) = default;
        estimator& operator=(estimator&&) = default;

    public:
        virtual size_t generate_models(
            const data_type* const __restrict data,
            const size_t data_size,
            model_type* const __restrict models
        ) const = 0;

        virtual void compute_residuals(
            const data_type* const __restrict data,
            const size_t data_size,
            const model_type& candidate,
            float* const __restrict residuals
        ) const = 0;
    };
}

#endif // ZEROSLAM_ESTIMATION_ROBUST_ESTIMATOR_HPP
