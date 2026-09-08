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
#ifndef ZEROSLAM_ESTIMATION_ROBUST_EVALUATE_LEAST_MEDIAN_OF_SQUARES_HPP
#define ZEROSLAM_ESTIMATION_ROBUST_EVALUATE_LEAST_MEDIAN_OF_SQUARES_HPP

#include "estimation/robust/evaluator.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::robust::evaluate {
    // The cost is the median squared residual; inliers are within a robust standard deviation band derived from it, so no threshold is needed.
    class least_median_of_squares final
        : public evaluator {
    private:
        size_t sample_size;

    public:
        explicit least_median_of_squares(const size_t minimum_sample_size);

    public:
        virtual float evaluate(
            const float* const __restrict residuals,
            const size_t residuals_size,
            size_t* const __restrict inliers,
            size_t& inliers_size
        ) const override final;

    private:
        static float nth_element(float* const __restrict data, const size_t data_size, const size_t n);
    };
}

#endif // ZEROSLAM_ESTIMATION_ROBUST_EVALUATE_LEAST_MEDIAN_OF_SQUARES_HPP
