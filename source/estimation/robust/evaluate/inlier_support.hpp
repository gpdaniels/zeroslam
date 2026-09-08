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
#ifndef ZEROSLAM_ESTIMATION_ROBUST_EVALUATE_INLIER_SUPPORT_HPP
#define ZEROSLAM_ESTIMATION_ROBUST_EVALUATE_INLIER_SUPPORT_HPP

#include "estimation/robust/evaluator.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::robust::evaluate {
    // Inliers are the residuals below the threshold; the cost is the number of outliers.
    class inlier_support final
        : public evaluator {
    private:
        float threshold;

    public:
        explicit inlier_support(const float residual_threshold);

    public:
        virtual float evaluate(
            const float* const __restrict residuals,
            const size_t residuals_size,
            size_t* const __restrict inliers,
            size_t& inliers_size
        ) const override final;
    };
}

#endif // ZEROSLAM_ESTIMATION_ROBUST_EVALUATE_INLIER_SUPPORT_HPP
