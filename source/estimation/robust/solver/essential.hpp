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
#ifndef ZEROSLAM_ESTIMATION_ROBUST_SOLVER_ESSENTIAL_HPP
#define ZEROSLAM_ESTIMATION_ROBUST_SOLVER_ESSENTIAL_HPP

#include "estimation/robust/estimate/essential.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::robust::solver {
    // Prebuilt consensus for the essential estimator with a sampler seeded from the data, the maximum likelihood evaluator and a fixed iteration budget, then a linear refit on the inliers kept when it lowers the cost; residuals and inliers must hold data_size entries.
    // The residual_threshold bounds the squared Sampson distance in normalised image units, so for a tolerance in pixels pass (pixels / f)^2; with isotropic noise of sigma pixels in both images an inlier's residual is (sigma / f)^2 times a chi-squared variable of one degree of freedom, so 3.84 (sigma / f)^2 keeps 95% of them.
    template <typename type>
    class essential final {
    public:
        using data_type = correspondence_2d_2d<type>;
        using model_type = estimate::model_essential<type>;

    public:
        static bool solve(
            const data_type* const __restrict correspondences,
            const size_t data_size,
            float* const __restrict residuals,
            size_t* const __restrict inliers,
            size_t& inliers_size,
            model_type& best_model,
            const float residual_threshold = 1.0e-5f
        );
    };

    extern template class essential<float>;
    extern template class essential<double>;
}

#endif // ZEROSLAM_ESTIMATION_ROBUST_SOLVER_ESSENTIAL_HPP
