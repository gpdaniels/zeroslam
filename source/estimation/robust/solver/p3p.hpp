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
#ifndef ZEROSLAM_ESTIMATION_ROBUST_SOLVER_P3P_HPP
#define ZEROSLAM_ESTIMATION_ROBUST_SOLVER_P3P_HPP

#include "estimation/robust/estimate/p3p.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::robust::solver {
    // Prebuilt consensus for the p3p estimator with a sampler seeded from the data, the maximum likelihood evaluator and the default iteration budget; residuals and inliers must hold data_size entries.
    // The residual_threshold bounds one minus the cosine of the angle between the observed and reprojected bearings, so for a tolerance in pixels pass 1 - cos(pixels / f), about (pixels / f)^2 / 2 near the principal point, and the default is 1 - cos(2.5e-3); with isotropic noise of sigma pixels there an inlier's residual is (sigma / f)^2 / 2 times a chi-squared variable of two degrees of freedom, so 3.0 (sigma / f)^2 keeps 95% of them.
    template <typename type>
    class p3p final {
    public:
        using data_type = correspondence_2d_3d<type>;
        using model_type = estimate::model_p3p<type>;

    public:
        static bool solve(
            const data_type* const __restrict correspondences,
            const size_t data_size,
            float* const __restrict residuals,
            size_t* const __restrict inliers,
            size_t& inliers_size,
            model_type& best_model,
            const float residual_threshold = 3.12499833e-6f
        );
    };

    extern template class p3p<float>;
    extern template class p3p<double>;
}

#endif // ZEROSLAM_ESTIMATION_ROBUST_SOLVER_P3P_HPP
