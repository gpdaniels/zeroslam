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
    // Prebuilt consensus for the p3p estimator with the default sampler, evaluator, threshold and iteration budget; residuals and inliers must hold data_size entries.
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
            model_type& best_model
        );
    };

    extern template class p3p<float>;
    extern template class p3p<double>;
}

#endif // ZEROSLAM_ESTIMATION_ROBUST_SOLVER_P3P_HPP
