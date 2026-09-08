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

#include "estimation/robust/solver/p3p.hpp"

#include "estimation/robust/consensus.hpp"
#include "estimation/robust/evaluate/maximum_likelihood.hpp"
#include "estimation/robust/sample/random.hpp"
#include "math/math.hpp"

namespace estimation::robust::solver {
    template <typename type>
    bool p3p<type>::solve(
        const data_type* const __restrict correspondences,
        const size_t data_size,
        float* const __restrict residuals,
        size_t* const __restrict inliers,
        size_t& inliers_size,
        model_type& best_model
    ) {
        const float probability_failure = 0.01f;
        const size_t iterations_minimum = 5;
        const size_t iterations_maximum = 300;

        const float threshold_angle_radians = 1.0e-3f;
        const float residual_threshold = static_cast<float>(1.0 - math::cos(static_cast<double>(threshold_angle_radians)));

        sample::random<3> sampler;
        estimate::p3p<type> estimator;
        const evaluate::maximum_likelihood support(residual_threshold);

        consensus<decltype(sampler), decltype(estimator), decltype(support)> estimate(
            sampler,
            estimator,
            support,
            probability_failure,
            iterations_minimum,
            iterations_maximum
        );

        return estimate.estimate(correspondences, data_size, residuals, inliers, inliers_size, best_model);
    }

    template class p3p<float>;
    template class p3p<double>;
}
