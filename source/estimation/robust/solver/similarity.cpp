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

#include "estimation/robust/solver/similarity.hpp"

#include "estimation/minimal/similarity_3_point.hpp"
#include "estimation/robust/consensus.hpp"
#include "estimation/robust/evaluate/maximum_likelihood.hpp"
#include "estimation/robust/sample/random.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace estimation::robust::solver {
    template <typename type>
    bool similarity<type>::solve(
        const data_type* const __restrict correspondences,
        const size_t data_size,
        const float residual_threshold,
        float* const __restrict residuals,
        size_t* const __restrict inliers,
        size_t& inliers_size,
        model_type& best_model
    ) {
        const float probability_failure = 0.01f;
        const size_t iterations_minimum = 5;
        const size_t iterations_maximum = 256;
        sample::random<3> sampler;
        estimate::similarity<type> estimator;
        const evaluate::maximum_likelihood support(residual_threshold);
        consensus<decltype(sampler), decltype(estimator), decltype(support)> estimate(
            sampler,
            estimator,
            support,
            probability_failure,
            iterations_minimum,
            iterations_maximum
        );
        if (!estimate.estimate(correspondences, data_size, residuals, inliers, inliers_size, best_model)) {
            return false;
        }

        std::vector<type> source(inliers_size * 3);
        std::vector<type> target(inliers_size * 3);
        for (size_t i = 0; i < inliers_size; ++i) {
            for (size_t axis = 0; axis < 3; ++axis) {
                source[(i * 3) + axis] = correspondences[inliers[i]].lhs[axis];
                target[(i * 3) + axis] = correspondences[inliers[i]].rhs[axis];
            }
        }
        type rotation[9];
        model_type refit_model;
        if (!minimal::similarity_3_point<type>::solve(source.data(), target.data(), inliers_size, rotation, refit_model.translation, refit_model.scale)) {
            return true;
        }
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                refit_model.rotation[row][column] = rotation[(row * 3) + column];
            }
        }
        std::vector<float> refit_residuals(data_size);
        std::vector<size_t> refit_inliers(data_size);
        size_t refit_inliers_size = 0;
        estimator.compute_residuals(correspondences, data_size, refit_model, refit_residuals.data());
        support.evaluate(refit_residuals.data(), data_size, refit_inliers.data(), refit_inliers_size);
        if (refit_inliers_size >= inliers_size) {
            best_model = refit_model;
            inliers_size = refit_inliers_size;
            for (size_t i = 0; i < data_size; ++i) {
                residuals[i] = refit_residuals[i];
            }
            for (size_t i = 0; i < inliers_size; ++i) {
                inliers[i] = refit_inliers[i];
            }
        }
        return true;
    }

    template class similarity<float>;
    template class similarity<double>;
}
