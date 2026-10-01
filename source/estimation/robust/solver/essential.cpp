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

#include "estimation/robust/solver/essential.hpp"

#include "estimation/robust/consensus.hpp"
#include "estimation/robust/evaluate/maximum_likelihood.hpp"
#include "estimation/robust/sample/random.hpp"
#include "math/math.hpp"
#include "math/matrix_decomposition_qr.hpp"
#include "math/matrix_decomposition_singular_value.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace estimation::robust::solver {
    template <typename type>
    bool essential<type>::solve(
        const data_type* const __restrict correspondences,
        const size_t data_size,
        float* const __restrict residuals,
        size_t* const __restrict inliers,
        size_t& inliers_size,
        model_type& best_model,
        const float residual_threshold
    ) {
        static_assert(sizeof(data_type) == 4 * sizeof(type), "The sampler seed hashes the correspondences, which must hold only their coordinates.");

        // A five point sample of noisy inliers is often far from the best model, so the adaptive stop is held back until enough samples are drawn.
        const float probability_failure = 0.01f;
        const size_t iterations_minimum = 100;
        const size_t iterations_maximum = 300;

        sample::random<5> sampler(sample::random<5>::seed_from(correspondences, data_size));
        estimate::essential<type> estimator;
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

        if (inliers_size >= 8) {
            const float best_cost = support.evaluate(residuals, data_size, inliers, inliers_size);

            std::vector<type> design(inliers_size * 9);
            for (size_t i = 0; i < inliers_size; ++i) {
                const type lhs_x = correspondences[inliers[i]].lhs[0];
                const type lhs_y = correspondences[inliers[i]].lhs[1];
                const type rhs_x = correspondences[inliers[i]].rhs[0];
                const type rhs_y = correspondences[inliers[i]].rhs[1];
                design[(i * 9) + 0] = lhs_x * rhs_x;
                design[(i * 9) + 1] = lhs_y * rhs_x;
                design[(i * 9) + 2] = rhs_x;
                design[(i * 9) + 3] = lhs_x * rhs_y;
                design[(i * 9) + 4] = lhs_y * rhs_y;
                design[(i * 9) + 5] = rhs_y;
                design[(i * 9) + 6] = lhs_x;
                design[(i * 9) + 7] = lhs_y;
                design[(i * 9) + 8] = type(1);
            }

            // The design shares its singular values and right singular vectors with its triangular factor, which avoids squaring the condition number in the normal matrix.
            type betas[9] = {};
            std::vector<type> workspace(inliers_size);
            math::decompose_qr_householder_in_place(design.data(), static_cast<int>(inliers_size), 9, &betas[0], workspace.data());
            type triangular[9][9] = {};
            for (size_t row = 0; (row < 9) && (row < inliers_size); ++row) {
                for (size_t col = row; col < 9; ++col) {
                    triangular[row][col] = design[(row * 9) + col];
                }
            }
            type linear_u[9][9];
            type linear_s[9][9];
            type linear_vt[9][9];
            if (!math::decompose_singular_value(&triangular[0][0], 9, 9, &linear_u[0][0], &linear_s[0][0], &linear_vt[0][0])) {
                return true;
            }
            // Planar or otherwise degenerate inliers leave a null space of more than one dimension, any member of which is arbitrary.
            if (!(linear_s[7][7] >= type(1.0e-9) * linear_s[0][0])) {
                return true;
            }

            model_type refit_model;
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    refit_model.essential[row][col] = linear_vt[8][(row * 3) + col];
                }
            }
            type manifold_u[9];
            type manifold_s[9];
            type manifold_vt[9];
            if (!math::decompose_singular_value(&refit_model.essential[0][0], 3, 3, &manifold_u[0], &manifold_s[0], &manifold_vt[0])) {
                return true;
            }
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    refit_model.essential[row][col] = (manifold_u[(row * 3) + 0] * manifold_vt[(0 * 3) + col]) + (manifold_u[(row * 3) + 1] * manifold_vt[(1 * 3) + col]);
                }
            }

            // The refit replaces the consensus model only when it lowers the same cost.
            std::vector<float> refit_residuals(data_size);
            std::vector<size_t> refit_inliers(data_size);
            size_t refit_inliers_size = 0;
            estimator.compute_residuals(correspondences, data_size, refit_model, refit_residuals.data());
            const float refit_cost = support.evaluate(refit_residuals.data(), data_size, refit_inliers.data(), refit_inliers_size);
            if ((refit_cost < best_cost) && (refit_inliers_size >= 5)) {
                best_model = refit_model;
                inliers_size = refit_inliers_size;
                for (size_t i = 0; i < data_size; ++i) {
                    residuals[i] = refit_residuals[i];
                }
                for (size_t i = 0; i < inliers_size; ++i) {
                    inliers[i] = refit_inliers[i];
                }
            }
        }
        return true;
    }

    template class essential<float>;
    template class essential<double>;
}
