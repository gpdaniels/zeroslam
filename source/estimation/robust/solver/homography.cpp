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

#include "estimation/robust/solver/homography.hpp"

#include "estimation/robust/consensus.hpp"
#include "estimation/robust/evaluate/maximum_likelihood.hpp"
#include "estimation/robust/sample/random.hpp"
#include "math/math.hpp"
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
    bool homography<type>::solve(
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
        const float residual_threshold = 2.0e-5f;

        sample::random<4> sampler;
        estimate::homography<type> estimator;
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

        if (inliers_size >= 4) {
            constexpr static const auto matrix_multiply = [](const type* lhs, const type* rhs, type* result) {
                for (size_t y = 0; y < 3; ++y) {
                    for (size_t x = 0; x < 3; ++x) {
                        type sum = 0;
                        for (size_t k = 0; k < 3; ++k) {
                            sum += lhs[(y * 3) + k] * rhs[(k * 3) + x];
                        }
                        result[(y * 3) + x] = sum;
                    }
                }
            };

            type centroid_lhs_x = 0;
            type centroid_lhs_y = 0;
            type centroid_rhs_x = 0;
            type centroid_rhs_y = 0;
            for (size_t i = 0; i < inliers_size; ++i) {
                centroid_lhs_x += correspondences[inliers[i]].lhs[0];
                centroid_lhs_y += correspondences[inliers[i]].lhs[1];
                centroid_rhs_x += correspondences[inliers[i]].rhs[0];
                centroid_rhs_y += correspondences[inliers[i]].rhs[1];
            }
            centroid_lhs_x /= type(inliers_size);
            centroid_lhs_y /= type(inliers_size);
            centroid_rhs_x /= type(inliers_size);
            centroid_rhs_y /= type(inliers_size);

            type mean_distance_lhs = 0;
            type mean_distance_rhs = 0;
            for (size_t i = 0; i < inliers_size; ++i) {
                const type dx_lhs = correspondences[inliers[i]].lhs[0] - centroid_lhs_x;
                const type dy_lhs = correspondences[inliers[i]].lhs[1] - centroid_lhs_y;
                const type dx_rhs = correspondences[inliers[i]].rhs[0] - centroid_rhs_x;
                const type dy_rhs = correspondences[inliers[i]].rhs[1] - centroid_rhs_y;
                mean_distance_lhs += math::sqrt(dx_lhs * dx_lhs + dy_lhs * dy_lhs);
                mean_distance_rhs += math::sqrt(dx_rhs * dx_rhs + dy_rhs * dy_rhs);
            }
            mean_distance_lhs /= type(inliers_size);
            mean_distance_rhs /= type(inliers_size);

            if ((mean_distance_lhs < type(1.0e-12)) || (mean_distance_rhs < type(1.0e-12))) {
                return true;
            }

            const type scale_lhs = math::sqrt(type(2.0)) / mean_distance_lhs;
            const type scale_rhs = math::sqrt(type(2.0)) / mean_distance_rhs;

            std::vector<type> lhs_normalized(inliers_size * 2);
            std::vector<type> rhs_normalized(inliers_size * 2);
            for (size_t i = 0; i < inliers_size; ++i) {
                lhs_normalized[(i * 2) + 0] = scale_lhs * (correspondences[inliers[i]].lhs[0] - centroid_lhs_x);
                lhs_normalized[(i * 2) + 1] = scale_lhs * (correspondences[inliers[i]].lhs[1] - centroid_lhs_y);
                rhs_normalized[(i * 2) + 0] = scale_rhs * (correspondences[inliers[i]].rhs[0] - centroid_rhs_x);
                rhs_normalized[(i * 2) + 1] = scale_rhs * (correspondences[inliers[i]].rhs[1] - centroid_rhs_y);
            }

            std::vector<type> constraint(inliers_size * 18);
            for (size_t i = 0; i < inliers_size; ++i) {
                const type u = rhs_normalized[(i * 2) + 0];
                const type v = rhs_normalized[(i * 2) + 1];
                const type up = lhs_normalized[(i * 2) + 0];
                const type vp = lhs_normalized[(i * 2) + 1];
                constraint[(i * 18) + 0] = 0;
                constraint[(i * 18) + 1] = 0;
                constraint[(i * 18) + 2] = 0;
                constraint[(i * 18) + 3] = -u;
                constraint[(i * 18) + 4] = -v;
                constraint[(i * 18) + 5] = -1;
                constraint[(i * 18) + 6] = vp * u;
                constraint[(i * 18) + 7] = vp * v;
                constraint[(i * 18) + 8] = vp;
                constraint[(i * 18) + 9] = u;
                constraint[(i * 18) + 10] = v;
                constraint[(i * 18) + 11] = 1;
                constraint[(i * 18) + 12] = 0;
                constraint[(i * 18) + 13] = 0;
                constraint[(i * 18) + 14] = 0;
                constraint[(i * 18) + 15] = -up * u;
                constraint[(i * 18) + 16] = -up * v;
                constraint[(i * 18) + 17] = -up;
            }

            type block[9][9];
            for (size_t y = 0; y < 9; ++y) {
                for (size_t x = 0; x < 9; ++x) {
                    type sum = 0;
                    for (size_t i = 0; i < inliers_size * 2; ++i) {
                        sum += constraint[(i * 9) + y] * constraint[(i * 9) + x];
                    }
                    block[y][x] = sum;
                }
            }
            type u_matrix[9][9];
            type s_matrix[9][9];
            type vt_matrix[9][9];
            if (!math::decompose_singular_value(&block[0][0], 9, 9, &u_matrix[0][0], &s_matrix[0][0], &vt_matrix[0][0])) {
                return true;
            }
            if (s_matrix[7][7] < type(1.0e-9) * s_matrix[0][0]) {
                return true;
            }

            type homography_normalized[9];
            for (size_t i = 0; i < 9; ++i) {
                homography_normalized[i] = vt_matrix[8][i];
            }

            const type transform_rhs[9] = {
                scale_rhs,
                0,
                -scale_rhs * centroid_rhs_x,
                0,
                scale_rhs,
                -scale_rhs * centroid_rhs_y,
                0,
                0,
                1
            };
            const type transform_lhs_inverse[9] = {
                type(1.0) / scale_lhs,
                0,
                centroid_lhs_x,
                0,
                type(1.0) / scale_lhs,
                centroid_lhs_y,
                0,
                0,
                1
            };
            type homography_temp[9];
            matrix_multiply(&homography_normalized[0], &transform_rhs[0], homography_temp);
            type homography_denormalized[9];
            matrix_multiply(&transform_lhs_inverse[0], homography_temp, homography_denormalized);

            type frobenius_squared = 0;
            for (size_t i = 0; i < 9; ++i) {
                frobenius_squared += homography_denormalized[i] * homography_denormalized[i];
            }
            if (frobenius_squared < type(1.0e-30)) {
                return true;
            }
            bool valid = true;
            const type frobenius_inverse = type(1.0) / math::sqrt(frobenius_squared);
            for (size_t i = 0; i < 9; ++i) {
                homography_denormalized[i] *= frobenius_inverse;
                if (math::isnan(homography_denormalized[i]) || math::isinf(homography_denormalized[i])) {
                    valid = false;
                }
            }
            if (!valid) {
                return true;
            }

            model_type refit_model;
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    refit_model.homography[row][col] = homography_denormalized[(row * 3) + col];
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
        }
        return true;
    }

    template class homography<float>;
    template class homography<double>;
}
