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

#include "estimation/robust/evaluate/inlier_support.hpp"

namespace estimation::robust::evaluate {
    inlier_support::inlier_support(const float residual_threshold)
        : threshold(residual_threshold) {
    }

    float inlier_support::evaluate(
        const float* const __restrict residuals,
        const size_t residuals_size,
        size_t* const __restrict inliers,
        size_t& inliers_size
    ) const {
        inliers_size = 0;
        for (size_t i = 0; i < residuals_size; ++i) {
            if (residuals[i] < this->threshold) {
                inliers[inliers_size++] = i;
            }
        }
        return static_cast<float>(residuals_size - inliers_size);
    }
}
