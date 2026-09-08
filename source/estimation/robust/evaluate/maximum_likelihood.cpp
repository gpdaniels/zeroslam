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

#include "estimation/robust/evaluate/maximum_likelihood.hpp"

namespace estimation::robust::evaluate {
    maximum_likelihood::maximum_likelihood(const float residual_threshold)
        : threshold(residual_threshold) {
    }

    float maximum_likelihood::evaluate(
        const float* const __restrict residuals,
        const size_t residuals_size,
        size_t* const __restrict inliers,
        size_t& inliers_size
    ) const {
        inliers_size = 0;
        float cost = 0.0f;
        for (size_t i = 0; i < residuals_size; ++i) {
            if (residuals[i] < this->threshold) {
                inliers[inliers_size++] = i;
                cost += residuals[i];
            }
            else {
                cost += this->threshold;
            }
        }
        return cost;
    }
}
