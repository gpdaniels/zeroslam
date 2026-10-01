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

#include "estimation/robust/evaluate/least_median_of_squares.hpp"

#include "core/assert.hpp"
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace estimation::robust::evaluate {
    least_median_of_squares::least_median_of_squares(const size_t minimum_sample_size)
        : sample_size(minimum_sample_size) {
    }

    float least_median_of_squares::evaluate(
        const float* const __restrict residuals,
        const size_t residuals_size,
        size_t* const __restrict inliers,
        size_t& inliers_size
    ) const {
        ASSERT(this->sample_size < residuals_size, "The median needs more residuals than the sample size.");
        // A non-finite residual orders as infinitely large, so a median is never NaN and a model with one can still be replaced.
        const auto squared = [residuals](const size_t index) -> float {
            const float value = residuals[index] * residuals[index];
            return math::isnan(value) ? math::inf<float>() : value;
        };
        const auto squared_less = [&squared](const size_t lhs, const size_t rhs) -> bool {
            return squared(lhs) < squared(rhs);
        };
        // The inlier indices, which hold residuals_size entries, are ordered while the median is found, so nothing is allocated.
        for (size_t i = 0; i < residuals_size; ++i) {
            inliers[i] = i;
        }
        size_t* const middle = inliers + (residuals_size / 2);
        std::nth_element(inliers, middle, inliers + residuals_size, squared_less);
        float median_squared = squared(*middle);
        if ((residuals_size % 2) == 0) {
            median_squared = 0.5f * (median_squared + squared(*std::max_element(inliers, middle, squared_less)));
        }
        // Rousseeuw's robust scale estimate with the small sample correction, at two and a half standard deviations.
        const float threshold = 2.5f * 1.4826f * (1.0f + (5.0f / static_cast<float>(residuals_size - this->sample_size))) * math::sqrt(median_squared);
        const float threshold_squared = threshold * threshold;
        inliers_size = 0;
        for (size_t i = 0; i < residuals_size; ++i) {
            if ((residuals[i] * residuals[i]) < threshold_squared) {
                inliers[inliers_size++] = i;
            }
        }
        return median_squared;
    }
}
