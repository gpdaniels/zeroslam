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

#include <vector>

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
        std::vector<float> residuals_squared(residuals_size);
        for (size_t i = 0; i < residuals_size; ++i) {
            residuals_squared[i] = residuals[i] * residuals[i];
        }
        float median_squared = least_median_of_squares::nth_element(residuals_squared.data(), residuals_size, residuals_size / 2);
        if ((residuals_size % 2) == 0) {
            median_squared = 0.5f * (median_squared + least_median_of_squares::nth_element(residuals_squared.data(), residuals_size, (residuals_size / 2) - 1));
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

    float least_median_of_squares::nth_element(float* const __restrict data, const size_t data_size, const size_t n) {
        // Quickselect on the values that are at least the pivot moving left, so the nth largest ends at index n.
        size_t start = 0;
        size_t end = data_size - 1;
        for (;;) {
            if (end == start) {
                return data[start];
            }
            size_t store = start;
            const float pivot = data[end];
            for (size_t i = start; i < end; ++i) {
                if (data[i] >= pivot) {
                    const float temp = data[store];
                    data[store] = data[i];
                    data[i] = temp;
                    ++store;
                }
            }
            data[end] = data[store];
            data[store] = pivot;
            if (n == store) {
                return data[store];
            }
            if (n < store) {
                end = store - 1;
            }
            else {
                start = store + 1;
            }
        }
    }
}
