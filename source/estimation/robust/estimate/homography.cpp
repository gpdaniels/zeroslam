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

#include "estimation/robust/estimate/homography.hpp"

#include "estimation/minimal/homography_4_point.hpp"
#include "math/math.hpp"

namespace estimation::robust::estimate {
    template <typename type>
    size_t homography<type>::generate_models(
        const correspondence_2d_2d<type>* const __restrict data,
        const size_t data_size,
        model* const __restrict models
    ) const {
        if (data_size < 4) {
            return 0;
        }
        const type lhs[4 * 2]{
            data[0].lhs[0],
            data[0].lhs[1],
            data[1].lhs[0],
            data[1].lhs[1],
            data[2].lhs[0],
            data[2].lhs[1],
            data[3].lhs[0],
            data[3].lhs[1],
        };
        const type rhs[4 * 2]{
            data[0].rhs[0],
            data[0].rhs[1],
            data[1].rhs[0],
            data[1].rhs[1],
            data[2].rhs[0],
            data[2].rhs[1],
            data[3].rhs[0],
            data[3].rhs[1],
        };
        type homography_matrix[3][3];
        if (!minimal::homography_4_point<type>::solve(&lhs[0], &rhs[0], &homography_matrix[0][0])) {
            return 0;
        }
        for (size_t y = 0; y < 3; ++y) {
            for (size_t x = 0; x < 3; ++x) {
                models[0].homography[y][x] = homography_matrix[y][x];
            }
        }
        return 1;
    }

    template <typename type>
    void homography<type>::compute_residuals(
        const correspondence_2d_2d<type>* const __restrict data,
        const size_t data_size,
        const model& candidate,
        float* const __restrict residuals
    ) const {
        const type* const h = &candidate.homography[0][0];
        const type determinant =
            h[0] * (h[4] * h[8] - h[5] * h[7]) -
            h[1] * (h[3] * h[8] - h[5] * h[6]) +
            h[2] * (h[3] * h[7] - h[4] * h[6]);

        if (math::abs(determinant) < type(1.0e-15)) {
            for (size_t i = 0; i < data_size; ++i) {
                residuals[i] = static_cast<float>(math::inf<float>());
            }
            return;
        }

        const type determinant_inverse = type(1.0) / determinant;
        const type homography_inverse[9] = {
            (h[4] * h[8] - h[5] * h[7]) * determinant_inverse,
            (h[2] * h[7] - h[1] * h[8]) * determinant_inverse,
            (h[1] * h[5] - h[2] * h[4]) * determinant_inverse,
            (h[5] * h[6] - h[3] * h[8]) * determinant_inverse,
            (h[0] * h[8] - h[2] * h[6]) * determinant_inverse,
            (h[2] * h[3] - h[0] * h[5]) * determinant_inverse,
            (h[3] * h[7] - h[4] * h[6]) * determinant_inverse,
            (h[1] * h[6] - h[0] * h[7]) * determinant_inverse,
            (h[0] * h[4] - h[1] * h[3]) * determinant_inverse
        };

        for (size_t i = 0; i < data_size; ++i) {
            const type lhs_x = data[i].lhs[0];
            const type lhs_y = data[i].lhs[1];
            const type rhs_x = data[i].rhs[0];
            const type rhs_y = data[i].rhs[1];

            const type forward_x = h[0] * rhs_x + h[1] * rhs_y + h[2];
            const type forward_y = h[3] * rhs_x + h[4] * rhs_y + h[5];
            const type forward_w = h[6] * rhs_x + h[7] * rhs_y + h[8];

            const type backward_x = homography_inverse[0] * lhs_x + homography_inverse[1] * lhs_y + homography_inverse[2];
            const type backward_y = homography_inverse[3] * lhs_x + homography_inverse[4] * lhs_y + homography_inverse[5];
            const type backward_w = homography_inverse[6] * lhs_x + homography_inverse[7] * lhs_y + homography_inverse[8];

            if ((math::abs(forward_w) < type(1.0e-12)) || (math::abs(backward_w) < type(1.0e-12))) {
                residuals[i] = static_cast<float>(math::inf<float>());
                continue;
            }

            const type forward_error_x = (forward_x / forward_w) - lhs_x;
            const type forward_error_y = (forward_y / forward_w) - lhs_y;
            const type backward_error_x = (backward_x / backward_w) - rhs_x;
            const type backward_error_y = (backward_y / backward_w) - rhs_y;

            const type symmetric_transfer_error =
                forward_error_x * forward_error_x + forward_error_y * forward_error_y +
                backward_error_x * backward_error_x + backward_error_y * backward_error_y;

            if (math::isnan(symmetric_transfer_error)) {
                residuals[i] = static_cast<float>(math::inf<float>());
            }
            else {
                residuals[i] = static_cast<float>(symmetric_transfer_error);
            }
        }
    }

    template class homography<float>;
    template class homography<double>;
}
