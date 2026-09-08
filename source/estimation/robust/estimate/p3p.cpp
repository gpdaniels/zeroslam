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

#include "estimation/robust/estimate/p3p.hpp"

#include "estimation/minimal/perspective_3_point.hpp"
#include "math/math.hpp"

namespace estimation::robust::estimate {
    template <typename type>
    size_t p3p<type>::generate_models(
        const correspondence_2d_3d<type>* const __restrict data,
        const size_t data_size,
        model* const __restrict models
    ) const {
        if (data_size < 3) {
            return 0;
        }

        type lhs_normalized[3][3];
        for (int i = 0; i < 3; ++i) {
            const type x = data[i].lhs[0];
            const type y = data[i].lhs[1];
            const type z = 1.0;
            const type norm = math::sqrt(x * x + y * y + z * z);
            lhs_normalized[i][0] = x / norm;
            lhs_normalized[i][1] = y / norm;
            lhs_normalized[i][2] = z / norm;
        }

        const type rhs[3][3] = {
            { data[0].rhs[0], data[0].rhs[1], data[0].rhs[2] },
            { data[1].rhs[0], data[1].rhs[1], data[1].rhs[2] },
            { data[2].rhs[0], data[2].rhs[1], data[2].rhs[2] },
        };

        type rotations[4][9];
        type translations[4][3];
        const size_t generated_models_count = static_cast<size_t>(minimal::perspective_3_point<type>::solve(lhs_normalized, rhs, rotations, translations));
        for (size_t i = 0; i < generated_models_count; ++i) {
            for (size_t y = 0; y < 3; ++y) {
                for (size_t x = 0; x < 3; ++x) {
                    models[i].rotation[y][x] = rotations[i][y * 3 + x];
                }
                models[i].translation[y] = translations[i][y];
            }
        }
        return generated_models_count;
    }

    template <typename type>
    void p3p<type>::compute_residuals(
        const correspondence_2d_3d<type>* const __restrict data,
        const size_t data_size,
        const model& candidate,
        float* const __restrict residuals
    ) const {
        constexpr static const auto matrix_multiply = [](const type* lhs, int lhs_width, int lhs_height, const type* rhs, int rhs_width, int rhs_height, type* result) {
            static_cast<void>(rhs_height);
            for (int lhs_y = 0; lhs_y < lhs_height; ++lhs_y) {
                for (int rhs_x = 0; rhs_x < rhs_width; ++rhs_x) {
                    type sum = 0;
                    for (int lhs_x_rhs_y = 0; lhs_x_rhs_y < lhs_width; ++lhs_x_rhs_y) {
                        sum += lhs[lhs_y * lhs_width + lhs_x_rhs_y] * rhs[lhs_x_rhs_y * rhs_width + rhs_x];
                    }
                    result[lhs_y * rhs_width + rhs_x] = sum;
                }
            }
        };

        for (size_t i = 0; i < data_size; ++i) {
            const type x = data[i].lhs[0];
            const type y = data[i].lhs[1];
            const type z = 1.0;
            const type lhs_point_norm = math::sqrt(x * x + y * y + z * z);
            const type lhs_point_normalized[3] = {
                x / lhs_point_norm,
                y / lhs_point_norm,
                z / lhs_point_norm
            };

            const type rhs_point[3] = {
                data[i].rhs[0],
                data[i].rhs[1],
                data[i].rhs[2]
            };

            type rhs_point_transformed[3];
            matrix_multiply(&candidate.rotation[0][0], 3, 3, &rhs_point[0], 1, 3, &rhs_point_transformed[0]);
            rhs_point_transformed[0] += candidate.translation[0];
            rhs_point_transformed[1] += candidate.translation[1];
            rhs_point_transformed[2] += candidate.translation[2];

            const type rhs_point_transformed_norm = math::sqrt(rhs_point_transformed[0] * rhs_point_transformed[0] + rhs_point_transformed[1] * rhs_point_transformed[1] + rhs_point_transformed[2] * rhs_point_transformed[2]);
            const type rhs_point_transformed_normalized[3] = {
                rhs_point_transformed[0] / rhs_point_transformed_norm,
                rhs_point_transformed[1] / rhs_point_transformed_norm,
                rhs_point_transformed[2] / rhs_point_transformed_norm
            };

            const type dot = rhs_point_transformed_normalized[0] * lhs_point_normalized[0] + rhs_point_transformed_normalized[1] * lhs_point_normalized[1] + rhs_point_transformed_normalized[2] * lhs_point_normalized[2];
            if (math::isnan(dot)) {
                residuals[i] = math::inf<float>();
            }
            else {
                residuals[i] = static_cast<float>(type(1) - dot);
            }
        }
    }

    template class p3p<float>;
    template class p3p<double>;
}
