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

#include "estimation/robust/estimate/essential.hpp"

#include "estimation/minimal/essential_5_point.hpp"
#include "geometry/sampson.hpp"

namespace estimation::robust::estimate {
    template <typename type>
    size_t essential<type>::generate_models(
        const correspondence_2d_2d<type>* const __restrict data,
        const size_t data_size,
        model* const __restrict models
    ) const {
        if (data_size < 5) {
            return 0;
        }
        const type lhs[5 * 2]{
            data[0].lhs[0],
            data[0].lhs[1],
            data[1].lhs[0],
            data[1].lhs[1],
            data[2].lhs[0],
            data[2].lhs[1],
            data[3].lhs[0],
            data[3].lhs[1],
            data[4].lhs[0],
            data[4].lhs[1],
        };
        const type rhs[5 * 2]{
            data[0].rhs[0],
            data[0].rhs[1],
            data[1].rhs[0],
            data[1].rhs[1],
            data[2].rhs[0],
            data[2].rhs[1],
            data[3].rhs[0],
            data[3].rhs[1],
            data[4].rhs[0],
            data[4].rhs[1],
        };
        type essentials[10][3][3];
        const size_t generated_models_count = static_cast<size_t>(minimal::essential_5_point<type>::solve(&lhs[0], &rhs[0], &essentials[0][0][0]));
        for (size_t i = 0; i < generated_models_count; ++i) {
            for (size_t y = 0; y < 3; ++y) {
                for (size_t x = 0; x < 3; ++x) {
                    models[i].essential[y][x] = essentials[i][y][x];
                }
            }
        }
        return generated_models_count;
    }

    template <typename type>
    void essential<type>::compute_residuals(
        const correspondence_2d_2d<type>* const __restrict data,
        const size_t data_size,
        const model& candidate,
        float* const __restrict residuals
    ) const {
        const math::matrix<type, 3, 3> epipolar(&candidate.essential[0][0]);
        for (size_t i = 0; i < data_size; ++i) {
            residuals[i] = static_cast<float>(geometry::sampson<type>::distance_squared(epipolar, data[i].lhs, data[i].rhs));
        }
    }

    template class essential<float>;
    template class essential<double>;
}
