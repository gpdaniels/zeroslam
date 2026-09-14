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

#include "estimation/robust/estimate/similarity.hpp"

#include "estimation/minimal/similarity_3_point.hpp"
#include "math/math.hpp"

namespace estimation::robust::estimate {
    template <typename type>
    size_t similarity<type>::generate_models(
        const correspondence_3d_3d<type>* const __restrict data,
        const size_t data_size,
        model* const __restrict models
    ) const {
        if (data_size < 3) {
            return 0;
        }
        const type lhs[3][3] = {
            { data[0].lhs[0], data[0].lhs[1], data[0].lhs[2] },
            { data[1].lhs[0], data[1].lhs[1], data[1].lhs[2] },
            { data[2].lhs[0], data[2].lhs[1], data[2].lhs[2] },
        };
        const type rhs[3][3] = {
            { data[0].rhs[0], data[0].rhs[1], data[0].rhs[2] },
            { data[1].rhs[0], data[1].rhs[1], data[1].rhs[2] },
            { data[2].rhs[0], data[2].rhs[1], data[2].rhs[2] },
        };
        type rotation[9];
        if (!minimal::similarity_3_point<type>::solve(&lhs[0][0], &rhs[0][0], 3, rotation, models[0].translation, models[0].scale)) {
            return 0;
        }
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                models[0].rotation[row][column] = rotation[(row * 3) + column];
            }
        }
        return 1;
    }

    template <typename type>
    void similarity<type>::compute_residuals(
        const correspondence_3d_3d<type>* const __restrict data,
        const size_t data_size,
        const model& candidate,
        float* const __restrict residuals
    ) const {
        for (size_t i = 0; i < data_size; ++i) {
            type distance_squared = 0;
            for (size_t row = 0; row < 3; ++row) {
                const type mapped = (candidate.scale * ((candidate.rotation[row][0] * data[i].lhs[0]) + (candidate.rotation[row][1] * data[i].lhs[1]) + (candidate.rotation[row][2] * data[i].lhs[2]))) + candidate.translation[row];
                const type difference = data[i].rhs[row] - mapped;
                distance_squared += difference * difference;
            }
            if (math::isnan(distance_squared)) {
                residuals[i] = math::inf<float>();
            }
            else {
                residuals[i] = static_cast<float>(math::sqrt(distance_squared));
            }
        }
    }

    template class similarity<float>;
    template class similarity<double>;
}
