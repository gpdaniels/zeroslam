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

#include "geometry/sampson.hpp"

namespace geometry {
    template <typename type>
    type sampson<type>::distance_squared(
        const math::matrix<type, 3, 3>& epipolar,
        const math::matrix<type, 2, 1>& lhs,
        const math::matrix<type, 2, 1>& rhs
    ) {
        const math::matrix<type, 3, 1> lhs_point{ { lhs[0], lhs[1], type(1) } };
        const math::matrix<type, 3, 1> rhs_point{ { rhs[0], rhs[1], type(1) } };
        const math::matrix<type, 3, 1> epipolar_lhs = epipolar * lhs_point;
        const math::matrix<type, 3, 1> epipolar_rhs = math::transpose(epipolar) * rhs_point;
        const type constraint = (rhs_point[0] * epipolar_lhs[0]) + (rhs_point[1] * epipolar_lhs[1]) + (rhs_point[2] * epipolar_lhs[2]);
        const type gradient_squared = (epipolar_lhs[0] * epipolar_lhs[0]) + (epipolar_lhs[1] * epipolar_lhs[1]) + (epipolar_rhs[0] * epipolar_rhs[0]) + (epipolar_rhs[1] * epipolar_rhs[1]);
        return (constraint * constraint) / gradient_squared;
    }

    template class sampson<float>;
    template class sampson<double>;
}
