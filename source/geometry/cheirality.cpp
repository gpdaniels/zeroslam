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

#include "geometry/cheirality.hpp"

namespace geometry {
    template <typename type>
    bool cheirality<type>::in_front(
        const math::matrix<type, 3, 4>& pose,
        const math::matrix<type, 3, 1>& ray,
        const math::matrix<type, 3, 1>& point
    ) {
        type projection = type(0);
        for (size_t row = 0; row < 3; ++row) {
            const type camera = (pose[row][0] * point[0]) + (pose[row][1] * point[1]) + (pose[row][2] * point[2]) + pose[row][3];
            projection += ray[row] * camera;
        }
        return projection > type(0);
    }

    template class cheirality<float>;
    template class cheirality<double>;
}
