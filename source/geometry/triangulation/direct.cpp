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

#include "geometry/triangulation/direct.hpp"

#include "math/math.hpp"

namespace geometry::triangulation {
    template <typename type>
    bool direct<type>::triangulate(
        const math::matrix<type, 3, 1>& lhs_ray,
        const math::matrix<type, 3, 4>& lhs_pose,
        const math::matrix<type, 3, 1>& rhs_ray,
        const math::matrix<type, 3, 4>& rhs_pose,
        math::matrix<type, 3, 1>& result
    ) {
        // The depth along the left ray follows from one row of rhs_ray x (R lhs_ray depth + t) = 0, choosing the better conditioned row.
        const math::matrix<type, 3, 3> lhs_rotation = math::get_block<type, 3, 3>(lhs_pose, 0, 0);
        const math::matrix<type, 3, 1> lhs_translation = math::get_block<type, 3, 1>(lhs_pose, 0, 3);
        const math::matrix<type, 3, 3> rhs_rotation = math::get_block<type, 3, 3>(rhs_pose, 0, 0);
        const math::matrix<type, 3, 1> rhs_translation = math::get_block<type, 3, 1>(rhs_pose, 0, 3);
        const math::matrix<type, 3, 3> lhs_rotation_transpose = math::transpose(lhs_rotation);
        const math::matrix<type, 3, 3> rotation = rhs_rotation * lhs_rotation_transpose;
        const math::matrix<type, 3, 1> translation = rhs_translation - (rotation * lhs_translation);

        const math::matrix<type, 3, 1> rotated = rotation * lhs_ray;
        const type scale = math::sqrt(rotated.get_length_squared() * rhs_ray.get_length_squared());
        if (!(scale > type(0))) {
            return false;
        }
        const type denominator_y = (rotated[1] * rhs_ray[2]) - (rotated[2] * rhs_ray[1]);
        const type denominator_x = (rotated[0] * rhs_ray[2]) - (rotated[2] * rhs_ray[0]);

        type depth;
        if (math::abs(denominator_y) > math::abs(denominator_x)) {
            if (math::abs(denominator_y) <= (direct::tolerance * scale)) {
                return false;
            }
            depth = ((translation[2] * rhs_ray[1]) - (translation[1] * rhs_ray[2])) / denominator_y;
        }
        else {
            if (math::abs(denominator_x) <= (direct::tolerance * scale)) {
                return false;
            }
            depth = ((translation[2] * rhs_ray[0]) - (translation[0] * rhs_ray[2])) / denominator_x;
        }

        const math::matrix<type, 3, 1> lhs_point{ { lhs_ray[0] * depth, lhs_ray[1] * depth, lhs_ray[2] * depth } };
        result = lhs_rotation_transpose * (lhs_point - lhs_translation);
        return true;
    }

    template <typename type>
    bool direct<type>::triangulate(
        const math::matrix<type, 2, 1>& lhs_point_normalised,
        const math::matrix<type, 3, 4>& lhs_pose,
        const math::matrix<type, 2, 1>& rhs_point_normalised,
        const math::matrix<type, 3, 4>& rhs_pose,
        math::matrix<type, 3, 1>& result
    ) {
        return direct::triangulate(
            math::matrix<type, 3, 1>{ { lhs_point_normalised[0], lhs_point_normalised[1], type(1) } },
            lhs_pose,
            math::matrix<type, 3, 1>{ { rhs_point_normalised[0], rhs_point_normalised[1], type(1) } },
            rhs_pose,
            result
        );
    }

    template class direct<float>;
    template class direct<double>;
}
