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

#include "geometry/triangulation/linear_least_squares.hpp"

#include "math/math.hpp"
#include "math/matrix_decomposition_singular_value.hpp"

namespace geometry::triangulation {
    template <typename type>
    bool linear_least_squares<type>::triangulate(
        const math::matrix<type, 3, 1>& lhs_ray,
        const math::matrix<type, 3, 4>& lhs_pose,
        const math::matrix<type, 3, 1>& rhs_ray,
        const math::matrix<type, 3, 4>& rhs_pose,
        math::matrix<type, 3, 1>& result
    ) {
        // The system is built in the lhs camera, [I | 0] and [R | t] with R = R_rhs R_lhs^T and t = t_rhs - R t_lhs, so its conditioning does not depend on how far the cameras are from the world origin.
        const math::matrix<type, 3, 3> lhs_rotation = math::get_block<type, 3, 3>(lhs_pose, 0, 0);
        const math::matrix<type, 3, 1> lhs_translation = math::get_block<type, 3, 1>(lhs_pose, 0, 3);
        const math::matrix<type, 3, 3> lhs_rotation_transpose = math::transpose(lhs_rotation);
        const math::matrix<type, 3, 3> rotation = math::get_block<type, 3, 3>(rhs_pose, 0, 0) * lhs_rotation_transpose;
        const math::matrix<type, 3, 1> translation = math::get_block<type, 3, 1>(rhs_pose, 0, 3) - (rotation * lhs_translation);
        const math::matrix<type, 3, 4> lhs_relative_pose{ { { type(1), type(0), type(0), type(0) }, { type(0), type(1), type(0), type(0) }, { type(0), type(0), type(1), type(0) } } };
        const math::matrix<type, 3, 4> rhs_relative_pose{ { { rotation[0][0], rotation[0][1], rotation[0][2], translation[0] }, { rotation[1][0], rotation[1][1], rotation[1][2], translation[1] }, { rotation[2][0], rotation[2][1], rotation[2][2], translation[2] } } };

        type matrix_a[4][4];
        if (!linear_least_squares::rows(lhs_ray, lhs_relative_pose, &matrix_a[0][0], &matrix_a[1][0]) || !linear_least_squares::rows(rhs_ray, rhs_relative_pose, &matrix_a[2][0], &matrix_a[3][0])) {
            return false;
        }

        type matrix_u[4][4];
        type matrix_s[4][4];
        type matrix_vt[4][4];
        if (!math::decompose_singular_value(&matrix_a[0][0], 4, 4, &matrix_u[0][0], &matrix_s[0][0], &matrix_vt[0][0])) {
            return false;
        }

        const type* point_homography = &matrix_vt[3][0];

        if (math::abs(point_homography[3]) < linear_least_squares::tolerance) {
            return false;
        }

        // Two equal smallest singular values mean the null space is not unique unless both vectors give the same point.
        if (math::abs(matrix_s[2][2] - matrix_s[3][3]) < linear_least_squares::tolerance * matrix_s[0][0]) {
            const type* point_homography_alternate = &matrix_vt[2][0];
            const type point_difference =
                math::abs(point_homography[0] - point_homography_alternate[0]) +
                math::abs(point_homography[1] - point_homography_alternate[1]) +
                math::abs(point_homography[2] - point_homography_alternate[2]) +
                math::abs(point_homography[3] - point_homography_alternate[3]);
            if (point_difference > linear_least_squares::tolerance) {
                return false;
            }
        }

        const math::matrix<type, 3, 1> lhs_point{ { point_homography[0] / point_homography[3], point_homography[1] / point_homography[3], point_homography[2] / point_homography[3] } };
        result = lhs_rotation_transpose * (lhs_point - lhs_translation);

        return true;
    }

    template <typename type>
    bool linear_least_squares<type>::triangulate(
        const math::matrix<type, 2, 1>& lhs_point_normalised,
        const math::matrix<type, 3, 4>& lhs_pose,
        const math::matrix<type, 2, 1>& rhs_point_normalised,
        const math::matrix<type, 3, 4>& rhs_pose,
        math::matrix<type, 3, 1>& result
    ) {
        return linear_least_squares::triangulate(
            math::matrix<type, 3, 1>{ { lhs_point_normalised[0], lhs_point_normalised[1], type(1) } },
            lhs_pose,
            math::matrix<type, 3, 1>{ { rhs_point_normalised[0], rhs_point_normalised[1], type(1) } },
            rhs_pose,
            result
        );
    }

    template <typename type>
    bool linear_least_squares<type>::rows(
        const math::matrix<type, 3, 1>& ray,
        const math::matrix<type, 3, 4>& pose,
        type* const row_a,
        type* const row_b
    ) {
        // Two rows of ray x (P X) = 0, taken against the dominant ray component so any bearing is well conditioned.
        const type x = math::abs(ray[0]);
        const type y = math::abs(ray[1]);
        const type z = math::abs(ray[2]);
        if (!((x > type(0)) || (y > type(0)) || (z > type(0)))) {
            return false;
        }
        for (size_t i = 0; i < 4; ++i) {
            if ((z >= x) && (z >= y)) {
                row_a[i] = ray[0] * pose[2][i] - ray[2] * pose[0][i];
                row_b[i] = ray[1] * pose[2][i] - ray[2] * pose[1][i];
            }
            else if (x >= y) {
                row_a[i] = ray[1] * pose[0][i] - ray[0] * pose[1][i];
                row_b[i] = ray[2] * pose[0][i] - ray[0] * pose[2][i];
            }
            else {
                row_a[i] = ray[0] * pose[1][i] - ray[1] * pose[0][i];
                row_b[i] = ray[2] * pose[1][i] - ray[1] * pose[2][i];
            }
        }
        return true;
    }

    template class linear_least_squares<float>;
    template class linear_least_squares<double>;
}
