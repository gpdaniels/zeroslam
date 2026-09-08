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

#include "geometry/triangulation/iteratively_reweighted_linear_least_squares.hpp"

#include "math/math.hpp"
#include "math/matrix_decomposition_singular_value.hpp"

namespace geometry::triangulation {
    template <typename type>
    bool iteratively_reweighted_linear_least_squares<type>::triangulate(
        const math::matrix<type, 3, 1>& lhs_ray,
        const math::matrix<type, 3, 4>& lhs_pose,
        const math::matrix<type, 3, 1>& rhs_ray,
        const math::matrix<type, 3, 4>& rhs_pose,
        math::matrix<type, 3, 1>& result
    ) {
        type matrix_rows[4][4];
        if (!iteratively_reweighted_linear_least_squares::rows(lhs_ray, lhs_pose, &matrix_rows[0][0], &matrix_rows[1][0]) || !iteratively_reweighted_linear_least_squares::rows(rhs_ray, rhs_pose, &matrix_rows[2][0], &matrix_rows[3][0])) {
            return false;
        }
        const type lhs_ray_length = math::sqrt(lhs_ray.get_length_squared());
        const type rhs_ray_length = math::sqrt(rhs_ray.get_length_squared());

        // Each pass divides a view's rows by the depth of the previous solution along its ray, so the algebraic residual approaches the angular error.
        type lhs_weight = type(1);
        type rhs_weight = type(1);

        for (unsigned int iteration = 0; iteration < iteratively_reweighted_linear_least_squares::maximum_iterations; ++iteration) {
            type matrix_a[4][4];
            for (size_t i = 0; i < 4; ++i) {
                matrix_a[0][i] = matrix_rows[0][i] / lhs_weight;
                matrix_a[1][i] = matrix_rows[1][i] / lhs_weight;
                matrix_a[2][i] = matrix_rows[2][i] / rhs_weight;
                matrix_a[3][i] = matrix_rows[3][i] / rhs_weight;
            }

            type matrix_u[4][4];
            type matrix_s[4][4];
            type matrix_vt[4][4];
            if (!math::decompose_singular_value(&matrix_a[0][0], 4, 4, &matrix_u[0][0], &matrix_s[0][0], &matrix_vt[0][0])) {
                return false;
            }

            const type* point_homography = &matrix_vt[3][0];

            if (math::abs(point_homography[3]) < iteratively_reweighted_linear_least_squares::tolerance) {
                return false;
            }

            if (math::abs(matrix_s[2][2] - matrix_s[3][3]) < iteratively_reweighted_linear_least_squares::tolerance * matrix_s[0][0]) {
                const type* point_homography_alternate = &matrix_vt[2][0];
                const type point_difference =
                    math::abs(point_homography[0] - point_homography_alternate[0]) +
                    math::abs(point_homography[1] - point_homography_alternate[1]) +
                    math::abs(point_homography[2] - point_homography_alternate[2]) +
                    math::abs(point_homography[3] - point_homography_alternate[3]);
                if (point_difference > iteratively_reweighted_linear_least_squares::tolerance) {
                    return false;
                }
            }

            result[0] = point_homography[0] / point_homography[3];
            result[1] = point_homography[1] / point_homography[3];
            result[2] = point_homography[2] / point_homography[3];

            type lhs_camera[3];
            type rhs_camera[3];
            for (size_t row = 0; row < 3; ++row) {
                lhs_camera[row] = (lhs_pose[row][0] * result[0]) + (lhs_pose[row][1] * result[1]) + (lhs_pose[row][2] * result[2]) + lhs_pose[row][3];
                rhs_camera[row] = (rhs_pose[row][0] * result[0]) + (rhs_pose[row][1] * result[1]) + (rhs_pose[row][2] * result[2]) + rhs_pose[row][3];
            }
            const type lhs_weight_new = ((lhs_ray[0] * lhs_camera[0]) + (lhs_ray[1] * lhs_camera[1]) + (lhs_ray[2] * lhs_camera[2])) / lhs_ray_length;
            const type rhs_weight_new = ((rhs_ray[0] * rhs_camera[0]) + (rhs_ray[1] * rhs_camera[1]) + (rhs_ray[2] * rhs_camera[2])) / rhs_ray_length;
            if ((math::abs(lhs_weight_new) < iteratively_reweighted_linear_least_squares::tolerance) || (math::abs(rhs_weight_new) < iteratively_reweighted_linear_least_squares::tolerance)) {
                return false;
            }

            if ((math::abs(lhs_weight - lhs_weight_new) < iteratively_reweighted_linear_least_squares::minimum_weight_change) && (math::abs(rhs_weight - rhs_weight_new) < iteratively_reweighted_linear_least_squares::minimum_weight_change)) {
                return true;
            }

            lhs_weight = lhs_weight_new;
            rhs_weight = rhs_weight_new;
        }

        return true;
    }

    template <typename type>
    bool iteratively_reweighted_linear_least_squares<type>::triangulate(
        const math::matrix<type, 2, 1>& lhs_point_normalised,
        const math::matrix<type, 3, 4>& lhs_pose,
        const math::matrix<type, 2, 1>& rhs_point_normalised,
        const math::matrix<type, 3, 4>& rhs_pose,
        math::matrix<type, 3, 1>& result
    ) {
        return iteratively_reweighted_linear_least_squares::triangulate(
            math::matrix<type, 3, 1>{ { lhs_point_normalised[0], lhs_point_normalised[1], type(1) } },
            lhs_pose,
            math::matrix<type, 3, 1>{ { rhs_point_normalised[0], rhs_point_normalised[1], type(1) } },
            rhs_pose,
            result
        );
    }

    template <typename type>
    bool iteratively_reweighted_linear_least_squares<type>::rows(
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

    template class iteratively_reweighted_linear_least_squares<float>;
    template class iteratively_reweighted_linear_least_squares<double>;
}
