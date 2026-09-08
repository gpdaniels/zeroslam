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

#include "geometry/triangulation/cost_function.hpp"

#include "math/math.hpp"

namespace geometry::triangulation {
    template <typename type>
    bool cost_function<type>::triangulate(
        const math::matrix<type, 3, 1>& lhs_ray,
        const math::matrix<type, 3, 4>& lhs_pose,
        const math::matrix<type, 3, 1>& rhs_ray,
        const math::matrix<type, 3, 4>& rhs_pose,
        math::matrix<type, 3, 1>& result
    ) {
        // Minimises the squared perpendicular distances to both rays, X^T W1 X + (R X + t)^T W2 (R X + t) with Wi = I - ri ri^T for unit rays in camera 1
        // coordinates, whose closed form is X = -(W1 + R^T W2 R)^-1 R^T W2 t.
        const type lhs_ray_length_squared = lhs_ray.get_length_squared();
        const type rhs_ray_length_squared = rhs_ray.get_length_squared();
        if (!((lhs_ray_length_squared > type(0)) && (rhs_ray_length_squared > type(0)))) {
            return false;
        }

        const math::matrix<type, 3, 3> lhs_rotation = math::get_block<type, 3, 3>(lhs_pose, 0, 0);
        const math::matrix<type, 3, 1> lhs_translation = math::get_block<type, 3, 1>(lhs_pose, 0, 3);
        const math::matrix<type, 3, 3> rhs_rotation = math::get_block<type, 3, 3>(rhs_pose, 0, 0);
        const math::matrix<type, 3, 1> rhs_translation = math::get_block<type, 3, 1>(rhs_pose, 0, 3);
        const math::matrix<type, 3, 3> lhs_rotation_transpose = math::transpose(lhs_rotation);
        const math::matrix<type, 3, 3> rotation = rhs_rotation * lhs_rotation_transpose;
        const math::matrix<type, 3, 1> translation = rhs_translation - (rotation * lhs_translation);

        type lhs_weight[3][3];
        type rhs_weight[3][3];
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                lhs_weight[row][col] = ((row == col) ? type(1) : type(0)) - ((lhs_ray[row] * lhs_ray[col]) / lhs_ray_length_squared);
                rhs_weight[row][col] = ((row == col) ? type(1) : type(0)) - ((rhs_ray[row] * rhs_ray[col]) / rhs_ray_length_squared);
            }
        }

        type rotation_transpose_w2[3][3];
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                rotation_transpose_w2[row][col] = (rotation[0][row] * rhs_weight[0][col]) + (rotation[1][row] * rhs_weight[1][col]) + (rotation[2][row] * rhs_weight[2][col]);
            }
        }

        type system[3][3];
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                system[row][col] = lhs_weight[row][col] + (rotation_transpose_w2[row][0] * rotation[0][col]) + (rotation_transpose_w2[row][1] * rotation[1][col]) + (rotation_transpose_w2[row][2] * rotation[2][col]);
            }
        }

        const type determinant =
            (system[0][0] * ((system[1][1] * system[2][2]) - (system[1][2] * system[2][1]))) -
            (system[0][1] * ((system[1][0] * system[2][2]) - (system[1][2] * system[2][0]))) +
            (system[0][2] * ((system[1][0] * system[2][1]) - (system[1][1] * system[2][0])));
        if (math::abs(determinant) < cost_function::determinant_tolerance) {
            return false;
        }

        type right_hand_side[3];
        for (size_t row = 0; row < 3; ++row) {
            right_hand_side[row] = -((rotation_transpose_w2[row][0] * translation[0]) + (rotation_transpose_w2[row][1] * translation[1]) + (rotation_transpose_w2[row][2] * translation[2]));
        }

        type inverse[3][3];
        inverse[0][0] = ((system[1][1] * system[2][2]) - (system[1][2] * system[2][1])) / determinant;
        inverse[0][1] = ((system[0][2] * system[2][1]) - (system[0][1] * system[2][2])) / determinant;
        inverse[0][2] = ((system[0][1] * system[1][2]) - (system[0][2] * system[1][1])) / determinant;
        inverse[1][0] = ((system[1][2] * system[2][0]) - (system[1][0] * system[2][2])) / determinant;
        inverse[1][1] = ((system[0][0] * system[2][2]) - (system[0][2] * system[2][0])) / determinant;
        inverse[1][2] = ((system[0][2] * system[1][0]) - (system[0][0] * system[1][2])) / determinant;
        inverse[2][0] = ((system[1][0] * system[2][1]) - (system[1][1] * system[2][0])) / determinant;
        inverse[2][1] = ((system[0][1] * system[2][0]) - (system[0][0] * system[2][1])) / determinant;
        inverse[2][2] = ((system[0][0] * system[1][1]) - (system[0][1] * system[1][0])) / determinant;

        math::matrix<type, 3, 1> lhs_point;
        for (size_t row = 0; row < 3; ++row) {
            lhs_point[row] = (inverse[row][0] * right_hand_side[0]) + (inverse[row][1] * right_hand_side[1]) + (inverse[row][2] * right_hand_side[2]);
        }

        result = lhs_rotation_transpose * (lhs_point - lhs_translation);
        return true;
    }

    template <typename type>
    bool cost_function<type>::triangulate(
        const math::matrix<type, 2, 1>& lhs_point_normalised,
        const math::matrix<type, 3, 4>& lhs_pose,
        const math::matrix<type, 2, 1>& rhs_point_normalised,
        const math::matrix<type, 3, 4>& rhs_pose,
        math::matrix<type, 3, 1>& result
    ) {
        return cost_function::triangulate(
            math::matrix<type, 3, 1>{ { lhs_point_normalised[0], lhs_point_normalised[1], type(1) } },
            lhs_pose,
            math::matrix<type, 3, 1>{ { rhs_point_normalised[0], rhs_point_normalised[1], type(1) } },
            rhs_pose,
            result
        );
    }

    template class cost_function<float>;
    template class cost_function<double>;
}
