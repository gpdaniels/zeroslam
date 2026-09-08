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

#include "geometry/homography.hpp"

#include "math/matrix_decomposition_singular_value.hpp"

namespace geometry {
    template <typename type>
    bool homography<type>::from_plane(
        const math::matrix<type, 3, 3>& intrinsics_lhs,
        const math::matrix<type, 3, 3>& intrinsics_rhs,
        const math::matrix<type, 3, 3>& rotation,
        const math::matrix<type, 3, 1>& translation,
        const math::matrix<type, 3, 1>& plane_normal,
        const type plane_distance,
        math::matrix<type, 3, 3>& homography
    ) {
        if (!(math::abs(plane_distance) > type(0))) {
            return false;
        }
        math::matrix<type, 3, 3> intrinsics_lhs_inverse;
        if (!math::invert(intrinsics_lhs, intrinsics_lhs_inverse)) {
            return false;
        }
        math::matrix<type, 3, 3> euclidean = rotation;
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                euclidean[row][col] += (translation[row] * plane_normal[col]) / plane_distance;
            }
        }
        homography = intrinsics_rhs * euclidean * intrinsics_lhs_inverse;
        return true;
    }

    template <typename type>
    size_t homography<type>::decompose(
        const math::matrix<type, 3, 3>& intrinsics_lhs,
        const math::matrix<type, 3, 3>& intrinsics_rhs,
        const math::matrix<type, 3, 3>& homography,
        math::matrix<type, 3, 3>* const rotations,
        math::matrix<type, 3, 1>* const translations,
        math::matrix<type, 3, 1>* const normals
    ) {
        // Faugeras and Lustman via the singular values of the euclidean homography R + t n^T, scaled so the middle singular value is one.
        math::matrix<type, 3, 3> intrinsics_rhs_inverse;
        if (!math::invert(intrinsics_rhs, intrinsics_rhs_inverse)) {
            return 0;
        }
        math::matrix<type, 3, 3> euclidean = intrinsics_rhs_inverse * homography * intrinsics_lhs;

        type matrix_u[3][3];
        type matrix_s[3][3];
        type matrix_vt[3][3];
        if (!math::decompose_singular_value(euclidean.data(), 3, 3, &matrix_u[0][0], &matrix_s[0][0], &matrix_vt[0][0])) {
            return 0;
        }
        if (!(matrix_s[1][1] > type(0))) {
            return 0;
        }
        const type sigma_1 = matrix_s[0][0] / matrix_s[1][1];
        const type sigma_3 = matrix_s[2][2] / matrix_s[1][1];
        euclidean = euclidean * (type(1) / matrix_s[1][1]);

        // A homography of two cameras on the same side of the plane has a positive determinant.
        const type determinant =
            euclidean[0][0] * (euclidean[1][1] * euclidean[2][2] - euclidean[1][2] * euclidean[2][1]) -
            euclidean[0][1] * (euclidean[1][0] * euclidean[2][2] - euclidean[1][2] * euclidean[2][0]) +
            euclidean[0][2] * (euclidean[1][0] * euclidean[2][1] - euclidean[1][1] * euclidean[2][0]);
        if (determinant < type(0)) {
            euclidean = -euclidean;
        }

        math::matrix<type, 3, 1> v_1{ { matrix_vt[0][0], matrix_vt[0][1], matrix_vt[0][2] } };
        const math::matrix<type, 3, 1> v_2{ { matrix_vt[1][0], matrix_vt[1][1], matrix_vt[1][2] } };
        math::matrix<type, 3, 1> v_3{ { matrix_vt[2][0], matrix_vt[2][1], matrix_vt[2][2] } };
        const type determinant_v =
            v_1[0] * (v_2[1] * v_3[2] - v_2[2] * v_3[1]) -
            v_1[1] * (v_2[0] * v_3[2] - v_2[2] * v_3[0]) +
            v_1[2] * (v_2[0] * v_3[1] - v_2[1] * v_3[0]);
        if (determinant_v < type(0)) {
            v_3 = -v_3;
        }

        if ((sigma_1 - sigma_3) <= homography::rotation_tolerance) {
            rotations[0] = euclidean;
            translations[0] = math::matrix<type, 3, 1>::zero();
            normals[0] = math::matrix<type, 3, 1>::zero();
            return 1;
        }

        const type weight_1 = math::sqrt(math::max(type(1) - (sigma_3 * sigma_3), type(0)));
        const type weight_3 = math::sqrt(math::max((sigma_1 * sigma_1) - type(1), type(0)));
        const type scale = math::sqrt((sigma_1 * sigma_1) - (sigma_3 * sigma_3));
        const math::matrix<type, 3, 1> u_candidates[2] = {
            ((v_1 * weight_1) + (v_3 * weight_3)) * (type(1) / scale),
            ((v_1 * weight_1) - (v_3 * weight_3)) * (type(1) / scale)
        };

        for (size_t candidate = 0; candidate < 2; ++candidate) {
            const math::matrix<type, 3, 1>& u = u_candidates[candidate];
            const math::matrix<type, 3, 1> normal{ { v_2[1] * u[2] - v_2[2] * u[1], v_2[2] * u[0] - v_2[0] * u[2], v_2[0] * u[1] - v_2[1] * u[0] } };
            const math::matrix<type, 3, 1> mapped_v_2 = euclidean * v_2;
            const math::matrix<type, 3, 1> mapped_u = euclidean * u;
            const math::matrix<type, 3, 1> mapped_normal{ { mapped_v_2[1] * mapped_u[2] - mapped_v_2[2] * mapped_u[1], mapped_v_2[2] * mapped_u[0] - mapped_v_2[0] * mapped_u[2], mapped_v_2[0] * mapped_u[1] - mapped_v_2[1] * mapped_u[0] } };

            math::matrix<type, 3, 3> rotation;
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    rotation[row][col] = (mapped_v_2[row] * v_2[col]) + (mapped_u[row] * u[col]) + (mapped_normal[row] * normal[col]);
                }
            }
            const math::matrix<type, 3, 1> translation = (euclidean - rotation) * normal;

            rotations[candidate] = rotation;
            translations[candidate] = translation;
            normals[candidate] = normal;
            rotations[candidate + 2] = rotation;
            translations[candidate + 2] = -translation;
            normals[candidate + 2] = -normal;
        }
        return 4;
    }

    template <typename type>
    bool homography<type>::transform(
        const math::matrix<type, 3, 3>& homography,
        const type x,
        const type y,
        type& transformed_x,
        type& transformed_y
    ) {
        const type mapped_x = (homography[0][0] * x) + (homography[0][1] * y) + homography[0][2];
        const type mapped_y = (homography[1][0] * x) + (homography[1][1] * y) + homography[1][2];
        const type mapped_w = (homography[2][0] * x) + (homography[2][1] * y) + homography[2][2];
        const type scale = math::sqrt((mapped_x * mapped_x) + (mapped_y * mapped_y) + (mapped_w * mapped_w));
        if (math::abs(mapped_w) <= (homography::depth_tolerance * scale)) {
            return false;
        }
        transformed_x = mapped_x / mapped_w;
        transformed_y = mapped_y / mapped_w;
        return true;
    }

    template class homography<float>;
    template class homography<double>;
}
