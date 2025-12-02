/*
Copyright (C) 2025 Geoffrey Daniels. https://gpdaniels.com/

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

#pragma once
#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include "matrix.hpp"
#include "matrix_decomposition_singular_value.hpp"

namespace geometry {
    static inline bool triangulate(
        const matrix::matrix<double, 3, 1>& lhs_ray,
        const matrix::matrix<double, 3, 4>& lhs_pose,
        const matrix::matrix<double, 3, 1>& rhs_ray,
        const matrix::matrix<double, 3, 4>& rhs_pose,
        matrix::matrix<double, 3, 1>& result
    ) {
        constexpr static const double tolerance = 1e-8;
        // Build system matrix.
        double matrix_a[4][4];
        for (int i = 0; i < 4; ++i) {
            matrix_a[0][i] = (lhs_ray[0] / lhs_ray[2]) * lhs_pose[2][i] - lhs_pose[0][i];
            matrix_a[1][i] = (lhs_ray[1] / lhs_ray[2]) * lhs_pose[2][i] - lhs_pose[1][i];
            matrix_a[2][i] = (rhs_ray[0] / rhs_ray[2]) * rhs_pose[2][i] - rhs_pose[0][i];
            matrix_a[3][i] = (rhs_ray[1] / rhs_ray[2]) * rhs_pose[2][i] - rhs_pose[1][i];
        }

        // Solve matrix using svd decomposition.
        double matrix_u[4][4];
        double matrix_s[4][4];
        double matrix_vt[4][4];
        matrix::decompose_singular_value(&matrix_a[0][0], 4, 4, &matrix_u[0][0], &matrix_s[0][0], &matrix_vt[0][0]);

        // Extract the homographic point from the results.
        const double* point_homography = &matrix_vt[3][0];

        // If the fourth dimension of the homographic point is zero return false.
        if (math::abs(point_homography[3]) < tolerance) {
            return false;
        }

        // Check for non-unique solutions by comparing the two lowest singular values.
        if (math::abs(matrix_s[2][2] - matrix_s[3][3]) < tolerance) {
            // If the singular values are non-unique check if the associated points are different.
            const double* point_homography_alternate = &matrix_vt[2][0];
            const double point_difference =
                math::abs(point_homography[0] - point_homography_alternate[0]) +
                math::abs(point_homography[1] - point_homography_alternate[1]) +
                math::abs(point_homography[2] - point_homography_alternate[2]) +
                math::abs(point_homography[3] - point_homography_alternate[3]);
            // If the points are different return false as there is no unique solution.
            if (point_difference > tolerance) {
                return false;
            }
        }

        // Convert to euclidean coordinates.
        result[0] = point_homography[0] / point_homography[3];
        result[1] = point_homography[1] / point_homography[3];
        result[2] = point_homography[2] / point_homography[3];

        return true;
    }

    static inline bool triangulate(
        const matrix::matrix<double, 2, 1>& lhs_point_normalised,
        const matrix::matrix<double, 3, 4>& lhs_pose,
        const matrix::matrix<double, 2, 1>& rhs_point_normalised,
        const matrix::matrix<double, 3, 4>& rhs_pose,
        matrix::matrix<double, 3, 1>& result
    ) {
        return triangulate(
            matrix::matrix<double, 3, 1>{ { lhs_point_normalised[0], lhs_point_normalised[1], 1 } },
            lhs_pose,
            matrix::matrix<double, 3, 1>{ { rhs_point_normalised[0], rhs_point_normalised[1], 1 } },
            rhs_pose,
            result
        );
    }
}

#endif // GEOMETRY_HPP
