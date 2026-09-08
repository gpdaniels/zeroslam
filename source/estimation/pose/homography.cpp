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

#include "estimation/pose/homography.hpp"

#include "geometry/cheirality.hpp"
#include "geometry/triangulation/linear_least_squares.hpp"
#include "math/math.hpp"
#include "math/matrix.hpp"
#include "math/matrix_decomposition_singular_value.hpp"

namespace estimation::pose {
    template <typename type>
    bool homography<type>::recover(
        const type* const __restrict homography,
        const type* const __restrict lhs_points,
        const type* const __restrict rhs_points,
        const size_t point_count,
        type* const __restrict rotation,
        type* const __restrict translation,
        type* const __restrict triangulated_points,
        size_t* const __restrict support_count
    ) {
        constexpr static const auto matrix_multiply_3x3 = [](const type* const __restrict lhs, const type* const __restrict rhs, type* const __restrict result) {
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    type sum = type(0);
                    for (int k = 0; k < 3; ++k) {
                        sum += lhs[row * 3 + k] * rhs[k * 3 + col];
                    }
                    result[row * 3 + col] = sum;
                }
            }
        };
        constexpr static const auto determinant_3x3 = [](const type* const __restrict m) -> type {
            return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
        };

        const type homography_determinant = determinant_3x3(homography);
        if (math::abs(homography_determinant) < type(1.0e-15)) {
            return false;
        }
        const type homography_determinant_inverse = type(1.0) / homography_determinant;
        type euclidean_homography[9] = {
            (homography[4] * homography[8] - homography[5] * homography[7]) * homography_determinant_inverse,
            (homography[2] * homography[7] - homography[1] * homography[8]) * homography_determinant_inverse,
            (homography[1] * homography[5] - homography[2] * homography[4]) * homography_determinant_inverse,
            (homography[5] * homography[6] - homography[3] * homography[8]) * homography_determinant_inverse,
            (homography[0] * homography[8] - homography[2] * homography[6]) * homography_determinant_inverse,
            (homography[2] * homography[3] - homography[0] * homography[5]) * homography_determinant_inverse,
            (homography[3] * homography[7] - homography[4] * homography[6]) * homography_determinant_inverse,
            (homography[1] * homography[6] - homography[0] * homography[7]) * homography_determinant_inverse,
            (homography[0] * homography[4] - homography[1] * homography[3]) * homography_determinant_inverse
        };

        type projective_depth_sum = type(0);
        for (size_t point_index = 0; point_index < point_count; ++point_index) {
            projective_depth_sum += euclidean_homography[6] * lhs_points[point_index * 2 + 0] + euclidean_homography[7] * lhs_points[point_index * 2 + 1] + euclidean_homography[8];
        }
        if (projective_depth_sum < type(0)) {
            for (int i = 0; i < 9; ++i) {
                euclidean_homography[i] = -euclidean_homography[i];
            }
        }

        type matrix_u[9];
        type matrix_s[9];
        type matrix_vt[9];
        if (!math::decompose_singular_value(&euclidean_homography[0], 3, 3, &matrix_u[0], &matrix_s[0], &matrix_vt[0])) {
            return false;
        }
        const type d1 = matrix_s[0];
        const type d2 = matrix_s[4];
        const type d3 = matrix_s[8];

        if ((d1 - d2 < type(1.0e-7) * d1) || (d2 - d3 < type(1.0e-7) * d1) || (d3 <= type(0))) {
            return false;
        }

        const type sign_factor = ((determinant_3x3(matrix_u) < type(0)) ? type(-1) : type(1)) * ((determinant_3x3(matrix_vt) < type(0)) ? type(-1) : type(1));

        type matrix_v[9];
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                matrix_v[row * 3 + col] = matrix_vt[col * 3 + row];
            }
        }

        const type d1_squared = d1 * d1;
        const type d2_squared = d2 * d2;
        const type d3_squared = d3 * d3;

        const type aux1 = math::sqrt((d1_squared - d2_squared) / (d1_squared - d3_squared));
        const type aux3 = math::sqrt((d2_squared - d3_squared) / (d1_squared - d3_squared));
        const type x1_signs[4] = { aux1, aux1, -aux1, -aux1 };
        const type x3_signs[4] = { aux3, -aux3, aux3, -aux3 };

        type candidate_rotations[8][9];
        type candidate_translations[8][3];
        int candidate_count = 0;

        const auto emit_candidate = [&](const type rotation_prime[9], const type translation_prime[3], const type normal_prime[3]) {
            type normal[3];
            for (int row = 0; row < 3; ++row) {
                normal[row] = matrix_v[row * 3 + 0] * normal_prime[0] + matrix_v[row * 3 + 1] * normal_prime[1] + matrix_v[row * 3 + 2] * normal_prime[2];
            }
            if (normal[2] < type(0)) {
                return;
            }

            type u_rotation_prime[9];
            matrix_multiply_3x3(matrix_u, rotation_prime, u_rotation_prime);
            type candidate_rotation[9];
            matrix_multiply_3x3(u_rotation_prime, matrix_vt, candidate_rotation);

            type candidate_translation[3];
            for (int row = 0; row < 3; ++row) {
                candidate_translation[row] = matrix_u[row * 3 + 0] * translation_prime[0] + matrix_u[row * 3 + 1] * translation_prime[1] + matrix_u[row * 3 + 2] * translation_prime[2];
            }
            const type translation_norm = math::sqrt(candidate_translation[0] * candidate_translation[0] + candidate_translation[1] * candidate_translation[1] + candidate_translation[2] * candidate_translation[2]);
            if (translation_norm < type(1.0e-12)) {
                return;
            }
            const type translation_norm_inverse = type(1.0) / translation_norm;

            for (int i = 0; i < 9; ++i) {
                candidate_rotations[candidate_count][i] = sign_factor * candidate_rotation[i];
            }
            for (int row = 0; row < 3; ++row) {
                candidate_translations[candidate_count][row] = candidate_translation[row] * translation_norm_inverse;
            }
            ++candidate_count;
        };

        {
            const type sin_theta = math::sqrt((d1_squared - d2_squared) * (d2_squared - d3_squared)) / ((d1 + d3) * d2);
            const type cos_theta = (d2_squared + d1 * d3) / ((d1 + d3) * d2);
            const type sin_theta_signs[4] = { sin_theta, -sin_theta, -sin_theta, sin_theta };
            for (int i = 0; i < 4; ++i) {
                const type rotation_prime[9] = {
                    cos_theta,
                    type(0),
                    -sin_theta_signs[i],
                    type(0),
                    type(1),
                    type(0),
                    sin_theta_signs[i],
                    type(0),
                    cos_theta
                };
                const type translation_prime[3] = { x1_signs[i] * (d1 - d3), type(0), -x3_signs[i] * (d1 - d3) };
                const type normal_prime[3] = { x1_signs[i], type(0), x3_signs[i] };
                emit_candidate(rotation_prime, translation_prime, normal_prime);
            }
        }

        {
            const type sin_phi = math::sqrt((d1_squared - d2_squared) * (d2_squared - d3_squared)) / ((d1 - d3) * d2);
            const type cos_phi = (d1 * d3 - d2_squared) / ((d1 - d3) * d2);
            const type sin_phi_signs[4] = { sin_phi, -sin_phi, -sin_phi, sin_phi };
            for (int i = 0; i < 4; ++i) {
                const type rotation_prime[9] = {
                    cos_phi,
                    type(0),
                    sin_phi_signs[i],
                    type(0),
                    type(-1),
                    type(0),
                    sin_phi_signs[i],
                    type(0),
                    -cos_phi
                };
                const type translation_prime[3] = { x1_signs[i] * (d1 + d3), type(0), x3_signs[i] * (d1 + d3) };
                const type normal_prime[3] = { x1_signs[i], type(0), x3_signs[i] };
                emit_candidate(rotation_prime, translation_prime, normal_prime);
            }
        }

        if (candidate_count == 0) {
            return false;
        }

        const type lhs_camera_pose[3][4] = {
            { type(1), type(0), type(0), type(0) },
            { type(0), type(1), type(0), type(0) },
            { type(0), type(0), type(1), type(0) }
        };

        type** const triangulated_points_set = new type*[static_cast<size_t>(candidate_count)];
        size_t* const triangulated_points_valid_counts = new size_t[static_cast<size_t>(candidate_count)];
        for (int candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
            triangulated_points_set[candidate_index] = new type[point_count * 3]{};
            triangulated_points_valid_counts[candidate_index] = size_t(0);
        }

        for (int candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
            const type rhs_camera_pose[3][4] = {
                { candidate_rotations[candidate_index][0], candidate_rotations[candidate_index][1], candidate_rotations[candidate_index][2], candidate_translations[candidate_index][0] },
                { candidate_rotations[candidate_index][3], candidate_rotations[candidate_index][4], candidate_rotations[candidate_index][5], candidate_translations[candidate_index][1] },
                { candidate_rotations[candidate_index][6], candidate_rotations[candidate_index][7], candidate_rotations[candidate_index][8], candidate_translations[candidate_index][2] }
            };

            for (size_t point_index = 0; point_index < point_count; ++point_index) {
                math::matrix<type, 3, 1> triangulated_result = math::matrix<type, 3, 1>::zero();
                const bool is_valid = geometry::cheirality<type>::template triangulate<geometry::triangulation::linear_least_squares<type>>(
                    math::matrix<type, 3, 1>{ { lhs_points[point_index * 2], lhs_points[point_index * 2 + 1], type(1) } },
                    math::matrix<type, 3, 4>(&lhs_camera_pose[0][0]),
                    math::matrix<type, 3, 1>{ { rhs_points[point_index * 2], rhs_points[point_index * 2 + 1], type(1) } },
                    math::matrix<type, 3, 4>(&rhs_camera_pose[0][0]),
                    triangulated_result
                );
                triangulated_points_set[candidate_index][point_index * 3 + 0] = triangulated_result[0];
                triangulated_points_set[candidate_index][point_index * 3 + 1] = triangulated_result[1];
                triangulated_points_set[candidate_index][point_index * 3 + 2] = triangulated_result[2];
                if (is_valid) {
                    triangulated_points_valid_counts[candidate_index] += size_t(1);
                }
            }
        }

        int best_candidate_index = 0;
        for (int candidate_index = 1; candidate_index < candidate_count; ++candidate_index) {
            if (triangulated_points_valid_counts[candidate_index] > triangulated_points_valid_counts[best_candidate_index]) {
                best_candidate_index = candidate_index;
            }
        }
        size_t runner_up_count = 0;
        for (int candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
            if (candidate_index == best_candidate_index) {
                continue;
            }
            if (triangulated_points_valid_counts[candidate_index] > runner_up_count) {
                runner_up_count = triangulated_points_valid_counts[candidate_index];
            }
        }

        const size_t best_count = triangulated_points_valid_counts[best_candidate_index];
        const bool success = (best_count > 0) && (size_t(10) * runner_up_count <= size_t(7) * best_count);

        for (int i = 0; i < 9; ++i) {
            rotation[i] = candidate_rotations[best_candidate_index][i];
        }
        for (int row = 0; row < 3; ++row) {
            translation[row] = candidate_translations[best_candidate_index][row];
        }
        for (size_t point_index = 0; point_index < point_count; ++point_index) {
            triangulated_points[point_index * 3 + 0] = triangulated_points_set[best_candidate_index][point_index * 3 + 0];
            triangulated_points[point_index * 3 + 1] = triangulated_points_set[best_candidate_index][point_index * 3 + 1];
            triangulated_points[point_index * 3 + 2] = triangulated_points_set[best_candidate_index][point_index * 3 + 2];
        }
        if (support_count != nullptr) {
            *support_count = best_count;
        }

        for (int candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
            delete[] triangulated_points_set[candidate_index];
        }
        delete[] triangulated_points_set;
        delete[] triangulated_points_valid_counts;

        return success;
    }

    template class homography<float>;
    template class homography<double>;
}
