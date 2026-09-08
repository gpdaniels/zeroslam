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

#include "estimation/minimal/perspective_3_point.hpp"

#include "math/math.hpp"

namespace estimation::minimal {
    template <typename type>
    int perspective_3_point<type>::solve(
        const type image_points_in[3][3],
        const type world_points_in[3][3],
        type rotations[4][9],
        type translations[4][3]
    ) {
        constexpr static const auto solve_quadratic_real_roots = [](
                                                                     const type b,
                                                                     const type c,
                                                                     type& r1,
                                                                     type& r2
                                                                 ) -> bool {
            const type threshold = type(-1.0e-12);
            const type v = b * b - type(4.0) * c;
            if (v < threshold) {
                r1 = r2 = type(-0.5) * b;
                return v >= type(0);
            }
            if (v > threshold && v < type(0)) {
                r1 = type(-0.5) * b;
                r2 = type(-2);
                return true;
            }
            const type y = math::sqrt(v);
            if (b < type(0)) {
                r1 = type(0.5) * (-b + y);
                r2 = type(0.5) * (-b - y);
            }
            else {
                r1 = type(2.0) * c / (-b + y);
                r2 = type(2.0) * c / (-b - y);
            }
            return true;
        };

        constexpr static const auto solve_cubic_single_real_root = [](
                                                                       const type c2,
                                                                       const type c1,
                                                                       const type c0,
                                                                       type& root
                                                                   ) -> bool {
            const type a = c1 - c2 * c2 / type(3.0);
            const type b = (type(2.0) * c2 * c2 * c2 - type(9.0) * c2 * c1) / type(27.0) + c0;
            type c = b * b / type(4.0) + a * a * a / type(27.0);
            if (c != type(0)) {
                if (c > type(0)) {
                    c = math::sqrt(c);
                    const type bh = -b * type(0.5);
                    const type val1 = bh + c;
                    const type val2 = bh - c;
                    const type cbrt1 = math::copysign(math::pow(math::abs(val1), type(1.0) / type(3.0)), val1);
                    const type cbrt2 = math::copysign(math::pow(math::abs(val2), type(1.0) / type(3.0)), val2);
                    root = cbrt1 + cbrt2 - c2 / type(3.0);
                    return true;
                }
                else {
                    c = type(3.0) * b / (type(2.0) * a) * math::sqrt(type(-3.0) / a);
                    root = type(2.0) * math::sqrt(-a / type(3.0)) * math::cos(math::acos(c) / type(3.0)) - c2 / type(3.0);
                }
            }
            else {
                root = -c2 / type(3.0) + (a != type(0) ? (type(3.0) * b / a) : type(0));
            }
            return false;
        };

        constexpr static const auto compute_null_space_factorization = [](
                                                                           type matrix_c[3][3],
                                                                           type pq[2][3]
                                                                       ) -> void {
            type matrix_c_adj[3][3];

            matrix_c_adj[0][0] = matrix_c[1][2] * matrix_c[2][1] - matrix_c[1][1] * matrix_c[2][2];
            matrix_c_adj[1][1] = matrix_c[0][2] * matrix_c[2][0] - matrix_c[0][0] * matrix_c[2][2];
            matrix_c_adj[2][2] = matrix_c[0][1] * matrix_c[1][0] - matrix_c[0][0] * matrix_c[1][1];
            matrix_c_adj[0][1] = matrix_c[0][1] * matrix_c[2][2] - matrix_c[0][2] * matrix_c[2][1];
            matrix_c_adj[0][2] = matrix_c[0][2] * matrix_c[1][1] - matrix_c[0][1] * matrix_c[1][2];
            matrix_c_adj[1][0] = matrix_c_adj[0][1];
            matrix_c_adj[1][2] = matrix_c[0][0] * matrix_c[1][2] - matrix_c[0][2] * matrix_c[1][0];
            matrix_c_adj[2][0] = matrix_c_adj[0][2];
            matrix_c_adj[2][1] = matrix_c_adj[1][2];

            type v[3];
            if (matrix_c_adj[0][0] > matrix_c_adj[1][1]) {
                if (matrix_c_adj[0][0] > matrix_c_adj[2][2]) {
                    const type inv = type(1.0) / math::sqrt(matrix_c_adj[0][0]);
                    v[0] = matrix_c_adj[0][0] * inv;
                    v[1] = matrix_c_adj[1][0] * inv;
                    v[2] = matrix_c_adj[2][0] * inv;
                }
                else {
                    const type inv = type(1.0) / math::sqrt(matrix_c_adj[2][2]);
                    v[0] = matrix_c_adj[0][2] * inv;
                    v[1] = matrix_c_adj[1][2] * inv;
                    v[2] = matrix_c_adj[2][2] * inv;
                }
            }
            else if (matrix_c_adj[1][1] > matrix_c_adj[2][2]) {
                const type inv = type(1.0) / math::sqrt(matrix_c_adj[1][1]);
                v[0] = matrix_c_adj[0][1] * inv;
                v[1] = matrix_c_adj[1][1] * inv;
                v[2] = matrix_c_adj[2][1] * inv;
            }
            else {
                const type inv = type(1.0) / math::sqrt(matrix_c_adj[2][2]);
                v[0] = matrix_c_adj[0][2] * inv;
                v[1] = matrix_c_adj[1][2] * inv;
                v[2] = matrix_c_adj[2][2] * inv;
            }

            matrix_c[0][1] -= v[2];
            matrix_c[0][2] += v[1];
            matrix_c[1][2] -= v[0];
            matrix_c[1][0] += v[2];
            matrix_c[2][0] -= v[1];
            matrix_c[2][1] += v[0];

            pq[0][0] = matrix_c[0][0];
            pq[0][1] = matrix_c[1][0];
            pq[0][2] = matrix_c[2][0];
            pq[1][0] = matrix_c[0][0];
            pq[1][1] = matrix_c[0][1];
            pq[1][2] = matrix_c[0][2];
        };

        constexpr static const auto refine_distances_newton = [](
                                                                  type& lambda1,
                                                                  type& lambda2,
                                                                  type& lambda3,
                                                                  const type a12,
                                                                  const type a13,
                                                                  const type a23,
                                                                  const type b12,
                                                                  const type b13,
                                                                  const type b23
                                                              ) -> void {
            for (int iter = 0; iter < 5; ++iter) {
                const type r1 = (lambda1 * lambda1 - type(2.0) * lambda1 * lambda2 * b12 + lambda2 * lambda2 - a12);
                const type r2 = (lambda1 * lambda1 - type(2.0) * lambda1 * lambda3 * b13 + lambda3 * lambda3 - a13);
                const type r3 = (lambda2 * lambda2 - type(2.0) * lambda2 * lambda3 * b23 + lambda3 * lambda3 - a23);
                if (math::abs(r1) + math::abs(r2) + math::abs(r3) < type(1.0e-10)) {
                    return;
                }
                const type x11 = lambda1 - lambda2 * b12;
                const type x12 = lambda2 - lambda1 * b12;
                const type x21 = lambda1 - lambda3 * b13;
                const type x23 = lambda3 - lambda1 * b13;
                const type x32 = lambda2 - lambda3 * b23;
                const type x33 = lambda3 - lambda2 * b23;
                const type determinant_j_inv = type(0.5) / (x11 * x23 * x32 + x12 * x21 * x33);
                lambda1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * determinant_j_inv;
                lambda2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * determinant_j_inv;
                lambda3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * determinant_j_inv;
            }
        };
        type world_points[3][3];
        type image_points[3][3];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                world_points[i][j] = world_points_in[i][j];
                image_points[i][j] = image_points_in[i][j];
            }
        }

        type world_vector_01[3] = {
            world_points[0][0] - world_points[1][0],
            world_points[0][1] - world_points[1][1],
            world_points[0][2] - world_points[1][2]
        };
        type world_vector_02[3] = {
            world_points[0][0] - world_points[2][0],
            world_points[0][1] - world_points[2][1],
            world_points[0][2] - world_points[2][2]
        };
        type world_vector_12[3] = {
            world_points[1][0] - world_points[2][0],
            world_points[1][1] - world_points[2][1],
            world_points[1][2] - world_points[2][2]
        };

        type world_dist_sq_01 = world_vector_01[0] * world_vector_01[0] + world_vector_01[1] * world_vector_01[1] + world_vector_01[2] * world_vector_01[2];
        type world_dist_sq_02 = world_vector_02[0] * world_vector_02[0] + world_vector_02[1] * world_vector_02[1] + world_vector_02[2] * world_vector_02[2];
        type world_dist_sq_12 = world_vector_12[0] * world_vector_12[0] + world_vector_12[1] * world_vector_12[1] + world_vector_12[2] * world_vector_12[2];

        if (world_dist_sq_01 > world_dist_sq_02) {
            if (world_dist_sq_01 > world_dist_sq_12) {
                for (int j = 0; j < 3; ++j) {
                    const type temp_image = image_points[0][j];
                    image_points[0][j] = image_points[2][j];
                    image_points[2][j] = temp_image;
                }
                for (int j = 0; j < 3; ++j) {
                    const type temp_world = world_points[0][j];
                    world_points[0][j] = world_points[2][j];
                    world_points[2][j] = temp_world;
                }
                const type temp_dist = world_dist_sq_01;
                world_dist_sq_01 = world_dist_sq_12;
                world_dist_sq_12 = temp_dist;
                for (int j = 0; j < 3; ++j) {
                    world_vector_01[j] = -world_vector_12[j];
                    world_vector_02[j] = -world_vector_02[j];
                }
            }
        }
        else if (world_dist_sq_02 > world_dist_sq_12) {
            for (int j = 0; j < 3; ++j) {
                const type temp_image = image_points[0][j];
                image_points[0][j] = image_points[1][j];
                image_points[1][j] = temp_image;
            }
            for (int j = 0; j < 3; ++j) {
                const type temp_world = world_points[0][j];
                world_points[0][j] = world_points[1][j];
                world_points[1][j] = temp_world;
            }
            const type temp_dist = world_dist_sq_02;
            world_dist_sq_02 = world_dist_sq_12;
            world_dist_sq_12 = temp_dist;
            for (int j = 0; j < 3; ++j) {
                world_vector_01[j] = -world_vector_01[j];
                world_vector_02[j] = world_vector_12[j];
            }
        }

        const type world_dist_sq_12_inv = type(1.0) / world_dist_sq_12;
        const type ratio_a = world_dist_sq_01 * world_dist_sq_12_inv;
        const type ratio_b = world_dist_sq_02 * world_dist_sq_12_inv;

        const type image_cos_01 = image_points[0][0] * image_points[1][0] + image_points[0][1] * image_points[1][1] + image_points[0][2] * image_points[1][2];
        const type image_cos_02 = image_points[0][0] * image_points[2][0] + image_points[0][1] * image_points[2][1] + image_points[0][2] * image_points[2][2];
        const type image_cos_12 = image_points[1][0] * image_points[2][0] + image_points[1][1] * image_points[2][1] + image_points[1][2] * image_points[2][2];

        const type image_sin_sq_12 = type(1.0) - image_cos_12 * image_cos_12;
        const type image_cos_sq_02_minus_1 = image_cos_02 * image_cos_02 - type(1.0);
        const type image_cos_sq_01_minus_1 = image_cos_01 * image_cos_01 - type(1.0);
        const type ratio_ab = ratio_a * ratio_b;
        const type ratio_b_sq = ratio_b * ratio_b;
        const type ratio_a_sq = ratio_a * ratio_a;
        const type term_m013 = type(-2.0) + type(2.0) * image_cos_01 * image_cos_02 * image_cos_12;
        const type ratio_b_sq_sin_sq_12 = ratio_b_sq * image_sin_sq_12;
        const type ratio_a_sq_sin_sq_12 = ratio_a_sq * image_sin_sq_12;
        const type ratio_ab_sin_sq_12_2 = type(2.0) * ratio_ab * image_sin_sq_12;

        const type k3_inv = type(1.0) / (ratio_b_sq_sin_sq_12 + ratio_b * image_cos_sq_02_minus_1);
        const type k2 = k3_inv * ((type(-1.0) + ratio_a) * image_cos_sq_02_minus_1 + ratio_ab_sin_sq_12_2 + ratio_b_sq_sin_sq_12 + ratio_b * term_m013);
        const type k1 = k3_inv * (ratio_a_sq_sin_sq_12 + ratio_ab_sin_sq_12_2 + ratio_a * term_m013 + (type(-1.0) + ratio_b) * image_cos_sq_01_minus_1);
        const type k0 = k3_inv * (ratio_a_sq_sin_sq_12 + ratio_a * image_cos_sq_01_minus_1);

        type real_root_s;
        const bool found_single_real_root = solve_cubic_single_real_root(k2, k1, k0, real_root_s);

        type matrix_c[3][3];
        matrix_c[0][0] = -ratio_a + real_root_s * (type(1.0) - ratio_b);
        matrix_c[0][1] = -image_cos_02 * real_root_s;
        matrix_c[0][2] = ratio_a * image_cos_12 + ratio_b * image_cos_12 * real_root_s;
        matrix_c[1][0] = matrix_c[0][1];
        matrix_c[1][1] = real_root_s + type(1.0);
        matrix_c[1][2] = -image_cos_01;
        matrix_c[2][0] = matrix_c[0][2];
        matrix_c[2][1] = matrix_c[1][2];
        matrix_c[2][2] = -ratio_a - ratio_b * real_root_s + type(1.0);

        type pq_factors[2][3];
        compute_null_space_factorization(matrix_c, pq_factors);

        type world_vector_01_cross_02[3];
        world_vector_01_cross_02[0] = world_vector_01[1] * world_vector_02[2] - world_vector_01[2] * world_vector_02[1];
        world_vector_01_cross_02[1] = world_vector_01[2] * world_vector_02[0] - world_vector_01[0] * world_vector_02[2];
        world_vector_01_cross_02[2] = world_vector_01[0] * world_vector_02[1] - world_vector_01[1] * world_vector_02[0];

        type world_basis_matrix[3][3] = {
            { world_vector_01[0], world_vector_02[0], world_vector_01_cross_02[0] },
            { world_vector_01[1], world_vector_02[1], world_vector_01_cross_02[1] },
            { world_vector_01[2], world_vector_02[2], world_vector_01_cross_02[2] }
        };

        const type det_world_basis =
            world_basis_matrix[0][0] * (world_basis_matrix[1][1] * world_basis_matrix[2][2] - world_basis_matrix[2][1] * world_basis_matrix[1][2]) - world_basis_matrix[0][1] * (world_basis_matrix[1][0] * world_basis_matrix[2][2] - world_basis_matrix[1][2] * world_basis_matrix[2][0]) + world_basis_matrix[0][2] * (world_basis_matrix[1][0] * world_basis_matrix[2][1] - world_basis_matrix[1][1] * world_basis_matrix[2][0]);

        if (math::abs(det_world_basis) < type(1.0e-15)) {
            return 0;
        }

        const type det_world_basis_inv = type(1.0) / det_world_basis;
        type world_basis_matrix_inv[3][3];
        world_basis_matrix_inv[0][0] = (world_basis_matrix[1][1] * world_basis_matrix[2][2] - world_basis_matrix[2][1] * world_basis_matrix[1][2]) * det_world_basis_inv;
        world_basis_matrix_inv[0][1] = (world_basis_matrix[0][2] * world_basis_matrix[2][1] - world_basis_matrix[0][1] * world_basis_matrix[2][2]) * det_world_basis_inv;
        world_basis_matrix_inv[0][2] = (world_basis_matrix[0][1] * world_basis_matrix[1][2] - world_basis_matrix[0][2] * world_basis_matrix[1][1]) * det_world_basis_inv;
        world_basis_matrix_inv[1][0] = (world_basis_matrix[1][2] * world_basis_matrix[2][0] - world_basis_matrix[1][0] * world_basis_matrix[2][2]) * det_world_basis_inv;
        world_basis_matrix_inv[1][1] = (world_basis_matrix[0][0] * world_basis_matrix[2][2] - world_basis_matrix[0][2] * world_basis_matrix[2][0]) * det_world_basis_inv;
        world_basis_matrix_inv[1][2] = (world_basis_matrix[1][0] * world_basis_matrix[0][2] - world_basis_matrix[0][0] * world_basis_matrix[1][2]) * det_world_basis_inv;
        world_basis_matrix_inv[2][0] = (world_basis_matrix[1][0] * world_basis_matrix[2][1] - world_basis_matrix[2][0] * world_basis_matrix[1][1]) * det_world_basis_inv;
        world_basis_matrix_inv[2][1] = (world_basis_matrix[2][0] * world_basis_matrix[0][1] - world_basis_matrix[0][0] * world_basis_matrix[2][1]) * det_world_basis_inv;
        world_basis_matrix_inv[2][2] = (world_basis_matrix[0][0] * world_basis_matrix[1][1] - world_basis_matrix[1][0] * world_basis_matrix[0][1]) * det_world_basis_inv;

        int num_solutions = 0;

        for (int i = 0; i < 2; ++i) {
            const type factor_p0 = pq_factors[i][0];
            const type factor_p1 = pq_factors[i][1];
            const type factor_p2 = pq_factors[i][2];

            const bool should_switch_12 = math::abs(factor_p0) <= math::abs(factor_p1);

            if (should_switch_12) {
                const type weight_w0 = -factor_p0 / factor_p1;
                const type weight_w1 = -factor_p2 / factor_p1;
                const type coeff_ca = type(1.0) / (weight_w1 * weight_w1 - ratio_b);
                const type coeff_cb = type(2.0) * (ratio_b * image_cos_12 - image_cos_02 * weight_w1 + weight_w0 * weight_w1) * coeff_ca;
                const type coeff_cc = (weight_w0 * weight_w0 - type(2.0) * image_cos_02 * weight_w0 - ratio_b + type(1.0)) * coeff_ca;
                type solution_taus[2];
                if (!solve_quadratic_real_roots(coeff_cb, coeff_cc, solution_taus[0], solution_taus[1])) {
                    continue;
                }
                for (int ti = 0; ti < 2; ++ti) {
                    const type tau = solution_taus[ti];
                    if (tau <= type(0)) {
                        continue;
                    }
                    type dist_d2 = math::sqrt(world_dist_sq_12 / (tau * (tau - type(2.0) * image_cos_12) + type(1.0)));
                    type dist_d1 = tau * dist_d2;
                    type dist_d0 = weight_w0 * dist_d2 + weight_w1 * dist_d1;
                    if (dist_d0 < type(0)) {
                        continue;
                    }

                    refine_distances_newton(dist_d0, dist_d1, dist_d2, world_dist_sq_01, world_dist_sq_02, world_dist_sq_12, image_cos_01, image_cos_02, image_cos_12);

                    type vector_v1[3];
                    type vector_v2[3];
                    for (int j = 0; j < 3; ++j) {
                        vector_v1[j] = dist_d0 * image_points[0][j] - dist_d1 * image_points[1][j];
                        vector_v2[j] = dist_d0 * image_points[0][j] - dist_d2 * image_points[2][j];
                    }
                    type vector_v1_cross_v2[3];
                    vector_v1_cross_v2[0] = vector_v1[1] * vector_v2[2] - vector_v1[2] * vector_v2[1];
                    vector_v1_cross_v2[1] = vector_v1[2] * vector_v2[0] - vector_v1[0] * vector_v2[2];
                    vector_v1_cross_v2[2] = vector_v1[0] * vector_v2[1] - vector_v1[1] * vector_v2[0];

                    type image_basis_matrix[3][3] = {
                        { vector_v1[0], vector_v2[0], vector_v1_cross_v2[0] },
                        { vector_v1[1], vector_v2[1], vector_v1_cross_v2[1] },
                        { vector_v1[2], vector_v2[2], vector_v1_cross_v2[2] }
                    };

                    type solution_rotation[9];
                    for (int row = 0; row < 3; ++row) {
                        for (int col = 0; col < 3; ++col) {
                            solution_rotation[row * 3 + col] = image_basis_matrix[row][0] * world_basis_matrix_inv[0][col] + image_basis_matrix[row][1] * world_basis_matrix_inv[1][col] + image_basis_matrix[row][2] * world_basis_matrix_inv[2][col];
                        }
                    }

                    type solution_translation[3];
                    for (int row = 0; row < 3; ++row) {
                        solution_translation[row] = dist_d0 * image_points[0][row] - (solution_rotation[row * 3 + 0] * world_points[0][0] + solution_rotation[row * 3 + 1] * world_points[0][1] + solution_rotation[row * 3 + 2] * world_points[0][2]);
                    }

                    for (int k = 0; k < 9; ++k)
                        rotations[num_solutions][k] = solution_rotation[k];
                    for (int k = 0; k < 3; ++k)
                        translations[num_solutions][k] = solution_translation[k];
                    ++num_solutions;
                }
            }
            else {
                const type weight_w0 = -factor_p1 / factor_p0;
                const type weight_w1 = -factor_p2 / factor_p0;
                const type coeff_ca = type(1.0) / (-ratio_a * weight_w1 * weight_w1 + type(2.0) * ratio_a * image_cos_12 * weight_w1 - ratio_a + type(1.0));
                const type coeff_cb = type(2.0) * (ratio_a * image_cos_12 * weight_w0 - image_cos_01 - ratio_a * weight_w0 * weight_w1) * coeff_ca;
                const type coeff_cc = (type(1.0) - ratio_a * weight_w0 * weight_w0) * coeff_ca;
                type solution_taus[2];
                if (!solve_quadratic_real_roots(coeff_cb, coeff_cc, solution_taus[0], solution_taus[1])) {
                    continue;
                }
                for (int ti = 0; ti < 2; ++ti) {
                    const type tau = solution_taus[ti];
                    if (tau <= type(0)) {
                        continue;
                    }
                    type dist_d0 = math::sqrt(world_dist_sq_01 / (tau * (tau - type(2.0) * image_cos_01) + type(1.0)));
                    type dist_d1 = tau * dist_d0;
                    type dist_d2 = weight_w0 * dist_d0 + weight_w1 * dist_d1;

                    if (dist_d2 < type(0)) {
                        continue;
                    }

                    refine_distances_newton(dist_d0, dist_d1, dist_d2, world_dist_sq_01, world_dist_sq_02, world_dist_sq_12, image_cos_01, image_cos_02, image_cos_12);

                    type vector_v1[3];
                    type vector_v2[3];
                    for (int j = 0; j < 3; ++j) {
                        vector_v1[j] = dist_d0 * image_points[0][j] - dist_d1 * image_points[1][j];
                        vector_v2[j] = dist_d0 * image_points[0][j] - dist_d2 * image_points[2][j];
                    }
                    type vector_v1_cross_v2[3];
                    vector_v1_cross_v2[0] = vector_v1[1] * vector_v2[2] - vector_v1[2] * vector_v2[1];
                    vector_v1_cross_v2[1] = vector_v1[2] * vector_v2[0] - vector_v1[0] * vector_v2[2];
                    vector_v1_cross_v2[2] = vector_v1[0] * vector_v2[1] - vector_v1[1] * vector_v2[0];

                    type image_basis_matrix[3][3] = {
                        { vector_v1[0], vector_v2[0], vector_v1_cross_v2[0] },
                        { vector_v1[1], vector_v2[1], vector_v1_cross_v2[1] },
                        { vector_v1[2], vector_v2[2], vector_v1_cross_v2[2] }
                    };

                    type solution_rotation[9];
                    for (int row = 0; row < 3; ++row) {
                        for (int col = 0; col < 3; ++col) {
                            solution_rotation[row * 3 + col] = image_basis_matrix[row][0] * world_basis_matrix_inv[0][col] + image_basis_matrix[row][1] * world_basis_matrix_inv[1][col] + image_basis_matrix[row][2] * world_basis_matrix_inv[2][col];
                        }
                    }

                    type solution_translation[3];
                    for (int row = 0; row < 3; ++row) {
                        solution_translation[row] = dist_d0 * image_points[0][row] - (solution_rotation[row * 3 + 0] * world_points[0][0] + solution_rotation[row * 3 + 1] * world_points[0][1] + solution_rotation[row * 3 + 2] * world_points[0][2]);
                    }

                    for (int k = 0; k < 9; ++k)
                        rotations[num_solutions][k] = solution_rotation[k];
                    for (int k = 0; k < 3; ++k)
                        translations[num_solutions][k] = solution_translation[k];
                    ++num_solutions;
                }
            }

            if (num_solutions > 0 && found_single_real_root) {
                break;
            }
        }

        return num_solutions;
    }

    template class perspective_3_point<float>;
    template class perspective_3_point<double>;
}
