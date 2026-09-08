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

#include "estimation/pose/essential.hpp"

#include "geometry/cheirality.hpp"
#include "geometry/essential.hpp"
#include "geometry/triangulation/linear_least_squares.hpp"
#include "math/matrix.hpp"

namespace estimation::pose {
    template <typename type>
    bool essential<type>::recover(
        const type* const __restrict essential,
        const type* const __restrict lhs_points,
        const type* const __restrict rhs_points,
        const size_t point_count,
        type* const __restrict rotation,
        type* const __restrict translation,
        type* const __restrict triangulated_points,
        size_t* const __restrict support_count
    ) {
        math::matrix<type, 3, 3> rotations[4];
        math::matrix<type, 3, 1> translations[4];
        if (geometry::essential<type>::decompose(math::matrix<type, 3, 3>(essential), rotations, translations) != 4) {
            return false;
        }

        const type lhs_camera_pose[3][4] = {
            { type(1), type(0), type(0), type(0) },
            { type(0), type(1), type(0), type(0) },
            { type(0), type(0), type(1), type(0) },
        };

        type* const triangulated_points_set[4] = {
            new type[point_count * 3]{},
            new type[point_count * 3]{},
            new type[point_count * 3]{},
            new type[point_count * 3]{}
        };

        size_t triangulated_points_valid_counts[4] = {
            size_t(0),
            size_t(0),
            size_t(0),
            size_t(0)
        };

        for (size_t solution_index = 0; solution_index < 4; ++solution_index) {
            {
                const type rhs_camera_pose[3][4] = {
                    { rotations[solution_index][0][0], rotations[solution_index][0][1], rotations[solution_index][0][2], translations[solution_index][0] },
                    { rotations[solution_index][1][0], rotations[solution_index][1][1], rotations[solution_index][1][2], translations[solution_index][1] },
                    { rotations[solution_index][2][0], rotations[solution_index][2][1], rotations[solution_index][2][2], translations[solution_index][2] },
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
                    triangulated_points_set[solution_index][point_index * 3 + 0] = triangulated_result[0];
                    triangulated_points_set[solution_index][point_index * 3 + 1] = triangulated_result[1];
                    triangulated_points_set[solution_index][point_index * 3 + 2] = triangulated_result[2];
                    if (is_valid) {
                        triangulated_points_valid_counts[solution_index] += size_t(1);
                    }
                }
            }
        }

        size_t best_solution_index = 0;
        for (size_t solution_index = 1; solution_index < 4; ++solution_index) {
            if (triangulated_points_valid_counts[solution_index] > triangulated_points_valid_counts[best_solution_index]) {
                best_solution_index = solution_index;
            }
        }

        size_t runner_up_count = 0;
        for (size_t solution_index = 0; solution_index < 4; ++solution_index) {
            if (solution_index == best_solution_index) {
                continue;
            }
            if (triangulated_points_valid_counts[solution_index] > runner_up_count) {
                runner_up_count = triangulated_points_valid_counts[solution_index];
            }
        }

        const size_t best_count = triangulated_points_valid_counts[best_solution_index];
        const bool success = (best_count > 0) && (size_t(10) * runner_up_count <= size_t(7) * best_count);

        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                rotation[row * 3 + col] = rotations[best_solution_index][row][col];
            }
            translation[row] = translations[best_solution_index][row];
        }
        for (size_t point_index = 0; point_index < point_count; ++point_index) {
            triangulated_points[point_index * 3 + 0] = triangulated_points_set[best_solution_index][point_index * 3 + 0];
            triangulated_points[point_index * 3 + 1] = triangulated_points_set[best_solution_index][point_index * 3 + 1];
            triangulated_points[point_index * 3 + 2] = triangulated_points_set[best_solution_index][point_index * 3 + 2];
        }
        if (support_count != nullptr) {
            *support_count = best_count;
        }

        delete[] triangulated_points_set[0];
        delete[] triangulated_points_set[1];
        delete[] triangulated_points_set[2];
        delete[] triangulated_points_set[3];

        return success;
    }

    template class essential<float>;
    template class essential<double>;
}
