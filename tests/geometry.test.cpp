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

#include "geometry.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-8) {
    if (std::isnan(lhs) && std::isnan(rhs))
        return true;
    if (std::isnan(lhs) != std::isnan(rhs))
        return false;
    if (std::isinf(lhs) != std::isinf(rhs))
        return false;
    if (std::signbit(lhs + epsilon) != std::signbit(rhs + epsilon))
        return false;
    if (std::isinf(lhs) && std::isinf(rhs))
        return true;
    return (std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon);
}

template <typename array_type>
static inline bool are_values_approx(const array_type& lhs, const array_type& rhs, unsigned long long int length, double epsilon = 1e-8) {
    for (size_t index = 0; index < length; ++index) {
        if (!is_value_approx(lhs[index], rhs[index], epsilon)) {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);
    {
        // triangulate
        constexpr static const auto create_rotation_matrix_x =
            [](
                double angle_radians
            ) -> matrix::matrix<double, 3, 3> {
            return matrix::matrix<double, 3, 3>{ { { 1, 0, 0 }, { 0, +std::cos(angle_radians), -std::sin(angle_radians) }, { 0, +std::sin(angle_radians), +std::cos(angle_radians) } } };
        };
        constexpr static const auto create_rotation_matrix_y =
            [](
                double angle_radians
            ) -> matrix::matrix<double, 3, 3> {
            return matrix::matrix<double, 3, 3>{ { { +std::cos(angle_radians), 0, +std::sin(angle_radians) }, { 0, 1, 0 }, { -std::sin(angle_radians), 0, +std::cos(angle_radians) } } };
        };
        constexpr static const auto create_rotation_matrix_z =
            [](
                double angle_radians
            ) -> matrix::matrix<double, 3, 3> {
            return matrix::matrix<double, 3, 3>{ { { +std::cos(angle_radians), -std::sin(angle_radians), 0 }, { +std::sin(angle_radians), +std::cos(angle_radians), 0 }, { 0, 0, 1 } } };
        };
        constexpr static const auto create_projection_matrix =
            [](
                const matrix::matrix<double, 3, 3> matrix_camera,
                const matrix::matrix<double, 3, 3> matrix_rotation,
                const matrix::matrix<double, 3, 1> vector_translation
            ) -> matrix::matrix<double, 3, 4> {
            matrix::matrix<double, 3, 3> matrix_camera_rotation = matrix_camera * matrix_rotation;
            matrix::matrix<double, 3, 1> vector_camera_translation = matrix_camera * vector_translation;
            return matrix::matrix<double, 3, 4>{ { { matrix_camera_rotation[0][0], matrix_camera_rotation[0][1], matrix_camera_rotation[0][2], vector_camera_translation[0] }, { matrix_camera_rotation[1][0], matrix_camera_rotation[1][1], matrix_camera_rotation[1][2], vector_camera_translation[1] }, { matrix_camera_rotation[2][0], matrix_camera_rotation[2][1], matrix_camera_rotation[2][2], vector_camera_translation[2] } } };
        };
        constexpr static const auto project_world_to_camera =
            [](
                const matrix::matrix<double, 3, 3> matrix_rotation,
                const matrix::matrix<double, 3, 1> vector_translation,
                const matrix::matrix<double, 3, 1> world_xyz
            ) -> matrix::matrix<double, 3, 1> {
            const matrix::matrix<double, 3, 1> camera_xyz = matrix_rotation * world_xyz;
            return matrix::matrix<double, 3, 1>{ { camera_xyz[0] + vector_translation[0], camera_xyz[1] + vector_translation[1], camera_xyz[2] + vector_translation[2] } };
        };
        constexpr static const auto project_camera_to_image =
            [](
                const matrix::matrix<double, 3, 3> matrix_camera,
                const matrix::matrix<double, 3, 1> camera_xyz
            ) -> matrix::matrix<double, 2, 1> {
            const matrix::matrix<double, 3, 1> point_xyz = matrix_camera * camera_xyz;
            return matrix::matrix<double, 2, 1>{ { point_xyz[0] / point_xyz[2], point_xyz[1] / point_xyz[2] } };
        };
        constexpr static const auto reprojection_error =
            [](
                const matrix::matrix<double, 2, 1> lhs_point_xy,
                const matrix::matrix<double, 3, 4> lhs_pose,
                const matrix::matrix<double, 2, 1> rhs_point_xy,
                const matrix::matrix<double, 3, 4> rhs_pose,
                const matrix::matrix<double, 3, 1> world_xyz
            ) -> double {
            const matrix::matrix<double, 4, 1> world_point_homogeneous{ { world_xyz[0], world_xyz[1], world_xyz[2], 1 } };
            const matrix::matrix<double, 3, 1> lhs_projected_point_homogeneous = lhs_pose * world_point_homogeneous;
            const matrix::matrix<double, 3, 1> rhs_projected_point_homogeneous = rhs_pose * world_point_homogeneous;
            const matrix::matrix<double, 2, 1> lhs_projected_point = { { lhs_projected_point_homogeneous[0] / lhs_projected_point_homogeneous[2], lhs_projected_point_homogeneous[1] / lhs_projected_point_homogeneous[2] } };
            const matrix::matrix<double, 2, 1> rhs_projected_point = { { rhs_projected_point_homogeneous[0] / rhs_projected_point_homogeneous[2], rhs_projected_point_homogeneous[1] / rhs_projected_point_homogeneous[2] } };
            const matrix::matrix<double, 2, 1> lhs_distance = { { lhs_projected_point[0] - lhs_point_xy[0], lhs_projected_point[1] - lhs_point_xy[1] } };
            const matrix::matrix<double, 2, 1> rhs_distance = { { rhs_projected_point[0] - rhs_point_xy[0], rhs_projected_point[1] - rhs_point_xy[1] } };
            const double lhs_l2_distance = lhs_distance.get_length_squared();
            const double rhs_l2_distance = rhs_distance.get_length_squared();
            return (lhs_l2_distance + rhs_l2_distance);
        };
        constexpr static const auto test_triangulation =
            [](
                const matrix::matrix<double, 3, 3> matrix_camera,
                const matrix::matrix<double, 3, 3> lhs_rotation,
                const matrix::matrix<double, 3, 1> lhs_translation,
                const matrix::matrix<double, 3, 3> rhs_rotation,
                const matrix::matrix<double, 3, 1> rhs_translation,
                const matrix::matrix<double, 3, 1> world_point_expected
            ) {
                const matrix::matrix<double, 3, 4> lhs_pose = create_projection_matrix(matrix_camera, lhs_rotation, lhs_translation);
                const matrix::matrix<double, 3, 4> rhs_pose = create_projection_matrix(matrix_camera, rhs_rotation, rhs_translation);
                const matrix::matrix<double, 3, 1> lhs_camera_point = project_world_to_camera(lhs_rotation, lhs_translation, world_point_expected);
                const matrix::matrix<double, 3, 1> rhs_camera_point = project_world_to_camera(rhs_rotation, rhs_translation, world_point_expected);
                const matrix::matrix<double, 2, 1> lhs_image_point = project_camera_to_image(matrix_camera, lhs_camera_point);
                const matrix::matrix<double, 2, 1> rhs_image_point = project_camera_to_image(matrix_camera, rhs_camera_point);
                matrix::matrix<double, 3, 1> world_point_triangulated;
                REQUIRE(geometry::triangulate(lhs_image_point, lhs_pose, rhs_image_point, rhs_pose, world_point_triangulated));
                REQUIRE(are_values_approx<const double*>(world_point_triangulated.data(), world_point_expected.data(), 3));
                const double error = reprojection_error(lhs_image_point, lhs_pose, rhs_image_point, rhs_pose, world_point_triangulated);
                REQUIRE(error < 1e-9);
            };

        const matrix::matrix<double, 3, 3> matrix_camera = { { { 1.0, 0.0, 0.5 }, { 0.0, 1.0, 0.5 }, { 0.0, 0.0, 1.0 } } };
        const matrix::matrix<double, 3, 3> lhs_rotation_x = create_rotation_matrix_x(M_PI * +0.33);
        const matrix::matrix<double, 3, 3> lhs_rotation_z = create_rotation_matrix_z(M_PI * +0.15);
        const matrix::matrix<double, 3, 1> lhs_translation = { { 0.9, 1.4, 5.2 } };
        const matrix::matrix<double, 3, 3> lhs_rotation = lhs_rotation_x * lhs_rotation_z;
        const matrix::matrix<double, 3, 3> rhs_rotation_z = create_rotation_matrix_z(M_PI * -0.12);
        const matrix::matrix<double, 3, 3> rhs_rotation_y = create_rotation_matrix_y(M_PI * +0.45);
        const matrix::matrix<double, 3, 1> rhs_translation = { { -3.3, -0.1, 7.0 } };
        const matrix::matrix<double, 3, 3> rhs_rotation = rhs_rotation_z * rhs_rotation_y;
        constexpr static const int world_points_count = 5;
        const matrix::matrix<double, 3, 1> world_points_expected[world_points_count] = {
            { { 0.0, 0.0, 0.0 } },
            { { -1.0, 0.0, 1.0 } },
            { { -5.0, 5.0, 0.0 } },
            { { 1.0, 2.0, -1.0 } },
            { { 0.0, 0.0, 50.0 } }
        };
        for (int i = 0; i < world_points_count; ++i) {
            test_triangulation(matrix_camera, lhs_rotation, lhs_translation, rhs_rotation, rhs_translation, world_points_expected[i]);
        }
    }

    return EXIT_SUCCESS;
}