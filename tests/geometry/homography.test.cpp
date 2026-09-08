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

#include "core/random_pcg.hpp"
#include "math/lie.hpp"
#include "math/math.hpp"

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
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

static inline math::matrix<double, 3, 3> make_intrinsics(double fx, double fy, double cx, double cy) {
    return math::matrix<double, 3, 3>{ { { fx, 0.0, cx }, { 0.0, fy, cy }, { 0.0, 0.0, 1.0 } } };
}

static inline math::matrix<double, 3, 3> make_rotation(core::random_pcg& random, double maximum_angle) {
    return math::so3<double>::rotation(random.get_random(-maximum_angle, maximum_angle), random.get_random(-maximum_angle, maximum_angle), random.get_random(-maximum_angle, maximum_angle)).get_matrix();
}

static inline math::matrix<double, 3, 1> make_translation(core::random_pcg& random, double maximum) {
    return math::matrix<double, 3, 1>{ { random.get_random(-maximum, maximum), random.get_random(-maximum, maximum), random.get_random(-maximum, maximum) } };
}

static inline math::matrix<float, 3, 3> to_float(const math::matrix<double, 3, 3>& value) {
    math::matrix<float, 3, 3> result;
    for (size_t row = 0; row < 3; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            result[row][col] = static_cast<float>(value[row][col]);
        }
    }
    return result;
}

static inline math::matrix<float, 3, 1> to_float(const math::matrix<double, 3, 1>& value) {
    return math::matrix<float, 3, 1>{ { static_cast<float>(value[0]), static_cast<float>(value[1]), static_cast<float>(value[2]) } };
}

static inline bool matches_pose(const math::matrix<double, 3, 3>& rotation, const math::matrix<double, 3, 1>& translation, const math::matrix<double, 3, 3>& expected_rotation, const math::matrix<double, 3, 1>& expected_translation, double epsilon) {
    for (size_t row = 0; row < 3; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            if (!is_value_approx(rotation[row][col], expected_rotation[row][col], epsilon)) {
                return false;
            }
        }
    }
    for (size_t row = 0; row < 3; ++row) {
        if (!is_value_approx(translation[row], expected_translation[row], epsilon)) {
            return false;
        }
    }
    return true;
}

static inline math::matrix<double, 3, 1> project(const math::matrix<double, 3, 3>& intrinsics, const math::matrix<double, 3, 1>& camera_xyz) {
    const math::matrix<double, 3, 1> pixel = intrinsics * camera_xyz;
    return math::matrix<double, 3, 1>{ { pixel[0] / pixel[2], pixel[1] / pixel[2], 1.0 } };
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const math::matrix<double, 3, 3> intrinsics_lhs = make_intrinsics(525.0, 520.0, 320.0, 240.0);
    const math::matrix<double, 3, 3> intrinsics_rhs = make_intrinsics(610.0, 605.0, 310.0, 250.0);

    // Points on the plane map exactly, points off it do not.
    {
        core::random_pcg random(0x5eed0007ull);
        size_t on_plane = 0;
        size_t off_plane = 0;
        for (int trial = 0; trial < 32; ++trial) {
            const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.6);
            const math::matrix<double, 3, 1> translation = make_translation(random, 0.5);
            math::matrix<double, 3, 1> plane_normal{ { random.get_random(-0.3, 0.3), random.get_random(-0.3, 0.3), 1.0 } };
            plane_normal = plane_normal * (1.0 / std::sqrt(plane_normal.get_length_squared()));
            const double plane_distance = random.get_random(3.0, 6.0);

            math::matrix<double, 3, 3> homography;
            REQUIRE(geometry::homography<double>::from_plane(intrinsics_lhs, intrinsics_rhs, rotation, translation, plane_normal, plane_distance, homography));

            for (int sample = 0; sample < 32; ++sample) {
                const double x = random.get_random(-1.5, 1.5);
                const double y = random.get_random(-1.5, 1.5);
                const double z = (plane_distance - (plane_normal[0] * x) - (plane_normal[1] * y)) / plane_normal[2];
                const math::matrix<double, 3, 1> lhs_camera{ { x, y, z } };
                const math::matrix<double, 3, 1> rhs_camera = (rotation * lhs_camera) + translation;
                if ((lhs_camera[2] <= 0.1) || (rhs_camera[2] <= 0.1)) {
                    continue;
                }
                const math::matrix<double, 3, 1> lhs_pixel = project(intrinsics_lhs, lhs_camera);
                const math::matrix<double, 3, 1> rhs_pixel = project(intrinsics_rhs, rhs_camera);
                double mapped_x = 0.0;
                double mapped_y = 0.0;
                REQUIRE(geometry::homography<double>::transform(homography, lhs_pixel[0], lhs_pixel[1], mapped_x, mapped_y));
                REQUIRE(is_value_approx(mapped_x, rhs_pixel[0], 1e-9));
                REQUIRE(is_value_approx(mapped_y, rhs_pixel[1], 1e-9));
                ++on_plane;

                const math::matrix<double, 3, 1> lhs_camera_off{ { x, y, z + 1.0 } };
                const math::matrix<double, 3, 1> rhs_camera_off = (rotation * lhs_camera_off) + translation;
                if (rhs_camera_off[2] <= 0.1) {
                    continue;
                }
                const math::matrix<double, 3, 1> lhs_pixel_off = project(intrinsics_lhs, lhs_camera_off);
                const math::matrix<double, 3, 1> rhs_pixel_off = project(intrinsics_rhs, rhs_camera_off);
                REQUIRE(geometry::homography<double>::transform(homography, lhs_pixel_off[0], lhs_pixel_off[1], mapped_x, mapped_y));
                const double parallax = std::sqrt(((mapped_x - rhs_pixel_off[0]) * (mapped_x - rhs_pixel_off[0])) + ((mapped_y - rhs_pixel_off[1]) * (mapped_y - rhs_pixel_off[1])));
                if (std::sqrt(translation.get_length_squared()) > 0.1) {
                    REQUIRE(parallax > 1e-3);
                }
                ++off_plane;
            }
        }
        REQUIRE(on_plane > 500);
        REQUIRE(off_plane > 500);
    }

    // Decomposition recovers the pose and the plane among four candidates.
    {
        core::random_pcg random(0x5eed000dull);
        for (int trial = 0; trial < 64; ++trial) {
            const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.8);
            const math::matrix<double, 3, 1> translation = make_translation(random, 1.0);
            math::matrix<double, 3, 1> plane_normal{ { random.get_random(-0.5, 0.5), random.get_random(-0.5, 0.5), 1.0 } };
            plane_normal = plane_normal * (1.0 / std::sqrt(plane_normal.get_length_squared()));
            const double plane_distance = random.get_random(2.0, 6.0);
            const double scale = random.get_random(0.5, 2.0) * ((random.get_random(0.0, 1.0) < 0.5) ? -1.0 : 1.0);

            math::matrix<double, 3, 3> homography;
            REQUIRE(geometry::homography<double>::from_plane(intrinsics_lhs, intrinsics_rhs, rotation, translation, plane_normal, plane_distance, homography));
            homography = homography * scale;

            math::matrix<double, 3, 3> rotations[4];
            math::matrix<double, 3, 1> translations[4];
            math::matrix<double, 3, 1> normals[4];
            const size_t count = geometry::homography<double>::decompose(intrinsics_lhs, intrinsics_rhs, homography, rotations, translations, normals);
            REQUIRE(count == 4);
            const math::matrix<double, 3, 1> expected_translation = translation * (1.0 / plane_distance);
            size_t matching = 0;
            for (size_t solution = 0; solution < count; ++solution) {
                const math::matrix<double, 3, 3> orthogonality = math::transpose(rotations[solution]) * rotations[solution];
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t col = 0; col < 3; ++col) {
                        REQUIRE(is_value_approx(orthogonality[row][col], (row == col) ? 1.0 : 0.0, 1e-6));
                    }
                }
                REQUIRE(is_value_approx(normals[solution].get_length_squared(), 1.0, 1e-6));
                math::matrix<double, 3, 3> recomposed;
                REQUIRE(geometry::homography<double>::from_plane(intrinsics_lhs, intrinsics_rhs, rotations[solution], translations[solution], normals[solution], 1.0, recomposed));
                double ratio = 0.0;
                double weight = 0.0;
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t col = 0; col < 3; ++col) {
                        ratio += recomposed[row][col] * homography[row][col];
                        weight += recomposed[row][col] * recomposed[row][col];
                    }
                }
                ratio /= weight;
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t col = 0; col < 3; ++col) {
                        REQUIRE(is_value_approx(ratio * recomposed[row][col], homography[row][col], 1e-6));
                    }
                }
                bool matches = matches_pose(rotations[solution], translations[solution], rotation, expected_translation, 1e-6);
                for (size_t row = 0; row < 3; ++row) {
                    matches = matches && is_value_approx(normals[solution][row], plane_normal[row], 1e-6);
                }
                if (matches) {
                    ++matching;
                }
            }
            REQUIRE(matching == 1);
        }
    }

    // Pure rotation is independent of the plane and decomposes to one solution.
    {
        core::random_pcg random(0x5eed0008ull);
        const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.4);
        const math::matrix<double, 3, 1> normal_a{ { 0.0, 0.0, 1.0 } };
        const math::matrix<double, 3, 1> normal_b{ { 0.6, 0.0, 0.8 } };
        math::matrix<double, 3, 3> homography_a;
        math::matrix<double, 3, 3> homography_b;
        REQUIRE(geometry::homography<double>::from_plane(intrinsics_lhs, intrinsics_rhs, rotation, math::matrix<double, 3, 1>::zero(), normal_a, 2.0, homography_a));
        REQUIRE(geometry::homography<double>::from_plane(intrinsics_lhs, intrinsics_rhs, rotation, math::matrix<double, 3, 1>::zero(), normal_b, 7.0, homography_b));
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                REQUIRE(is_value_approx(homography_a[row][col], homography_b[row][col], 1e-12));
            }
        }
        math::matrix<double, 3, 3> rotations[4];
        math::matrix<double, 3, 1> translations[4];
        math::matrix<double, 3, 1> normals[4];
        REQUIRE(geometry::homography<double>::decompose(intrinsics_lhs, intrinsics_rhs, homography_a, rotations, translations, normals) == 1);
        REQUIRE(matches_pose(rotations[0], translations[0], rotation, math::matrix<double, 3, 1>::zero(), 1e-9));
    }

    // Degenerate inputs.
    {
        math::matrix<double, 3, 3> homography;
        const math::matrix<double, 3, 1> normal{ { 0.0, 0.0, 1.0 } };
        const math::matrix<double, 3, 1> translation{ { 0.1, 0.0, 0.0 } };
        REQUIRE(!geometry::homography<double>::from_plane(intrinsics_lhs, intrinsics_rhs, math::matrix<double, 3, 3>::identity(), translation, normal, 0.0, homography));
        REQUIRE(!geometry::homography<double>::from_plane(math::matrix<double, 3, 3>::zero(), intrinsics_rhs, math::matrix<double, 3, 3>::identity(), translation, normal, 1.0, homography));

        math::matrix<double, 3, 3> rotations[4];
        math::matrix<double, 3, 1> translations[4];
        math::matrix<double, 3, 1> normals[4];
        REQUIRE(geometry::homography<double>::decompose(intrinsics_lhs, intrinsics_rhs, math::matrix<double, 3, 3>::zero(), rotations, translations, normals) == 0);
        REQUIRE(geometry::homography<double>::decompose(intrinsics_lhs, math::matrix<double, 3, 3>::zero(), math::matrix<double, 3, 3>::identity(), rotations, translations, normals) == 0);

        const math::matrix<double, 3, 3> to_infinity{ { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 1.0, 0.0, 0.0 } } };
        double mapped_x = 0.0;
        double mapped_y = 0.0;
        REQUIRE(!geometry::homography<double>::transform(to_infinity, 0.0, 5.0, mapped_x, mapped_y));
        REQUIRE(geometry::homography<double>::transform(to_infinity, 2.0, 5.0, mapped_x, mapped_y));
        REQUIRE(is_value_approx(mapped_x, 1.0));
        REQUIRE(is_value_approx(mapped_y, 2.5));
    }

    // Single precision.
    {
        core::random_pcg random(0x5eed000eull);
        const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.5);
        const math::matrix<double, 3, 1> translation{ { 0.4, -0.3, 0.2 } };
        const math::matrix<double, 3, 1> plane_normal{ { 0.0, 0.0, 1.0 } };
        math::matrix<float, 3, 3> homography;
        REQUIRE(geometry::homography<float>::from_plane(to_float(intrinsics_lhs), to_float(intrinsics_rhs), to_float(rotation), to_float(translation), to_float(plane_normal), 4.0f, homography));
        math::matrix<float, 3, 3> rotations[4];
        math::matrix<float, 3, 1> translations[4];
        math::matrix<float, 3, 1> normals[4];
        REQUIRE(geometry::homography<float>::decompose(to_float(intrinsics_lhs), to_float(intrinsics_rhs), homography, rotations, translations, normals) == 4);
        size_t matching = 0;
        for (size_t solution = 0; solution < 4; ++solution) {
            bool matches = true;
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    matches = matches && is_value_approx(static_cast<double>(rotations[solution][row][col]), rotation[row][col], 1e-3);
                }
                matches = matches && is_value_approx(static_cast<double>(translations[solution][row]), translation[row] / 4.0, 1e-3);
                matches = matches && is_value_approx(static_cast<double>(normals[solution][row]), plane_normal[row], 1e-3);
            }
            if (matches) {
                ++matching;
            }
        }
        REQUIRE(matching == 1);
    }

    return EXIT_SUCCESS;
}
