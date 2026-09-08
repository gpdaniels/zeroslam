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

#include "geometry/fundamental.hpp"

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

    // Epipolar constraint.
    {
        core::random_pcg random(0x5eed0003ull);
        for (int trial = 0; trial < 32; ++trial) {
            const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.6);
            const math::matrix<double, 3, 1> translation = make_translation(random, 0.5);
            math::matrix<double, 3, 3> fundamental;
            REQUIRE(geometry::fundamental<double>::from_poses(intrinsics_lhs, intrinsics_rhs, rotation, translation, fundamental));

            double fundamental_norm = 0.0;
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    fundamental_norm += fundamental[row][col] * fundamental[row][col];
                }
            }
            fundamental_norm = std::sqrt(fundamental_norm);

            for (int sample = 0; sample < 32; ++sample) {
                const math::matrix<double, 3, 1> lhs_camera{ { random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0), random.get_random(2.0, 6.0) } };
                const math::matrix<double, 3, 1> rhs_camera = (rotation * lhs_camera) + translation;
                if (rhs_camera[2] <= 0.1) {
                    continue;
                }
                const math::matrix<double, 3, 1> lhs_pixel = project(intrinsics_lhs, lhs_camera);
                const math::matrix<double, 3, 1> rhs_pixel = project(intrinsics_rhs, rhs_camera);
                double constraint = 0.0;
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t col = 0; col < 3; ++col) {
                        constraint += rhs_pixel[row] * fundamental[row][col] * lhs_pixel[col];
                    }
                }
                REQUIRE(std::abs(constraint) < (1e-9 * fundamental_norm * std::sqrt(lhs_pixel.get_length_squared()) * std::sqrt(rhs_pixel.get_length_squared())));
            }
        }
    }

    // Round trip through the essential matrix and decomposition back to the pose.
    {
        core::random_pcg random(0x5eed0004ull);
        for (int trial = 0; trial < 32; ++trial) {
            const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.6);
            math::matrix<double, 3, 1> translation = make_translation(random, 0.5);
            translation = translation * (1.0 / std::sqrt(translation.get_length_squared()));
            math::matrix<double, 3, 3> fundamental;
            REQUIRE(geometry::fundamental<double>::from_poses(intrinsics_lhs, intrinsics_rhs, rotation, translation, fundamental));

            const math::matrix<double, 3, 3> translation_skew{ { { 0.0, -translation[2], +translation[1] }, { +translation[2], 0.0, -translation[0] }, { -translation[1], +translation[0], 0.0 } } };
            const math::matrix<double, 3, 3> essential = translation_skew * rotation;
            math::matrix<double, 3, 3> from_essential;
            REQUIRE(geometry::fundamental<double>::from_essential(intrinsics_lhs, intrinsics_rhs, essential, from_essential));
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    REQUIRE(is_value_approx(from_essential[row][col], fundamental[row][col], 1e-12));
                }
            }

            math::matrix<double, 3, 3> rotations[4];
            math::matrix<double, 3, 1> translations[4];
            REQUIRE(geometry::fundamental<double>::decompose(intrinsics_lhs, intrinsics_rhs, fundamental, rotations, translations) == 4);
            size_t matching = 0;
            for (size_t solution = 0; solution < 4; ++solution) {
                if (matches_pose(rotations[solution], translations[solution], rotation, translation, 1e-6)) {
                    ++matching;
                }
            }
            REQUIRE(matching == 1);
        }
    }

    // Epipoles against the projected camera centres.
    {
        core::random_pcg random(0x5eed0001ull);
        for (int trial = 0; trial < 32; ++trial) {
            const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.6);
            const math::matrix<double, 3, 1> translation{ { random.get_random(-0.5, 0.5), random.get_random(-0.5, 0.5), random.get_random(0.2, 0.8) } };
            math::matrix<double, 3, 3> fundamental;
            REQUIRE(geometry::fundamental<double>::from_poses(intrinsics_lhs, intrinsics_rhs, rotation, translation, fundamental));

            double epipole_x = 0.0;
            double epipole_y = 0.0;
            double epipole_z = 0.0;
            REQUIRE(geometry::fundamental<double>::epipole_rhs(fundamental, epipole_x, epipole_y, epipole_z));
            const math::matrix<double, 3, 1> epipole_rhs{ { epipole_x, epipole_y, epipole_z } };
            const math::matrix<double, 3, 1> residual_rhs = math::transpose(fundamental) * epipole_rhs;
            REQUIRE(std::sqrt(residual_rhs.get_length_squared()) < (1e-12 * std::sqrt(epipole_rhs.get_length_squared())));
            const math::matrix<double, 3, 1> expected_rhs = intrinsics_rhs * translation;
            const math::matrix<double, 3, 1> cross_rhs{ { (epipole_rhs[1] * expected_rhs[2]) - (epipole_rhs[2] * expected_rhs[1]), (epipole_rhs[2] * expected_rhs[0]) - (epipole_rhs[0] * expected_rhs[2]), (epipole_rhs[0] * expected_rhs[1]) - (epipole_rhs[1] * expected_rhs[0]) } };
            REQUIRE(std::sqrt(cross_rhs.get_length_squared()) < (1e-9 * std::sqrt(epipole_rhs.get_length_squared()) * std::sqrt(expected_rhs.get_length_squared())));

            REQUIRE(geometry::fundamental<double>::epipole_lhs(fundamental, epipole_x, epipole_y, epipole_z));
            const math::matrix<double, 3, 1> epipole_lhs{ { epipole_x, epipole_y, epipole_z } };
            const math::matrix<double, 3, 1> residual_lhs = fundamental * epipole_lhs;
            REQUIRE(std::sqrt(residual_lhs.get_length_squared()) < (1e-12 * std::sqrt(epipole_lhs.get_length_squared())));
            const math::matrix<double, 3, 1> expected_lhs = intrinsics_lhs * (math::transpose(rotation) * translation);
            const math::matrix<double, 3, 1> cross_lhs{ { (epipole_lhs[1] * expected_lhs[2]) - (epipole_lhs[2] * expected_lhs[1]), (epipole_lhs[2] * expected_lhs[0]) - (epipole_lhs[0] * expected_lhs[2]), (epipole_lhs[0] * expected_lhs[1]) - (epipole_lhs[1] * expected_lhs[0]) } };
            REQUIRE(std::sqrt(cross_lhs.get_length_squared()) < (1e-9 * std::sqrt(epipole_lhs.get_length_squared()) * std::sqrt(expected_lhs.get_length_squared())));
        }
    }

    // Degenerate inputs.
    {
        double epipole_x = 0.0;
        double epipole_y = 0.0;
        double epipole_z = 0.0;
        REQUIRE(!geometry::fundamental<double>::epipole_rhs(math::matrix<double, 3, 3>::zero(), epipole_x, epipole_y, epipole_z));
        REQUIRE(!geometry::fundamental<double>::epipole_lhs(math::matrix<double, 3, 3>::zero(), epipole_x, epipole_y, epipole_z));

        const double lhs[3] = { 0.3, -0.7, 1.1 };
        const double rhs[3] = { 2.0, 0.5, -1.5 };
        math::matrix<double, 3, 3> rank_one;
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                rank_one[row][col] = lhs[row] * rhs[col];
            }
        }
        REQUIRE(!geometry::fundamental<double>::epipole_rhs(rank_one, epipole_x, epipole_y, epipole_z));
        REQUIRE(!geometry::fundamental<double>::epipole_lhs(rank_one, epipole_x, epipole_y, epipole_z));

        math::matrix<double, 3, 3> fundamental;
        const math::matrix<double, 3, 1> translation{ { 0.1, 0.2, 0.3 } };
        REQUIRE(!geometry::fundamental<double>::from_poses(math::matrix<double, 3, 3>::zero(), intrinsics_rhs, math::matrix<double, 3, 3>::identity(), translation, fundamental));
        REQUIRE(!geometry::fundamental<double>::from_poses(intrinsics_lhs, math::matrix<double, 3, 3>::zero(), math::matrix<double, 3, 3>::identity(), translation, fundamental));
        REQUIRE(!geometry::fundamental<double>::from_essential(math::matrix<double, 3, 3>::zero(), intrinsics_rhs, math::matrix<double, 3, 3>::identity(), fundamental));
    }

    // Single precision.
    {
        core::random_pcg random(0x5eed0009ull);
        const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.6);
        const math::matrix<double, 3, 1> translation{ { 0.3, -0.2, 0.4 } };
        math::matrix<double, 3, 3> fundamental;
        REQUIRE(geometry::fundamental<double>::from_poses(intrinsics_lhs, intrinsics_rhs, rotation, translation, fundamental));
        math::matrix<float, 3, 3> fundamental_float;
        REQUIRE(geometry::fundamental<float>::from_poses(to_float(intrinsics_lhs), to_float(intrinsics_rhs), to_float(rotation), to_float(translation), fundamental_float));
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                REQUIRE(is_value_approx(static_cast<double>(fundamental_float[row][col]), fundamental[row][col], 1e-4));
            }
        }
        float epipole_x = 0.0f;
        float epipole_y = 0.0f;
        float epipole_z = 0.0f;
        REQUIRE(geometry::fundamental<float>::epipole_rhs(fundamental_float, epipole_x, epipole_y, epipole_z));
        REQUIRE(!geometry::fundamental<float>::epipole_rhs(math::matrix<float, 3, 3>::zero(), epipole_x, epipole_y, epipole_z));
    }

    return EXIT_SUCCESS;
}
