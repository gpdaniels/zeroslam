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

#include "geometry/essential.hpp"

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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const math::matrix<double, 3, 3> intrinsics_lhs = make_intrinsics(525.0, 520.0, 320.0, 240.0);
    const math::matrix<double, 3, 3> intrinsics_rhs = make_intrinsics(610.0, 605.0, 310.0, 250.0);

    // Epipolar constraint on normalised coordinates.
    {
        core::random_pcg random(0x5eed0005ull);
        for (int trial = 0; trial < 32; ++trial) {
            const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.6);
            const math::matrix<double, 3, 1> translation = make_translation(random, 0.5);
            math::matrix<double, 3, 3> essential;
            geometry::essential<double>::from_poses(rotation, translation, essential);

            double essential_norm = 0.0;
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    essential_norm += essential[row][col] * essential[row][col];
                }
            }
            essential_norm = std::sqrt(essential_norm);
            REQUIRE(is_value_approx(essential_norm, std::sqrt(2.0 * translation.get_length_squared()), 1e-12));

            for (int sample = 0; sample < 32; ++sample) {
                const math::matrix<double, 3, 1> lhs_camera{ { random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0), random.get_random(2.0, 6.0) } };
                const math::matrix<double, 3, 1> rhs_camera = (rotation * lhs_camera) + translation;
                if (rhs_camera[2] <= 0.1) {
                    continue;
                }
                const math::matrix<double, 3, 1> lhs_normalised{ { lhs_camera[0] / lhs_camera[2], lhs_camera[1] / lhs_camera[2], 1.0 } };
                const math::matrix<double, 3, 1> rhs_normalised{ { rhs_camera[0] / rhs_camera[2], rhs_camera[1] / rhs_camera[2], 1.0 } };
                double constraint = 0.0;
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t col = 0; col < 3; ++col) {
                        constraint += rhs_normalised[row] * essential[row][col] * lhs_normalised[col];
                    }
                }
                REQUIRE(std::abs(constraint) < (1e-12 * essential_norm * std::sqrt(lhs_normalised.get_length_squared()) * std::sqrt(rhs_normalised.get_length_squared())));
            }
        }
    }

    // Round trip through the fundamental matrix.
    {
        core::random_pcg random(0x5eed0006ull);
        math::matrix<double, 3, 3> intrinsics_lhs_inverse;
        math::matrix<double, 3, 3> intrinsics_rhs_inverse;
        REQUIRE(math::invert(intrinsics_lhs, intrinsics_lhs_inverse));
        REQUIRE(math::invert(intrinsics_rhs, intrinsics_rhs_inverse));
        for (int trial = 0; trial < 32; ++trial) {
            const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.6);
            const math::matrix<double, 3, 1> translation = make_translation(random, 0.5);
            math::matrix<double, 3, 3> essential;
            geometry::essential<double>::from_poses(rotation, translation, essential);
            const math::matrix<double, 3, 3> fundamental = math::transpose(intrinsics_rhs_inverse) * essential * intrinsics_lhs_inverse;
            math::matrix<double, 3, 3> from_fundamental;
            geometry::essential<double>::from_fundamental(intrinsics_lhs, intrinsics_rhs, fundamental, from_fundamental);
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    REQUIRE(is_value_approx(from_fundamental[row][col], essential[row][col], 1e-9));
                }
            }
        }
    }

    // Decomposition recovers the pose among four candidates.
    {
        core::random_pcg random(0x5eed000aull);
        for (int trial = 0; trial < 64; ++trial) {
            const math::matrix<double, 3, 3> rotation = make_rotation(random, 1.0);
            math::matrix<double, 3, 1> translation = make_translation(random, 1.0);
            translation = translation * (1.0 / std::sqrt(translation.get_length_squared()));
            math::matrix<double, 3, 3> essential;
            geometry::essential<double>::from_poses(rotation, translation, essential);

            math::matrix<double, 3, 3> rotations[4];
            math::matrix<double, 3, 1> translations[4];
            REQUIRE(geometry::essential<double>::decompose(essential, rotations, translations) == 4);
            size_t matching = 0;
            for (size_t solution = 0; solution < 4; ++solution) {
                REQUIRE(is_value_approx(translations[solution].get_length_squared(), 1.0, 1e-9));
                const math::matrix<double, 3, 3> orthogonality = math::transpose(rotations[solution]) * rotations[solution];
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t col = 0; col < 3; ++col) {
                        REQUIRE(is_value_approx(orthogonality[row][col], (row == col) ? 1.0 : 0.0, 1e-9));
                    }
                }
                math::matrix<double, 3, 3> recomposed;
                geometry::essential<double>::from_poses(rotations[solution], translations[solution], recomposed);
                double sign = 0.0;
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t col = 0; col < 3; ++col) {
                        sign += recomposed[row][col] * essential[row][col];
                    }
                }
                sign = (sign < 0.0) ? -1.0 : 1.0;
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t col = 0; col < 3; ++col) {
                        REQUIRE(is_value_approx(sign * recomposed[row][col], essential[row][col], 1e-9));
                    }
                }
                if (matches_pose(rotations[solution], translations[solution], rotation, translation, 1e-8)) {
                    ++matching;
                }
            }
            REQUIRE(matching == 1);
            REQUIRE(translations[1] == -translations[0]);
            REQUIRE(translations[3] == -translations[2]);
            REQUIRE(rotations[1] == rotations[0]);
            REQUIRE(rotations[3] == rotations[2]);
        }
        math::matrix<double, 3, 3> rotations[4];
        math::matrix<double, 3, 1> translations[4];
        REQUIRE(geometry::essential<double>::decompose(math::matrix<double, 3, 3>::zero(), rotations, translations) == 0);
    }

    // Axis aligned translations with a zero row in the matrix.
    {
        for (size_t axis = 0; axis < 3; ++axis) {
            math::matrix<double, 3, 1> translation = math::matrix<double, 3, 1>::zero();
            translation[axis] = 1.0;
            math::matrix<double, 3, 3> essential;
            geometry::essential<double>::from_poses(math::matrix<double, 3, 3>::identity(), translation, essential);
            math::matrix<double, 3, 3> rotations[4];
            math::matrix<double, 3, 1> translations[4];
            REQUIRE(geometry::essential<double>::decompose(essential, rotations, translations) == 4);
            size_t matching = 0;
            for (size_t solution = 0; solution < 4; ++solution) {
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t col = 0; col < 3; ++col) {
                        REQUIRE(std::isfinite(rotations[solution][row][col]));
                    }
                    REQUIRE(std::isfinite(translations[solution][row]));
                }
                if (matches_pose(rotations[solution], translations[solution], math::matrix<double, 3, 3>::identity(), translation, 1e-9)) {
                    ++matching;
                }
            }
            REQUIRE(matching == 1);
        }
    }

    // Epipoles are the translation directions.
    {
        core::random_pcg random(0x5eed000bull);
        for (int trial = 0; trial < 32; ++trial) {
            const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.6);
            const math::matrix<double, 3, 1> translation = make_translation(random, 0.5);
            math::matrix<double, 3, 3> essential;
            geometry::essential<double>::from_poses(rotation, translation, essential);

            double epipole_x = 0.0;
            double epipole_y = 0.0;
            double epipole_z = 0.0;
            REQUIRE(geometry::essential<double>::epipole_rhs(essential, epipole_x, epipole_y, epipole_z));
            const math::matrix<double, 3, 1> epipole_rhs{ { epipole_x, epipole_y, epipole_z } };
            const math::matrix<double, 3, 1> cross_rhs{ { (epipole_rhs[1] * translation[2]) - (epipole_rhs[2] * translation[1]), (epipole_rhs[2] * translation[0]) - (epipole_rhs[0] * translation[2]), (epipole_rhs[0] * translation[1]) - (epipole_rhs[1] * translation[0]) } };
            REQUIRE(std::sqrt(cross_rhs.get_length_squared()) < (1e-9 * std::sqrt(epipole_rhs.get_length_squared()) * std::sqrt(translation.get_length_squared())));

            REQUIRE(geometry::essential<double>::epipole_lhs(essential, epipole_x, epipole_y, epipole_z));
            const math::matrix<double, 3, 1> epipole_lhs{ { epipole_x, epipole_y, epipole_z } };
            const math::matrix<double, 3, 1> expected_lhs = math::transpose(rotation) * translation;
            const math::matrix<double, 3, 1> cross_lhs{ { (epipole_lhs[1] * expected_lhs[2]) - (epipole_lhs[2] * expected_lhs[1]), (epipole_lhs[2] * expected_lhs[0]) - (epipole_lhs[0] * expected_lhs[2]), (epipole_lhs[0] * expected_lhs[1]) - (epipole_lhs[1] * expected_lhs[0]) } };
            REQUIRE(std::sqrt(cross_lhs.get_length_squared()) < (1e-9 * std::sqrt(epipole_lhs.get_length_squared()) * std::sqrt(expected_lhs.get_length_squared())));
        }
    }

    // Zero translation gives a zero matrix.
    {
        math::matrix<double, 3, 3> essential;
        geometry::essential<double>::from_poses(math::matrix<double, 3, 3>::identity(), math::matrix<double, 3, 1>::zero(), essential);
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                REQUIRE(essential[row][col] == 0.0);
            }
        }
    }

    // Single precision.
    {
        core::random_pcg random(0x5eed000cull);
        const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.6);
        const math::matrix<double, 3, 1> translation{ { 0.6, 0.0, 0.8 } };
        math::matrix<float, 3, 3> essential;
        geometry::essential<float>::from_poses(to_float(rotation), to_float(translation), essential);
        math::matrix<float, 3, 3> rotations[4];
        math::matrix<float, 3, 1> translations[4];
        REQUIRE(geometry::essential<float>::decompose(essential, rotations, translations) == 4);
        size_t matching = 0;
        for (size_t solution = 0; solution < 4; ++solution) {
            bool matches = true;
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    matches = matches && is_value_approx(static_cast<double>(rotations[solution][row][col]), rotation[row][col], 1e-4);
                }
                matches = matches && is_value_approx(static_cast<double>(translations[solution][row]), translation[row], 1e-4);
            }
            if (matches) {
                ++matching;
            }
        }
        REQUIRE(matching == 1);
    }

    return EXIT_SUCCESS;
}
