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

#include "geometry/sampson.hpp"

#include "core/random_pcg.hpp"
#include "geometry/essential.hpp"
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

static inline math::matrix<double, 3, 3> make_rotation(core::random_pcg& random, double maximum_angle) {
    return math::so3<double>::rotation(random.get_random(-maximum_angle, maximum_angle), random.get_random(-maximum_angle, maximum_angle), random.get_random(-maximum_angle, maximum_angle)).get_matrix();
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    core::random_pcg random(0x5eed0030ull);
    for (int trial = 0; trial < 32; ++trial) {
        const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.5);
        const math::matrix<double, 3, 1> translation{ { random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0), random.get_random(0.5, 1.5) } };
        math::matrix<double, 3, 3> essential;
        geometry::essential<double>::from_poses(rotation, translation, essential);

        const math::matrix<double, 3, 1> world{ { random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0), random.get_random(3.0, 8.0) } };
        const math::matrix<double, 3, 1> camera = (rotation * world) + translation;
        const math::matrix<double, 2, 1> lhs{ { world[0] / world[2], world[1] / world[2] } };
        const math::matrix<double, 2, 1> rhs{ { camera[0] / camera[2], camera[1] / camera[2] } };

        // Zero on a true correspondence.
        REQUIRE(geometry::sampson<double>::distance_squared(essential, lhs, rhs) < 1e-20);

        // A small displacement across the epipolar line gives approximately its squared size, and grows quadratically.
        const math::matrix<double, 3, 1> line = essential * math::matrix<double, 3, 1>{ { lhs[0], lhs[1], 1.0 } };
        const math::matrix<double, 2, 1> normal{ { line[0], line[1] } };
        const double normal_length = std::sqrt(normal.get_length_squared());
        const math::matrix<double, 2, 1> rhs_near = rhs + (normal * (1e-4 / normal_length));
        const math::matrix<double, 2, 1> rhs_far = rhs + (normal * (2e-4 / normal_length));
        const double distance_near = geometry::sampson<double>::distance_squared(essential, lhs, rhs_near);
        const double distance_far = geometry::sampson<double>::distance_squared(essential, lhs, rhs_far);
        REQUIRE(distance_near > 1e-9);
        REQUIRE(distance_near < 4e-8);
        REQUIRE(is_value_approx(distance_far / distance_near, 4.0, 1e-2));

        // Symmetric under swapping the views with the transposed matrix.
        REQUIRE(is_value_approx(geometry::sampson<double>::distance_squared(math::transpose(essential), rhs_near, lhs), distance_near, 1e-9));
    }

    // Single precision.
    {
        const math::matrix<float, 3, 3> essential{ { { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 1.0f, 0.0f } } };
        const math::matrix<float, 2, 1> lhs{ { 0.2f, 0.1f } };
        const math::matrix<float, 2, 1> rhs{ { 0.7f, 0.1f } };
        REQUIRE(geometry::sampson<float>::distance_squared(essential, lhs, rhs) < 1e-12f);
        const math::matrix<float, 2, 1> rhs_off{ { 0.7f, 0.3f } };
        REQUIRE(geometry::sampson<float>::distance_squared(essential, lhs, rhs_off) > 1e-3f);
    }

    return EXIT_SUCCESS;
}
