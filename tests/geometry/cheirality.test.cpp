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

#include "geometry/cheirality.hpp"

#include "core/random_pcg.hpp"
#include "geometry/triangulation/cost_function.hpp"
#include "geometry/triangulation/direct.hpp"
#include "geometry/triangulation/linear_least_squares.hpp"
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

static inline math::matrix<double, 3, 4> make_pose(const math::matrix<double, 3, 3>& rotation, const math::matrix<double, 3, 1>& translation) {
    math::matrix<double, 3, 4> pose;
    math::set_block(pose, 0, 0, rotation);
    math::set_block(pose, 0, 3, translation);
    return pose;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // In front means along the ray, whatever its direction.
    {
        const math::matrix<double, 3, 4> identity = make_pose(math::matrix<double, 3, 3>::identity(), math::matrix<double, 3, 1>::zero());
        const math::matrix<double, 3, 1> forward{ { 0.1, -0.2, 1.0 } };
        REQUIRE(geometry::cheirality<double>::in_front(identity, forward, math::matrix<double, 3, 1>{ { 0.2, -0.4, 2.0 } }));
        REQUIRE(!geometry::cheirality<double>::in_front(identity, forward, math::matrix<double, 3, 1>{ { -0.2, 0.4, -2.0 } }));
        const math::matrix<double, 3, 1> backward{ { 0.0, 0.0, -1.0 } };
        REQUIRE(geometry::cheirality<double>::in_front(identity, backward, math::matrix<double, 3, 1>{ { 0.0, 0.0, -3.0 } }));
        REQUIRE(!geometry::cheirality<double>::in_front(identity, backward, math::matrix<double, 3, 1>{ { 0.0, 0.0, 3.0 } }));
        const math::matrix<double, 3, 1> sideways{ { 1.0, 0.0, 0.0 } };
        REQUIRE(geometry::cheirality<double>::in_front(identity, sideways, math::matrix<double, 3, 1>{ { 5.0, 0.0, 0.0 } }));
        REQUIRE(!geometry::cheirality<double>::in_front(identity, sideways, math::matrix<double, 3, 1>{ { -5.0, 0.0, 0.0 } }));
        const math::matrix<double, 3, 4> moved = make_pose(math::matrix<double, 3, 3>::identity(), math::matrix<double, 3, 1>{ { 0.0, 0.0, -4.0 } });
        REQUIRE(!geometry::cheirality<double>::in_front(moved, forward, math::matrix<double, 3, 1>{ { 0.0, 0.0, 3.0 } }));
        REQUIRE(geometry::cheirality<double>::in_front(moved, forward, math::matrix<double, 3, 1>{ { 0.0, 0.0, 5.0 } }));
    }

    // Triangulation agrees with the true point and the sign of the rays decides the verdict.
    {
        core::random_pcg random(0x5eed0031ull);
        size_t accepted = 0;
        size_t rejected = 0;
        for (int trial = 0; trial < 64; ++trial) {
            const math::matrix<double, 3, 4> lhs_pose = make_pose(make_rotation(random, 0.5), math::matrix<double, 3, 1>{ { random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0), random.get_random(-0.5, 0.5) } });
            const math::matrix<double, 3, 4> rhs_pose = make_pose(make_rotation(random, 0.5), math::matrix<double, 3, 1>{ { random.get_random(-1.0, 1.0) + 2.0, random.get_random(-1.0, 1.0), random.get_random(-0.5, 0.5) } });
            const math::matrix<double, 3, 1> world{ { random.get_random(-2.0, 2.0), random.get_random(-2.0, 2.0), random.get_random(3.0, 8.0) } };
            math::matrix<double, 3, 1> lhs_ray = (math::get_block<double, 3, 3>(lhs_pose, 0, 0) * world) + math::get_block<double, 3, 1>(lhs_pose, 0, 3);
            math::matrix<double, 3, 1> rhs_ray = (math::get_block<double, 3, 3>(rhs_pose, 0, 0) * world) + math::get_block<double, 3, 1>(rhs_pose, 0, 3);
            math::matrix<double, 3, 1> point;
            REQUIRE(geometry::cheirality<double>::triangulate<geometry::triangulation::linear_least_squares<double>>(lhs_ray, lhs_pose, rhs_ray, rhs_pose, point));
            REQUIRE(is_value_approx(point[0], world[0], 1e-6));
            REQUIRE(is_value_approx(point[1], world[1], 1e-6));
            REQUIRE(is_value_approx(point[2], world[2], 1e-6));
            math::matrix<double, 3, 1> point_direct;
            REQUIRE(geometry::cheirality<double>::triangulate<geometry::triangulation::direct<double>>(lhs_ray, lhs_pose, rhs_ray, rhs_pose, point_direct));
            REQUIRE(is_value_approx(point_direct[2], world[2], 1e-6));
            math::matrix<double, 3, 1> point_cost;
            REQUIRE(geometry::cheirality<double>::triangulate<geometry::triangulation::cost_function<double>>(lhs_ray, lhs_pose, rhs_ray, rhs_pose, point_cost));
            REQUIRE(is_value_approx(point_cost[2], world[2], 1e-6));
            ++accepted;
            rhs_ray = -rhs_ray;
            REQUIRE(!geometry::cheirality<double>::triangulate<geometry::triangulation::linear_least_squares<double>>(lhs_ray, lhs_pose, rhs_ray, rhs_pose, point));
            REQUIRE(is_value_approx(point[2], world[2], 1e-6));
            REQUIRE(!geometry::cheirality<double>::triangulate<geometry::triangulation::direct<double>>(lhs_ray, lhs_pose, rhs_ray, rhs_pose, point_direct));
            ++rejected;
        }
        REQUIRE(accepted == 64);
        REQUIRE(rejected == 64);
    }

    // Parallel rays cannot be triangulated.
    {
        const math::matrix<double, 3, 4> lhs_pose = make_pose(math::matrix<double, 3, 3>::identity(), math::matrix<double, 3, 1>::zero());
        const math::matrix<double, 3, 4> rhs_pose = make_pose(math::matrix<double, 3, 3>::identity(), math::matrix<double, 3, 1>{ { 1.0, 0.0, 0.0 } });
        const math::matrix<double, 3, 1> ray{ { 0.0, 0.0, 1.0 } };
        math::matrix<double, 3, 1> point{ { 7.0, 7.0, 7.0 } };
        REQUIRE(!geometry::cheirality<double>::triangulate<geometry::triangulation::linear_least_squares<double>>(ray, lhs_pose, ray, rhs_pose, point));
    }

    return EXIT_SUCCESS;
}
