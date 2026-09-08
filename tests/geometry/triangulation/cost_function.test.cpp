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

#include "geometry/triangulation/cost_function.hpp"

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

static inline math::matrix<double, 3, 4> make_pose(const math::matrix<double, 3, 3>& rotation, const math::matrix<double, 3, 1>& translation) {
    math::matrix<double, 3, 4> pose;
    math::set_block(pose, 0, 0, rotation);
    math::set_block(pose, 0, 3, translation);
    return pose;
}

static inline math::matrix<double, 3, 4> make_random_pose(core::random_pcg& random) {
    const math::matrix<double, 3, 3> rotation = math::so3<double>::rotation(random.get_random(-0.5, 0.5), random.get_random(-0.5, 0.5), random.get_random(-0.5, 0.5)).get_matrix();
    const math::matrix<double, 3, 1> translation{ { random.get_random(-1.5, 1.5), random.get_random(-1.5, 1.5), random.get_random(-0.5, 0.5) } };
    return make_pose(rotation, translation);
}

static inline math::matrix<double, 3, 1> transform(const math::matrix<double, 3, 4>& pose, const math::matrix<double, 3, 1>& world) {
    return math::matrix<double, 3, 1>{ { (pose[0][0] * world[0]) + (pose[0][1] * world[1]) + (pose[0][2] * world[2]) + pose[0][3],
                                         (pose[1][0] * world[0]) + (pose[1][1] * world[1]) + (pose[1][2] * world[2]) + pose[1][3],
                                         (pose[2][0] * world[0]) + (pose[2][1] * world[1]) + (pose[2][2] * world[2]) + pose[2][3] } };
}

static inline math::matrix<double, 2, 1> normalise(const math::matrix<double, 3, 1>& camera) {
    return math::matrix<double, 2, 1>{ { camera[0] / camera[2], camera[1] / camera[2] } };
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // Exact recovery.
    {
        core::random_pcg random(0x5eed0010ull);
        size_t recovered = 0;
        for (int trial = 0; trial < 64; ++trial) {
            const math::matrix<double, 3, 4> lhs_pose = make_random_pose(random);
            const math::matrix<double, 3, 4> rhs_pose = make_random_pose(random);
            for (int sample = 0; sample < 16; ++sample) {
                const math::matrix<double, 3, 1> world{ { random.get_random(-2.0, 2.0), random.get_random(-2.0, 2.0), random.get_random(2.0, 8.0) } };
                const math::matrix<double, 3, 1> lhs_camera = transform(lhs_pose, world);
                const math::matrix<double, 3, 1> rhs_camera = transform(rhs_pose, world);
                if ((lhs_camera[2] < 0.5) || (rhs_camera[2] < 0.5)) {
                    continue;
                }
                const math::matrix<double, 3, 1> baseline = math::get_block<double, 3, 1>(lhs_pose, 0, 3) - math::get_block<double, 3, 1>(rhs_pose, 0, 3);
                if (baseline.get_length_squared() < 0.25) {
                    continue;
                }
                math::matrix<double, 3, 1> result;
                REQUIRE(geometry::triangulation::cost_function<double>::triangulate(normalise(lhs_camera), lhs_pose, normalise(rhs_camera), rhs_pose, result));
                REQUIRE(is_value_approx(result[0], world[0], 1e-7));
                REQUIRE(is_value_approx(result[1], world[1], 1e-7));
                REQUIRE(is_value_approx(result[2], world[2], 1e-7));
                math::matrix<double, 3, 1> result_from_rays;
                REQUIRE(geometry::triangulation::cost_function<double>::triangulate(lhs_camera, lhs_pose, rhs_camera * 0.5, rhs_pose, result_from_rays));
                REQUIRE(is_value_approx(result_from_rays[0], world[0], 1e-7));
                REQUIRE(is_value_approx(result_from_rays[1], world[1], 1e-7));
                REQUIRE(is_value_approx(result_from_rays[2], world[2], 1e-7));
                ++recovered;
            }
        }
        REQUIRE(recovered > 500);
    }

    // Noisy observations stay close.
    {
        core::random_pcg random(0x5eed0011ull);
        const double noise = 1e-3;
        double worst = 0.0;
        size_t checked = 0;
        for (int trial = 0; trial < 64; ++trial) {
            const math::matrix<double, 3, 4> lhs_pose = make_random_pose(random);
            const math::matrix<double, 3, 4> rhs_pose = make_random_pose(random);
            const math::matrix<double, 3, 1> baseline = math::get_block<double, 3, 1>(lhs_pose, 0, 3) - math::get_block<double, 3, 1>(rhs_pose, 0, 3);
            if (baseline.get_length_squared() < 1.0) {
                continue;
            }
            for (int sample = 0; sample < 16; ++sample) {
                const math::matrix<double, 3, 1> world{ { random.get_random(-2.0, 2.0), random.get_random(-2.0, 2.0), random.get_random(2.0, 6.0) } };
                const math::matrix<double, 3, 1> lhs_camera = transform(lhs_pose, world);
                const math::matrix<double, 3, 1> rhs_camera = transform(rhs_pose, world);
                if ((lhs_camera[2] < 0.5) || (rhs_camera[2] < 0.5)) {
                    continue;
                }
                math::matrix<double, 2, 1> lhs_point = normalise(lhs_camera);
                math::matrix<double, 2, 1> rhs_point = normalise(rhs_camera);
                lhs_point[0] += random.get_random(-noise, noise);
                lhs_point[1] += random.get_random(-noise, noise);
                rhs_point[0] += random.get_random(-noise, noise);
                rhs_point[1] += random.get_random(-noise, noise);
                math::matrix<double, 3, 1> result;
                REQUIRE(geometry::triangulation::cost_function<double>::triangulate(lhs_point, lhs_pose, rhs_point, rhs_pose, result));
                const double error = std::sqrt((result - world).get_length_squared());
                worst = math::max(worst, error);
                ++checked;
            }
        }
        REQUIRE(checked > 300);
        REQUIRE(worst < 0.2);
    }

    // Degenerate rays and coincident views.
    {
        const math::matrix<double, 3, 4> lhs_pose = { { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 } } };
        const math::matrix<double, 3, 4> rhs_pose = { { { 1, 0, 0, -2.0 }, { 0, 1, 0, 0.3 }, { 0, 0, 1, 0.5 } } };
        const math::matrix<double, 3, 1> zero_ray = math::matrix<double, 3, 1>::zero();
        const math::matrix<double, 3, 1> normal_ray = { { 0.1, 0.0, 1.0 } };
        math::matrix<double, 3, 1> result;
        REQUIRE(!geometry::triangulation::cost_function<double>::triangulate(zero_ray, lhs_pose, normal_ray, rhs_pose, result));
        REQUIRE(!geometry::triangulation::cost_function<double>::triangulate(normal_ray, lhs_pose, zero_ray, rhs_pose, result));
        REQUIRE(!geometry::triangulation::cost_function<double>::triangulate(normal_ray, lhs_pose, normal_ray, lhs_pose, result));

        const math::matrix<double, 3, 1> lhs_ray = { { 1.0, 0.5, 4.0 } };
        const math::matrix<double, 3, 1> rhs_ray = { { -1.0, 0.8, 4.5 } };
        REQUIRE(geometry::triangulation::cost_function<double>::triangulate(lhs_ray, lhs_pose, rhs_ray, rhs_pose, result));
        REQUIRE(is_value_approx(result[0], 1.0));
        REQUIRE(is_value_approx(result[1], 0.5));
        REQUIRE(is_value_approx(result[2], 4.0));
    }

    // Rays of any direction: sideways, backwards and arbitrarily scaled.
    {
        core::random_pcg random(0x5eed0013ull);
        size_t sideways = 0;
        size_t backwards = 0;
        for (int trial = 0; trial < 64; ++trial) {
            const math::matrix<double, 3, 4> lhs_pose = make_random_pose(random);
            const math::matrix<double, 3, 4> rhs_pose = make_random_pose(random);
            const math::matrix<double, 3, 1> baseline = math::get_block<double, 3, 1>(lhs_pose, 0, 3) - math::get_block<double, 3, 1>(rhs_pose, 0, 3);
            if (baseline.get_length_squared() < 1.0) {
                continue;
            }
            for (int sample = 0; sample < 16; ++sample) {
                const math::matrix<double, 3, 1> world{ { random.get_random(-6.0, 6.0), random.get_random(-6.0, 6.0), random.get_random(-6.0, 6.0) } };
                math::matrix<double, 3, 1> lhs_camera = transform(lhs_pose, world);
                math::matrix<double, 3, 1> rhs_camera = transform(rhs_pose, world);
                if ((sample % 4) == 0) {
                    lhs_camera[2] = 0.0;
                    const math::matrix<double, 3, 1> lhs_translation = math::get_block<double, 3, 1>(lhs_pose, 0, 3);
                    const math::matrix<double, 3, 1> moved = math::transpose(math::get_block<double, 3, 3>(lhs_pose, 0, 0)) * (lhs_camera - lhs_translation);
                    rhs_camera = transform(rhs_pose, moved);
                    if ((lhs_camera.get_length_squared() < 0.25) || (rhs_camera.get_length_squared() < 0.25)) {
                        continue;
                    }
                    math::matrix<double, 3, 1> result;
                    REQUIRE(geometry::triangulation::cost_function<double>::triangulate(lhs_camera * random.get_random(0.5, 2.0), lhs_pose, rhs_camera * random.get_random(-2.0, -0.5), rhs_pose, result));
                    REQUIRE(is_value_approx(result[0], moved[0], 1e-6));
                    REQUIRE(is_value_approx(result[1], moved[1], 1e-6));
                    REQUIRE(is_value_approx(result[2], moved[2], 1e-6));
                    ++sideways;
                    continue;
                }
                if ((lhs_camera.get_length_squared() < 0.25) || (rhs_camera.get_length_squared() < 0.25)) {
                    continue;
                }
                if ((lhs_camera[2] < 0.0) || (rhs_camera[2] < 0.0)) {
                    ++backwards;
                }
                math::matrix<double, 3, 1> result;
                REQUIRE(geometry::triangulation::cost_function<double>::triangulate(lhs_camera * random.get_random(0.5, 2.0), lhs_pose, rhs_camera * random.get_random(-2.0, -0.5), rhs_pose, result));
                REQUIRE(is_value_approx(result[0], world[0], 1e-6));
                REQUIRE(is_value_approx(result[1], world[1], 1e-6));
                REQUIRE(is_value_approx(result[2], world[2], 1e-6));
            }
        }
        REQUIRE(sideways > 100);
        REQUIRE(backwards > 100);
    }

    // Single precision.
    {
        core::random_pcg random(0x5eed0012ull);
        size_t recovered = 0;
        for (int trial = 0; trial < 16; ++trial) {
            const math::matrix<double, 3, 4> lhs_pose = make_random_pose(random);
            const math::matrix<double, 3, 4> rhs_pose = make_random_pose(random);
            const math::matrix<double, 3, 1> baseline = math::get_block<double, 3, 1>(lhs_pose, 0, 3) - math::get_block<double, 3, 1>(rhs_pose, 0, 3);
            if (baseline.get_length_squared() < 1.0) {
                continue;
            }
            math::matrix<float, 3, 4> lhs_pose_float;
            math::matrix<float, 3, 4> rhs_pose_float;
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 4; ++col) {
                    lhs_pose_float[row][col] = static_cast<float>(lhs_pose[row][col]);
                    rhs_pose_float[row][col] = static_cast<float>(rhs_pose[row][col]);
                }
            }
            for (int sample = 0; sample < 8; ++sample) {
                const math::matrix<double, 3, 1> world{ { random.get_random(-2.0, 2.0), random.get_random(-2.0, 2.0), random.get_random(2.0, 6.0) } };
                const math::matrix<double, 3, 1> lhs_camera = transform(lhs_pose, world);
                const math::matrix<double, 3, 1> rhs_camera = transform(rhs_pose, world);
                if ((lhs_camera[2] < 0.5) || (rhs_camera[2] < 0.5)) {
                    continue;
                }
                const math::matrix<float, 2, 1> lhs_point{ { static_cast<float>(lhs_camera[0] / lhs_camera[2]), static_cast<float>(lhs_camera[1] / lhs_camera[2]) } };
                const math::matrix<float, 2, 1> rhs_point{ { static_cast<float>(rhs_camera[0] / rhs_camera[2]), static_cast<float>(rhs_camera[1] / rhs_camera[2]) } };
                math::matrix<float, 3, 1> result;
                REQUIRE(geometry::triangulation::cost_function<float>::triangulate(lhs_point, lhs_pose_float, rhs_point, rhs_pose_float, result));
                REQUIRE(is_value_approx(static_cast<double>(result[0]), world[0], 1e-3));
                REQUIRE(is_value_approx(static_cast<double>(result[1]), world[1], 1e-3));
                REQUIRE(is_value_approx(static_cast<double>(result[2]), world[2], 1e-3));
                ++recovered;
            }
        }
        REQUIRE(recovered > 50);
    }

    return EXIT_SUCCESS;
}
