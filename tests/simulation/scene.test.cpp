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

#include "simulation/scene.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        simulation::random_stream a(7);
        simulation::random_stream b(7);
        for (int i = 0; i < 1000; ++i) {
            const double value_a = a.uniform();
            const double value_b = b.uniform();
            REQUIRE(value_a == value_b);
            REQUIRE(value_a >= 0.0);
            REQUIRE(value_a < 1.0);
        }
        simulation::random_stream c(8);
        bool any_different = false;
        for (int i = 0; i < 16; ++i) {
            any_different = any_different || (a.next() != c.next());
        }
        REQUIRE(any_different);
    }

    {
        simulation::options options;
        options.seed = 123;
        options.noise_sigma = 0.7;
        options.outlier_fraction = 0.1;
        const simulation::scene first = simulation::scene::generate(options);
        const simulation::scene second = simulation::scene::generate(options);
        REQUIRE(first.points.size() == second.points.size());
        REQUIRE(first.poses.size() == second.poses.size());
        REQUIRE(first.observations.size() == second.observations.size());
        REQUIRE(std::memcmp(first.points.data(), second.points.data(), first.points.size() * sizeof(first.points[0])) == 0);
        for (size_t i = 0; i < first.observations.size(); ++i) {
            REQUIRE(first.observations[i].x == second.observations[i].x);
            REQUIRE(first.observations[i].y == second.observations[i].y);
            REQUIRE(first.observations[i].outlier == second.observations[i].outlier);
        }
        options.seed = 124;
        const simulation::scene third = simulation::scene::generate(options);
        REQUIRE(std::memcmp(first.points.data(), third.points.data(), first.points.size() * sizeof(first.points[0])) != 0);
    }

    {
        simulation::options options;
        const simulation::scene scene = simulation::scene::generate(options);
        REQUIRE(!scene.observations.empty());
        REQUIRE(scene.observations.size() == scene.observations_exact.size());
        for (size_t i = 0; i < scene.observations.size(); ++i) {
            const simulation::observation& measured = scene.observations[i];
            const simulation::observation& exact = scene.observations_exact[i];
            REQUIRE(measured.x == exact.x);
            REQUIRE(measured.y == exact.y);
            REQUIRE(!measured.outlier);
            const math::matrix<double, 3, 1> in_camera = scene.poses[static_cast<size_t>(exact.frame_index)] * scene.points[static_cast<size_t>(exact.point_index)];
            REQUIRE(in_camera[2] > 0.0);
            math::matrix<double, 2, 1> pixel;
            REQUIRE(scene.camera.project(in_camera.data(), pixel.data()));
            REQUIRE(pixel[0] == exact.x);
            REQUIRE(pixel[1] == exact.y);
            REQUIRE(exact.x >= 0.5);
            REQUIRE(exact.x <= options.image_width - 0.5);
            REQUIRE(exact.y >= 0.5);
            REQUIRE(exact.y <= options.image_height - 0.5);
        }
        REQUIRE(scene.observations.size() > static_cast<size_t>(options.point_count * options.frame_count) / 2);
    }

    {
        simulation::options options;
        const simulation::scene scene = simulation::scene::generate(options);
        for (const math::se3<double>& pose : scene.poses) {
            const math::matrix<double, 3, 3> rotation = pose.rotation().get_matrix();
            const math::matrix<double, 3, 3> identity_check = math::transpose(rotation) * rotation;
            for (size_t r = 0; r < 3; ++r) {
                for (size_t c = 0; c < 3; ++c) {
                    REQUIRE(math::abs(identity_check[r][c] - ((r == c) ? 1.0 : 0.0)) < 1.0e-12);
                }
            }
            const math::matrix<double, 3, 1> origin_in_camera = pose * math::matrix<double, 3, 1>::zero();
            REQUIRE(origin_in_camera[2] > 0.0);
        }
    }

    {
        simulation::options options;
        options.noise_sigma = 0.5;
        const simulation::scene scene = simulation::scene::generate(options);
        REQUIRE(scene.observations.size() == scene.observations_exact.size());
        double sum_squared = 0.0;
        for (size_t i = 0; i < scene.observations.size(); ++i) {
            const double delta_x = scene.observations[i].x - scene.observations_exact[i].x;
            const double delta_y = scene.observations[i].y - scene.observations_exact[i].y;
            sum_squared += delta_x * delta_x + delta_y * delta_y;
            REQUIRE(!scene.observations[i].outlier);
        }
        const double rms = math::sqrt(sum_squared / static_cast<double>(2 * scene.observations.size()));
        REQUIRE(rms > 0.25);
        REQUIRE(rms < 1.0);
    }

    {
        simulation::options options;
        options.point_count = 500;
        options.outlier_fraction = 0.2;
        const simulation::scene scene = simulation::scene::generate(options);
        size_t outliers = 0;
        for (const simulation::observation& measured : scene.observations) {
            outliers += measured.outlier ? 1u : 0u;
            REQUIRE(measured.x >= 0.5);
            REQUIRE(measured.x <= options.image_width - 0.5);
            REQUIRE(measured.y >= 0.5);
            REQUIRE(measured.y <= options.image_height - 0.5);
            if (measured.outlier) {
                REQUIRE(measured.x - std::floor(measured.x) == 0.5);
                REQUIRE(measured.y - std::floor(measured.y) == 0.5);
            }
        }
        const double fraction = static_cast<double>(outliers) / static_cast<double>(scene.observations.size());
        REQUIRE(fraction > 0.15);
        REQUIRE(fraction < 0.25);
    }

    return EXIT_SUCCESS;
}
