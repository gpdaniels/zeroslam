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

#include "core/coordinates.hpp"

#include "sensor/camera/pinhole.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>

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
        REQUIRE(core::to_pixel_centre(0.0f) == 0.5f);
        REQUIRE(core::to_pixel_centre(639.0f) == 639.5f);
        REQUIRE(core::to_pixel_centre(10.25f) == 10.75f);
        REQUIRE(core::to_pixel_index_position(core::to_pixel_centre(10.25f)) == 10.25f);
        for (int index = 0; index < 640; ++index) {
            REQUIRE(core::to_pixel_index(core::to_pixel_centre(static_cast<float>(index))) == index);
        }
    }

    {
        REQUIRE(core::to_pixel_index(10.5f) == 10);
        REQUIRE(core::to_pixel_index(10.999f) == 10);
        REQUIRE(core::to_pixel_index(11.0f) == 11);
        REQUIRE(core::to_pixel_index(0.25f) == 0);
        REQUIRE(core::to_pixel_index(-0.25f) == -1);
        REQUIRE(core::to_pixel_index(-1.0f) == -1);
    }

    {
        const double parameters[4] = { 500.0, 500.0, 640.0 / 2.0, 480.0 / 2.0 };
        const sensor::camera::pinhole<double> camera(&parameters[0], 4);
        const double axis[3] = { 0.0, 0.0, 1.0 };
        double projected[2] = {};
        REQUIRE(camera.project(&axis[0], &projected[0]));
        REQUIRE(projected[0] == 320.0);
        REQUIRE(projected[1] == 240.0);
        const double left[2] = { static_cast<double>(core::to_pixel_centre(319.0f)), 240.0 };
        const double right[2] = { static_cast<double>(core::to_pixel_centre(320.0f)), 240.0 };
        double ray_left[3] = {};
        double ray_right[3] = {};
        REQUIRE(camera.unproject(&left[0], &ray_left[0]));
        REQUIRE(camera.unproject(&right[0], &ray_right[0]));
        REQUIRE(ray_left[0] == -ray_right[0]);
        REQUIRE(ray_left[1] == 0.0);
        REQUIRE(ray_right[1] == 0.0);
    }

    return EXIT_SUCCESS;
}
