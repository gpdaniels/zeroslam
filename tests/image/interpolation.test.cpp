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

#include "image/interpolation.hpp"

#include "image/image.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>
#include <vector>

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

    const unsigned char data[3][4] = {
        { 0, 10, 20, 30 },
        { 40, 50, 60, 70 },
        { 80, 90, 100, 110 }
    };

    {
        REQUIRE(image::interpolation::nearest(&data[0][0], 4, 3, 0.0f, 0.0f) == 0);
        REQUIRE(image::interpolation::nearest(&data[0][0], 4, 3, 1.4f, 0.4f) == 10);
        REQUIRE(image::interpolation::nearest(&data[0][0], 4, 3, 1.6f, 0.6f) == 60);
        REQUIRE(image::interpolation::nearest(&data[0][0], 4, 3, 3.0f, 2.0f) == 110);
        REQUIRE(image::interpolation::nearest(&data[0][0], 4, 3, -5.0f, -5.0f) == 0);
        REQUIRE(image::interpolation::nearest(&data[0][0], 4, 3, 50.0f, 50.0f) == 110);
    }

    {
        REQUIRE(image::interpolation::linear(&data[0][0], 4, 3, 0.0f, 0.0f) == 0);
        REQUIRE(image::interpolation::linear(&data[0][0], 4, 3, 0.5f, 0.0f) == 5);
        REQUIRE(image::interpolation::linear(&data[0][0], 4, 3, 0.0f, 0.5f) == 20);
        REQUIRE(image::interpolation::linear(&data[0][0], 4, 3, 0.5f, 0.5f) == 25);
        REQUIRE(image::interpolation::linear(&data[0][0], 4, 3, 1.5f, 1.5f) == 75);
        REQUIRE(image::interpolation::linear(&data[0][0], 4, 3, 3.0f, 2.0f) == 110);
        REQUIRE(image::interpolation::linear(&data[0][0], 4, 3, -5.0f, -5.0f) == 0);
        REQUIRE(image::interpolation::linear(&data[0][0], 4, 3, 50.0f, 50.0f) == 110);
        REQUIRE(image::interpolation::linear(&data[0][0], 4, 3, 3.5f, 2.5f) == 110);
    }

    {
        std::vector<unsigned char> constant(6 * 5, 77);
        for (float y = -1.0f; y <= 5.0f; y += 0.25f) {
            for (float x = -1.0f; x <= 6.0f; x += 0.25f) {
                REQUIRE(image::interpolation::nearest(constant.data(), 6, 5, x, y) == 77);
                REQUIRE(image::interpolation::linear(constant.data(), 6, 5, x, y) == 77);
            }
        }
    }

    return EXIT_SUCCESS;
}
