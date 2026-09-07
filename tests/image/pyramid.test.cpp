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

#include "image/pyramid.hpp"

#include "image/blur.hpp"

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

static inline image::image make_textured(const size_t rows, const size_t cols) {
    image::image textured(rows, cols);
    unsigned int state = 12345u;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            state = (state * 1664525u) + 1013904223u;
            const unsigned int checker = (((i / 8) + (j / 8)) % 2) ? 200u : 40u;
            textured.get_data()[i * cols + j] = static_cast<unsigned char>((checker + ((state >> 16) % 32u)) % 256u);
        }
    }
    return textured;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        REQUIRE(image::pyramid::automatic_levels(640, 480) == 4);
        REQUIRE(image::pyramid::automatic_levels(752, 480) == 4);
        REQUIRE(image::pyramid::automatic_levels(1920, 1080) == 6);
        REQUIRE(image::pyramid::automatic_levels(160, 160) == 3);
        REQUIRE(image::pyramid::automatic_levels(100, 100) == 2);
        REQUIRE(image::pyramid::automatic_levels(8, 8) == 1);
        REQUIRE(image::pyramid::automatic_levels(1, 1) == 1);
    }

    {
        const image::pyramid empty;
        REQUIRE(empty.size() == 0);
    }

    {
        const image::image base = make_textured(480, 640);
        const image::pyramid built(base);
        REQUIRE(built.size() == 4);
        REQUIRE(built[0].get_cols() == 640);
        REQUIRE(built[0].get_rows() == 480);
        REQUIRE(built[1].get_cols() == 320);
        REQUIRE(built[1].get_rows() == 240);
        REQUIRE(built[2].get_cols() == 160);
        REQUIRE(built[2].get_rows() == 120);
        REQUIRE(built[3].get_cols() == 80);
        REQUIRE(built[3].get_rows() == 60);
        REQUIRE(&built.back() == &built[3]);
        REQUIRE(built.scale_x(0) == 1.0f);
        REQUIRE(built.scale_y(0) == 1.0f);
        REQUIRE(built.scale_x(1) == 2.0f);
        REQUIRE(built.scale_y(1) == 2.0f);
        REQUIRE(built.scale_x(2) == 4.0f);
        REQUIRE(built.scale_y(2) == 4.0f);
        REQUIRE(built.scale_x(3) == 8.0f);
        REQUIRE(built.scale_y(3) == 8.0f);
        for (size_t i = 0; i < 640 * 480; ++i) {
            REQUIRE(built[0].get_data()[i] == base.get_data()[i]);
        }
        image::image blurred(480, 640);
        image::blur::gaussian_5x5(base.get_data(), 640, 480, 640, blurred.get_data());
        for (size_t i = 0; i < 240; ++i) {
            for (size_t j = 0; j < 320; ++j) {
                REQUIRE(built[1].get_data()[i * 320 + j] == blurred.get_data()[(2 * i) * 640 + (2 * j)]);
            }
        }
    }

    {
        const image::image base = make_textured(75, 300);
        const image::pyramid built(base);
        REQUIRE(built.size() == 2);
        REQUIRE(built.back().get_cols() == 150);
        REQUIRE(built.back().get_rows() == 37);
    }

    {
        const image::image base = make_textured(130, 1000);
        const image::pyramid built(base);
        REQUIRE(built.size() == 3);
        REQUIRE(built[1].get_cols() == 500);
        REQUIRE(built[1].get_rows() == 65);
        REQUIRE(built[2].get_cols() == 250);
        REQUIRE(built[2].get_rows() == 32);
        REQUIRE(built.scale_x(1) == 2.0f);
        REQUIRE(built.scale_y(1) == 2.0f);
    }

    {
        const image::image base = make_textured(127, 1000);
        const image::pyramid built(base);
        REQUIRE(built.size() == 2);
    }

    {
        const image::image base = make_textured(125, 201);
        const image::pyramid built(base);
        REQUIRE(built.size() == 2);
        REQUIRE(built[1].get_cols() == 100);
        REQUIRE(built[1].get_rows() == 62);
    }

    return EXIT_SUCCESS;
}
