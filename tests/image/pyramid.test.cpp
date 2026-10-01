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

// A bright paraboloid centred on level 0 pixel (centre_x, centre_y), symmetric about it, so every level keeps its peak on the pixel the halving maps the centre to.
static inline image::image make_dot(const size_t rows, const size_t cols, const size_t centre_x, const size_t centre_y, const size_t radius) {
    image::image dot(rows, cols);
    const long long int limit = static_cast<long long int>(radius * radius);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const long long int offset_x = static_cast<long long int>(j) - static_cast<long long int>(centre_x);
            const long long int offset_y = static_cast<long long int>(i) - static_cast<long long int>(centre_y);
            const long long int squared = (offset_x * offset_x) + (offset_y * offset_y);
            dot.get_data()[i * cols + j] = static_cast<unsigned char>((squared < limit) ? ((250 * (limit - squared)) / limit) : 0);
        }
    }
    return dot;
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
        REQUIRE(image::pyramid::automatic_levels(0, 0) == 1);
        REQUIRE(image::pyramid::automatic_levels(0, 640) == 1);
        REQUIRE(image::pyramid::automatic_levels(480, 0) == 1);
        REQUIRE(image::pyramid::automatic_levels(31, 1000) == 1);
        REQUIRE(image::pyramid::automatic_levels(32, 32) == 1);
        REQUIRE(image::pyramid::automatic_levels(63, 1000) == 1);
        REQUIRE(image::pyramid::automatic_levels(64, 64) == 2);
        for (size_t smallest = 0; smallest < 4096; ++smallest) {
            size_t expected = 1;
            while ((smallest >> (expected + 5)) > 0) {
                ++expected;
            }
            REQUIRE(image::pyramid::automatic_levels(smallest, 4096) == expected);
        }
    }

    {
        const image::pyramid from_empty((image::image()));
        REQUIRE(from_empty.size() == 1);
        REQUIRE(from_empty[0].get_cols() == 0);
        REQUIRE(from_empty.scale_x(0) == 1.0f);
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

    {
        // Sizes that do not divide by 2^level: each level still halves the one above, so the scale is 2^level and level pixel (i, j) is level 0 pixel (i * 2^level, j * 2^level) up to the far edges.
        const size_t sizes[4][3] = { { 739, 458, 4 }, { 572, 758, 5 }, { 1241, 376, 4 }, { 1242, 375, 4 } };
        for (const auto& size : sizes) {
            const size_t cols = size[0];
            const size_t rows = size[1];
            const image::pyramid plain(make_textured(rows, cols));
            REQUIRE(plain.size() == size[2]);
            for (size_t level = 1; level < plain.size(); ++level) {
                REQUIRE(plain[level].get_cols() == plain[level - 1].get_cols() / 2);
                REQUIRE(plain[level].get_rows() == plain[level - 1].get_rows() / 2);
                REQUIRE(plain.scale_x(level) == static_cast<float>(1u << level));
                REQUIRE(plain.scale_y(level) == static_cast<float>(1u << level));
                const size_t level_cols = plain[level].get_cols();
                const size_t level_rows = plain[level].get_rows();
                const size_t positions[2][2] = { { level_cols - 4, level_rows - 4 }, { level_cols / 3, (2 * level_rows) / 3 } };
                for (const auto& position : positions) {
                    const size_t centre_x = position[0] << level;
                    const size_t centre_y = position[1] << level;
                    const image::pyramid dotted(make_dot(rows, cols, centre_x, centre_y, static_cast<size_t>(2) << level));
                    REQUIRE(dotted.size() == plain.size());
                    const unsigned char* const data = dotted[level].get_data();
                    size_t brightest = 0;
                    size_t ties = 0;
                    for (size_t index = 0; index < level_cols * level_rows; ++index) {
                        if (data[index] > data[brightest]) {
                            brightest = index;
                            ties = 1;
                        }
                        else if (data[index] == data[brightest]) {
                            ++ties;
                        }
                    }
                    REQUIRE(ties == 1);
                    REQUIRE(brightest % level_cols == position[0]);
                    REQUIRE(brightest / level_cols == position[1]);
                    REQUIRE(static_cast<float>(brightest % level_cols) * dotted.scale_x(level) == static_cast<float>(centre_x));
                    REQUIRE(static_cast<float>(brightest / level_cols) * dotted.scale_y(level) == static_cast<float>(centre_y));
                }
            }
        }
    }

    return EXIT_SUCCESS;
}
