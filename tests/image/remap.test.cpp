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

#include "image/remap.hpp"

#include "image/image.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
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

    constexpr static const size_t columns = 23;
    constexpr static const size_t rows = 17;
    // A source with a stride wider than its rows, so the table must step rows by the stride it is given.
    constexpr static const size_t stride = 29;
    std::vector<unsigned char> source(stride * rows, static_cast<unsigned char>(7));
    for (size_t y = 0; y < rows; ++y) {
        for (size_t x = 0; x < columns; ++x) {
            source[(y * stride) + x] = static_cast<unsigned char>((x * 7) + (y * 5));
        }
    }

    {
        // The identity table copies every pixel and marks every one valid.
        std::vector<float> source_x(columns * rows);
        std::vector<float> source_y(columns * rows);
        for (size_t y = 0; y < rows; ++y) {
            for (size_t x = 0; x < columns; ++x) {
                source_x[(y * columns) + x] = static_cast<float>(x);
                source_y[(y * columns) + x] = static_cast<float>(y);
            }
        }
        const image::remap identity(columns, rows, source_x.data(), source_y.data(), columns, rows);
        REQUIRE(!identity.empty());
        REQUIRE((identity.get_columns() == columns) && (identity.get_rows() == rows));
        std::vector<unsigned char> destination(columns * rows, static_cast<unsigned char>(0));
        identity.apply(source.data(), stride, destination.data());
        for (size_t y = 0; y < rows; ++y) {
            for (size_t x = 0; x < columns; ++x) {
                REQUIRE(destination[(y * columns) + x] == source[(y * stride) + x]);
                REQUIRE(identity.valid(x, y));
            }
        }
        REQUIRE(!identity.valid(columns, 0));
        REQUIRE(!identity.valid(0, rows));
    }

    {
        // Half a pixel right and a quarter down interpolates bilinearly, and positions outside the source are clamped to its border and marked invalid.
        std::vector<float> source_x(columns * rows);
        std::vector<float> source_y(columns * rows);
        for (size_t y = 0; y < rows; ++y) {
            for (size_t x = 0; x < columns; ++x) {
                source_x[(y * columns) + x] = static_cast<float>(x) + 0.5f;
                source_y[(y * columns) + x] = static_cast<float>(y) + 0.25f;
            }
        }
        source_x[0] = -3.0f;
        source_y[1] = 100.0f;
        const image::remap shifted(columns, rows, source_x.data(), source_y.data(), columns, rows);
        std::vector<unsigned char> destination(columns * rows, static_cast<unsigned char>(0));
        shifted.apply(source.data(), stride, destination.data());
        for (size_t y = 0; y + 1 < rows; ++y) {
            for (size_t x = 0; x + 1 < columns; ++x) {
                if ((y == 0) && (x < 2)) {
                    continue;
                }
                const double expected = ((static_cast<double>(x) + 0.5) * 7.0) + ((static_cast<double>(y) + 0.25) * 5.0);
                REQUIRE(std::abs(static_cast<double>(destination[(y * columns) + x]) - expected) <= 1.0);
                REQUIRE(shifted.valid(x, y));
            }
        }
        REQUIRE(!shifted.valid(0, 0));
        REQUIRE(!shifted.valid(1, 0));
        REQUIRE(std::abs(static_cast<double>(destination[0]) - 1.25) <= 1.0);
        REQUIRE(std::abs(static_cast<double>(destination[1]) - static_cast<double>(source[((rows - 1) * stride) + 1]) - 3.5) <= 1.0);
        // The last column reaches past the source's last pixel centre, so it is clamped and invalid.
        REQUIRE(!shifted.valid(columns - 1, 5));
    }

    {
        const image::remap empty;
        REQUIRE(empty.empty());
        const float position[1] = { 0.0f };
        const image::remap degenerate(1, 1, &position[0], &position[0], 1, 1);
        REQUIRE(degenerate.empty());
    }
    return EXIT_SUCCESS;
}
