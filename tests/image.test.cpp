/*
Copyright (C) 2025 Geoffrey Daniels. https://gpdaniels.com/

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

#include "image.hpp"

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
        image::image image;
    }
    {
        image::image image(10, 20);
        REQUIRE(image.get_rows() == 10);
        REQUIRE(image.get_cols() == 20);
    }
    {
        image::image image(10, 20);
        for (size_t i = 0; i < image.get_rows(); ++i) {
            for (size_t j = 0; j < image.get_cols(); ++j) {
                image.get_data()[i * image.get_cols() + j] = i * image.get_cols() + j;
            }
        }
        for (size_t i = 0; i < image.get_rows(); ++i) {
            for (size_t j = 0; j < image.get_cols(); ++j) {
                REQUIRE(image.get_data()[i * image.get_cols() + j] == i * image.get_cols() + j);
            }
        }
    }
    {
        unsigned char data[10][20] = {
            { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 },
            { 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 },
            { 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 },
            { 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79 },
            { 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99 },
            { 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119 },
            { 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139 },
            { 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159 },
            { 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179 },
            { 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199 }
        };
        image::image image(10, 20, &data[0][0]);
        for (size_t i = 0; i < image.get_rows(); ++i) {
            for (size_t j = 0; j < image.get_cols(); ++j) {
                REQUIRE(image.get_data()[i * image.get_cols() + j] == i * image.get_cols() + j);
            }
        }
    }
    {
        image::image image(10, 20);
        for (size_t i = 0; i < image.get_rows(); ++i) {
            for (size_t j = 0; j < image.get_cols(); ++j) {
                image.get_data()[i * image.get_cols() + j] = (((i % 2) == 0) ^ ((j % 2) == 0)) * 100;
            }
        }
        image::image blurred(image.get_rows(), 20);
        image::blur(image.get_data(), image.get_cols(), image.get_rows(), image.get_cols(), blurred.get_data());
        for (size_t i = 0; i < blurred.get_rows(); ++i) {
            for (size_t j = 0; j < blurred.get_cols(); ++j) {
                if ((i < 3) || (i >= blurred.get_rows() - 3) || (j < 3) || (j >= blurred.get_cols() - 3)) {
                    REQUIRE(blurred.get_data()[i * blurred.get_cols() + j] == 0);
                }
                else {
                    REQUIRE(blurred.get_data()[i * blurred.get_cols() + j] == 50);
                }
            }
        }
    }
    {
        image::image image(10, 20);
        for (size_t i = 0; i < image.get_rows(); ++i) {
            for (size_t j = 0; j < image.get_cols(); ++j) {
                image.get_data()[i * image.get_cols() + j] = (((i % 2) == 0) ^ ((j % 2) == 0)) * 100;
            }
        }
        image::image resized(5, 10);
        image::resize(image.get_data(), image.get_cols(), image.get_rows(), resized.get_cols(), resized.get_rows(), resized.get_data(), image::interpolate_linear);
        for (size_t i = 0; i < resized.get_rows(); ++i) {
            for (size_t j = 0; j < resized.get_cols(); ++j) {
                REQUIRE(resized.get_data()[i * resized.get_cols() + j] == 50);
            }
        }
    }

    return EXIT_SUCCESS;
}