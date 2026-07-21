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
        constexpr static const int kernel_size = 7;
        constexpr static const int kernel[kernel_size] = { 18, 34, 49, 55, 49, 34, 18 };
        {
            // Narrower than the kernel radius.
            constexpr static const int width = 2;
            constexpr static const int height = 3;
            unsigned char data[height][width] = { { 1, 2 }, { 3, 4 }, { 5, 6 } };
            unsigned char convolved[height][width] = {};
            image::convolution_horizontal<kernel_size>(&data[0][0], width, height, width, &kernel[0], &convolved[0][0]);
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    REQUIRE(convolved[i][j] == 0);
                }
            }
        }
        {
            // Shorter than the kernel radius.
            constexpr static const int width = 3;
            constexpr static const int height = 2;
            unsigned char data[height][width] = { { 1, 2, 3 }, { 4, 5, 6 } };
            unsigned char convolved[height][width] = {};
            image::convolution_vertical<kernel_size>(&data[0][0], width, height, width, &kernel[0], &convolved[0][0]);
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    REQUIRE(convolved[i][j] == 0);
                }
            }
        }
        {
            // Single pixel in both dimensions (the most extreme narrower/shorter-than-kernel case).
            unsigned char data[1][1] = { { 42 } };
            unsigned char convolved[1][1] = {};
            image::convolution_horizontal<kernel_size>(&data[0][0], 1, 1, 1, &kernel[0], &convolved[0][0]);
            REQUIRE(convolved[0][0] == 0);
            image::convolution_vertical<kernel_size>(&data[0][0], 1, 1, 1, &kernel[0], &convolved[0][0]);
            REQUIRE(convolved[0][0] == 0);
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
    {
        // Upscaling reads source coordinates with negative offsets, these must be clamped in-bounds.
        image::image image(8, 8);
        for (size_t i = 0; i < image.get_rows(); ++i) {
            for (size_t j = 0; j < image.get_cols(); ++j) {
                image.get_data()[i * image.get_cols() + j] = i * image.get_cols() + j;
            }
        }
        image::image resized(16, 16);
        image::resize(image.get_data(), image.get_cols(), image.get_rows(), resized.get_cols(), resized.get_rows(), resized.get_data(), image::interpolate_linear);
        // The corners of the upscaled image must match the corners of the source image.
        REQUIRE(resized.get_data()[0 * resized.get_cols() + 0] == 0);
        REQUIRE(resized.get_data()[0 * resized.get_cols() + 15] == 7);
        REQUIRE(resized.get_data()[15 * resized.get_cols() + 0] == 56);
        REQUIRE(resized.get_data()[15 * resized.get_cols() + 15] == 63);
        // Every interpolated value must lie within the range of the source values.
        for (size_t i = 0; i < resized.get_rows(); ++i) {
            for (size_t j = 0; j < resized.get_cols(); ++j) {
                REQUIRE(resized.get_data()[i * resized.get_cols() + j] <= 63);
            }
        }
    }
    {
        // Downscaling regression, the interpolated values must not change.
        unsigned char data[4][4] = {
            { 0, 1, 2, 3 },
            { 4, 5, 6, 7 },
            { 8, 9, 10, 11 },
            { 12, 13, 14, 15 }
        };
        image::image image(4, 4, &data[0][0]);
        image::image resized(2, 2);
        image::resize(image.get_data(), image.get_cols(), image.get_rows(), resized.get_cols(), resized.get_rows(), resized.get_data(), image::interpolate_linear);
        REQUIRE(resized.get_data()[0 * resized.get_cols() + 0] == 3);
        REQUIRE(resized.get_data()[0 * resized.get_cols() + 1] == 5);
        REQUIRE(resized.get_data()[1 * resized.get_cols() + 0] == 11);
        REQUIRE(resized.get_data()[1 * resized.get_cols() + 1] == 13);
    }
    {
        // Copy-assignment of a bigger image onto a smaller non-empty image.
        image::image smaller(5, 10);
        for (size_t i = 0; i < smaller.get_rows() * smaller.get_cols(); ++i) {
            smaller.get_data()[i] = 7;
        }
        image::image bigger(10, 20);
        for (size_t i = 0; i < bigger.get_rows() * bigger.get_cols(); ++i) {
            bigger.get_data()[i] = i;
        }
        smaller = bigger;
        REQUIRE(smaller.get_rows() == 10);
        REQUIRE(smaller.get_cols() == 20);
        REQUIRE(smaller.get_data() != bigger.get_data());
        for (size_t i = 0; i < smaller.get_rows() * smaller.get_cols(); ++i) {
            REQUIRE(smaller.get_data()[i] == static_cast<unsigned char>(i));
        }
    }
    {
        // Copy-assignment of a smaller image onto a bigger non-empty image.
        image::image bigger(10, 20);
        for (size_t i = 0; i < bigger.get_rows() * bigger.get_cols(); ++i) {
            bigger.get_data()[i] = 7;
        }
        image::image smaller(5, 10);
        for (size_t i = 0; i < smaller.get_rows() * smaller.get_cols(); ++i) {
            smaller.get_data()[i] = i;
        }
        bigger = smaller;
        REQUIRE(bigger.get_rows() == 5);
        REQUIRE(bigger.get_cols() == 10);
        REQUIRE(bigger.get_data() != smaller.get_data());
        for (size_t i = 0; i < bigger.get_rows() * bigger.get_cols(); ++i) {
            REQUIRE(bigger.get_data()[i] == static_cast<unsigned char>(i));
        }
    }
    {
        // Copy-assignment from an empty image onto a non-empty image.
        image::image image(10, 20);
        image::image empty;
        image = empty;
        REQUIRE(image.get_rows() == 0);
        REQUIRE(image.get_cols() == 0);
        REQUIRE(image.get_data() == nullptr);
    }
    {
        // Self-assignment.
        image::image image(10, 20);
        for (size_t i = 0; i < image.get_rows() * image.get_cols(); ++i) {
            image.get_data()[i] = i;
        }
        image::image& self = image;
        image = self;
        REQUIRE(image.get_rows() == 10);
        REQUIRE(image.get_cols() == 20);
        for (size_t i = 0; i < image.get_rows() * image.get_cols(); ++i) {
            REQUIRE(image.get_data()[i] == static_cast<unsigned char>(i));
        }
    }
    {
        // Move-assignment onto a differently-sized non-empty image.
        image::image source(10, 20);
        for (size_t i = 0; i < source.get_rows() * source.get_cols(); ++i) {
            source.get_data()[i] = i;
        }
        unsigned char* source_data = source.get_data();
        image::image target(5, 10);
        target = static_cast<image::image&&>(source);
        REQUIRE(target.get_rows() == 10);
        REQUIRE(target.get_cols() == 20);
        REQUIRE(target.get_data() == source_data);
        for (size_t i = 0; i < target.get_rows() * target.get_cols(); ++i) {
            REQUIRE(target.get_data()[i] == static_cast<unsigned char>(i));
        }
        // The moved-from image must remain consistent: its dimensions must match the buffer it holds.
        REQUIRE(source.get_rows() == 5);
        REQUIRE(source.get_cols() == 10);
        REQUIRE(source.get_data() != nullptr);
        for (size_t i = 0; i < source.get_rows() * source.get_cols(); ++i) {
            source.get_data()[i] = 0;
        }
    }

    return EXIT_SUCCESS;
}