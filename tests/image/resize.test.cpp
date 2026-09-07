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

#include "image/resize.hpp"

#include "image/image.hpp"

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
        unsigned char data[4][4] = {
            { 0, 1, 2, 3 },
            { 4, 5, 6, 7 },
            { 8, 9, 10, 11 },
            { 12, 13, 14, 15 }
        };
        unsigned char decimated[2][2] = {};
        image::resize::decimate(&data[0][0], 4, 4, 2, 2, &decimated[0][0]);
        REQUIRE(decimated[0][0] == 0);
        REQUIRE(decimated[0][1] == 2);
        REQUIRE(decimated[1][0] == 8);
        REQUIRE(decimated[1][1] == 10);
        unsigned char odd[5][5];
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                odd[i][j] = static_cast<unsigned char>(i * 5 + j);
            }
        }
        unsigned char odd_decimated[2][2] = {};
        image::resize::decimate(&odd[0][0], 5, 5, 2, 2, &odd_decimated[0][0]);
        REQUIRE(odd_decimated[0][0] == 0);
        REQUIRE(odd_decimated[0][1] == 2);
        REQUIRE(odd_decimated[1][0] == 10);
        REQUIRE(odd_decimated[1][1] == 12);
        unsigned char thirds[1][9] = { { 0, 1, 2, 3, 4, 5, 6, 7, 8 } };
        unsigned char thirds_decimated[1][3] = {};
        image::resize::decimate(&thirds[0][0], 9, 1, 3, 1, &thirds_decimated[0][0]);
        REQUIRE(thirds_decimated[0][0] == 0);
        REQUIRE(thirds_decimated[0][1] == 3);
        REQUIRE(thirds_decimated[0][2] == 6);
        unsigned char same[2][2] = { { 1, 2 }, { 3, 4 } };
        unsigned char same_decimated[2][2] = {};
        image::resize::decimate(&same[0][0], 2, 2, 2, 2, &same_decimated[0][0]);
        REQUIRE(same_decimated[1][1] == 4);
    }

    {
        image::image image(10, 20);
        for (size_t i = 0; i < image.get_rows(); ++i) {
            for (size_t j = 0; j < image.get_cols(); ++j) {
                image.get_data()[i * image.get_cols() + j] = static_cast<unsigned char>((((i % 2) == 0) ^ ((j % 2) == 0)) * 100);
            }
        }
        image::image resized(5, 10);
        image::resize::linear(image.get_data(), image.get_cols(), image.get_rows(), resized.get_cols(), resized.get_rows(), resized.get_data());
        for (size_t i = 0; i < resized.get_rows(); ++i) {
            for (size_t j = 0; j < resized.get_cols(); ++j) {
                REQUIRE(resized.get_data()[i * resized.get_cols() + j] == 50);
            }
        }
    }
    {
        image::image image(8, 8);
        for (size_t i = 0; i < image.get_rows(); ++i) {
            for (size_t j = 0; j < image.get_cols(); ++j) {
                image.get_data()[i * image.get_cols() + j] = static_cast<unsigned char>(i * image.get_cols() + j);
            }
        }
        image::image resized(16, 16);
        image::resize::linear(image.get_data(), image.get_cols(), image.get_rows(), resized.get_cols(), resized.get_rows(), resized.get_data());
        REQUIRE(resized.get_data()[0 * resized.get_cols() + 0] == 0);
        REQUIRE(resized.get_data()[0 * resized.get_cols() + 15] == 7);
        REQUIRE(resized.get_data()[15 * resized.get_cols() + 0] == 56);
        REQUIRE(resized.get_data()[15 * resized.get_cols() + 15] == 63);
        for (size_t i = 0; i < resized.get_rows(); ++i) {
            for (size_t j = 0; j < resized.get_cols(); ++j) {
                REQUIRE(resized.get_data()[i * resized.get_cols() + j] <= 63);
            }
        }
    }
    {
        unsigned char data[4][4] = {
            { 0, 1, 2, 3 },
            { 4, 5, 6, 7 },
            { 8, 9, 10, 11 },
            { 12, 13, 14, 15 }
        };
        image::image image(4, 4, &data[0][0]);
        image::image resized(2, 2);
        image::resize::linear(image.get_data(), image.get_cols(), image.get_rows(), resized.get_cols(), resized.get_rows(), resized.get_data());
        REQUIRE(resized.get_data()[0 * resized.get_cols() + 0] == 3);
        REQUIRE(resized.get_data()[0 * resized.get_cols() + 1] == 5);
        REQUIRE(resized.get_data()[1 * resized.get_cols() + 0] == 11);
        REQUIRE(resized.get_data()[1 * resized.get_cols() + 1] == 13);
    }
    {
        constexpr static const unsigned char constant = 100;
        image::image image(3, 3);
        for (size_t i = 0; i < image.get_rows() * image.get_cols(); ++i) {
            image.get_data()[i] = constant;
        }
        image::image resized(7, 7);
        image::resize::linear(image.get_data(), image.get_cols(), image.get_rows(), resized.get_cols(), resized.get_rows(), resized.get_data());
        for (size_t i = 0; i < resized.get_rows() * resized.get_cols(); ++i) {
            REQUIRE(resized.get_data()[i] == constant);
        }
    }
    {
        unsigned char data[2][4] = {
            { 0, 8, 16, 24 },
            { 0, 8, 16, 24 }
        };
        image::image image(2, 4, &data[0][0]);
        image::image resized(2, 8);
        image::resize::linear(image.get_data(), image.get_cols(), image.get_rows(), resized.get_cols(), resized.get_rows(), resized.get_data());
        const unsigned char expected[8] = { 0, 2, 6, 10, 14, 18, 22, 24 };
        for (size_t i = 0; i < resized.get_rows(); ++i) {
            for (size_t j = 0; j < resized.get_cols(); ++j) {
                REQUIRE(resized.get_data()[i * resized.get_cols() + j] == expected[j]);
            }
        }
    }

    return EXIT_SUCCESS;
}
