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

#include "image/blur.hpp"

#include "core/cpu.hpp"
#include "core/timestamp.hpp"
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

namespace image {
    void gaussian_5x5_cpu(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
    void gaussian_7x7_cpu(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
#if defined(ZEROSLAM_SIMD_AVX2)
    void gaussian_5x5_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
    void gaussian_7x7_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
#endif
#if defined(ZEROSLAM_SIMD_NEON)
    void gaussian_5x5_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
    void gaussian_7x7_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
#endif
}

static void fill_test_pattern(std::vector<unsigned char>& data, const size_t width, const size_t height) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            data[y * width + x] = static_cast<unsigned char>((x * 17 + y * 31) % 256);
            if (((x / 64) + (y / 64)) % 5 == 0) {
                data[y * width + x] = 255;
            }
            else if (((x / 64) + (y / 64)) % 5 == 1) {
                data[y * width + x] = 0;
            }
        }
    }
}

template <typename blur_type>
static void require_matches_cpu(blur_type blur_cpu, blur_type blur_tier, const char* const name) {
    constexpr size_t width = 640;
    constexpr size_t height = 480;
    std::vector<unsigned char> data(width * height);
    fill_test_pattern(data, width, height);
    std::vector<unsigned char> blurred_cpu(width * height, 0);
    std::vector<unsigned char> blurred_tier(width * height, 0);
    blur_cpu(data.data(), width, height, width, blurred_cpu.data());
    blur_tier(data.data(), width, height, width, blurred_tier.data());
    for (size_t i = 0; i < width * height; ++i) {
        REQUIRE(blurred_cpu[i] == blurred_tier[i]);
    }
    constexpr int iteration_count = 200;
    const long long int start_cpu = core::timestamp();
    for (int i = 0; i < iteration_count; ++i) {
        blur_cpu(data.data(), width, height, width, blurred_cpu.data());
    }
    const long long int start_tier = core::timestamp();
    for (int i = 0; i < iteration_count; ++i) {
        blur_tier(data.data(), width, height, width, blurred_tier.data());
    }
    const long long int end_tier = core::timestamp();
    const double time_cpu = static_cast<double>(start_tier - start_cpu) * 1.0e-6;
    const double time_tier = static_cast<double>(end_tier - start_tier) * 1.0e-6;
    std::printf("%s benchmark (%d iterations): cpu %.1f ms, simd %.1f ms, speedup %.2fx\n", name, iteration_count, time_cpu, time_tier, (time_tier > 0.0) ? (time_cpu / time_tier) : 0.0);
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        image::image image(10, 20);
        for (size_t i = 0; i < image.get_rows(); ++i) {
            for (size_t j = 0; j < image.get_cols(); ++j) {
                image.get_data()[i * image.get_cols() + j] = static_cast<unsigned char>((((i % 2) == 0) ^ ((j % 2) == 0)) * 100);
            }
        }
        image::image blurred(image.get_rows(), image.get_cols());
        image::blur::gaussian_5x5(image.get_data(), static_cast<int>(image.get_cols()), static_cast<int>(image.get_rows()), static_cast<int>(image.get_cols()), blurred.get_data());
        for (size_t i = 0; i < blurred.get_rows() * blurred.get_cols(); ++i) {
            REQUIRE(blurred.get_data()[i] == 50);
        }
        image::blur::gaussian_7x7(image.get_data(), static_cast<int>(image.get_cols()), static_cast<int>(image.get_rows()), static_cast<int>(image.get_cols()), blurred.get_data());
        for (size_t i = 0; i < blurred.get_rows() * blurred.get_cols(); ++i) {
            REQUIRE(blurred.get_data()[i] == 50);
        }
    }

    {
        constexpr static const int width = 40;
        constexpr static const int height = 9;
        unsigned char data[height][width];
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                data[i][j] = 200;
            }
        }
        unsigned char blurred[height][width] = {};
        image::blur::gaussian_5x5(&data[0][0], width, height, width, &blurred[0][0]);
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(blurred[i][j] == 200);
            }
        }
        image::blur::gaussian_7x7(&data[0][0], width, height, width, &blurred[0][0]);
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                REQUIRE(blurred[i][j] == 200);
            }
        }
        unsigned char step[height][width];
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                step[i][j] = static_cast<unsigned char>((j < width / 2) ? 0 : 160);
            }
        }
        image::blur::gaussian_5x5(&step[0][0], width, height, width, &blurred[0][0]);
        for (int i = 0; i < height; ++i) {
            REQUIRE(blurred[i][0] == 0);
            REQUIRE(blurred[i][width / 2 - 3] == 0);
            REQUIRE(blurred[i][width / 2 - 2] == 10);
            REQUIRE(blurred[i][width / 2 - 1] == 50);
            REQUIRE(blurred[i][width / 2] == 110);
            REQUIRE(blurred[i][width / 2 + 1] == 150);
            REQUIRE(blurred[i][width / 2 + 2] == 160);
            REQUIRE(blurred[i][width - 1] == 160);
        }
        image::blur::gaussian_7x7(&step[0][0], width, height, width, &blurred[0][0]);
        for (int i = 0; i < height; ++i) {
            REQUIRE(blurred[i][0] == 0);
            REQUIRE(blurred[i][width / 2 - 4] == 0);
            REQUIRE(blurred[i][width / 2 - 3] == 3);
            REQUIRE(blurred[i][width / 2 - 2] == 18);
            REQUIRE(blurred[i][width / 2 - 1] == 55);
            REQUIRE(blurred[i][width / 2] == 105);
            REQUIRE(blurred[i][width / 2 + 1] == 143);
            REQUIRE(blurred[i][width / 2 + 2] == 158);
            REQUIRE(blurred[i][width / 2 + 3] == 160);
            REQUIRE(blurred[i][width - 1] == 160);
        }
    }

    {
        constexpr static const int width = 7;
        constexpr static const int height = 5;
        unsigned char data[height][width] = {};
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                data[i][j] = static_cast<unsigned char>((i == 2) && (j == 3) ? 255 : 0);
            }
        }
        unsigned char blurred[height][width] = {};
        image::blur::gaussian_5x5(&data[0][0], width, height, width, &blurred[0][0]);
        REQUIRE(blurred[2][3] == 36);
        REQUIRE(blurred[2][2] == 24);
        REQUIRE(blurred[2][1] == 6);
        REQUIRE(blurred[2][0] == 0);
        REQUIRE(blurred[1][3] == 24);
        REQUIRE(blurred[0][3] == 12);
        REQUIRE(blurred[1][2] == 16);
        REQUIRE(blurred[0][0] == 0);
        REQUIRE(blurred[4][3] == 12);
        REQUIRE(blurred[0][1] == 2);
        unsigned char reflected[height][width] = {};
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                reflected[i][j] = static_cast<unsigned char>((i == 0) && (j == 0) ? 255 : 0);
            }
        }
        image::blur::gaussian_5x5(&reflected[0][0], width, height, width, &blurred[0][0]);
        REQUIRE(blurred[0][0] == 36);
        REQUIRE(blurred[0][1] == 24);
        REQUIRE(blurred[0][2] == 6);
        REQUIRE(blurred[1][0] == 24);
        REQUIRE(blurred[2][0] == 6);
        REQUIRE(blurred[1][1] == 16);
        REQUIRE(blurred[2][2] == 1);
        REQUIRE(blurred[3][3] == 0);
    }

    {
        unsigned char tiny[2][3] = { { 10, 20, 30 }, { 40, 50, 60 } };
        unsigned char tiny_blurred[2][3] = {};
        unsigned char tiny_cpu[2][3] = {};
        image::blur::gaussian_5x5(&tiny[0][0], 3, 2, 3, &tiny_blurred[0][0]);
        image::gaussian_5x5_cpu(&tiny[0][0], 3, 2, 3, &tiny_cpu[0][0]);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                REQUIRE(tiny_blurred[i][j] == tiny_cpu[i][j]);
                REQUIRE(tiny_blurred[i][j] >= 10);
                REQUIRE(tiny_blurred[i][j] <= 60);
            }
        }
        image::blur::gaussian_7x7(&tiny[0][0], 3, 2, 3, &tiny_blurred[0][0]);
        image::gaussian_7x7_cpu(&tiny[0][0], 3, 2, 3, &tiny_cpu[0][0]);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                REQUIRE(tiny_blurred[i][j] == tiny_cpu[i][j]);
            }
        }
        unsigned char single = 42;
        unsigned char single_blurred = 0;
        image::blur::gaussian_5x5(&single, 1, 1, 1, &single_blurred);
        REQUIRE(single_blurred == 42);
        image::blur::gaussian_7x7(&single, 1, 1, 1, &single_blurred);
        REQUIRE(single_blurred == 42);
        unsigned char constant[3][3] = { { 9, 9, 9 }, { 9, 9, 9 }, { 9, 9, 9 } };
        unsigned char constant_blurred[3][3] = {};
        image::blur::gaussian_7x7(&constant[0][0], 3, 3, 3, &constant_blurred[0][0]);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                REQUIRE(constant_blurred[i][j] == 9);
            }
        }
    }

    {
        constexpr static const int width = 37;
        constexpr static const int height = 11;
        unsigned char data[height][width];
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                data[i][j] = static_cast<unsigned char>((i * 29 + j * 53 + (i * j) % 7) % 256);
            }
        }
        unsigned char blurred_cpu[height][width] = {};
        unsigned char blurred[height][width] = {};
        image::gaussian_5x5_cpu(&data[0][0], width, height, width, &blurred_cpu[0][0]);
        image::blur::gaussian_5x5(&data[0][0], width, height, width, &blurred[0][0]);
        for (int i = 0; i < height * width; ++i) {
            REQUIRE((&blurred[0][0])[i] == (&blurred_cpu[0][0])[i]);
        }
        image::gaussian_7x7_cpu(&data[0][0], width, height, width, &blurred_cpu[0][0]);
        image::blur::gaussian_7x7(&data[0][0], width, height, width, &blurred[0][0]);
        for (int i = 0; i < height * width; ++i) {
            REQUIRE((&blurred[0][0])[i] == (&blurred_cpu[0][0])[i]);
        }
    }

#if defined(ZEROSLAM_SIMD_AVX2)
    if (core::cpu::has_avx2()) {
        require_matches_cpu(image::gaussian_5x5_cpu, image::gaussian_5x5_avx2, "gaussian_5x5 avx2");
        require_matches_cpu(image::gaussian_7x7_cpu, image::gaussian_7x7_avx2, "gaussian_7x7 avx2");
    }
#endif

#if defined(ZEROSLAM_SIMD_NEON)
    if (core::cpu::has_neon()) {
        require_matches_cpu(image::gaussian_5x5_cpu, image::gaussian_5x5_neon, "gaussian_5x5 neon");
        require_matches_cpu(image::gaussian_7x7_cpu, image::gaussian_7x7_neon, "gaussian_7x7 neon");
    }
#endif

    return EXIT_SUCCESS;
}
