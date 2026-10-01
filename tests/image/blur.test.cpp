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
#include "image/resize.hpp"

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
    void gaussian_5x5_decimate_cpu(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data, const int target_stride);
    void gaussian_7x7_cpu(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
#if defined(ZEROSLAM_SIMD_AVX2)
    void gaussian_5x5_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
    void gaussian_5x5_decimate_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data, const int target_stride);
    void gaussian_7x7_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
#endif
#if defined(ZEROSLAM_SIMD_NEON)
    void gaussian_5x5_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
    void gaussian_5x5_decimate_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data, const int target_stride);
    void gaussian_7x7_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
#endif
}

using blur_function = void (*)(const unsigned char* __restrict const, const int, const int, const int, unsigned char* __restrict const);
using decimate_function = void (*)(const unsigned char* __restrict const, const int, const int, const int, unsigned char* __restrict const, const int);

struct blur_tier final {
    const char* name;
    blur_function gaussian_5x5;
    blur_function gaussian_7x7;
    decimate_function gaussian_5x5_decimate;
};

static std::vector<blur_tier> available_tiers() {
    std::vector<blur_tier> tiers;
    tiers.push_back({ "cpu", &image::gaussian_5x5_cpu, &image::gaussian_7x7_cpu, &image::gaussian_5x5_decimate_cpu });
    tiers.push_back({ "dispatched", &image::blur::gaussian_5x5, &image::blur::gaussian_7x7, &image::blur::gaussian_5x5_decimate });
#if defined(ZEROSLAM_SIMD_AVX2)
    if (core::cpu::has_avx2()) {
        tiers.push_back({ "avx2", &image::gaussian_5x5_avx2, &image::gaussian_7x7_avx2, &image::gaussian_5x5_decimate_avx2 });
    }
#endif
#if defined(ZEROSLAM_SIMD_NEON)
    if (core::cpu::has_neon()) {
        tiers.push_back({ "neon", &image::gaussian_5x5_neon, &image::gaussian_7x7_neon, &image::gaussian_5x5_decimate_neon });
    }
#endif
    return tiers;
}

static int reflect_index(int index, const int size) {
    if (size == 1) {
        return 0;
    }
    while ((index < 0) || (index >= size)) {
        if (index < 0) {
            index = -index;
        }
        if (index >= size) {
            index = (2 * size) - 2 - index;
        }
    }
    return index;
}

// The blur from its definition, the outer product of the binomial taps at every pixel with the edges reflected, the target rows stride apart like the source.
static void reference_blur(const int kernel_size, const unsigned char* const data, const int width, const int height, const int stride, unsigned char* const blurred) {
    const int taps_5x5[5] = { 1, 4, 6, 4, 1 };
    const int taps_7x7[7] = { 1, 6, 15, 20, 15, 6, 1 };
    const int* const taps = (kernel_size == 5) ? &taps_5x5[0] : &taps_7x7[0];
    const int shift = (kernel_size == 5) ? 8 : 12;
    const int radius = kernel_size / 2;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            for (int i = 0; i < kernel_size; ++i) {
                for (int j = 0; j < kernel_size; ++j) {
                    sum += taps[i] * taps[j] * data[(reflect_index(y + i - radius, height) * stride) + reflect_index(x + j - radius, width)];
                }
            }
            blurred[(y * stride) + x] = static_cast<unsigned char>((sum + (1 << (shift - 1))) >> shift);
        }
    }
}

static unsigned char next_random_byte(unsigned int& state) {
    state = (state * 1664525u) + 1013904223u;
    return static_cast<unsigned char>(state >> 24);
}

// Every tier's blurs against the definition, and the decimating blur against the blur then resize::decimate, the padding past each row left alone.
static void require_matches_reference(const std::vector<blur_tier>& tiers, const int width, const int height, const int padding, unsigned int& state) {
    constexpr static const unsigned char untouched = 0x5A;
    const int stride = width + padding;
    const size_t size = static_cast<size_t>(stride) * static_cast<size_t>(height);
    std::vector<unsigned char> data(size);
    for (unsigned char& value : data) {
        value = next_random_byte(state);
    }
    for (const int kernel_size : { 5, 7 }) {
        std::vector<unsigned char> expected(size, untouched);
        reference_blur(kernel_size, data.data(), width, height, stride, expected.data());
        for (const blur_tier& tier : tiers) {
            std::vector<unsigned char> blurred(size, untouched);
            ((kernel_size == 5) ? tier.gaussian_5x5 : tier.gaussian_7x7)(data.data(), width, height, stride, blurred.data());
            REQUIRE(blurred == expected);
        }
    }
    std::vector<unsigned char> blurred(size, untouched);
    reference_blur(5, data.data(), width, height, stride, blurred.data());
    std::vector<unsigned char> packed(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            packed[(static_cast<size_t>(y) * static_cast<size_t>(width)) + static_cast<size_t>(x)] = blurred[(static_cast<size_t>(y) * static_cast<size_t>(stride)) + static_cast<size_t>(x)];
        }
    }
    const int target_width = width / 2;
    const int target_height = height / 2;
    const int target_stride = target_width + padding;
    std::vector<unsigned char> decimated(static_cast<size_t>(target_width) * static_cast<size_t>(target_height));
    image::resize::decimate(packed.data(), static_cast<size_t>(width), static_cast<size_t>(height), static_cast<size_t>(target_width), static_cast<size_t>(target_height), decimated.data());
    std::vector<unsigned char> expected(static_cast<size_t>(target_stride) * static_cast<size_t>(target_height), untouched);
    for (int y = 0; y < target_height; ++y) {
        for (int x = 0; x < target_width; ++x) {
            expected[(static_cast<size_t>(y) * static_cast<size_t>(target_stride)) + static_cast<size_t>(x)] = decimated[(static_cast<size_t>(y) * static_cast<size_t>(target_width)) + static_cast<size_t>(x)];
        }
    }
    for (const blur_tier& tier : tiers) {
        std::vector<unsigned char> target(expected.size(), untouched);
        tier.gaussian_5x5_decimate(data.data(), width, height, stride, target.data(), target_stride);
        REQUIRE(target == expected);
    }
}

// The pyramid's step, one blur then decimate against the decimating blur.
static void benchmark_decimate(const blur_tier& tier, const int width, const int height) {
    const size_t size = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t target_size = static_cast<size_t>(width / 2) * static_cast<size_t>(height / 2);
    std::vector<unsigned char> data(size);
    unsigned int state = 17u;
    for (unsigned char& value : data) {
        value = next_random_byte(state);
    }
    std::vector<unsigned char> blurred(size);
    std::vector<unsigned char> separate(target_size);
    std::vector<unsigned char> combined(target_size);
    constexpr static const int iteration_count = 50;
    const long long int start_separate = core::timestamp();
    for (int i = 0; i < iteration_count; ++i) {
        tier.gaussian_5x5(data.data(), width, height, width, blurred.data());
        image::resize::decimate(blurred.data(), static_cast<size_t>(width), static_cast<size_t>(height), static_cast<size_t>(width / 2), static_cast<size_t>(height / 2), separate.data());
    }
    const long long int start_combined = core::timestamp();
    for (int i = 0; i < iteration_count; ++i) {
        tier.gaussian_5x5_decimate(data.data(), width, height, width, combined.data(), width / 2);
    }
    const long long int end_combined = core::timestamp();
    REQUIRE(separate == combined);
    const double time_separate = static_cast<double>(start_combined - start_separate) * 1.0e-3 / iteration_count;
    const double time_combined = static_cast<double>(end_combined - start_combined) * 1.0e-3 / iteration_count;
    std::printf("gaussian_5x5_decimate %s benchmark %dx%d: blur then decimate %.1f us, decimating blur %.1f us, speedup %.2fx\n", tier.name, width, height, time_separate, time_combined, (time_combined > 0.0) ? (time_separate / time_combined) : 0.0);
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

    {
        const std::vector<blur_tier> tiers = available_tiers();
        unsigned int state = 2463534242u;
        for (int width = 1; width <= 70; ++width) {
            for (int height = 1; height <= 9; ++height) {
                for (const int padding : { 0, 1, 13 }) {
                    require_matches_reference(tiers, width, height, padding, state);
                }
            }
        }
        for (const int width : { 1023, 1024, 1025, 1026, 1031, 2048, 2049, 2050, 2100 }) {
            for (const int height : { 1, 2, 5, 6 }) {
                for (const int padding : { 0, 7 }) {
                    require_matches_reference(tiers, width, height, padding, state);
                }
            }
        }
        for (const int size : { 0, -3 }) {
            unsigned char data[4] = { 1, 2, 3, 4 };
            unsigned char target[4] = { 9, 9, 9, 9 };
            for (const blur_tier& tier : tiers) {
                tier.gaussian_5x5(&data[0], size, 2, 2, &target[0]);
                tier.gaussian_7x7(&data[0], 2, size, 2, &target[0]);
                tier.gaussian_5x5_decimate(&data[0], size, 2, 2, &target[0], 1);
                tier.gaussian_5x5_decimate(&data[0], 1, 4, 1, &target[0], 1);
                tier.gaussian_5x5_decimate(&data[0], 4, 1, 4, &target[0], 2);
            }
            for (int i = 0; i < 4; ++i) {
                REQUIRE(target[i] == 9);
            }
        }
        for (const blur_tier& tier : tiers) {
            if (tier.name[0] == 'd') {
                continue;
            }
            benchmark_decimate(tier, 640, 480);
            benchmark_decimate(tier, 752, 480);
            benchmark_decimate(tier, 1241, 376);
            benchmark_decimate(tier, 739, 458);
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
