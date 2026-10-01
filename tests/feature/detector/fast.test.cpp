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

#include "feature/detector/fast.hpp"

#include "core/cpu.hpp"
#include "core/timestamp.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

namespace feature::detector {
    size_t detect_cpu(const unsigned char* __restrict const data, const int width, const int height, const int stride, const int threshold, const size_t feature_point_buffer_size, point* __restrict feature_point_buffer);
#if defined(ZEROSLAM_SIMD_AVX2)
    size_t detect_avx2(const unsigned char* __restrict const data, const int width, const int height, const int stride, const int threshold, const size_t feature_point_buffer_size, point* __restrict feature_point_buffer);
#endif
#if defined(ZEROSLAM_SIMD_NEON)
    size_t detect_neon(const unsigned char* __restrict const data, const int width, const int height, const int stride, const int threshold, const size_t feature_point_buffer_size, point* __restrict feature_point_buffer);
#endif
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        constexpr static const size_t data_width = 128;
        constexpr static const size_t data_height = 64;
        unsigned char data[data_height][data_width] = {};
        for (size_t y = 0; y < data_height; ++y) {
            for (size_t x = 0; x < data_width; ++x) {
                data[y][x] = static_cast<unsigned char>(((x > data_width / 4) && (x < (data_width * 3) / 4) && (y > data_height / 4) && (y < (data_height * 3) / 4)) * 255);
            }
        }
        feature::point features_opencv[24] = {
            { 33, 17, 0, 0, 0 },
            { 34, 17, 0, 0, 0 },
            { 35, 17, 0, 0, 0 },
            { 93, 17, 0, 0, 0 },
            { 94, 17, 0, 0, 0 },
            { 95, 17, 0, 0, 0 },
            { 33, 18, 0, 0, 0 },
            { 34, 18, 0, 0, 0 },
            { 94, 18, 0, 0, 0 },
            { 95, 18, 0, 0, 0 },
            { 33, 19, 0, 0, 0 },
            { 95, 19, 0, 0, 0 },
            { 33, 45, 0, 0, 0 },
            { 95, 45, 0, 0, 0 },
            { 33, 46, 0, 0, 0 },
            { 34, 46, 0, 0, 0 },
            { 94, 46, 0, 0, 0 },
            { 95, 46, 0, 0, 0 },
            { 33, 47, 0, 0, 0 },
            { 34, 47, 0, 0, 0 },
            { 35, 47, 0, 0, 0 },
            { 93, 47, 0, 0, 0 },
            { 94, 47, 0, 0, 0 },
            { 95, 47, 0, 0, 0 }
        };
        feature::point features[data_width * data_height];
        const size_t features_10_count = feature::detector::fast::detect(&data[0][0], data_width, data_height, data_width, 10, data_width * data_height, features);
        REQUIRE(features_10_count == 24);
        for (size_t i = 0; i < 24; ++i) {
            if (i >= features_10_count)
                continue;
            REQUIRE(features[i].x == features_opencv[i].x);
            REQUIRE(features[i].y == features_opencv[i].y);
        }
        const size_t features_255_count = feature::detector::fast::detect(&data[0][0], data_width, data_height, data_width, 255, data_width * data_height, features);
        REQUIRE(features_255_count == 0);
    }
    {
        // Every tier finds the same corners for any size, stride and threshold, and none writes past the buffer it is given.
        using detect_function = size_t (*)(const unsigned char*, const int, const int, const int, const int, const size_t, feature::point*);
        std::vector<detect_function> tiers = { &feature::detector::detect_cpu };
#if defined(ZEROSLAM_SIMD_AVX2)
        if (core::cpu::has_avx2()) {
            tiers.push_back(&feature::detector::detect_avx2);
        }
#endif
#if defined(ZEROSLAM_SIMD_NEON)
        if (core::cpu::has_neon()) {
            tiers.push_back(&feature::detector::detect_neon);
        }
#endif
        const feature::point sentinel = { -1.0f, -1.0f, -1.0f, -1.0f, -1 };
        unsigned long long state = 12345;
        for (int width : { 7, 8, 16, 21, 38, 39, 40, 71, 100 }) {
            for (int padding : { 0, 5 }) {
                const int height = 23;
                const int stride = width + padding;
                std::vector<unsigned char> data(static_cast<size_t>(stride * height));
                for (unsigned char& value : data) {
                    state = state * 6364136223846793005ull + 1442695040888963407ull;
                    value = static_cast<unsigned char>(state >> 56);
                }
                for (int threshold : { 0, 1, 7, 20, 60, 128, 254, 255 }) {
                    std::vector<feature::point> expected(static_cast<size_t>(width * height), sentinel);
                    const size_t expected_count = feature::detector::detect_cpu(data.data(), width, height, stride, threshold, expected.size(), expected.data());
                    for (detect_function detect : tiers) {
                        std::vector<feature::point> features(static_cast<size_t>(width * height), sentinel);
                        REQUIRE(detect(data.data(), width, height, stride, threshold, features.size(), features.data()) == expected_count);
                        for (size_t i = 0; i < expected_count; ++i) {
                            REQUIRE((features[i].x == expected[i].x) && (features[i].y == expected[i].y));
                        }
                        const size_t buffer_sizes[4] = { 0, 1, 3, expected_count / 2 };
                        for (const size_t buffer_size : buffer_sizes) {
                            if (buffer_size >= expected_count) {
                                continue;
                            }
                            std::vector<feature::point> limited(buffer_size + 1, sentinel);
                            REQUIRE(detect(data.data(), width, height, stride, threshold, buffer_size, limited.data()) == buffer_size);
                            for (size_t i = 0; i < buffer_size; ++i) {
                                REQUIRE((limited[i].x == expected[i].x) && (limited[i].y == expected[i].y));
                            }
                            REQUIRE(limited[buffer_size].response == sentinel.response);
                        }
                    }
                }
                // The dispatcher clamps thresholds outside the pixel range, a tier holding it in a byte would wrap them instead.
                std::vector<feature::point> clamped(static_cast<size_t>(width * height));
                std::vector<feature::point> features(static_cast<size_t>(width * height));
                for (const int threshold : { -7, 256, 300, 1000 }) {
                    const size_t clamped_count = feature::detector::detect_cpu(data.data(), width, height, stride, (threshold < 0) ? 0 : 255, clamped.size(), clamped.data());
                    REQUIRE(feature::detector::fast::detect(data.data(), width, height, stride, threshold, features.size(), features.data()) == clamped_count);
                    for (size_t i = 0; i < clamped_count; ++i) {
                        REQUIRE((features[i].x == clamped[i].x) && (features[i].y == clamped[i].y));
                    }
                }
            }
        }
    }
#if defined(ZEROSLAM_SIMD_AVX2)
    {
        if (core::cpu::has_avx2()) {
            constexpr static const size_t data_width = 640;
            constexpr static const size_t data_height = 480;
            std::vector<unsigned char> data(data_width * data_height);
            for (size_t y = 0; y < data_height; ++y) {
                for (size_t x = 0; x < data_width; ++x) {
                    data[y * data_width + x] = static_cast<unsigned char>((x * 17 + y * 31) % 256);
                    if ((x / 16 + y / 16) % 2 == 0) {
                        data[y * data_width + x] = static_cast<unsigned char>((data[y * data_width + x] + 128) % 256);
                    }
                }
            }
            std::vector<feature::point> features_scalar(data_width * data_height);
            std::vector<feature::point> features_simd(data_width * data_height);
            const size_t count_scalar = feature::detector::detect_cpu(data.data(), data_width, data_height, data_width, 20, features_scalar.size(), features_scalar.data());
            const size_t count_simd = feature::detector::detect_avx2(data.data(), data_width, data_height, data_width, 20, features_simd.size(), features_simd.data());
            REQUIRE(count_scalar > 0);
            REQUIRE(count_scalar == count_simd);
            for (size_t i = 0; i < count_scalar; ++i) {
                REQUIRE(features_scalar[i].x == features_simd[i].x);
                REQUIRE(features_scalar[i].y == features_simd[i].y);
            }
            const size_t count_dispatched = feature::detector::fast::detect(data.data(), data_width, data_height, data_width, 20, features_simd.size(), features_simd.data());
            REQUIRE(count_dispatched == count_scalar);
            constexpr int iteration_count = 100;
            const long long int start_scalar = core::timestamp();
            for (int i = 0; i < iteration_count; ++i) {
                feature::detector::detect_cpu(data.data(), data_width, data_height, data_width, 20, features_scalar.size(), features_scalar.data());
            }
            const long long int start_simd = core::timestamp();
            for (int i = 0; i < iteration_count; ++i) {
                feature::detector::detect_avx2(data.data(), data_width, data_height, data_width, 20, features_simd.size(), features_simd.data());
            }
            const long long int end_simd = core::timestamp();
            std::printf("Detect benchmark (%d iterations): scalar %.1f ms, avx2 %.1f ms\n", iteration_count, static_cast<double>(start_simd - start_scalar) * 1.0e-6, static_cast<double>(end_simd - start_simd) * 1.0e-6);
        }
    }
#endif
#if defined(ZEROSLAM_SIMD_NEON)
    {
        if (core::cpu::has_neon()) {
            constexpr static const size_t data_width = 640;
            constexpr static const size_t data_height = 480;
            std::vector<unsigned char> data(data_width * data_height);
            for (size_t y = 0; y < data_height; ++y) {
                for (size_t x = 0; x < data_width; ++x) {
                    data[y * data_width + x] = static_cast<unsigned char>((x * 17 + y * 31) % 256);
                    if ((x / 16 + y / 16) % 2 == 0) {
                        data[y * data_width + x] = static_cast<unsigned char>((data[y * data_width + x] + 128) % 256);
                    }
                }
            }
            std::vector<feature::point> features_scalar(data_width * data_height);
            std::vector<feature::point> features_simd(data_width * data_height);
            const size_t count_scalar = feature::detector::detect_cpu(data.data(), data_width, data_height, data_width, 20, features_scalar.size(), features_scalar.data());
            const size_t count_simd = feature::detector::detect_neon(data.data(), data_width, data_height, data_width, 20, features_simd.size(), features_simd.data());
            REQUIRE(count_scalar > 0);
            REQUIRE(count_scalar == count_simd);
            for (size_t i = 0; i < count_scalar; ++i) {
                REQUIRE(features_scalar[i].x == features_simd[i].x);
                REQUIRE(features_scalar[i].y == features_simd[i].y);
            }
            const size_t count_dispatched = feature::detector::fast::detect(data.data(), data_width, data_height, data_width, 20, features_simd.size(), features_simd.data());
            REQUIRE(count_dispatched == count_scalar);
            constexpr int iteration_count = 100;
            const long long int start_scalar = core::timestamp();
            for (int i = 0; i < iteration_count; ++i) {
                feature::detector::detect_cpu(data.data(), data_width, data_height, data_width, 20, features_scalar.size(), features_scalar.data());
            }
            const long long int start_simd = core::timestamp();
            for (int i = 0; i < iteration_count; ++i) {
                feature::detector::detect_neon(data.data(), data_width, data_height, data_width, 20, features_simd.size(), features_simd.data());
            }
            const long long int end_simd = core::timestamp();
            std::printf("Detect benchmark (%d iterations): scalar %.1f ms, neon %.1f ms\n", iteration_count, static_cast<double>(start_simd - start_scalar) * 1.0e-6, static_cast<double>(end_simd - start_simd) * 1.0e-6);
        }
    }
#endif

    return EXIT_SUCCESS;
}
