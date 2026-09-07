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

#include "feature/suppressor/fast.hpp"

#include "core/cpu.hpp"
#include "core/sort.hpp"
#include "feature/detector/fast.hpp"
#include "feature/score/fast.hpp"

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

namespace feature::suppressor {
    size_t suppress_cpu(const point* __restrict const features, const size_t features_count, const size_t max_row, point* __restrict const features_suppressed);
#if defined(ZEROSLAM_SIMD_AVX2)
    size_t suppress_avx2(const point* __restrict const features, const size_t features_count, const size_t max_row, point* __restrict const features_suppressed);
#endif
#if defined(ZEROSLAM_SIMD_NEON)
    size_t suppress_neon(const point* __restrict const features, const size_t features_count, const size_t max_row, point* __restrict const features_suppressed);
#endif
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        {
            const feature::point features_detected_sorted_by_y[12] = {
                { 0, 0, 0, 0.0f, 0 },
                { 1, 0, 1, 0.1f, 0 },
                { 2, 0, 2, 0.2f, 0 },
                { 3, 0, 3, 0.3f, 0 },
                { 4, 0, 4, 0.4f, 0 },
                { 1, 1, 2, 0.5f, 0 },
                { 2, 1, 4, 0.6f, 0 },
                { 3, 1, 3, 0.7f, 0 },
                { 1, 2, 1, 0.8f, 0 },
                { 2, 2, 1, 0.9f, 0 },
                { 3, 2, 8, 1.0f, 0 },
                { 2, 20, 0, 1.1f, 0 },
            };
            feature::point features_suppressed[12] = {};
            const size_t features = feature::suppressor::fast::suppress(&features_detected_sorted_by_y[0], 12, 21, &features_suppressed[0]);
            REQUIRE(features == 3);
            REQUIRE(features_suppressed[0].x == features_detected_sorted_by_y[4].x);
            REQUIRE(features_suppressed[0].y == features_detected_sorted_by_y[4].y);
            REQUIRE(features_suppressed[0].response == features_detected_sorted_by_y[4].response);
            REQUIRE(features_suppressed[1].x == features_detected_sorted_by_y[10].x);
            REQUIRE(features_suppressed[1].y == features_detected_sorted_by_y[10].y);
            REQUIRE(features_suppressed[1].response == features_detected_sorted_by_y[10].response);
            REQUIRE(features_suppressed[2].x == features_detected_sorted_by_y[11].x);
            REQUIRE(features_suppressed[2].y == features_detected_sorted_by_y[11].y);
            REQUIRE(features_suppressed[2].response == features_detected_sorted_by_y[11].response);
        }

        {
            const feature::point features_detected_sorted_by_y[12] = {
                { 0, 0, 0, 0.0f, 0 },
                { 2, 2, 0, 0.1f, 0 },
                { 4, 4, 0, 0.2f, 0 },
                { 8, 8, 0, 0.3f, 0 },
                { 10, 10, 0, 0.4f, 0 },
                { 12, 12, 0, 0.5f, 0 },
                { 14, 14, 0, 0.6f, 0 },
                { 16, 16, 0, 0.7f, 0 },
                { 18, 18, 0, 0.8f, 0 },
                { 20, 20, 0, 0.9f, 0 },
                { 22, 22, 0, 1.0f, 0 },
                { 24, 24, 0, 1.1f, 0 },
            };
            feature::point features_suppressed[12] = {};
            const size_t features = feature::suppressor::fast::suppress(&features_detected_sorted_by_y[0], 12, 25, &features_suppressed[0]);
            REQUIRE(features == 12);
        }

        {
            const feature::point features_detected_sorted_by_y[2] = {
                { 10, 0, 1.0f, 0.0f, 0 },
                { 10, 4, 2.0f, 0.0f, 0 },
            };
            feature::point features_suppressed[2] = {};
            const size_t features = feature::suppressor::fast::suppress(&features_detected_sorted_by_y[0], 2, 5, &features_suppressed[0]);
            REQUIRE(features == 2);
            REQUIRE(features_suppressed[0].x == features_detected_sorted_by_y[0].x);
            REQUIRE(features_suppressed[0].y == features_detected_sorted_by_y[0].y);
            REQUIRE(features_suppressed[1].x == features_detected_sorted_by_y[1].x);
            REQUIRE(features_suppressed[1].y == features_detected_sorted_by_y[1].y);
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
            std::vector<feature::point> features(data_width * data_height);
            const size_t count = feature::detector::fast::detect(data.data(), data_width, data_height, data_width, 20, features.size(), features.data());
            REQUIRE(count > 0);
            for (size_t i = 0; i < count; ++i) {
                features[i].response = feature::score::fast::score(data.data() + static_cast<size_t>(features[i].y) * data_width + static_cast<size_t>(features[i].x), data_width);
            }
            core::sort::quick(features.data(), count, [](const feature::point& lhs, const feature::point& rhs) {
                return (lhs.y < rhs.y) || ((lhs.y == rhs.y) && (lhs.x < rhs.x));
            });
            std::vector<feature::point> suppressed_scalar(count);
            std::vector<feature::point> suppressed_simd(count);
            const size_t count_scalar = feature::suppressor::suppress_cpu(features.data(), count, data_height, suppressed_scalar.data());
            const size_t count_simd = feature::suppressor::suppress_avx2(features.data(), count, data_height, suppressed_simd.data());
            REQUIRE(count_scalar == count_simd);
            for (size_t i = 0; i < count_scalar; ++i) {
                REQUIRE(suppressed_scalar[i].x == suppressed_simd[i].x);
                REQUIRE(suppressed_scalar[i].y == suppressed_simd[i].y);
                REQUIRE(suppressed_scalar[i].response == suppressed_simd[i].response);
            }
            REQUIRE(feature::suppressor::fast::suppress(features.data(), count, data_height, suppressed_simd.data()) == count_scalar);
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
            std::vector<feature::point> features(data_width * data_height);
            const size_t count = feature::detector::fast::detect(data.data(), data_width, data_height, data_width, 20, features.size(), features.data());
            REQUIRE(count > 0);
            for (size_t i = 0; i < count; ++i) {
                features[i].response = feature::score::fast::score(data.data() + static_cast<size_t>(features[i].y) * data_width + static_cast<size_t>(features[i].x), data_width);
            }
            core::sort::quick(features.data(), count, [](const feature::point& lhs, const feature::point& rhs) {
                return (lhs.y < rhs.y) || ((lhs.y == rhs.y) && (lhs.x < rhs.x));
            });
            std::vector<feature::point> suppressed_scalar(count);
            std::vector<feature::point> suppressed_simd(count);
            const size_t count_scalar = feature::suppressor::suppress_cpu(features.data(), count, data_height, suppressed_scalar.data());
            const size_t count_simd = feature::suppressor::suppress_neon(features.data(), count, data_height, suppressed_simd.data());
            REQUIRE(count_scalar == count_simd);
            for (size_t i = 0; i < count_scalar; ++i) {
                REQUIRE(suppressed_scalar[i].x == suppressed_simd[i].x);
                REQUIRE(suppressed_scalar[i].y == suppressed_simd[i].y);
                REQUIRE(suppressed_scalar[i].response == suppressed_simd[i].response);
            }
            REQUIRE(feature::suppressor::fast::suppress(features.data(), count, data_height, suppressed_simd.data()) == count_scalar);
        }
    }
#endif

    return EXIT_SUCCESS;
}
