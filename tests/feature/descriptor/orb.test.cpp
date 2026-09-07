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

#include "feature/descriptor/orb.hpp"

#include "core/cpu.hpp"
#include "feature/angle/orb.hpp"
#include "feature/refiner/subpixel.hpp"
#include "match/distance/hamming.hpp"
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-8) {
    if (std::isnan(lhs) && std::isnan(rhs))
        return true;
    if (std::isnan(lhs) != std::isnan(rhs))
        return false;
    if (std::isinf(lhs) != std::isinf(rhs))
        return false;
    if (std::signbit(lhs + epsilon) != std::signbit(rhs + epsilon))
        return false;
    if (std::isinf(lhs) && std::isinf(rhs))
        return true;
    return (std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon);
}

template <typename array_type>
static inline bool are_values_approx(const array_type& lhs, const array_type& rhs, unsigned long long int length, double epsilon = 1e-8) {
    for (size_t index = 0; index < length; ++index) {
        if (!is_value_approx(static_cast<double>(lhs[index]), static_cast<double>(rhs[index]), epsilon)) {
            return false;
        }
    }
    return true;
}

namespace feature::descriptor {
    void describe_cpu(const unsigned char* __restrict const data, const int stride, const float angle_radians, binary<256>& descriptor);
#if defined(ZEROSLAM_SIMD_AVX2)
    void describe_avx2(const unsigned char* __restrict const data, const int stride, const float angle_radians, binary<256>& descriptor);
#endif
#if defined(ZEROSLAM_SIMD_NEON)
    void describe_neon(const unsigned char* __restrict const data, const int stride, const float angle_radians, binary<256>& descriptor);
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
                data[y][x] = static_cast<unsigned char>(x + y);
            }
        }
        feature::descriptor::binary<256> descriptor_expected = {
            0x06,
            0x41,
            0x10,
            0x05,
            0x51,
            0x4E,
            0xB2,
            0x28,
            0x2C,
            0xA4,
            0x4B,
            0x01,
            0xCC,
            0x12,
            0x0D,
            0xC4,
            0x90,
            0x1D,
            0x82,
            0x63,
            0x44,
            0x90,
            0x51,
            0x4C,
            0x25,
            0x34,
            0x0A,
            0x07,
            0x00,
            0x37,
            0x0D,
            0x86
        };
        feature::descriptor::binary<256> descriptor;
        feature::descriptor::orb::describe(&data[data_height / 2][data_width / 2], data_width, 190.0f * (3.141592653589793f / 180.0f), descriptor);
        REQUIRE(are_values_approx(&descriptor.data[0], &descriptor_expected.data[0], 256 / 8));
    }
    {
        constexpr static const int data_size = 64;
        constexpr static const int data_center = data_size / 2;
        unsigned char data[data_size][data_size];
        unsigned char data_rotated[data_size][data_size];
        for (int y = 0; y < data_size; ++y) {
            for (int x = 0; x < data_size; ++x) {
                const unsigned char corner = static_cast<unsigned char>(((x >= data_center) && (y >= data_center)) * 128);
                const unsigned char texture = static_cast<unsigned char>((x * 7 + y * 13) % 64);
                data[y][x] = static_cast<unsigned char>(corner + texture);
                data_rotated[y][x] = 0;
            }
        }
        for (int v = -data_center + 1; v < data_center; ++v) {
            for (int u = -data_center + 1; u < data_center; ++u) {
                data_rotated[data_center + v][data_center + u] = data[data_center - u][data_center + v];
            }
        }
        constexpr static const float pi = 3.141592653589793f;
        feature::descriptor::binary<256> descriptor_0a;
        feature::descriptor::binary<256> descriptor_0b;
        feature::descriptor::binary<256> descriptor_90;
        feature::descriptor::orb::describe(&data[data_center][data_center], data_size, 0.0f, descriptor_0a);
        feature::descriptor::orb::describe(&data[data_center][data_center], data_size, 0.0f, descriptor_0b);
        feature::descriptor::orb::describe(&data_rotated[data_center][data_center], data_size, pi / 2.0f, descriptor_90);
        REQUIRE(match::distance::hamming::distance(descriptor_0a, descriptor_0b) == 0);
        REQUIRE(match::distance::hamming::distance(descriptor_0a, descriptor_90) <= 40);
    }
    {
        constexpr static const int border = 25;
        constexpr static const int data_size = 64;
        unsigned char* data = new unsigned char[data_size * data_size];
        for (int y = 0; y < data_size; ++y) {
            for (int x = 0; x < data_size; ++x) {
                data[y * data_size + x] = static_cast<unsigned char>((x * 5 + y * 11) % 256);
            }
        }
        const int positions[2] = { border, data_size - 1 - border };
        const float offsets[3] = { -4.0f, 0.0f, +4.0f };
        for (int py : positions) {
            for (int px : positions) {
                const unsigned char* feature = data + py * data_size + px;
                for (float oy : offsets) {
                    for (float ox : offsets) {
                        unsigned char patch[41][41];
                        feature::refiner::subpixel::patch_41x41_bilinear(feature, data_size, ox, oy, &patch[0][0]);
                        feature::descriptor::binary<256> descriptor;
                        feature::descriptor::orb::describe(&patch[20][20], 41, feature::angle::orb::dominant_angle(&patch[20][20], 41), descriptor);
                        for (int dy = -20; dy <= 20; ++dy) {
                            for (int dx = -20; dx <= 20; ++dx) {
                                REQUIRE(patch[20 + dy][20 + dx] == data[(py + static_cast<int>(oy) + dy) * data_size + (px + static_cast<int>(ox) + dx)]);
                            }
                        }
                    }
                }
                feature::descriptor::binary<256> descriptor;
                feature::descriptor::orb::describe(feature, data_size, feature::angle::orb::dominant_angle(feature, data_size), descriptor);
            }
        }
        {
            const int py = data_size / 2;
            const int px = data_size / 2;
            const unsigned char* feature = data + py * data_size + px;
            unsigned char patch[41][41];
            feature::refiner::subpixel::patch_41x41_bilinear(feature, data_size, -0.25f, +0.5f, &patch[0][0]);
            const float expected =
                0.25f * 0.5f * static_cast<float>(data[(py + 0) * data_size + (px - 1)]) +
                0.75f * 0.5f * static_cast<float>(data[(py + 0) * data_size + (px + 0)]) +
                0.25f * 0.5f * static_cast<float>(data[(py + 1) * data_size + (px - 1)]) +
                0.75f * 0.5f * static_cast<float>(data[(py + 1) * data_size + (px + 0)]);
            REQUIRE(patch[20][20] == static_cast<unsigned char>(math::round(expected)));
        }
        delete[] data;
    }
    {
        constexpr static const int data_size = 96;
        constexpr static const int data_center = data_size / 2;
        constexpr static const auto render = [](unsigned char* image, int cx, int cy) {
            for (int y = 0; y < data_size; ++y) {
                for (int x = 0; x < data_size; ++x) {
                    const int rx = x - cx;
                    const int ry = y - cy;
                    const int texture = (rx * 7 + ry * 13 + rx * ry) & 0x3F;
                    const int corner = ((rx >= 0) && (ry >= 0)) ? 160 : 0;
                    image[y * data_size + x] = static_cast<unsigned char>(texture + corner);
                }
            }
        };

        unsigned char data_base[data_size * data_size];
        unsigned char data_shifted[data_size * data_size];
        constexpr static const int shift_x = 5;
        constexpr static const int shift_y = 3;
        render(&data_base[0], data_center, data_center);
        render(&data_shifted[0], data_center + shift_x, data_center + shift_y);

        feature::descriptor::binary<256> descriptor_base;
        feature::descriptor::orb::describe(
            &data_base[data_center * data_size + data_center],
            data_size,
            feature::angle::orb::dominant_angle(&data_base[data_center * data_size + data_center], data_size),
            descriptor_base
        );

        feature::descriptor::binary<256> descriptor_shifted;
        feature::descriptor::orb::describe(
            &data_shifted[(data_center + shift_y) * data_size + (data_center + shift_x)],
            data_size,
            feature::angle::orb::dominant_angle(&data_shifted[(data_center + shift_y) * data_size + (data_center + shift_x)], data_size),
            descriptor_shifted
        );
        const unsigned int distance_integer_shift = match::distance::hamming::distance(descriptor_base, descriptor_shifted);
        REQUIRE(distance_integer_shift == 0);

        unsigned char patch_unshifted[41 * 41];
        unsigned char patch_subpixel[41 * 41];
        feature::refiner::subpixel::patch_41x41_bilinear(&data_base[data_center * data_size + data_center], data_size, 0.0f, 0.0f, &patch_unshifted[0]);
        feature::refiner::subpixel::patch_41x41_bilinear(&data_base[data_center * data_size + data_center], data_size, 0.25f, 0.25f, &patch_subpixel[0]);
        feature::descriptor::binary<256> descriptor_unshifted;
        feature::descriptor::binary<256> descriptor_subpixel;
        feature::descriptor::orb::describe(&patch_unshifted[20 * 41 + 20], 41, feature::angle::orb::dominant_angle(&patch_unshifted[20 * 41 + 20], 41), descriptor_unshifted);
        feature::descriptor::orb::describe(&patch_subpixel[20 * 41 + 20], 41, feature::angle::orb::dominant_angle(&patch_subpixel[20 * 41 + 20], 41), descriptor_subpixel);
        const unsigned int distance_subpixel_shift = match::distance::hamming::distance(descriptor_unshifted, descriptor_subpixel);
        REQUIRE(distance_subpixel_shift < 40);

        unsigned char data_unrelated[data_size * data_size];
        for (int y = 0; y < data_size; ++y) {
            for (int x = 0; x < data_size; ++x) {
                data_unrelated[y * data_size + x] = static_cast<unsigned char>((x * 5 + y * 11 + 37) % 256);
            }
        }
        feature::descriptor::binary<256> descriptor_unrelated;
        feature::descriptor::orb::describe(
            &data_unrelated[data_center * data_size + data_center],
            data_size,
            feature::angle::orb::dominant_angle(&data_unrelated[data_center * data_size + data_center], data_size),
            descriptor_unrelated
        );
        const unsigned int distance_unrelated = match::distance::hamming::distance(descriptor_base, descriptor_unrelated);
        REQUIRE(distance_unrelated > 100);
        REQUIRE(distance_integer_shift < distance_unrelated);
        REQUIRE(distance_subpixel_shift * 2 < distance_unrelated);
    }
#if defined(ZEROSLAM_SIMD_AVX2)
    {
        if (core::cpu::has_avx2() && core::cpu::has_popcnt()) {
            constexpr static const int data_width = 128;
            constexpr static const int data_height = 128;
            unsigned char data[data_height][data_width];
            for (int y = 0; y < data_height; ++y) {
                for (int x = 0; x < data_width; ++x) {
                    data[y][x] = static_cast<unsigned char>((x * 7 + y * 13 + (x * y) / 5) % 256);
                }
            }
            for (int step = 0; step < 64; ++step) {
                const float angle = static_cast<float>(step) * 0.1f - 3.0f;
                const unsigned char* const at = &data[data_height / 2 + (step % 5)][data_width / 2 + (step % 7)];
                feature::descriptor::binary<256> scalar;
                feature::descriptor::binary<256> simd;
                feature::descriptor::describe_cpu(at, data_width, angle, scalar);
                feature::descriptor::describe_avx2(at, data_width, angle, simd);
                for (size_t i = 0; i < 32; ++i) {
                    REQUIRE(scalar.data[i] == simd.data[i]);
                }
                feature::descriptor::orb::describe(at, data_width, angle, simd);
                for (size_t i = 0; i < 32; ++i) {
                    REQUIRE(scalar.data[i] == simd.data[i]);
                }
            }
        }
    }
#endif
#if defined(ZEROSLAM_SIMD_NEON)
    {
        if (core::cpu::has_neon()) {
            constexpr static const int data_width = 128;
            constexpr static const int data_height = 128;
            unsigned char data[data_height][data_width];
            for (int y = 0; y < data_height; ++y) {
                for (int x = 0; x < data_width; ++x) {
                    data[y][x] = static_cast<unsigned char>((x * 7 + y * 13 + (x * y) / 5) % 256);
                }
            }
            for (int step = 0; step < 64; ++step) {
                const float angle = static_cast<float>(step) * 0.1f - 3.0f;
                const unsigned char* const at = &data[data_height / 2 + (step % 5)][data_width / 2 + (step % 7)];
                feature::descriptor::binary<256> scalar;
                feature::descriptor::binary<256> simd;
                feature::descriptor::describe_cpu(at, data_width, angle, scalar);
                feature::descriptor::describe_neon(at, data_width, angle, simd);
                for (size_t i = 0; i < 32; ++i) {
                    REQUIRE(scalar.data[i] == simd.data[i]);
                }
                feature::descriptor::orb::describe(at, data_width, angle, simd);
                for (size_t i = 0; i < 32; ++i) {
                    REQUIRE(scalar.data[i] == simd.data[i]);
                }
            }
        }
    }
#endif

    return EXIT_SUCCESS;
}
