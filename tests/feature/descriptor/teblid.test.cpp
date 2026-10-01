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

#include "feature/descriptor/teblid.hpp"

#include "feature/angle/orb.hpp"
#include "match/distance/hamming.hpp"

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

static double texture(double x, double y) {
    return 128.0 + 55.0 * std::sin(0.21 * x) * std::sin(0.19 * y) + 35.0 * std::cos(0.15 * x + 0.10 * y) + 25.0 * std::sin(0.30 * x - 0.12 * y);
}

static std::vector<unsigned char> render(const int size, const double angle, const double shift_x, const double shift_y) {
    std::vector<unsigned char> image(static_cast<size_t>(size * size));
    const double centre = 0.5 * static_cast<double>(size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const double dx = static_cast<double>(x) - centre;
            const double dy = static_cast<double>(y) - centre;
            const double sx = (std::cos(angle) * dx) + (std::sin(angle) * dy) + centre + shift_x;
            const double sy = (-std::sin(angle) * dx) + (std::cos(angle) * dy) + centre + shift_y;
            image[static_cast<size_t>((y * size) + x)] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, texture(sx, sy) + 0.5)));
        }
    }
    return image;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);
    constexpr static const int size = 96;
    const int centre = size / 2;
    const std::vector<unsigned char> plain = render(size, 0.0, 0.0, 0.0);
    const unsigned char* const at_centre = plain.data() + (centre * size) + centre;
    const float angle = feature::angle::orb::dominant_angle(at_centre, size);

    feature::descriptor::binary<256> first;
    feature::descriptor::binary<256> again;
    feature::descriptor::teblid::describe(at_centre, size, angle, first);
    feature::descriptor::teblid::describe(at_centre, size, angle, again);
    REQUIRE(match::distance::hamming::distance(first, again) == 0);

    unsigned int set_bits = 0;
    for (int byte = 0; byte < 32; ++byte) {
        for (int bit = 0; bit < 8; ++bit) {
            set_bits += (static_cast<unsigned int>(first.data[byte]) >> bit) & 1u;
        }
    }
    REQUIRE((set_bits > 40) && (set_bits < 216));

    const std::vector<unsigned char> shifted = render(size, 0.0, 0.4, -0.3);
    feature::descriptor::binary<256> descriptor_shifted;
    feature::descriptor::teblid::describe(shifted.data() + (centre * size) + centre, size, feature::angle::orb::dominant_angle(shifted.data() + (centre * size) + centre, size), descriptor_shifted);
    REQUIRE(match::distance::hamming::distance(first, descriptor_shifted) < 40);

    const std::vector<unsigned char> rotated = render(size, 0.6, 0.0, 0.0);
    const unsigned char* const rotated_centre = rotated.data() + (centre * size) + centre;
    feature::descriptor::binary<256> descriptor_rotated;
    feature::descriptor::teblid::describe(rotated_centre, size, feature::angle::orb::dominant_angle(rotated_centre, size), descriptor_rotated);
    REQUIRE(match::distance::hamming::distance(first, descriptor_rotated) < 70);

    const unsigned char* const elsewhere = plain.data() + ((centre + 17) * size) + centre - 23;
    feature::descriptor::binary<256> descriptor_elsewhere;
    feature::descriptor::teblid::describe(elsewhere, size, feature::angle::orb::dominant_angle(elsewhere, size), descriptor_elsewhere);
    REQUIRE(match::distance::hamming::distance(first, descriptor_elsewhere) > 70);

    {
        // The level sums, on a region narrower than its stride.
        const int width = 7;
        const int height = 5;
        const int stride = 9;
        std::vector<unsigned char> data(static_cast<size_t>(stride * height));
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<unsigned char>((i * 37) % 251);
        }
        std::vector<unsigned int> sums(static_cast<size_t>((width + 1) * (height + 1)), 12345u);
        feature::descriptor::teblid::integral(data.data(), width, height, stride, sums.data());
        for (int y = 0; y <= height; ++y) {
            for (int x = 0; x <= width; ++x) {
                unsigned int expected = 0;
                for (int v = 0; v < y; ++v) {
                    for (int u = 0; u < x; ++u) {
                        expected += data[static_cast<size_t>(v * stride + u)];
                    }
                }
                REQUIRE(sums[static_cast<size_t>(y * (width + 1) + x)] == expected);
            }
        }
    }

    {
        std::vector<unsigned int> sums(static_cast<size_t>((size + 1) * (size + 1)));
        feature::descriptor::teblid::integral(plain.data(), size, size, size, sums.data());
        const int radius = feature::descriptor::teblid::window_radius;

        // Integral centres match the per keypoint description bit for bit, and a wrapped sum leaves every box sum exact.
        std::vector<unsigned int> wrapped(sums);
        for (unsigned int& sum : wrapped) {
            sum += 0xFFFFF000u;
        }
        for (int y = radius; y < size - radius; y += 3) {
            for (int x = radius; x < size - radius; x += 5) {
                const float keypoint_angle = static_cast<float>(x * 7 + y * 3) * 0.1f;
                feature::descriptor::binary<256> expected;
                feature::descriptor::binary<256> described;
                feature::descriptor::binary<256> described_wrapped;
                feature::descriptor::teblid::describe(plain.data() + (y * size) + x, size, keypoint_angle, expected);
                feature::descriptor::teblid::describe_integral(sums.data(), size + 1, static_cast<float>(x), static_cast<float>(y), keypoint_angle, described);
                feature::descriptor::teblid::describe_integral(wrapped.data(), size + 1, static_cast<float>(x), static_cast<float>(y), keypoint_angle, described_wrapped);
                REQUIRE(match::distance::hamming::distance(expected, described) == 0);
                REQUIRE(match::distance::hamming::distance(expected, described_wrapped) == 0);
                if ((x < size - radius - 1) && (y < size - radius - 1)) {
                    feature::descriptor::teblid::describe_integral(sums.data(), size + 1, static_cast<float>(x) + 0.37f, static_cast<float>(y) + 0.81f, keypoint_angle, described);
                    feature::descriptor::teblid::describe_integral(wrapped.data(), size + 1, static_cast<float>(x) + 0.37f, static_cast<float>(y) + 0.81f, keypoint_angle, described_wrapped);
                    REQUIRE(match::distance::hamming::distance(described, described_wrapped) == 0);
                }
            }
        }

        // A fractional centre blends the four surrounding integral centres, so a bit they all agree on keeps its value.
        for (int y = radius; y < size - radius - 1; y += 7) {
            for (int x = radius; x < size - radius - 1; x += 7) {
                const float keypoint_angle = static_cast<float>(x - y) * 0.05f;
                feature::descriptor::binary<256> corners[4];
                for (int corner = 0; corner < 4; ++corner) {
                    feature::descriptor::teblid::describe(plain.data() + ((y + corner / 2) * size) + x + (corner % 2), size, keypoint_angle, corners[corner]);
                }
                feature::descriptor::binary<256> fractional;
                feature::descriptor::teblid::describe_integral(sums.data(), size + 1, static_cast<float>(x) + 0.25f, static_cast<float>(y) + 0.5f, keypoint_angle, fractional);
                for (int byte = 0; byte < 32; ++byte) {
                    const unsigned int agree_set = static_cast<unsigned int>(corners[0].data[byte] & corners[1].data[byte] & corners[2].data[byte] & corners[3].data[byte]);
                    const unsigned int agree_clear = static_cast<unsigned int>(~(corners[0].data[byte] | corners[1].data[byte] | corners[2].data[byte] | corners[3].data[byte])) & 0xFFu;
                    REQUIRE((static_cast<unsigned int>(fractional.data[byte]) & agree_set) == agree_set);
                    REQUIRE((static_cast<unsigned int>(fractional.data[byte]) & agree_clear) == 0u);
                }
            }
        }

        // Content shifted by a fraction of a pixel is found again by describing at the fractional centre, not at the nearest pixel.
        int fractional_total = 0;
        int nearest_total = 0;
        for (int shift = 0; shift < 3; ++shift) {
            const double shift_x = 0.13 + 0.3 * static_cast<double>(shift);
            const double shift_y = -0.41 + 0.35 * static_cast<double>(shift);
            const std::vector<unsigned char> moved = render(size, 0.0, shift_x, shift_y);
            std::vector<unsigned int> moved_sums(sums.size());
            feature::descriptor::teblid::integral(moved.data(), size, size, size, moved_sums.data());
            for (int y = centre - 12; y <= centre + 12; y += 12) {
                for (int x = centre - 12; x <= centre + 12; x += 12) {
                    const float keypoint_angle = feature::angle::orb::dominant_angle(plain.data() + (y * size) + x, size);
                    feature::descriptor::binary<256> reference;
                    feature::descriptor::binary<256> fractional;
                    feature::descriptor::binary<256> nearest;
                    feature::descriptor::teblid::describe(plain.data() + (y * size) + x, size, keypoint_angle, reference);
                    const float moved_x = static_cast<float>(static_cast<double>(x) - shift_x);
                    const float moved_y = static_cast<float>(static_cast<double>(y) - shift_y);
                    feature::descriptor::teblid::describe_integral(moved_sums.data(), size + 1, moved_x, moved_y, keypoint_angle, fractional);
                    feature::descriptor::teblid::describe_integral(moved_sums.data(), size + 1, std::floor(moved_x + 0.5f), std::floor(moved_y + 0.5f), keypoint_angle, nearest);
                    const unsigned int fractional_distance = match::distance::hamming::distance(reference, fractional);
                    REQUIRE(fractional_distance <= 8);
                    fractional_total += static_cast<int>(fractional_distance);
                    nearest_total += static_cast<int>(match::distance::hamming::distance(reference, nearest));
                }
            }
        }
        REQUIRE(4 * fractional_total < nearest_total);
    }
    return EXIT_SUCCESS;
}
