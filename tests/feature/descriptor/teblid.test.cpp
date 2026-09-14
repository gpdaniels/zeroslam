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
    return EXIT_SUCCESS;
}
