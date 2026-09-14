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

#include "feature/detector/mser.hpp"

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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);
    constexpr static const int width = 160;
    constexpr static const int height = 120;
    std::vector<unsigned char> image(static_cast<size_t>(width * height), static_cast<unsigned char>(140));
    std::vector<feature::point> features(64);
    feature::detector::mser::options settings;
    REQUIRE(feature::detector::mser::detect(image.data(), width, height, width, settings, features.size(), features.data()) == 0);

    const int discs[3][3] = { { 40, 40, 9 }, { 110, 70, 14 }, { 60, 95, 7 } };
    const unsigned char shade[3] = { 40, 60, 230 };
    unsigned int seed = 12345u;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            seed = (seed * 1103515245u) + 12345u;
            int value = 130 + (x / 8) + static_cast<int>((seed >> 16) % 7) - 3;
            for (int d = 0; d < 3; ++d) {
                const int dx = x - discs[d][0];
                const int dy = y - discs[d][1];
                if ((dx * dx) + (dy * dy) <= discs[d][2] * discs[d][2]) {
                    value = shade[d] + static_cast<int>((seed >> 8) % 5) - 2;
                }
            }
            image[static_cast<size_t>((y * width) + x)] = static_cast<unsigned char>(value);
        }
    }
    const size_t count = feature::detector::mser::detect(image.data(), width, height, width, settings, features.size(), features.data());
    REQUIRE(count >= 3);
    for (int d = 0; d < 3; ++d) {
        bool found = false;
        for (size_t i = 0; i < count; ++i) {
            const double dx = static_cast<double>(features[i].x) - discs[d][0];
            const double dy = static_cast<double>(features[i].y) - discs[d][1];
            if ((std::sqrt((dx * dx) + (dy * dy)) < 2.0) && (std::abs(static_cast<double>(features[i].angle) - discs[d][2]) < 2.0)) {
                found = true;
            }
        }
        REQUIRE(found);
    }
    for (size_t i = 0; i < count; ++i) {
        bool near = false;
        for (int d = 0; d < 3; ++d) {
            const double dx = static_cast<double>(features[i].x) - discs[d][0];
            const double dy = static_cast<double>(features[i].y) - discs[d][1];
            near = near || (std::sqrt((dx * dx) + (dy * dy)) < 3.0);
        }
        REQUIRE(near);
    }
    return EXIT_SUCCESS;
}
