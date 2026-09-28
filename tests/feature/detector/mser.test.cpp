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

class region final {
public:
    double sum_x = 0.0;
    double sum_y = 0.0;
    int area = 0;

public:
    void add(const int x, const int y) {
        this->sum_x += static_cast<double>(x);
        this->sum_y += static_cast<double>(y);
        ++this->area;
    }
};

static int count_matches(const std::vector<feature::point>& features, const size_t count, const region& expected, const double centre_tolerance) {
    int matches = 0;
    for (size_t i = 0; i < count; ++i) {
        const double dx = static_cast<double>(features[i].x) - (expected.sum_x / static_cast<double>(expected.area));
        const double dy = static_cast<double>(features[i].y) - (expected.sum_y / static_cast<double>(expected.area));
        const double area = 3.14159265358979323846 * static_cast<double>(features[i].angle) * static_cast<double>(features[i].angle);
        if ((std::sqrt((dx * dx) + (dy * dy)) < centre_tolerance) && (std::abs(area - static_cast<double>(expected.area)) < 0.5)) {
            ++matches;
        }
    }
    return matches;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        constexpr static const int width = 160;
        constexpr static const int height = 120;
        std::vector<unsigned char> image(static_cast<size_t>(width * height), static_cast<unsigned char>(140));
        std::vector<feature::point> features(64);
        feature::detector::mser::options settings;
        REQUIRE(feature::detector::mser::detect(image.data(), width, height, width, settings, features.size(), features.data()) == 0);
    }

    {
        constexpr static const int width = 200;
        constexpr static const int height = 120;
        std::vector<unsigned char> image(static_cast<size_t>(width * height), static_cast<unsigned char>(200));
        region regions[7];
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                unsigned char value = 200;
                const int distance = ((x - 40) * (x - 40)) + ((y - 60) * (y - 60));
                if (distance <= 30 * 30) {
                    value = 120;
                    regions[0].add(x, y);
                }
                if (distance <= 18 * 18) {
                    value = 70;
                    regions[1].add(x, y);
                }
                if (distance <= 8 * 8) {
                    value = 20;
                    regions[2].add(x, y);
                }
                const bool left_arm = (x >= 80) && (x < 92) && (y >= 15) && (y < 85);
                const bool right_arm = (x >= 120) && (x < 132) && (y >= 15) && (y < 85);
                const bool bar = (x >= 80) && (x < 132) && (y >= 85) && (y < 95);
                if (left_arm || right_arm || bar) {
                    value = 60;
                    regions[3].add(x, y);
                }
                if ((x >= 123) && (x < 129) && (y >= 35) && (y < 55)) {
                    value = 10;
                    regions[4].add(x, y);
                }
                if ((x >= 150) && (x < 170) && (y >= 50) && (y < 66)) {
                    value = 250;
                    regions[5].add(x, y);
                }
                if ((x >= 154) && (x < 166) && (y >= 20) && (y < 32)) {
                    value = 180;
                    regions[6].add(x, y);
                }
                image[static_cast<size_t>((y * width) + x)] = value;
            }
        }
        std::vector<feature::point> features(64);
        feature::detector::mser::options settings;
        const size_t count = feature::detector::mser::detect(image.data(), width, height, width, settings, features.size(), features.data());
        REQUIRE(count == 7);
        for (const region& expected : regions) {
            REQUIRE(count_matches(features, count, expected, 1.0e-3) == 1);
        }
    }

    {
        constexpr static const int width = 160;
        constexpr static const int height = 120;
        std::vector<unsigned char> image(static_cast<size_t>(width * height));
        const int discs[3][3] = { { 40, 40, 9 }, { 110, 70, 14 }, { 60, 95, 7 } };
        const unsigned char shade[3] = { 40, 60, 230 };
        region regions[3];
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
                        regions[d].add(x, y);
                    }
                }
                image[static_cast<size_t>((y * width) + x)] = static_cast<unsigned char>(value);
            }
        }
        std::vector<feature::point> features(64);
        feature::detector::mser::options settings;
        const size_t count = feature::detector::mser::detect(image.data(), width, height, width, settings, features.size(), features.data());
        REQUIRE(count == 3);
        for (const region& expected : regions) {
            REQUIRE(count_matches(features, count, expected, 1.0e-3) == 1);
        }
    }

    return EXIT_SUCCESS;
}
