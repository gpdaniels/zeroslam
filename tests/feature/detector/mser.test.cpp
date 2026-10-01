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

#include "core/timestamp.hpp"

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

static bool is_match(const feature::point& feature, const region& expected, const double centre_tolerance) {
    const double dx = static_cast<double>(feature.x) - (expected.sum_x / static_cast<double>(expected.area));
    const double dy = static_cast<double>(feature.y) - (expected.sum_y / static_cast<double>(expected.area));
    const double area = 3.14159265358979323846 * static_cast<double>(feature.angle) * static_cast<double>(feature.angle);
    return (std::sqrt((dx * dx) + (dy * dy)) < centre_tolerance) && (std::abs(area - static_cast<double>(expected.area)) < 0.5);
}

static int count_matches(const std::vector<feature::point>& features, const size_t count, const region& expected, const double centre_tolerance) {
    int matches = 0;
    for (size_t i = 0; i < count; ++i) {
        if (is_match(features[i], expected, centre_tolerance)) {
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

    {
        // Every detection must be an extremal region, a 4-connected component of the pixels at or below a level or at or above it, found here by flood filling every level.
        constexpr static const int width = 64;
        constexpr static const int height = 48;
        std::vector<unsigned char> image(static_cast<size_t>(width * height));
        unsigned int seed = 779u;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                seed = (seed * 1103515245u) + 12345u;
                const double wave = (60.0 * std::sin(0.35 * static_cast<double>(x)) * std::sin(0.41 * static_cast<double>(y))) + (40.0 * std::cos((0.23 * static_cast<double>(x)) + (0.17 * static_cast<double>(y))));
                const int value = 128 + static_cast<int>(wave) + (2 * (static_cast<int>((seed >> 16) % 9) - 4));
                image[static_cast<size_t>((y * width) + x)] = static_cast<unsigned char>(((value < 0) ? 0 : ((value > 255) ? 255 : value)) / 8 * 8);
            }
        }
        std::vector<feature::point> features(4096);
        feature::detector::mser::options settings;
        settings.minimum_area = 5;
        const size_t count = feature::detector::mser::detect(image.data(), width, height, width, settings, features.size(), features.data());
        REQUIRE(count > 100);

        bool present[256] = {};
        for (const unsigned char value : image) {
            present[value] = true;
        }
        std::vector<region> extremal_regions;
        std::vector<unsigned char> visited(image.size());
        std::vector<int> stack;
        for (int polarity = 0; polarity < 2; ++polarity) {
            for (int level = 0; level < 256; ++level) {
                if (!present[level]) {
                    continue;
                }
                const auto inside = [&](const int index) {
                    const int value = image[static_cast<size_t>(index)];
                    return (polarity == 0) ? (value <= level) : (value >= level);
                };
                for (unsigned char& entry : visited) {
                    entry = 0;
                }
                for (int start = 0; start < width * height; ++start) {
                    if ((visited[static_cast<size_t>(start)] != 0) || !inside(start)) {
                        continue;
                    }
                    region component;
                    visited[static_cast<size_t>(start)] = 1;
                    stack.push_back(start);
                    while (!stack.empty()) {
                        const int index = stack.back();
                        stack.pop_back();
                        const int x = index % width;
                        const int y = index / width;
                        component.add(x, y);
                        const int neighbours[4][2] = { { x - 1, y }, { x + 1, y }, { x, y - 1 }, { x, y + 1 } };
                        for (const int (&neighbour)[2] : neighbours) {
                            if ((neighbour[0] < 0) || (neighbour[0] >= width) || (neighbour[1] < 0) || (neighbour[1] >= height)) {
                                continue;
                            }
                            const int next = (neighbour[1] * width) + neighbour[0];
                            if ((visited[static_cast<size_t>(next)] == 0) && inside(next)) {
                                visited[static_cast<size_t>(next)] = 1;
                                stack.push_back(next);
                            }
                        }
                    }
                    extremal_regions.push_back(component);
                }
            }
        }
        for (size_t i = 0; i < count; ++i) {
            bool extremal = false;
            for (const region& candidate : extremal_regions) {
                extremal = extremal || is_match(features[i], candidate, 1.0e-3);
            }
            REQUIRE(extremal);
        }
    }

    {
        constexpr static const int width = 640;
        constexpr static const int height = 480;
        std::vector<unsigned char> image(static_cast<size_t>(width * height));
        unsigned int seed = 3u;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                seed = (seed * 1103515245u) + 12345u;
                image[static_cast<size_t>((y * width) + x)] = static_cast<unsigned char>((((x / 13) + (y / 11)) % 2) * 120 + 60 + static_cast<int>((seed >> 16) % 20));
            }
        }
        std::vector<feature::point> features(50000);
        feature::detector::mser::options settings;
        constexpr int iteration_count = 10;
        size_t count = 0;
        const long long int start = core::timestamp();
        for (int i = 0; i < iteration_count; ++i) {
            count = feature::detector::mser::detect(image.data(), width, height, width, settings, features.size(), features.data());
        }
        const long long int end = core::timestamp();
        REQUIRE(count > 0);
        std::printf("Detect benchmark (%d iterations of %dx%d): %.1f ms, %zu regions\n", iteration_count, width, height, static_cast<double>(end - start) * 1.0e-6, count);
    }

    return EXIT_SUCCESS;
}
