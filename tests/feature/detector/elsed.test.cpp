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

#include "feature/detector/elsed.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static double point_line_distance(const double px, const double py, const double a, const double b, const double c) {
    return std::abs((a * px) + (b * py) + c);
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    constexpr static const int width = 128;
    constexpr static const int height = 128;

    {
        std::vector<unsigned char> image(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
        std::vector<feature::detector::elsed::segment> segments(64);
        REQUIRE(feature::detector::elsed::detect(image.data(), width, height, width, feature::detector::elsed::options(), segments.data(), segments.size()) == 0);
    }

    {
        std::vector<unsigned char> image(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
        for (int y = 40; y < 80; ++y) {
            for (int x = 30; x < 90; ++x) {
                image[(static_cast<size_t>(y) * width) + static_cast<size_t>(x)] = 255;
            }
        }
        std::vector<feature::detector::elsed::segment> segments(256);
        const size_t count = feature::detector::elsed::detect(image.data(), width, height, width, feature::detector::elsed::options(), segments.data(), segments.size());
        REQUIRE(count >= 4);

        const double sides[4][3] = {
            { 1.0, 0.0, -30.0 },
            { 1.0, 0.0, -90.0 },
            { 0.0, 1.0, -40.0 },
            { 0.0, 1.0, -80.0 }
        };
        int matched_sides = 0;
        for (int side = 0; side < 4; ++side) {
            bool matched = false;
            for (size_t i = 0; i < count; ++i) {
                const double endpoint_error_1 = point_line_distance(static_cast<double>(segments[i].x1), static_cast<double>(segments[i].y1), sides[side][0], sides[side][1], sides[side][2]);
                const double endpoint_error_2 = point_line_distance(static_cast<double>(segments[i].x2), static_cast<double>(segments[i].y2), sides[side][0], sides[side][1], sides[side][2]);
                if ((endpoint_error_1 < 1.0) && (endpoint_error_2 < 1.0) && (segments[i].length > 20.0f)) {
                    matched = true;
                    break;
                }
            }
            matched_sides += matched;
        }
        REQUIRE(matched_sides == 4);

        std::vector<feature::detector::elsed::segment> segments_repeat(256);
        const size_t count_repeat = feature::detector::elsed::detect(image.data(), width, height, width, feature::detector::elsed::options(), segments_repeat.data(), segments_repeat.size());
        REQUIRE(count_repeat == count);
        REQUIRE(std::memcmp(segments.data(), segments_repeat.data(), count * sizeof(feature::detector::elsed::segment)) == 0);
    }

    {
        std::vector<unsigned char> image(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if ((y - x) > -8 && (y - x) < 8) {
                    image[(static_cast<size_t>(y) * width) + static_cast<size_t>(x)] = 200;
                }
            }
        }
        std::vector<feature::detector::elsed::segment> segments(256);
        const size_t count = feature::detector::elsed::detect(image.data(), width, height, width, feature::detector::elsed::options(), segments.data(), segments.size());
        REQUIRE(count >= 1);
        bool diagonal_found = false;
        for (size_t i = 0; i < count; ++i) {
            const double dx = static_cast<double>(segments[i].x2 - segments[i].x1);
            const double dy = static_cast<double>(segments[i].y2 - segments[i].y1);
            const double length = std::sqrt((dx * dx) + (dy * dy));
            if (length < 40.0) {
                continue;
            }
            const double alignment = std::abs(((dx + dy) / std::sqrt(2.0)) / length);
            if (alignment > 0.996) {
                diagonal_found = true;
                break;
            }
        }
        REQUIRE(diagonal_found);
    }

    return EXIT_SUCCESS;
}
