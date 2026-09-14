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

#include "feature/detector/curvature_extrema.hpp"

#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdint>
#include <cstdio>
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

    {
        const int width = 24;
        const int height = 16;
        std::vector<std::int64_t> kappa(static_cast<size_t>(width * height), 0);
        std::vector<feature::detector::curvature_extrema::extremum> extrema;
        feature::detector::curvature_extrema::options settings;
        settings.quantile = 0.0f;

        REQUIRE(feature::detector::curvature_extrema::detect(kappa.data(), width, height, width, settings, extrema) == 0);

        kappa[7 * width + 10] = 500;
        std::int64_t threshold = -1;
        REQUIRE(feature::detector::curvature_extrema::detect(kappa.data(), width, height, width, settings, extrema, &threshold) == 1);
        REQUIRE(extrema[0].cell_x == 10);
        REQUIRE(extrema[0].cell_y == 7);
        REQUIRE(extrema[0].sign == 1);
        REQUIRE(extrema[0].response == 500);
        REQUIRE(extrema[0].x == 10.0f);
        REQUIRE(extrema[0].y == 7.0f);
        REQUIRE(threshold == 0);

        kappa[7 * width + 10] = -500;
        REQUIRE(feature::detector::curvature_extrema::detect(kappa.data(), width, height, width, settings, extrema) == 1);
        REQUIRE(extrema[0].sign == -1);
        REQUIRE(extrema[0].response == 500);
    }

    {
        const int width = 16;
        const int height = 16;
        std::vector<std::int64_t> kappa(static_cast<size_t>(width * height), 0);
        kappa[2 * width + 2] = 100;
        std::vector<feature::detector::curvature_extrema::extremum> extrema;
        feature::detector::curvature_extrema::options settings;
        settings.quantile = 0.0f;
        settings.border = 2;
        REQUIRE(feature::detector::curvature_extrema::detect(kappa.data(), width, height, width, settings, extrema) == 1);
        settings.border = 3;
        REQUIRE(feature::detector::curvature_extrema::detect(kappa.data(), width, height, width, settings, extrema) == 0);
        settings.border = 0;
        REQUIRE(feature::detector::curvature_extrema::detect(kappa.data(), width, height, width, settings, extrema) == 1);
    }

    {
        const int width = 64;
        const int height = 12;
        std::vector<std::int64_t> kappa(static_cast<size_t>(width * height), 0);
        for (int i = 0; i < 10; ++i) {
            kappa[static_cast<size_t>(5 * width + (4 + 5 * i))] = 100 * (i + 1);
        }
        std::vector<feature::detector::curvature_extrema::extremum> extrema;
        feature::detector::curvature_extrema::options settings;
        settings.quantile = 0.5f;
        std::int64_t threshold = 0;
        REQUIRE(feature::detector::curvature_extrema::detect(kappa.data(), width, height, width, settings, extrema, &threshold) == 6);
        REQUIRE(threshold == 500);
        for (size_t i = 0; i < extrema.size(); ++i) {
            REQUIRE(extrema[i].response == 100 * static_cast<std::int64_t>(i + 5));
            if (i > 0) {
                REQUIRE(extrema[i].cell_x > extrema[i - 1].cell_x);
            }
        }
    }

    {
        const int width = 64;
        const int height = 12;
        std::vector<std::int64_t> kappa(static_cast<size_t>(width * height), 0);
        for (int i = 0; i < 10; ++i) {
            kappa[static_cast<size_t>(5 * width + (4 + 5 * i))] = 100 * (i + 1);
        }
        std::vector<feature::detector::curvature_extrema::extremum> extrema;
        feature::detector::curvature_extrema::options settings;
        settings.quantile = 0.0f;
        settings.maximum = 3;
        REQUIRE(feature::detector::curvature_extrema::detect(kappa.data(), width, height, width, settings, extrema) == 3);
        REQUIRE(extrema[0].response == 800);
        REQUIRE(extrema[1].response == 900);
        REQUIRE(extrema[2].response == 1000);
    }

    {
        const int width = 32;
        const int height = 24;
        const double peak_x = 12.3;
        const double peak_y = 9.6;
        std::vector<std::int64_t> kappa(static_cast<size_t>(width * height), 0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const double du = static_cast<double>(x) - peak_x;
                const double dv = static_cast<double>(y) - peak_y;
                kappa[static_cast<size_t>(y * width + x)] = static_cast<std::int64_t>(200000.0 - 1000.0 * (du * du + dv * dv));
            }
        }
        std::vector<feature::detector::curvature_extrema::extremum> extrema;
        feature::detector::curvature_extrema::options settings;
        settings.quantile = 0.0f;
        settings.subpixel = true;
        REQUIRE(feature::detector::curvature_extrema::detect(kappa.data(), width, height, width, settings, extrema) == 1);
        REQUIRE(extrema[0].cell_x == 12);
        REQUIRE(extrema[0].cell_y == 10);
        REQUIRE(extrema[0].sign == 1);
        REQUIRE(math::abs(extrema[0].x - static_cast<float>(peak_x)) < 0.05f);
        REQUIRE(math::abs(extrema[0].y - static_cast<float>(peak_y)) < 0.05f);
        settings.subpixel = false;
        REQUIRE(feature::detector::curvature_extrema::detect(kappa.data(), width, height, width, settings, extrema) == 1);
        REQUIRE(extrema[0].x == 12.0f);
        REQUIRE(extrema[0].y == 10.0f);
    }

    {
        const int width = 64;
        const int height = 64;
        const int corner = 32;
        std::vector<unsigned char> data(static_cast<size_t>(width * height));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const bool bright = ((x < corner) ^ (y < corner));
                data[static_cast<size_t>(y * width + x)] = bright ? 200 : 40;
            }
        }
        std::vector<std::int64_t> kappa;
        std::vector<feature::detector::curvature_extrema::extremum> extrema;
        feature::detector::curvature_extrema::options settings;
        settings.quantile = 0.0f;
        const size_t count = feature::detector::curvature_extrema::detect(data.data(), width, height, width, settings, kappa, extrema);
        REQUIRE(count > 0);
        REQUIRE(kappa.size() == data.size());
        for (const feature::detector::curvature_extrema::extremum& found : extrema) {
            REQUIRE(math::abs(found.x - static_cast<float>(corner)) < 8.0f);
            REQUIRE(math::abs(found.y - static_cast<float>(corner)) < 8.0f);
        }
        std::vector<std::int64_t> kappa_repeat;
        std::vector<feature::detector::curvature_extrema::extremum> extrema_repeat;
        REQUIRE(feature::detector::curvature_extrema::detect(data.data(), width, height, width, settings, kappa_repeat, extrema_repeat) == count);
        for (size_t i = 0; i < extrema.size(); ++i) {
            REQUIRE(extrema[i].x == extrema_repeat[i].x);
            REQUIRE(extrema[i].y == extrema_repeat[i].y);
            REQUIRE(extrema[i].sign == extrema_repeat[i].sign);
            REQUIRE(extrema[i].response == extrema_repeat[i].response);
        }
    }

    {
        const int width = 96;
        const int height = 72;
        const int shift_x = 5;
        const int shift_y = 3;
        std::vector<unsigned char> base(static_cast<size_t>((width + shift_x) * (height + shift_y)));
        unsigned long long state = 7;
        for (size_t i = 0; i < base.size(); ++i) {
            state += 0x9E3779B97F4A7C15ull;
            unsigned long long z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            base[i] = static_cast<unsigned char>((z ^ (z >> 31)) & 0xFFull);
        }
        const int base_stride = width + shift_x;
        std::vector<unsigned char> frame_a(static_cast<size_t>(width * height));
        std::vector<unsigned char> frame_b(static_cast<size_t>(width * height));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                frame_a[static_cast<size_t>(y * width + x)] = base[static_cast<size_t>((y + shift_y) * base_stride + (x + shift_x))];
                frame_b[static_cast<size_t>(y * width + x)] = base[static_cast<size_t>(y * base_stride + x)];
            }
        }
        std::vector<std::int64_t> kappa_a;
        std::vector<std::int64_t> kappa_b;
        std::vector<feature::detector::curvature_extrema::extremum> extrema_a;
        std::vector<feature::detector::curvature_extrema::extremum> extrema_b;
        feature::detector::curvature_extrema::options settings;
        settings.quantile = 0.5f;
        feature::detector::curvature_extrema::detect(frame_a.data(), width, height, width, settings, kappa_a, extrema_a);
        feature::detector::curvature_extrema::detect(frame_b.data(), width, height, width, settings, kappa_b, extrema_b);
        REQUIRE(extrema_a.size() > 20);
        size_t interior = 0;
        size_t matched = 0;
        for (const feature::detector::curvature_extrema::extremum& found : extrema_a) {
            const int expected_x = found.cell_x + shift_x;
            const int expected_y = found.cell_y + shift_y;
            if ((expected_x < 8) || (expected_x >= width - 8) || (expected_y < 8) || (expected_y >= height - 8)) {
                continue;
            }
            ++interior;
            for (const feature::detector::curvature_extrema::extremum& candidate : extrema_b) {
                if ((candidate.cell_x == expected_x) && (candidate.cell_y == expected_y) && (candidate.sign == found.sign) && (candidate.response == found.response)) {
                    ++matched;
                    break;
                }
            }
        }
        REQUIRE(interior > 10);
        REQUIRE(matched * 10 >= interior * 9);
    }

    std::printf("All curvature extrema tests passed.\n");
    return 0;
}
