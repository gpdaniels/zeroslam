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

#include "image/derivative.hpp"

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

namespace {
    void reference_sobel(
        const unsigned char* data,
        const int width,
        const int height,
        const int stride,
        std::int16_t* dx,
        std::int16_t* dy,
        std::int16_t* dxx,
        std::int16_t* dyy,
        std::int16_t* dxy
    ) {
        const int kernel_dx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
        const int kernel_dy[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
        const int kernel_dxx[3][3] = { { 1, -2, 1 }, { 2, -4, 2 }, { 1, -2, 1 } };
        const int kernel_dyy[3][3] = { { 1, 2, 1 }, { -2, -4, -2 }, { 1, 2, 1 } };
        const int kernel_dxy[3][3] = { { 1, 0, -1 }, { 0, 0, 0 }, { -1, 0, 1 } };
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const int index = y * stride + x;
                if ((y == 0) || (y == height - 1) || (x == 0) || (x == width - 1)) {
                    dx[index] = 0;
                    dy[index] = 0;
                    dxx[index] = 0;
                    dyy[index] = 0;
                    dxy[index] = 0;
                    continue;
                }
                int sum_dx = 0;
                int sum_dy = 0;
                int sum_dxx = 0;
                int sum_dyy = 0;
                int sum_dxy = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        const int value = data[(y + ky) * stride + (x + kx)];
                        sum_dx += kernel_dx[ky + 1][kx + 1] * value;
                        sum_dy += kernel_dy[ky + 1][kx + 1] * value;
                        sum_dxx += kernel_dxx[ky + 1][kx + 1] * value;
                        sum_dyy += kernel_dyy[ky + 1][kx + 1] * value;
                        sum_dxy += kernel_dxy[ky + 1][kx + 1] * value;
                    }
                }
                dx[index] = static_cast<std::int16_t>(sum_dx);
                dy[index] = static_cast<std::int16_t>(sum_dy);
                dxx[index] = static_cast<std::int16_t>(sum_dxx);
                dyy[index] = static_cast<std::int16_t>(sum_dyy);
                dxy[index] = static_cast<std::int16_t>(sum_dxy);
            }
        }
    }

    unsigned char next_random_byte(unsigned long long& state) {
        state += 0x9E3779B97F4A7C15ull;
        unsigned long long z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z = z ^ (z >> 31);
        return static_cast<unsigned char>(z & 0xFFull);
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        const int width = 16;
        const int height = 12;
        std::vector<unsigned char> data(static_cast<size_t>(width * height), 137);
        std::vector<std::int16_t> dx(data.size());
        std::vector<std::int16_t> dy(data.size());
        std::vector<std::int16_t> dxx(data.size());
        std::vector<std::int16_t> dyy(data.size());
        std::vector<std::int16_t> dxy(data.size());
        std::vector<std::int64_t> kappa(data.size());
        image::derivative::curvature(data.data(), width, height, width, dx.data(), dy.data(), dxx.data(), dyy.data(), dxy.data(), kappa.data());
        for (size_t i = 0; i < data.size(); ++i) {
            REQUIRE(dx[i] == 0);
            REQUIRE(dy[i] == 0);
            REQUIRE(dxx[i] == 0);
            REQUIRE(dyy[i] == 0);
            REQUIRE(dxy[i] == 0);
            REQUIRE(kappa[i] == 0);
        }
    }

    {
        const int width = 32;
        const int height = 8;
        std::vector<unsigned char> data(static_cast<size_t>(width * height));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                data[static_cast<size_t>(y * width + x)] = static_cast<unsigned char>(x);
            }
        }
        std::vector<std::int16_t> dx(data.size());
        std::vector<std::int16_t> dy(data.size());
        std::vector<std::int16_t> dxx(data.size());
        std::vector<std::int16_t> dyy(data.size());
        std::vector<std::int16_t> dxy(data.size());
        image::derivative::sobel(data.data(), width, height, width, dx.data(), dy.data(), dxx.data(), dyy.data(), dxy.data());
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                const size_t index = static_cast<size_t>(y * width + x);
                REQUIRE(dx[index] == 8);
                REQUIRE(dy[index] == 0);
                REQUIRE(dxx[index] == 0);
                REQUIRE(dyy[index] == 0);
                REQUIRE(dxy[index] == 0);
            }
        }
    }

    {
        const int width = 8;
        const int height = 24;
        std::vector<unsigned char> data(static_cast<size_t>(width * height));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                data[static_cast<size_t>(y * width + x)] = static_cast<unsigned char>(3 * y);
            }
        }
        std::vector<std::int16_t> dx(data.size());
        std::vector<std::int16_t> dy(data.size());
        std::vector<std::int16_t> dxx(data.size());
        std::vector<std::int16_t> dyy(data.size());
        std::vector<std::int16_t> dxy(data.size());
        image::derivative::sobel(data.data(), width, height, width, dx.data(), dy.data(), dxx.data(), dyy.data(), dxy.data());
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                const size_t index = static_cast<size_t>(y * width + x);
                REQUIRE(dx[index] == 0);
                REQUIRE(dy[index] == 24);
                REQUIRE(dxx[index] == 0);
                REQUIRE(dyy[index] == 0);
                REQUIRE(dxy[index] == 0);
            }
        }
    }

    {
        const int width = 16;
        const int height = 14;
        std::vector<unsigned char> data(static_cast<size_t>(width * height));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                data[static_cast<size_t>(y * width + x)] = static_cast<unsigned char>(x * y);
            }
        }
        std::vector<std::int16_t> dx(data.size());
        std::vector<std::int16_t> dy(data.size());
        std::vector<std::int16_t> dxx(data.size());
        std::vector<std::int16_t> dyy(data.size());
        std::vector<std::int16_t> dxy(data.size());
        image::derivative::sobel(data.data(), width, height, width, dx.data(), dy.data(), dxx.data(), dyy.data(), dxy.data());
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                const size_t index = static_cast<size_t>(y * width + x);
                REQUIRE(dx[index] == 8 * y);
                REQUIRE(dy[index] == 8 * x);
                REQUIRE(dxx[index] == 0);
                REQUIRE(dyy[index] == 0);
                REQUIRE(dxy[index] == 4);
            }
        }
    }

    {
        const int width = 16;
        const int height = 8;
        std::vector<unsigned char> data(static_cast<size_t>(width * height));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                data[static_cast<size_t>(y * width + x)] = static_cast<unsigned char>(x * x);
            }
        }
        std::vector<std::int16_t> dx(data.size());
        std::vector<std::int16_t> dy(data.size());
        std::vector<std::int16_t> dxx(data.size());
        std::vector<std::int16_t> dyy(data.size());
        std::vector<std::int16_t> dxy(data.size());
        std::vector<std::int64_t> kappa(data.size());
        image::derivative::sobel(data.data(), width, height, width, dx.data(), dy.data(), dxx.data(), dyy.data(), dxy.data());
        image::derivative::curvature(dx.data(), dy.data(), dxx.data(), dyy.data(), dxy.data(), width, height, width, kappa.data());
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                const size_t index = static_cast<size_t>(y * width + x);
                REQUIRE(dx[index] == 16 * x);
                REQUIRE(dy[index] == 0);
                REQUIRE(dxx[index] == 8);
                REQUIRE(dyy[index] == 0);
                REQUIRE(dxy[index] == 0);
                REQUIRE(kappa[index] == 0);
            }
        }
    }

    {
        unsigned long long state = 42;
        for (int trial = 0; trial < 4; ++trial) {
            const int width = 33 + 7 * trial;
            const int height = 21 + 5 * trial;
            std::vector<unsigned char> data(static_cast<size_t>(width * height));
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = next_random_byte(state);
            }
            std::vector<std::int16_t> dx(data.size());
            std::vector<std::int16_t> dy(data.size());
            std::vector<std::int16_t> dxx(data.size());
            std::vector<std::int16_t> dyy(data.size());
            std::vector<std::int16_t> dxy(data.size());
            std::vector<std::int64_t> kappa(data.size());
            image::derivative::curvature(data.data(), width, height, width, dx.data(), dy.data(), dxx.data(), dyy.data(), dxy.data(), kappa.data());

            std::vector<std::int16_t> reference_dx(data.size());
            std::vector<std::int16_t> reference_dy(data.size());
            std::vector<std::int16_t> reference_dxx(data.size());
            std::vector<std::int16_t> reference_dyy(data.size());
            std::vector<std::int16_t> reference_dxy(data.size());
            reference_sobel(data.data(), width, height, width, reference_dx.data(), reference_dy.data(), reference_dxx.data(), reference_dyy.data(), reference_dxy.data());

            for (size_t i = 0; i < data.size(); ++i) {
                REQUIRE(dx[i] == reference_dx[i]);
                REQUIRE(dy[i] == reference_dy[i]);
                REQUIRE(dxx[i] == reference_dxx[i]);
                REQUIRE(dyy[i] == reference_dyy[i]);
                REQUIRE(dxy[i] == reference_dxy[i]);
                REQUIRE(dx[i] >= -1020);
                REQUIRE(dx[i] <= 1020);
                REQUIRE(dy[i] >= -1020);
                REQUIRE(dy[i] <= 1020);
                REQUIRE(dxx[i] >= -2040);
                REQUIRE(dxx[i] <= 2040);
                REQUIRE(dyy[i] >= -2040);
                REQUIRE(dyy[i] <= 2040);
                REQUIRE(dxy[i] >= -510);
                REQUIRE(dxy[i] <= 510);
                const std::int64_t first_x = dx[i];
                const std::int64_t first_y = dy[i];
                const std::int64_t expected = (first_y * first_y) * dxx[i] - 2 * (first_x * first_y) * dxy[i] + (first_x * first_x) * dyy[i];
                REQUIRE(kappa[i] == expected);
            }

            std::vector<std::int64_t> kappa_repeat(data.size());
            image::derivative::curvature(data.data(), width, height, width, dx.data(), dy.data(), dxx.data(), dyy.data(), dxy.data(), kappa_repeat.data());
            for (size_t i = 0; i < data.size(); ++i) {
                REQUIRE(kappa[i] == kappa_repeat[i]);
            }
        }
    }

    {
        const int width = 31;
        const int height = 31;
        const int centre = 15;
        std::vector<unsigned char> data(static_cast<size_t>(width * height));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const bool bright = ((x < centre) ^ (y < centre));
                data[static_cast<size_t>(y * width + x)] = bright ? 200 : 40;
            }
        }
        std::vector<std::int16_t> dx(data.size());
        std::vector<std::int16_t> dy(data.size());
        std::vector<std::int16_t> dxx(data.size());
        std::vector<std::int16_t> dyy(data.size());
        std::vector<std::int16_t> dxy(data.size());
        std::vector<std::int64_t> kappa(data.size());
        image::derivative::curvature(data.data(), width, height, width, dx.data(), dy.data(), dxx.data(), dyy.data(), dxy.data(), kappa.data());
        std::int64_t near_corner_maximum = 0;
        std::int64_t far_from_corner_maximum = 0;
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                const std::int64_t magnitude = math::abs(kappa[static_cast<size_t>(y * width + x)]);
                const bool near_corner = (math::abs(x - centre) <= 2) && (math::abs(y - centre) <= 2);
                if (near_corner) {
                    near_corner_maximum = math::max(near_corner_maximum, magnitude);
                }
                else if ((math::abs(x - centre) > 4) && (math::abs(y - centre) > 4)) {
                    far_from_corner_maximum = math::max(far_from_corner_maximum, magnitude);
                }
            }
        }
        REQUIRE(near_corner_maximum > 0);
        REQUIRE(far_from_corner_maximum == 0);
    }

    std::printf("All derivative tests passed.\n");
    return 0;
}
