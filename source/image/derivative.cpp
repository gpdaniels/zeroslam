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

namespace image {
    void derivative::sobel(
        const unsigned char* __restrict const data,
        const int width,
        const int height,
        const int stride,
        std::int16_t* __restrict const dx,
        std::int16_t* __restrict const dy,
        std::int16_t* __restrict const dxx,
        std::int16_t* __restrict const dyy,
        std::int16_t* __restrict const dxy
    ) {
        if ((width <= 0) || (height <= 0)) {
            return;
        }
        for (int x = 0; x < width; ++x) {
            dx[x] = 0;
            dy[x] = 0;
            dxx[x] = 0;
            dyy[x] = 0;
            dxy[x] = 0;
            const int bottom = (height - 1) * stride + x;
            dx[bottom] = 0;
            dy[bottom] = 0;
            dxx[bottom] = 0;
            dyy[bottom] = 0;
            dxy[bottom] = 0;
        }
        for (int y = 1; y < height - 1; ++y) {
            dx[y * stride] = 0;
            dy[y * stride] = 0;
            dxx[y * stride] = 0;
            dyy[y * stride] = 0;
            dxy[y * stride] = 0;
            if (width > 1) {
                dx[y * stride + width - 1] = 0;
                dy[y * stride + width - 1] = 0;
                dxx[y * stride + width - 1] = 0;
                dyy[y * stride + width - 1] = 0;
                dxy[y * stride + width - 1] = 0;
            }
            const unsigned char* __restrict const row_above = data + (y - 1) * stride;
            const unsigned char* __restrict const row_centre = data + y * stride;
            const unsigned char* __restrict const row_below = data + (y + 1) * stride;
            for (int x = 1; x < width - 1; ++x) {
                const int p00 = row_above[x - 1];
                const int p01 = row_above[x];
                const int p02 = row_above[x + 1];
                const int p10 = row_centre[x - 1];
                const int p11 = row_centre[x];
                const int p12 = row_centre[x + 1];
                const int p20 = row_below[x - 1];
                const int p21 = row_below[x];
                const int p22 = row_below[x + 1];
                const int index = y * stride + x;
                dx[index] = static_cast<std::int16_t>((p02 - p00) + 2 * (p12 - p10) + (p22 - p20));
                dy[index] = static_cast<std::int16_t>((p20 - p00) + 2 * (p21 - p01) + (p22 - p02));
                dxx[index] = static_cast<std::int16_t>((p00 - 2 * p01 + p02) + 2 * (p10 - 2 * p11 + p12) + (p20 - 2 * p21 + p22));
                dyy[index] = static_cast<std::int16_t>((p00 - 2 * p10 + p20) + 2 * (p01 - 2 * p11 + p21) + (p02 - 2 * p12 + p22));
                dxy[index] = static_cast<std::int16_t>((p00 - p02 - p20 + p22));
            }
        }
    }

    void derivative::curvature(
        const std::int16_t* __restrict const dx,
        const std::int16_t* __restrict const dy,
        const std::int16_t* __restrict const dxx,
        const std::int16_t* __restrict const dyy,
        const std::int16_t* __restrict const dxy,
        const int width,
        const int height,
        const int stride,
        std::int64_t* __restrict const kappa
    ) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const int index = y * stride + x;
                const std::int64_t first_x = dx[index];
                const std::int64_t first_y = dy[index];
                kappa[index] = (first_y * first_y) * dxx[index] - 2 * (first_x * first_y) * dxy[index] + (first_x * first_x) * dyy[index];
            }
        }
    }

    void derivative::curvature(
        const unsigned char* __restrict const data,
        const int width,
        const int height,
        const int stride,
        std::int16_t* __restrict const dx,
        std::int16_t* __restrict const dy,
        std::int16_t* __restrict const dxx,
        std::int16_t* __restrict const dyy,
        std::int16_t* __restrict const dxy,
        std::int64_t* __restrict const kappa
    ) {
        derivative::sobel(data, width, height, stride, dx, dy, dxx, dyy, dxy);
        derivative::curvature(dx, dy, dxx, dyy, dxy, width, height, stride, kappa);
    }
}
