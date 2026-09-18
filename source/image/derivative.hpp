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

#pragma once
#ifndef ZEROSLAM_IMAGE_DERIVATIVE_HPP
#define ZEROSLAM_IMAGE_DERIVATIVE_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdint>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace image {
    class derivative final {
    public:
        static void sobel(
            const unsigned char* __restrict const data,
            const int width,
            const int height,
            const int stride,
            std::int16_t* __restrict const dx,
            std::int16_t* __restrict const dy,
            std::int16_t* __restrict const dxx,
            std::int16_t* __restrict const dyy,
            std::int16_t* __restrict const dxy
        );

        static void curvature(
            const std::int16_t* __restrict const dx,
            const std::int16_t* __restrict const dy,
            const std::int16_t* __restrict const dxx,
            const std::int16_t* __restrict const dyy,
            const std::int16_t* __restrict const dxy,
            const int width,
            const int height,
            const int stride,
            std::int64_t* __restrict const kappa
        );

        static void curvature(
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
        );
    };
}

#endif // ZEROSLAM_IMAGE_DERIVATIVE_HPP
