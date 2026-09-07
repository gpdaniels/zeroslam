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
#ifndef ZEROSLAM_IMAGE_RESIZE_HPP
#define ZEROSLAM_IMAGE_RESIZE_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace image {
    class resize final {
    public:
        static void decimate(
            const unsigned char* __restrict const source_data,
            const size_t source_width,
            const size_t source_height,
            const size_t target_width,
            const size_t target_height,
            unsigned char* __restrict const target_data
        );

        static void nearest(
            const unsigned char* __restrict const source_data,
            const size_t source_width,
            const size_t source_height,
            const size_t target_width,
            const size_t target_height,
            unsigned char* __restrict const target_data
        );

        static void linear(
            const unsigned char* __restrict const source_data,
            const size_t source_width,
            const size_t source_height,
            const size_t target_width,
            const size_t target_height,
            unsigned char* __restrict const target_data
        );
    };
}

#endif // ZEROSLAM_IMAGE_RESIZE_HPP
