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
#ifndef ZEROSLAM_IMAGE_BLUR_HPP
#define ZEROSLAM_IMAGE_BLUR_HPP

namespace image {
    class blur final {
    public:
        static void gaussian_5x5(
            const unsigned char* __restrict const source_data,
            const int source_width,
            const int source_height,
            const int source_stride,
            unsigned char* __restrict const target_data
        );

        static void gaussian_7x7(
            const unsigned char* __restrict const source_data,
            const int source_width,
            const int source_height,
            const int source_stride,
            unsigned char* __restrict const target_data
        );
    };
}

#endif // ZEROSLAM_IMAGE_BLUR_HPP
