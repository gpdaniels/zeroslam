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
#ifndef ZEROSLAM_IMAGE_INTERPOLATION_HPP
#define ZEROSLAM_IMAGE_INTERPOLATION_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace image {
    class interpolation final {
    public:
        using function_type = unsigned char (*)(
            const unsigned char* __restrict data,
            size_t source_width,
            size_t source_height,
            float offset_x,
            float offset_y
        );

        static unsigned char nearest(
            const unsigned char* __restrict data,
            size_t source_width,
            size_t source_height,
            float offset_x,
            float offset_y
        );

        static unsigned char linear(
            const unsigned char* __restrict data,
            size_t source_width,
            size_t source_height,
            float offset_x,
            float offset_y
        );
    };
}

#endif // ZEROSLAM_IMAGE_INTERPOLATION_HPP
