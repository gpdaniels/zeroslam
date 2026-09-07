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
#ifndef ZEROSLAM_FEATURE_REFINER_SUBPIXEL_HPP
#define ZEROSLAM_FEATURE_REFINER_SUBPIXEL_HPP

namespace feature::refiner {
    class subpixel final {
    public:
        static bool refine(
            const unsigned char* __restrict const data,
            const int stride,
            float& offset_x,
            float& offset_y
        );

        static void patch_41x41_bilinear(
            const unsigned char* __restrict const data,
            const int stride,
            const float offset_x,
            const float offset_y,
            unsigned char* __restrict patch
        );
    };
}

#endif // ZEROSLAM_FEATURE_REFINER_SUBPIXEL_HPP
