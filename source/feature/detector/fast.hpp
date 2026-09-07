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
#ifndef ZEROSLAM_FEATURE_DETECTOR_FAST_HPP
#define ZEROSLAM_FEATURE_DETECTOR_FAST_HPP

#include "feature/point.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace feature::detector {
    class fast final {
    public:
        static size_t detect(
            const unsigned char* __restrict const data,
            const int width,
            const int height,
            const int stride,
            const int threshold,
            const size_t feature_point_buffer_size,
            point* __restrict feature_point_buffer
        );
    };
}

#endif // ZEROSLAM_FEATURE_DETECTOR_FAST_HPP
