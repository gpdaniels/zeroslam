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
#ifndef ZEROSLAM_FEATURE_DESCRIPTOR_TEBLID_HPP
#define ZEROSLAM_FEATURE_DESCRIPTOR_TEBLID_HPP

#include "feature/descriptor/binary.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace feature::descriptor {
    class teblid final {
    public:
        struct test final {
            int x1;
            int y1;
            int x2;
            int y2;
            int radius;
            float threshold;
        };

        constexpr static const int test_count = 256;
        constexpr static const int patch_size = 32;
        constexpr static const float keypoint_size = 31.0f;
        constexpr static const int window_radius = 25;

        static const test tests[test_count];

    public:
        static void describe(
            const unsigned char* __restrict const data,
            const int stride,
            const float angle_radians,
            binary<256>& descriptor
        );
    };
}

#endif // ZEROSLAM_FEATURE_DESCRIPTOR_TEBLID_HPP
