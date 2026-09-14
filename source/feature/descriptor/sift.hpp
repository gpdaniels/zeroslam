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
#ifndef ZEROSLAM_FEATURE_DESCRIPTOR_SIFT_HPP
#define ZEROSLAM_FEATURE_DESCRIPTOR_SIFT_HPP

#include "feature/descriptor/binary.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace feature::descriptor {
    class sift final {
    public:
        constexpr static const int cells = 4;
        constexpr static const int bins = 8;
        constexpr static const int samples = 16;
        constexpr static const int dimensions = cells * cells * bins;
        constexpr static const float sample_spacing = 2.0f;
        constexpr static const int radius = 24;
        constexpr static const float clip = 0.2f;

    public:
        static void describe_float(
            const unsigned char* __restrict const data,
            const int stride,
            const float angle_radians,
            const float* const affine,
            float (&vector)[dimensions]
        );

        static void binarise(const float (&vector)[dimensions], binary<256>& descriptor);

        static void describe(
            const unsigned char* __restrict const data,
            const int stride,
            const float angle_radians,
            binary<256>& descriptor
        );
    };
}

#endif // ZEROSLAM_FEATURE_DESCRIPTOR_SIFT_HPP
