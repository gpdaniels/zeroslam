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
#ifndef ZEROSLAM_IMAGE_PYRAMID_HPP
#define ZEROSLAM_IMAGE_PYRAMID_HPP

#include "image/image.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace image {
    class pyramid final {
    public:
        constexpr static const size_t minimum_dimension = 32;

    private:
        std::vector<image> images;
        std::vector<float> scales_x;
        std::vector<float> scales_y;

    public:
        pyramid();

        explicit pyramid(const image& base);

        static size_t automatic_levels(const size_t cols, const size_t rows);

        size_t size() const;

        const image& operator[](const size_t level) const;

        const image& back() const;

        float scale_x(const size_t level) const;

        float scale_y(const size_t level) const;
    };
}

#endif // ZEROSLAM_IMAGE_PYRAMID_HPP
