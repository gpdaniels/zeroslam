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
#ifndef ZEROSLAM_IMAGE_REMAP_HPP
#define ZEROSLAM_IMAGE_REMAP_HPP

#include "image/image.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace image {
    // Resamples images through a fixed table holding, for every destination pixel, the position in the source image it
    // takes its value from, in pixel index coordinates (pixel (x, y) is centred on (x, y)), interpolated bilinearly. A
    // position outside the source image is clamped to its border and the destination pixel marked invalid, so a detector
    // run on the result can drop what it finds there.
    class remap final {
    private:
        size_t columns;
        size_t rows;
        std::vector<unsigned int> bases_x;
        std::vector<unsigned int> bases_y;
        std::vector<unsigned char> weights_x;
        std::vector<unsigned char> weights_y;
        std::vector<unsigned char> validity;

    public:
        remap();

        // The table for a columns by rows destination, from source_x and source_y (columns * rows entries each, row by row)
        // into a source_columns by source_rows image.
        remap(const size_t destination_columns, const size_t destination_rows, const float* const source_x, const float* const source_y, const size_t source_columns, const size_t source_rows);

        bool empty() const;

        size_t get_columns() const;

        size_t get_rows() const;

        bool valid(const size_t x, const size_t y) const;

        // The destination, which the caller sizes to the table; the source must be the size the table was built for.
        void apply(const unsigned char* __restrict const source_data, const size_t source_stride, unsigned char* __restrict const destination_data) const;
    };
}

#endif // ZEROSLAM_IMAGE_REMAP_HPP
