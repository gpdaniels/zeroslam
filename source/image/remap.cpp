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

#include "image/remap.hpp"

#include "math/math.hpp"

namespace image {
    remap::remap()
        : columns(0)
        , rows(0)
        , bases_x()
        , bases_y()
        , weights_x()
        , weights_y()
        , validity() {
    }

    remap::remap(const size_t destination_columns, const size_t destination_rows, const float* const source_x, const float* const source_y, const size_t source_columns, const size_t source_rows)
        : columns(destination_columns)
        , rows(destination_rows)
        , bases_x(destination_columns * destination_rows, 0u)
        , bases_y(destination_columns * destination_rows, 0u)
        , weights_x(destination_columns * destination_rows, static_cast<unsigned char>(0))
        , weights_y(destination_columns * destination_rows, static_cast<unsigned char>(0))
        , validity(destination_columns * destination_rows, static_cast<unsigned char>(0)) {
        if ((source_columns < 2) || (source_rows < 2)) {
            this->columns = 0;
            this->rows = 0;
            return;
        }
        const float last_x = static_cast<float>(source_columns - 1);
        const float last_y = static_cast<float>(source_rows - 1);
        for (size_t index = 0; index < destination_columns * destination_rows; ++index) {
            const float x = source_x[index];
            const float y = source_y[index];
            this->validity[index] = ((x >= 0.0f) && (x <= last_x) && (y >= 0.0f) && (y <= last_y)) ? static_cast<unsigned char>(1) : static_cast<unsigned char>(0);
            // Note: Clamping to just inside the last column and row keeps the bilinear neighbours within the image.
            const float clamped_x = math::max(0.0f, math::min(x, last_x - (1.0f / 512.0f)));
            const float clamped_y = math::max(0.0f, math::min(y, last_y - (1.0f / 512.0f)));
            const size_t base_x = static_cast<size_t>(clamped_x);
            const size_t base_y = static_cast<size_t>(clamped_y);
            this->bases_x[index] = static_cast<unsigned int>(base_x);
            this->bases_y[index] = static_cast<unsigned int>(base_y);
            this->weights_x[index] = static_cast<unsigned char>(math::min(255.0f, ((clamped_x - static_cast<float>(base_x)) * 256.0f) + 0.5f));
            this->weights_y[index] = static_cast<unsigned char>(math::min(255.0f, ((clamped_y - static_cast<float>(base_y)) * 256.0f) + 0.5f));
        }
    }

    bool remap::empty() const {
        return (this->columns == 0) || (this->rows == 0);
    }

    size_t remap::get_columns() const {
        return this->columns;
    }

    size_t remap::get_rows() const {
        return this->rows;
    }

    bool remap::valid(const size_t x, const size_t y) const {
        return (x < this->columns) && (y < this->rows) && (this->validity[(y * this->columns) + x] != 0);
    }

    void remap::apply(const unsigned char* __restrict const source_data, const size_t source_stride, unsigned char* __restrict const destination_data) const {
        for (size_t y = 0; y < this->rows; ++y) {
            for (size_t x = 0; x < this->columns; ++x) {
                const size_t index = (y * this->columns) + x;
                const unsigned char* const top = source_data + (static_cast<size_t>(this->bases_y[index]) * source_stride) + this->bases_x[index];
                const unsigned char* const bottom = top + source_stride;
                const unsigned int wx = this->weights_x[index];
                const unsigned int wy = this->weights_y[index];
                const unsigned int upper = (static_cast<unsigned int>(top[0]) * (256u - wx)) + (static_cast<unsigned int>(top[1]) * wx);
                const unsigned int lower = (static_cast<unsigned int>(bottom[0]) * (256u - wx)) + (static_cast<unsigned int>(bottom[1]) * wx);
                destination_data[index] = static_cast<unsigned char>(((upper * (256u - wy)) + (lower * wy) + 32768u) >> 16u);
            }
        }
    }
}
