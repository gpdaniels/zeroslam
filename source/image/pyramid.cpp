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

#include "image/pyramid.hpp"

#include "core/assert.hpp"
#include "image/blur.hpp"
#include "math/math.hpp"

namespace image {
    pyramid::pyramid()
        : images()
        , scales_x()
        , scales_y() {
    }

    pyramid::pyramid(const image& base) {
        const size_t levels = pyramid::automatic_levels(base.get_cols(), base.get_rows());
        this->images.reserve(levels);
        this->scales_x.reserve(levels);
        this->scales_y.reserve(levels);
        this->images.push_back(base);
        this->scales_x.push_back(1.0f);
        this->scales_y.push_back(1.0f);
        for (size_t level = 1; level < levels; ++level) {
            // Each level keeps the even rows and columns of the blurred level above, so its pixel i is level 0 pixel i * 2^level whatever the size.
            const image& previous = this->images.back();
            const size_t target_cols = previous.get_cols() / 2;
            const size_t target_rows = previous.get_rows() / 2;
            if ((target_cols < pyramid::minimum_dimension) || (target_rows < pyramid::minimum_dimension)) {
                break;
            }
            image next(target_rows, target_cols);
            blur::gaussian_5x5_decimate(previous.get_data(), static_cast<int>(previous.get_cols()), static_cast<int>(previous.get_rows()), static_cast<int>(previous.get_cols()), next.get_data(), static_cast<int>(target_cols));
            this->images.push_back(static_cast<image&&>(next));
            const float scale = static_cast<float>(1u << level);
            this->scales_x.push_back(scale);
            this->scales_y.push_back(scale);
        }
    }

    size_t pyramid::automatic_levels(const size_t cols, const size_t rows) {
        const size_t smallest = math::min(cols, rows);
        if (smallest < pyramid::minimum_dimension) {
            return 1;
        }
        const int levels = static_cast<int>(math::floor(math::log(static_cast<double>(smallest)) / math::log(2.0))) - 4;
        return static_cast<size_t>(math::max(1, levels));
    }

    size_t pyramid::size() const {
        return this->images.size();
    }

    const image& pyramid::operator[](const size_t level) const {
        ASSERT(level < this->images.size(), "Pyramid level out of range.");
        return this->images[level];
    }

    const image& pyramid::back() const {
        ASSERT(!this->images.empty(), "The pyramid is empty.");
        return this->images.back();
    }

    float pyramid::scale_x(const size_t level) const {
        ASSERT(level < this->scales_x.size(), "Pyramid level out of range.");
        return this->scales_x[level];
    }

    float pyramid::scale_y(const size_t level) const {
        ASSERT(level < this->scales_y.size(), "Pyramid level out of range.");
        return this->scales_y[level];
    }
}
