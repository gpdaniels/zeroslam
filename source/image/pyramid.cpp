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
#include "image/resize.hpp"
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
        double inverse_scale = 1.0;
        for (size_t level = 1; level < levels; ++level) {
            inverse_scale *= 0.5;
            const size_t target_cols = static_cast<size_t>(static_cast<double>(base.get_cols()) * inverse_scale);
            const size_t target_rows = static_cast<size_t>(static_cast<double>(base.get_rows()) * inverse_scale);
            if ((target_cols < pyramid::minimum_dimension) || (target_rows < pyramid::minimum_dimension)) {
                break;
            }
            const image& previous = this->images.back();
            image blurred(previous.get_rows(), previous.get_cols());
            blur::gaussian_5x5(previous.get_data(), static_cast<int>(previous.get_cols()), static_cast<int>(previous.get_rows()), static_cast<int>(previous.get_cols()), blurred.get_data());
            image next(target_rows, target_cols);
            resize::decimate(blurred.get_data(), blurred.get_cols(), blurred.get_rows(), next.get_cols(), next.get_rows(), next.get_data());
            this->images.push_back(static_cast<image&&>(next));
            this->scales_x.push_back(static_cast<float>(static_cast<double>(base.get_cols()) / static_cast<double>(target_cols)));
            this->scales_y.push_back(static_cast<float>(static_cast<double>(base.get_rows()) / static_cast<double>(target_rows)));
        }
    }

    size_t pyramid::automatic_levels(const size_t cols, const size_t rows) {
        const int levels = static_cast<int>(math::floor(math::log(static_cast<double>(math::min(cols, rows))) / math::log(2.0))) - 4;
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
