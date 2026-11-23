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

#include "sensor/depth.hpp"

#include "math/math.hpp"

namespace sensor {
    depth::depth()
        : rows(0)
        , columns(0)
        , metres()
        , configuration() {
    }

    depth::depth(const size_t image_rows, const size_t image_columns, const float* const image_metres, const settings& depth_settings)
        : rows(image_rows)
        , columns(image_columns)
        , metres(image_rows * image_columns, 0.0f)
        , configuration(depth_settings) {
        for (size_t index = 0; index < this->metres.size(); ++index) {
            const double value = static_cast<double>(image_metres[index]);
            const bool within = math::isfinite(value) && (value > 0.0) && (value >= this->configuration.minimum) && (value <= this->configuration.maximum);
            this->metres[index] = within ? image_metres[index] : 0.0f;
        }
    }

    depth::depth(const size_t image_rows, const size_t image_columns, const unsigned short* const image_values, const double scale, const settings& depth_settings)
        : rows(image_rows)
        , columns(image_columns)
        , metres(image_rows * image_columns, 0.0f)
        , configuration(depth_settings) {
        for (size_t index = 0; index < this->metres.size(); ++index) {
            const double value = static_cast<double>(image_values[index]) * scale;
            const bool within = (image_values[index] != 0) && math::isfinite(value) && (value > 0.0) && (value >= this->configuration.minimum) && (value <= this->configuration.maximum);
            this->metres[index] = within ? static_cast<float>(value) : 0.0f;
        }
    }

    bool depth::empty() const {
        return this->metres.empty();
    }

    size_t depth::get_rows() const {
        return this->rows;
    }

    size_t depth::get_cols() const {
        return this->columns;
    }

    const depth::settings& depth::get_settings() const {
        return this->configuration;
    }

    float depth::at(const size_t row, const size_t column) const {
        if ((row >= this->rows) || (column >= this->columns)) {
            return 0.0f;
        }
        return this->metres[(row * this->columns) + column];
    }

    void depth::release() {
        this->metres = std::vector<float>();
        this->rows = 0;
        this->columns = 0;
    }

    bool depth::sample(const double x, const double y, double& value) const {
        if (this->metres.empty() || !(x >= 0.0) || !(y >= 0.0)) {
            return false;
        }
        // The pixel containing a centre position is its floor.
        const double column_position = math::floor(x);
        const double row_position = math::floor(y);
        if ((column_position >= static_cast<double>(this->columns)) || (row_position >= static_cast<double>(this->rows))) {
            return false;
        }
        const size_t column = static_cast<size_t>(column_position);
        const size_t row = static_cast<size_t>(row_position);
        const float centre = this->metres[(row * this->columns) + column];
        if (!(centre > 0.0f)) {
            return false;
        }
        const float bound = static_cast<float>(depth::discontinuity_fraction) * centre;
        for (size_t neighbour_row = (row > 0) ? (row - 1) : 0; (neighbour_row <= row + 1) && (neighbour_row < this->rows); ++neighbour_row) {
            for (size_t neighbour_column = (column > 0) ? (column - 1) : 0; (neighbour_column <= column + 1) && (neighbour_column < this->columns); ++neighbour_column) {
                const float neighbour = this->metres[(neighbour_row * this->columns) + neighbour_column];
                // A neighbour without a measurement says nothing: a structured light camera leaves the background beside a
                // foreground edge in shadow, and the foreground pixel next to it measures as well as any.
                if ((neighbour > 0.0f) && (math::abs(neighbour - centre) > bound)) {
                    return false;
                }
            }
        }
        value = static_cast<double>(centre);
        return true;
    }

    double depth::sigma(const double value) const {
        const noise_model& noise = this->configuration.noise;
        return math::max(depth::sigma_minimum, noise.constant + (noise.linear * value) + (noise.quadratic * value * value));
    }

    double depth::inverse_sigma(const double value) const {
        return this->sigma(value) / (value * value);
    }

    double depth::measured(const measure_kind measure, const double (&point)[3]) {
        if (measure == measure_kind::axial) {
            return point[2];
        }
        return math::sqrt((point[0] * point[0]) + (point[1] * point[1]) + (point[2] * point[2]));
    }
}
