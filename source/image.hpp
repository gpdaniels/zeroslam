/*
Copyright (C) 2025 Geoffrey Daniels. https://gpdaniels.com/

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
#ifndef IMAGE_HPP
#define IMAGE_HPP

#include "math.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace image {
    class image final {
    private:
        size_t rows;
        size_t cols;
        unsigned char* data;

    public:
        ~image() {
            delete[] data;
        }

        image()
            : rows(0)
            , cols(0)
            , data(nullptr) {
        }

        image(size_t image_rows, size_t image_cols)
            : rows(image_rows)
            , cols(image_cols)
            , data(new unsigned char[rows * cols]) {
        }

        image(size_t image_rows, size_t image_cols, unsigned char* image_data)
            : rows(image_rows)
            , cols(image_cols)
            , data(new unsigned char[image_rows * image_cols]) {
            for (size_t i = 0; i < this->rows * this->cols; ++i) {
                this->data[i] = image_data[i];
            }
        }

        image(const image& other)
            : rows(other.rows)
            , cols(other.cols)
            , data(new unsigned char[other.rows * other.cols]) {
            for (size_t i = 0; i < this->rows * this->cols; ++i) {
                this->data[i] = other.data[i];
            }
        }

        image(image&& other)
            : rows(other.rows)
            , cols(other.cols)
            , data(other.data) {
            other.data = nullptr;
        }

        const image& operator=(const image& other) {
            if (&other == this)
                return *this;
            this->rows = other.rows;
            this->cols = other.cols;
            this->data = new unsigned char[other.rows * other.cols];
            for (size_t i = 0; i < this->rows * this->cols; ++i) {
                this->data[i] = other.data[i];
            }
            return *this;
        }

        const image& operator=(image&& other) {
            if (&other == this)
                return *this;
            this->rows = other.rows;
            this->cols = other.cols;
            unsigned char* temp_data = this->data;
            this->data = other.data;
            other.data = temp_data;
            return *this;
        }

    public:
        size_t get_rows() const {
            return this->rows;
        }

        size_t get_cols() const {
            return this->cols;
        }

        unsigned char* get_data() const {
            return this->data;
        }
    };

    template <int kernel_size = 7>
    static inline void convolution_horizontal(
        const unsigned char* data,
        const int width,
        const int height,
        const int stride,
        const int* __restrict const kernel,
        unsigned char* __restrict const convolved
    ) {
        constexpr static const int kernel_radius = kernel_size / 2;
        int divisor = 0;
        for (int k = 0; k < kernel_size; ++k) {
            divisor += kernel[k];
        }
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < kernel_radius; ++x) {
                convolved[(y + 0) * stride + x] = 0;
            }
            for (int x = 0; x < width - (kernel_size - 1); ++x) {
                int sum = 0;
                for (int k = 0; k < kernel_size; ++k) {
                    sum += data[y * stride + (x + k)] * kernel[k];
                }
                convolved[y * stride + (x + kernel_radius)] = static_cast<unsigned char>((static_cast<float>(sum) / static_cast<float>(divisor)) + 0.5f);
            }
            for (int x = width - kernel_radius; x < width; ++x) {
                convolved[y * stride + x] = 0;
            }
        }
    }

    template <int kernel_size = 7>
    static inline void convolution_vertical(
        const unsigned char* __restrict const data,
        const int width,
        const int height,
        const int stride,
        const int* __restrict const kernel,
        unsigned char* __restrict const convolved
    ) {
        constexpr static const int kernel_radius = kernel_size / 2;
        int divisor = 0;
        for (int k = 0; k < kernel_size; ++k) {
            divisor += kernel[k];
        }
        for (int y = 0; y < kernel_radius; ++y) {
            for (int x = 0; x < width; ++x) {
                convolved[y * stride + x] = 0;
            }
        }
        for (int y = 0; y < height - (kernel_size - 1); ++y) {
            for (int x = 0; x < width; ++x) {
                int sum = 0;
                for (int k = 0; k < kernel_size; ++k) {
                    sum += data[(y + k) * stride + x] * kernel[k];
                }
                convolved[(y + kernel_radius) * stride + x] = static_cast<unsigned char>((static_cast<float>(sum) / static_cast<float>(divisor)) + 0.5f);
            }
        }
        for (int y = height - kernel_radius; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                convolved[y * stride + x] = 0;
            }
        }
    }

    static inline void blur(
        const unsigned char* __restrict const data,
        const int width,
        const int height,
        const int stride,
        unsigned char* __restrict const convolved
    ) {
        constexpr static const int kernel_size = 7;
        constexpr static const int kernel[kernel_size] = {
            18,
            34,
            49,
            55,
            49,
            34,
            18
        };
        unsigned char* vertically_convolved = new unsigned char[static_cast<unsigned long int>(height * stride)];
        convolution_vertical<kernel_size>(data, width, height, stride, kernel, vertically_convolved);
        convolution_horizontal<kernel_size>(vertically_convolved, width, height, stride, kernel, convolved);
        delete[] vertically_convolved;
    }

    static inline unsigned char interpolate_nearest(
        const unsigned char* __restrict data,
        size_t source_width,
        size_t source_height,
        float offset_x,
        float offset_y
    ) {
        const int nearest_x = math::max(0, math::min(math::round(offset_x), static_cast<int>(source_width) - 1));
        const int nearest_y = math::max(0, math::min(math::round(offset_y), static_cast<int>(source_height) - 1));
        return data[nearest_y * source_width + nearest_x];
    }

    static inline unsigned char interpolate_linear(
        const unsigned char* __restrict data,
        size_t source_width,
        size_t source_height,
        float offset_x,
        float offset_y
    ) {
        const int floor_x = math::floor(offset_x);
        const int floor_y = math::floor(offset_y);
        const int ceil_x = math::min(floor_x + 1, static_cast<int>(source_width) - 1);
        const int ceil_y = math::min(floor_y + 1, static_cast<int>(source_height) - 1);
        const float weight_x = offset_x - floor_x;
        const float weight_y = offset_y - floor_y;
        return static_cast<unsigned char>(
            math::round(
                static_cast<float>(data[floor_y * source_width + floor_x]) * (1.0f - weight_x) * (1.0f - weight_y) +
                static_cast<float>(data[floor_y * source_width + ceil_x]) * (weight_x) * (1.0f - weight_y) +
                static_cast<float>(data[ceil_y * source_width + floor_x]) * (1.0f - weight_x) * (weight_y) +
                static_cast<float>(data[ceil_y * source_width + ceil_x]) * (weight_x) * (weight_y)
            )
        );
    }

    using interpolate_function_type = unsigned char (*)(
        const unsigned char* __restrict data,
        size_t source_width,
        size_t source_height,
        float offset_x,
        float offset_y
    );

    static inline void resize(
        const unsigned char* __restrict const source_data,
        const size_t source_width,
        const size_t source_height,
        const size_t target_width,
        const size_t target_height,
        unsigned char* __restrict const target_data,
        interpolate_function_type interpolate_function = interpolate_nearest
    ) {
        for (size_t y = 0; y < target_height; ++y) {
            const float offset_y = ((static_cast<float>(y) + 0.5f) * (static_cast<float>(source_height) / static_cast<float>(target_height))) - 0.5f;
            for (size_t x = 0; x < target_width; ++x) {
                const float offset_x = ((static_cast<float>(x) + 0.5f) * (static_cast<float>(source_width) / static_cast<float>(target_width))) - 0.5f;
                target_data[y * target_width + x] = interpolate_function(source_data, source_width, source_height, offset_x, offset_y);
            }
        }
    }
}

#endif // IMAGE_HPP