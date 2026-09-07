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
#ifndef ZEROSLAM_IMAGE_IMAGE_HPP
#define ZEROSLAM_IMAGE_IMAGE_HPP

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

        image(size_t image_rows, size_t image_cols, const unsigned char* image_data)
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
            delete[] this->data;
            this->rows = other.rows;
            this->cols = other.cols;
            this->data = nullptr;
            if (other.data != nullptr) {
                this->data = new unsigned char[other.rows * other.cols];
                for (size_t i = 0; i < this->rows * this->cols; ++i) {
                    this->data[i] = other.data[i];
                }
            }
            return *this;
        }

        const image& operator=(image&& other) {
            if (&other == this)
                return *this;
            const size_t temp_rows = this->rows;
            const size_t temp_cols = this->cols;
            unsigned char* temp_data = this->data;
            this->rows = other.rows;
            this->cols = other.cols;
            this->data = other.data;
            other.rows = temp_rows;
            other.cols = temp_cols;
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
}

#endif // ZEROSLAM_IMAGE_IMAGE_HPP
