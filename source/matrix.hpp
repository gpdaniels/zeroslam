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
#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "assert.hpp"
#include "math.hpp"

namespace {
    using size_t = decltype(sizeof(0));
    template <bool condition, typename type_true, typename type_false>
    class conditional_type;

    template <typename type_true, typename type_false>
    class conditional_type<true, type_true, type_false> final {
    public:
        using type = type_true;
    };

    template <typename type_true, typename type_false>
    class conditional_type<false, type_true, type_false> final {
    public:
        using type = type_false;
    };
}

namespace matrix {
    template <typename template_type, size_t template_rows, size_t template_cols>
    class matrix final {
    public:
        using type = template_type;
        static_assert(template_rows != 0);
        static_assert(template_cols != 0);

    public:
        template <typename other_type, size_t other_rows, size_t other_cols>
        friend class matrix;

    private:
        constexpr static const size_t matrix_rows = template_rows;
        constexpr static const size_t matrix_cols = template_cols;
        constexpr static const size_t matrix_size = template_rows * template_cols;
        type matrix_data[template_rows * template_cols];

    public:
        ~matrix() {
        }

        constexpr matrix(size_t fixed_rows = matrix::matrix_rows, size_t fixed_cols = matrix::matrix_cols) {
            ASSERT(fixed_rows == matrix::matrix_rows, "Incorrect rows count for fixed size matrix.");
            ASSERT(fixed_cols == matrix::matrix_cols, "Incorrect cols count for fixed size matrix.");
            static_cast<void>(fixed_rows);
            static_cast<void>(fixed_cols);
        }

        constexpr matrix(const matrix& other) {
            for (size_t i = 0; i < matrix::matrix_size; ++i) {
                this->matrix_data[i] = other.matrix_data[i];
            }
        }

        constexpr matrix(matrix&& other) {
            for (size_t i = 0; i < matrix::matrix_size; ++i) {
                this->matrix_data[i] = other.matrix_data[i];
            }
        }

        constexpr const matrix& operator=(const matrix& other) {
            if (&other == this)
                return *this;
            for (size_t i = 0; i < matrix::matrix_size; ++i) {
                this->matrix_data[i] = other.matrix_data[i];
            }
            return *this;
        }

        constexpr const matrix& operator=(matrix&& other) {
            if (&other == this)
                return *this;
            for (size_t i = 0; i < matrix::matrix_size; ++i) {
                this->matrix_data[i] = other.matrix_data[i];
            }
            return *this;
        }

    public:
        explicit matrix(const type* values, size_t fixed_rows = matrix::matrix_rows, size_t fixed_cols = matrix::matrix_cols) {
            ASSERT(fixed_rows == matrix::matrix_rows, "Incorrect rows count for fixed size matrix.");
            ASSERT(fixed_cols == matrix::matrix_cols, "Incorrect cols count for fixed size matrix.");
            static_cast<void>(fixed_rows);
            static_cast<void>(fixed_cols);
            for (size_t i = 0; i < matrix::matrix_size; ++i) {
                this->matrix_data[i] = values[i];
            }
        }

    public:
        template <size_t value_size>
        constexpr matrix(const type (&values)[value_size]) {
            static_assert((matrix::matrix_rows == 1 && matrix::matrix_cols == value_size) || (matrix::matrix_rows == value_size && matrix::matrix_cols == 1), "Incorrectly size of single row/col brace encoded array.");
            for (size_t i = 0; i < matrix::matrix_size; ++i) {
                this->matrix_data[i] = values[i];
            }
        }

        constexpr matrix(const type (&values)[matrix::matrix_rows][matrix::matrix_cols]) {
            for (size_t r = 0; r < matrix::matrix_rows; ++r) {
                for (size_t c = 0; c < matrix::matrix_cols; ++c) {
                    this->matrix_data[r * matrix::matrix_cols + c] = values[r][c];
                }
            }
        }

    public:
        constexpr static inline matrix zero(size_t fixed_rows = matrix::matrix_rows, size_t fixed_cols = matrix::matrix_cols) {
            ASSERT(fixed_rows == matrix::matrix_rows, "Incorrect rows count for fixed size matrix.");
            ASSERT(fixed_cols == matrix::matrix_cols, "Incorrect cols count for fixed size matrix.");
            static_cast<void>(fixed_rows);
            static_cast<void>(fixed_cols);
            matrix result;
            for (size_t i = 0; i < matrix::matrix_size; ++i) {
                result.matrix_data[i] = 0;
            }
            return result;
        }

        constexpr static inline matrix identity(size_t fixed_rows = matrix::matrix_rows, size_t fixed_cols = matrix::matrix_cols) {
            ASSERT(fixed_rows == matrix::matrix_rows, "Incorrect rows count for fixed size matrix.");
            ASSERT(fixed_cols == matrix::matrix_cols, "Incorrect cols count for fixed size matrix.");
            static_cast<void>(fixed_rows);
            static_cast<void>(fixed_cols);
            matrix result = matrix::zero();
            for (size_t i = 0; i < math::min(matrix::matrix_rows, matrix::matrix_cols); ++i) {
                result.matrix_data[i * matrix::matrix_cols + i] = 1;
            }
            return result;
        }

    public:
        constexpr size_t rows() const {
            return matrix::matrix_rows;
        }

        constexpr size_t cols() const {
            return matrix::matrix_cols;
        }

        constexpr const type* data() const {
            return &this->matrix_data[0];
        }

        constexpr type* data() {
            return &this->matrix_data[0];
        }

        constexpr size_t size() const {
            return matrix::matrix_rows * matrix::matrix_cols;
        }

    public:
        constexpr type get_length_squared() const {
            type length_squared = 0;
            for (size_t i = 0; i < matrix::matrix_size; ++i) {
                length_squared += this->matrix_data[i] * this->matrix_data[i];
            }
            return length_squared;
        }

    public:
        constexpr typename conditional_type<(matrix::matrix_rows == 1 || matrix::matrix_cols == 1), const type&, const type*>::type operator[](size_t index) const {
            if constexpr (matrix::matrix_rows == 1 || matrix::matrix_cols == 1) {
                ASSERT((matrix::matrix_rows == 1) || (index < matrix::matrix_rows), "Row index out of bounds.");
                ASSERT((matrix::matrix_cols == 1) || (index < matrix::matrix_cols), "Col index out of bounds.");
                return this->matrix_data[index];
            }
            else {
                ASSERT(index < matrix::matrix_rows, "Row index out of bounds.");
                return &this->matrix_data[index * matrix::matrix_cols];
            }
        }

        constexpr typename conditional_type<(matrix::matrix_rows == 1 || matrix::matrix_cols == 1), type&, type*>::type operator[](size_t index) {
            if constexpr (matrix::matrix_rows == 1 || matrix::matrix_cols == 1) {
                ASSERT((matrix::matrix_rows == 1) || (index < matrix::matrix_rows), "Row index out of bounds.");
                ASSERT((matrix::matrix_cols == 1) || (index < matrix::matrix_cols), "Col index out of bounds.");
                return this->matrix_data[index];
            }
            else {
                ASSERT(index < matrix::matrix_rows, "Row index out of bounds.");
                return &this->matrix_data[index * matrix::matrix_cols];
            }
        }

    public:
        constexpr const type& operator()(size_t index_row, size_t index_col) const {
            ASSERT(index_row < matrix::matrix_rows, "Row index must be less than the height of the matrix.");
            ASSERT(index_col < matrix::matrix_cols, "Column index must be less than the width of the matrix.");
            return this->matrix_data[index_row * matrix::matrix_cols + index_col];
        }

        constexpr type& operator()(size_t index_row, size_t index_col) {
            ASSERT(index_row < matrix::matrix_rows, "Row index must be less than the height of the matrix.");
            ASSERT(index_col < matrix::matrix_cols, "Column index must be less than the width of the matrix.");
            return this->matrix_data[index_row * matrix::matrix_cols + index_col];
        }
    };

    template <typename type>
    class matrix<type, 0, 0> final {
    private:
        size_t matrix_rows;
        size_t matrix_cols;
        type* matrix_data;

    public:
        ~matrix() {
            delete[] matrix_data;
        }

        constexpr matrix()
            : matrix_rows(0)
            , matrix_cols(0)
            , matrix_data(nullptr) {
        }

        matrix(size_t rows, size_t cols)
            : matrix_rows(rows)
            , matrix_cols(cols)
            , matrix_data(new type[rows * cols]) {
        }

        matrix(const matrix& other)
            : matrix_rows(other.matrix_rows)
            , matrix_cols(other.matrix_cols)
            , matrix_data(new type[other.size()]) {
            for (size_t i = 0; i < this->size(); ++i) {
                this->matrix_data[i] = other.matrix_data[i];
            }
        }

        constexpr matrix(matrix&& other)
            : matrix_rows(other.matrix_rows)
            , matrix_cols(other.matrix_cols)
            , matrix_data(other.matrix_data) {
            other.matrix_data = nullptr;
        }

        const matrix& operator=(const matrix& other) {
            if (&other == this)
                return *this;
            if (this->size() != other.size()) {
                delete[] this->matrix_data;
                this->matrix_data = new type[other.size()];
            }
            this->matrix_rows = other.matrix_rows;
            this->matrix_cols = other.matrix_cols;
            for (size_t i = 0; i < this->size(); ++i) {
                this->matrix_data[i] = other.matrix_data[i];
            }
            return *this;
        }

        constexpr const matrix& operator=(matrix&& other) {
            if (&other == this)
                return *this;
            this->matrix_rows = other.matrix_rows;
            this->matrix_cols = other.matrix_cols;
            type* temp_data = this->matrix_data;
            this->matrix_data = other.matrix_data;
            other.matrix_data = temp_data;
            return *this;
        }

    public:
        matrix(size_t rows, size_t cols, const type* values)
            : matrix_rows(rows)
            , matrix_cols(cols)
            , matrix_data(new type[this->size()]) {
            for (size_t i = 0; i < this->size(); ++i) {
                this->matrix_data[i] = values[i];
            }
        }

    public:
        template <size_t value_size>
        matrix(const type (&values)[value_size]) = delete;

        template <size_t value_rows, size_t value_cols>
        matrix(const type (&values)[value_rows][value_cols]) {
            this->matrix_rows = value_rows;
            this->matrix_cols = value_cols;
            this->matrix_data = new type[value_rows * value_cols];
            for (size_t r = 0; r < value_rows; ++r) {
                for (size_t c = 0; c < value_cols; ++c) {
                    this->matrix_data[r * value_cols + c] = values[r][c];
                }
            }
        }

    public:
        static inline matrix zero(size_t rows, size_t cols) {
            matrix<type, 0, 0> result(rows, cols);
            for (size_t i = 0; i < result.size(); ++i) {
                result.matrix_data[i] = 0;
            }
            return result;
        }

        static inline matrix identity(size_t rows, size_t cols) {
            matrix<type, 0, 0> result = matrix<type, 0, 0>::zero(rows, cols);
            for (size_t i = 0; i < math::min(rows, cols); ++i) {
                result.matrix_data[i * cols + i] = 1;
            }
            return result;
        }

    public:
        constexpr size_t rows() const {
            return this->matrix_rows;
        }

        constexpr size_t cols() const {
            return this->matrix_cols;
        }

        constexpr const type* data() const {
            return this->matrix_data;
        }

        constexpr type* data() {
            return this->matrix_data;
        }

        constexpr size_t size() const {
            return this->matrix_rows * this->matrix_cols;
        }

    public:
        constexpr type get_length_squared() const {
            type length_squared = 0;
            for (size_t i = 0; i < this->size(); ++i) {
                length_squared += this->matrix_data[i] * this->matrix_data[i];
            }
            return length_squared;
        }

    public:
        constexpr const type* operator[](size_t index) const {
            ASSERT(index < this->matrix_rows, "Row index out of bounds.");
            return &this->matrix_data[index * this->matrix_cols];
        }

        constexpr type* operator[](size_t index) {
            ASSERT(index < this->matrix_rows, "Row index out of bounds.");
            return &this->matrix_data[index * this->matrix_cols];
        }

    public:
        constexpr const type& operator()(size_t index_row, size_t index_col) const {
            ASSERT(index_row < this->matrix_rows, "Row index must be less than the height of the matrix.");
            ASSERT(index_col < this->matrix_cols, "Column index must be less than the width of the matrix.");
            return this->matrix_data[index_row * this->matrix_cols + index_col];
        }

        constexpr type& operator()(size_t index_row, size_t index_col) {
            ASSERT(index_row < this->matrix_rows, "Row index must be less than the height of the matrix.");
            ASSERT(index_col < this->matrix_cols, "Column index must be less than the width of the matrix.");
            return this->matrix_data[index_row * this->matrix_cols + index_col];
        }
    };

    // Unary operator + and - operations.

    template <typename type, size_t rows, size_t cols>
    constexpr static inline matrix<type, rows, cols> operator+(const matrix<type, rows, cols>& value) {
        matrix<type, rows, cols> result = value;
        for (size_t i = 0; i < result.rows() * result.cols(); ++i) {
            result.data()[i] = +result.data()[i];
        }
        return result;
    }

    template <typename type, size_t rows, size_t cols>
    constexpr static inline matrix<type, rows, cols> operator-(const matrix<type, rows, cols>& value) {
        matrix<type, rows, cols> result = value;
        for (size_t i = 0; i < result.rows() * result.cols(); ++i) {
            result.data()[i] = -result.data()[i];
        }
        return result;
    }

    // Binary operator +, -, and * operations.

    template <typename type, size_t rows, size_t cols>
    constexpr static inline matrix<type, rows, cols> operator+(const matrix<type, rows, cols>& lhs, type rhs) {
        matrix<type, rows, cols> result = lhs;
        for (size_t i = 0; i < result.rows() * result.cols(); ++i) {
            result.data()[i] += rhs;
        }
        return result;
    }

    template <typename type, size_t rows, size_t cols>
    constexpr static inline matrix<type, rows, cols> operator+(type lhs, const matrix<type, rows, cols>& rhs) {
        matrix<type, rows, cols> result = rhs;
        for (size_t i = 0; i < result.rows() * result.cols(); ++i) {
            result.data()[i] += lhs;
        }
        return result;
    }

    template <typename type, size_t rows, size_t cols>
    constexpr static inline matrix<type, rows, cols> operator+(const matrix<type, rows, cols>& lhs, const matrix<type, rows, cols>& rhs) {
        ASSERT(lhs.rows() == rhs.rows(), "Mismated matrix sizes.");
        ASSERT(lhs.cols() == rhs.cols(), "Mismated matrix sizes.");
        matrix<type, rows, cols> result = lhs;
        for (size_t i = 0; i < result.rows() * result.cols(); ++i) {
            result.data()[i] += rhs.data()[i];
        }
        return result;
    }

    template <typename type, size_t rows, size_t cols>
    constexpr static inline matrix<type, rows, cols> operator-(const matrix<type, rows, cols>& lhs, type rhs) {
        matrix<type, rows, cols> result = lhs;
        for (size_t i = 0; i < result.rows() * result.cols(); ++i) {
            result.data()[i] -= rhs;
        }
        return result;
    }

    template <typename type, size_t rows, size_t cols>
    constexpr static inline matrix<type, rows, cols> operator-(type lhs, const matrix<type, rows, cols>& rhs) {
        matrix<type, rows, cols> result = rhs;
        for (size_t i = 0; i < result.rows() * result.cols(); ++i) {
            result.data()[i] -= lhs;
        }
        return result;
    }

    template <typename type, size_t rows, size_t cols>
    constexpr static inline matrix<type, rows, cols> operator-(const matrix<type, rows, cols>& lhs, const matrix<type, rows, cols>& rhs) {
        ASSERT(lhs.rows() == rhs.rows(), "Mismated matrix sizes.");
        ASSERT(lhs.cols() == rhs.cols(), "Mismated matrix sizes.");
        matrix<type, rows, cols> result = lhs;
        for (size_t i = 0; i < result.rows() * result.cols(); ++i) {
            result.data()[i] -= rhs.data()[i];
        }
        return result;
    }

    template <typename type, size_t rows, size_t cols>
    constexpr static inline matrix<type, rows, cols> operator*(const matrix<type, rows, cols>& lhs, const type rhs) {
        matrix<type, rows, cols> result = lhs;
        for (size_t i = 0; i < result.rows() * result.cols(); ++i) {
            result.data()[i] *= rhs;
        }
        return result;
    }

    template <typename type, size_t rows, size_t cols>
    constexpr static inline matrix<type, rows, cols> operator*(const type lhs, const matrix<type, rows, cols>& rhs) {
        matrix<type, rows, cols> result = rhs;
        for (size_t i = 0; i < result.rows() * result.cols(); ++i) {
            result.data()[i] *= lhs;
        }
        return result;
    }

    template <typename type, size_t lhs_rows, size_t lhs_cols, size_t rhs_rows, size_t rhs_cols>
    static inline matrix<type, (((lhs_rows != 0) && (lhs_cols != 0) && (rhs_rows != 0) && (rhs_cols != 0)) ? lhs_rows : 0), (((lhs_rows != 0) && (lhs_cols != 0) && (rhs_rows != 0) && (rhs_cols != 0)) ? rhs_cols : 0)>
    operator*(const matrix<type, lhs_rows, lhs_cols>& lhs, const matrix<type, rhs_rows, rhs_cols>& rhs) {
        if constexpr ((lhs_rows != 0) && (lhs_cols != 0) && (rhs_rows != 0) && (rhs_cols != 0)) {
            static_assert(lhs_cols == rhs_rows);
            constexpr static const size_t lhs_cols_rhs_rows = lhs_cols;
            matrix<type, lhs_rows, rhs_cols> result;
            for (size_t lhs_y = 0; lhs_y < lhs_rows; ++lhs_y) {
                for (size_t rhs_x = 0; rhs_x < rhs_cols; ++rhs_x) {
                    type sum = 0;
                    for (size_t lhs_x_rhs_y = 0; lhs_x_rhs_y < lhs_cols_rhs_rows; ++lhs_x_rhs_y) {
                        sum += lhs.data()[lhs_y * lhs_cols + lhs_x_rhs_y] * rhs.data()[lhs_x_rhs_y * rhs_cols + rhs_x];
                    }
                    result.data()[lhs_y * rhs_cols + rhs_x] = sum;
                }
            }
            return result;
        }
        else {
            ASSERT(lhs.cols() == rhs.rows(), "Invalid matrix sizes.");
            const size_t lhs_cols_rhs_rows = lhs.cols();
            const size_t result_rows = lhs.rows();
            const size_t result_cols = rhs.cols();
            matrix<type, 0, 0> result(result_rows, result_cols);
            for (size_t lhs_y = 0; lhs_y < lhs.rows(); ++lhs_y) {
                for (size_t rhs_x = 0; rhs_x < rhs.cols(); ++rhs_x) {
                    type sum = 0;
                    for (size_t lhs_x_rhs_y = 0; lhs_x_rhs_y < lhs_cols_rhs_rows; ++lhs_x_rhs_y) {
                        sum += lhs.data()[lhs_y * lhs.cols() + lhs_x_rhs_y] * rhs.data()[lhs_x_rhs_y * rhs.cols() + rhs_x];
                    }
                    result.data()[lhs_y * rhs.cols() + rhs_x] = sum;
                }
            }
            return result;
        }
    }

    // Binary operator == and != tests.

    template <typename type, size_t rows, size_t cols>
    constexpr static inline bool operator==(const matrix<type, rows, cols>& lhs, const matrix<type, rows, cols>& rhs) {
        ASSERT(lhs.rows() == rhs.rows(), "Mismated matrix sizes.");
        ASSERT(lhs.cols() == rhs.cols(), "Mismated matrix sizes.");
        for (size_t i = 0; i < lhs.rows() * lhs.cols(); ++i) {
            if (lhs.data()[i] != rhs.data()[i]) {
                return false;
            }
        }
        return true;
    }

    template <typename type, size_t rows, size_t cols>
    constexpr static inline bool operator!=(const matrix<type, rows, cols>& lhs, const matrix<type, rows, cols>& rhs) {
        return !(lhs == rhs);
    }

    // Sub-block access.

    template <typename type, size_t rows, size_t cols, size_t value_rows, size_t value_cols>
    static inline matrix<type, rows, cols> get_block(const matrix<type, value_rows, value_cols>& value, size_t row, size_t col) {
        ASSERT(cols <= value.cols(), "Block cols must be less or equal to matrix cols.");
        ASSERT(rows <= value.rows(), "Block rows must be less or equal to matrix rows.");
        ASSERT(col < value.cols(), "Col index must be less than the cols of the matrix.");
        ASSERT(row < value.rows(), "Row index must be less than the rows of the matrix.");
        ASSERT(col + cols <= value.cols(), "Cols of the requested block must be within the cols of the matrix.");
        ASSERT(row + rows <= value.rows(), "Rows of the requested block must be within the rows of the matrix.");
        matrix<type, rows, cols> block;
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                block.data()[r * block.cols() + c] = value.data()[(row + r) * value.cols() + (col + c)];
            }
        }
        return block;
    }

    template <typename type, size_t rows, size_t cols, size_t value_rows, size_t value_cols>
    constexpr static inline void set_block(matrix<type, value_rows, value_cols>& value, size_t row, size_t col, const matrix<type, rows, cols>& block) {
        ASSERT(block.rows() <= value.rows(), "Block cols must be less or equal to matrix rows.");
        ASSERT(block.cols() <= value.cols(), "Block rows must be less or equal to matrix cols.");
        ASSERT(row < value.rows(), "Row index must be less than the rows of the matrix.");
        ASSERT(col < value.cols(), "Col index must be less than the cols of the matrix.");
        ASSERT(row + block.rows() <= value.rows(), "All rows of the requested block must be within the rows of the matrix.");
        ASSERT(col + block.cols() <= value.cols(), "All cols of the requested block must be within the cols of the matrix.");
        for (size_t r = 0; r < block.rows(); ++r) {
            for (size_t c = 0; c < block.cols(); ++c) {
                value.data()[(row + r) * value.cols() + (col + c)] = block.data()[r * block.cols() + c];
            }
        }
    }

    template <typename type>
    constexpr static inline matrix<type, 0, 0> get_block(const matrix<type, 0, 0>& value, size_t row, size_t col, size_t rows, size_t cols) {
        ASSERT(cols <= value.cols(), "Block cols must be less or equal to matrix cols.");
        ASSERT(rows <= value.rows(), "Block rows must be less or equal to matrix rows.");
        ASSERT(col < value.cols(), "Col index must be less than the cols of the matrix.");
        ASSERT(row < value.rows(), "Row index must be less than the rows of the matrix.");
        ASSERT(col + cols <= value.cols(), "Cols of the requested block must be within the cols of the matrix.");
        ASSERT(row + rows <= value.rows(), "Rows of the requested block must be within the rows of the matrix.");
        matrix<type, 0, 0> block(rows, cols);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                block.data()[r * block.cols() + c] = value.data()[(row + r) * value.cols() + (col + c)];
            }
        }
        return block;
    }

    template <typename type>
    constexpr static inline void set_block(matrix<type, 0, 0>& value, size_t row, size_t col, const matrix<type, 0, 0>& block) {
        ASSERT(block.rows() <= value.rows(), "Block cols must be less or equal to matrix rows.");
        ASSERT(block.cols() <= value.cols(), "Block rows must be less or equal to matrix cols.");
        ASSERT(row < value.rows(), "Row index must be less than the rows of the matrix.");
        ASSERT(col < value.cols(), "Col index must be less than the cols of the matrix.");
        ASSERT(row + block.rows() <= value.rows(), "All rows of the requested block must be within the rows of the matrix.");
        ASSERT(col + block.cols() <= value.cols(), "All cols of the requested block must be within the cols of the matrix.");
        for (size_t r = 0; r < block.rows(); ++r) {
            for (size_t c = 0; c < block.cols(); ++c) {
                value.data()[(row + r) * value.cols() + (col + c)] = block.data()[r * block.cols() + c];
            }
        }
    }

    // General transpose function.

    template <typename type, size_t rows, size_t cols>
    constexpr static inline matrix<type, cols, rows> transpose(const matrix<type, rows, cols>& value) {
        matrix<type, cols, rows> result(value.cols(), value.rows());
        for (size_t i = 0; i < value.rows(); ++i) {
            for (size_t j = 0; j < value.cols(); ++j) {
                result[j][i] = value[i][j];
            }
        }
        return result;
    }

    // General invese funcion.

    template <typename type, size_t size>
    constexpr static inline matrix<type, size, size> invert(const matrix<type, size, size>& value) {
        ASSERT(value.rows() > 0, "Matrix must have rows to have an inverse.");
        ASSERT(value.cols() > 0, "Matrix must have cols to have an inverse.");
        ASSERT(value.rows() == value.cols(), "Matrix must be square to have an inverse.");
        // Generic NxN inversion using Gauss-Jordan.
        matrix working(value);
        matrix augmented = matrix<type, size, size>::identity(value.rows(), value.cols());
        // Forward elimination, moving down the diagonal from top left to bottom right.
        for (size_t i = 0; i < value.cols() - 1; ++i) {
            // Select the pivot row, this is the row with the largest element in the current column below the current row.
            size_t pivot_row_index = i;
            type pivot_scale = math::abs(working[i][i]);
            for (size_t j = i + 1; j < value.rows(); ++j) {
                const type potential_pivot_scale = math::abs(working[j][i]);
                if (potential_pivot_scale > pivot_scale) {
                    pivot_row_index = j;
                    pivot_scale = potential_pivot_scale;
                }
            }
            // If there is no valid pivot there is no valid inverse.
            if (pivot_scale < 1e-9) {
                return matrix<type, size, size>::zero(size, size);
            }
            // Swap current row with the pivot one if the pivot row is not the current row.
            if (pivot_row_index != i) {
                for (size_t j = 0; j < value.cols(); ++j) {
                    type swap_value = working[i][j];
                    working[i][j] = working[pivot_row_index][j];
                    working[pivot_row_index][j] = swap_value;
                    swap_value = augmented[i][j];
                    augmented[i][j] = augmented[pivot_row_index][j];
                    augmented[pivot_row_index][j] = swap_value;
                }
            }
            // Eliminate all numbers below the diagonal.
            for (size_t j = i + 1; j < value.rows(); ++j) {
                const type multiplier = working[j][i] / working[i][i];
                for (size_t k = 0; k < value.cols(); ++k) {
                    working[j][k] -= multiplier * working[i][k];
                    augmented[j][k] -= multiplier * augmented[i][k];
                }
                // Set the column value to exactly zero just in case of float precision issues.
                working[j][i] = 0;
            }
        }
        // Set matrix_data along the diagonal to one.
        for (size_t i = 0; i < value.cols(); ++i) {
            type divisor = working[i][i];
            for (size_t j = 0; j < value.cols(); ++j) {
                working[i][j] = working[i][j] / divisor;
                augmented[i][j] = augmented[i][j] / divisor;
            }
            // Set the diagonal value to exactly one just in case of float precision issues.
            working[i][i] = 1;
        }
        // Eliminate all numbers above the diagonal.
        for (size_t i = 0; i < value.rows() - 1; ++i) {
            for (size_t j = i + 1; j < value.rows(); ++j) {
                type constant = working[i][j];
                for (size_t k = 0; k < value.cols(); ++k) {
                    working[i][k] -= working[j][k] * constant;
                    augmented[i][k] -= augmented[j][k] * constant;
                }
                // Set the working cell value to exactly zero just in case of float precision issues.
                working[i][j] = 0;
            }
        }
        return augmented;
    }

    // Specialised inverse functions.

    template <typename type>
    inline matrix<type, 1, 1> invert(const matrix<type, 1, 1>& value) {
        const type determinant = value[0];
        if (math::abs(determinant) < 1e-6) {
            return matrix<type, 1, 1>::zero();
        }
        const type determinant_inverse = 1.0 / determinant;
        return { { determinant_inverse } };
    }

    template <typename type>
    inline matrix<type, 2, 2> invert(const matrix<type, 2, 2>& value) {
        const type determinant = (value[0][0] * value[1][1]) - (value[0][1] * value[1][0]);
        if (math::abs(determinant) < 1e-6) {
            return matrix<type, 2, 2>::zero();
        }
        const type determinant_inverse = 1.0 / determinant;
        return { { { +value[1][1] * determinant_inverse, -value[0][1] * determinant_inverse },
                   { -value[1][0] * determinant_inverse, +value[0][0] * determinant_inverse } } };
    }

    template <typename type>
    inline matrix<type, 3, 3> invert(const matrix<type, 3, 3>& value) {
        const type determinant =
            +value[0][0] * (value[1][1] * value[2][2] - value[2][1] * value[1][2]) - value[0][1] * (value[1][0] * value[2][2] - value[1][2] * value[2][0]) + value[0][2] * (value[1][0] * value[2][1] - value[1][1] * value[2][0]);
        if (math::abs(determinant) < 1e-6) {
            return matrix<type, 3, 3>::zero();
        }
        const type determinant_inverse = 1.0 / determinant;
        return { { { (value[1][1] * value[2][2] - value[2][1] * value[1][2]) * determinant_inverse,
                     (value[0][2] * value[2][1] - value[0][1] * value[2][2]) * determinant_inverse,
                     (value[0][1] * value[1][2] - value[0][2] * value[1][1]) * determinant_inverse },
                   { (value[1][2] * value[2][0] - value[1][0] * value[2][2]) * determinant_inverse,
                     (value[0][0] * value[2][2] - value[0][2] * value[2][0]) * determinant_inverse,
                     (value[1][0] * value[0][2] - value[0][0] * value[1][2]) * determinant_inverse },
                   { (value[1][0] * value[2][1] - value[2][0] * value[1][1]) * determinant_inverse,
                     (value[2][0] * value[0][1] - value[0][0] * value[2][1]) * determinant_inverse,
                     (value[0][0] * value[1][1] - value[1][0] * value[0][1]) * determinant_inverse } } };
    }
}

#endif // MATRIX_HPP
