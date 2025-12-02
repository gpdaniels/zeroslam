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
#ifndef MATRIX_SPARSE_BLOCK_DIAGONAL_HPP
#define MATRIX_SPARSE_BLOCK_DIAGONAL_HPP

#include "matrix.hpp"
#include "matrix_sparse_block.hpp"

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

namespace matrix {
    template <size_t template_block_size>
    class sparse_block_diagonal final {
    public:
        constexpr static const size_t block_size = template_block_size;
        using block_type = matrix<double, block_size, block_size>;

    private:
        std::vector<block_type> sparse_blocks_diagonal;

    public:
        size_t rows() const {
            return this->sparse_blocks_diagonal.size() * sparse_block_diagonal::block_size;
        }

        size_t cols() const {
            return this->sparse_blocks_diagonal.size() * sparse_block_diagonal::block_size;
        }

    public:
        double operator()(size_t index_row, size_t index_col) const {
            ASSERT(index_row < this->sparse_blocks_diagonal.size() * sparse_block_diagonal::block_size, "Row index must be less than the height of the matrix.");
            ASSERT(index_col < this->sparse_blocks_diagonal.size() * sparse_block_diagonal::block_size, "Column index must be less than the width of the matrix.");
            const size_t block_row = index_row / sparse_block_diagonal::block_size;
            const size_t block_col = index_col / sparse_block_diagonal::block_size;
            if (block_row != block_col) {
                return 0;
            }
            const size_t offset_row = index_row % sparse_block_diagonal::block_size;
            const size_t offset_col = index_col % sparse_block_diagonal::block_size;
            return this->sparse_blocks_diagonal[block_row][offset_row][offset_col];
        }

    public:
        const std::vector<block_type>& diagonal() const {
            return this->sparse_blocks_diagonal;
        }

        std::vector<block_type>& diagonal() {
            return this->sparse_blocks_diagonal;
        }

    public:
        matrix<double, 0, 0> multiply(const matrix<double, 0, 0>& rhs_vector) const {
            ASSERT(rhs_vector.cols() == 1, "RHS Vector must be of width 1.");
            ASSERT(rhs_vector.rows() == this->cols(), "RHS Vector must be same height as diagonal matrix.");
            matrix<double, 0, 0> result = matrix<double, 0, 0>::zero(this->sparse_blocks_diagonal.size() * sparse_block_diagonal::block_size, 1);
            for (size_t index = 0; index < this->sparse_blocks_diagonal.size(); ++index) {
                const int offset = index * sparse_block_diagonal::block_size;
                const block_type& block = this->sparse_blocks_diagonal[index];
                set_block(result, offset, 0, get_block(result, offset, 0, sparse_block_diagonal::block_size, 1) + block * get_block(rhs_vector, offset, 0, sparse_block_diagonal::block_size, 1));
            }
            return result;
        }

        matrix<double, 0, 0> multiply_right(const matrix<double, 0, 0>& lhs_vector) const {
            ASSERT(lhs_vector.cols() == this->rows(), "LHS Vector must be same width as diagonal matrix.");
            ASSERT(lhs_vector.rows() == 1, "LHS Vector must be of height 1.");
            matrix<double, 0, 0> result = matrix<double, 0, 0>::zero(1, this->sparse_blocks_diagonal.size() * sparse_block_diagonal::block_size);
            for (size_t index = 0; index < this->sparse_blocks_diagonal.size(); ++index) {
                const int offset = index * sparse_block_diagonal::block_size;
                const block_type& block = this->sparse_blocks_diagonal[index];
                set_block(result, 0, offset, get_block(result, 0, offset, 1, sparse_block_diagonal::block_size) + get_block(lhs_vector, 0, offset, 1, sparse_block_diagonal::block_size) * block);
            }
            return result;
        }
    };

    template <size_t block_rows, size_t block_cols>
    sparse_block<block_rows, block_cols> multiply(const sparse_block<block_rows, block_cols>& lhs, const sparse_block_diagonal<block_cols>& rhs) {
        ASSERT(lhs.cols() == rhs.rows(), "LHS cols must equal RHS rows.");
        sparse_block<block_rows, block_cols> result(lhs.rows(), rhs.cols());
        result.blocks().reserve(lhs.blocks().size());
        for (typename std::unordered_map<typename sparse_block<block_rows, block_cols>::template block_key<size_t, size_t>, typename sparse_block<block_rows, block_cols>::block_type>::const_iterator lhs_iterator = lhs.blocks().begin(); lhs_iterator != lhs.blocks().end(); ++lhs_iterator) {
            typename std::unordered_map<typename sparse_block<block_rows, block_cols>::template block_key<size_t, size_t>, typename sparse_block<block_rows, block_cols>::block_type>::iterator result_iterator = result.blocks().find(lhs_iterator->first);
            if (result_iterator == result.blocks().end()) {
                result.blocks()[lhs_iterator->first] = lhs_iterator->second * rhs.diagonal()[lhs_iterator->first.second];
            }
            else {
                result_iterator->second = result_iterator->second + lhs_iterator->second * rhs.diagonal()[lhs_iterator->first.second];
            }
        }
        return result;
    }

    template <size_t block_size>
    sparse_block<block_size, block_size> subtract(const sparse_block_diagonal<block_size>& lhs, const sparse_block<block_size, block_size>& rhs) {
        ASSERT(lhs.rows() == rhs.rows(), "LHS rows must equal RHS rows.");
        ASSERT(lhs.cols() == rhs.cols(), "LHS cols must equal RHS cols.");
        sparse_block<block_size, block_size> result(lhs.rows(), lhs.cols());
        result.blocks().reserve(rhs.blocks().size());
        for (size_t index = 0; index < lhs.diagonal().size(); ++index) {
            result.blocks()[{ index, index }] = lhs.diagonal()[index];
        }
        for (typename std::unordered_map<typename sparse_block<block_size, block_size>::template block_key<size_t, size_t>, typename sparse_block<block_size, block_size>::block_type>::const_iterator rhs_iterator = rhs.blocks().begin(); rhs_iterator != rhs.blocks().end(); ++rhs_iterator) {
            typename std::unordered_map<typename sparse_block<block_size, block_size>::template block_key<size_t, size_t>, typename sparse_block<block_size, block_size>::block_type>::iterator result_iterator = result.blocks().find(rhs_iterator->first);
            if (result_iterator == result.blocks().end()) {
                result.blocks()[rhs_iterator->first] = -rhs_iterator->second;
            }
            else {
                result_iterator->second = result_iterator->second - rhs_iterator->second;
            }
        }
        return result;
    }
}

#endif // MATRIX_SPARSE_BLOCK_DIAGONAL_HPP