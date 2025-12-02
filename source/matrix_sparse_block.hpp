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
#ifndef MATRIX_SPARSE_BLOCK_HPP
#define MATRIX_SPARSE_BLOCK_HPP

#include "matrix.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <unordered_map>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace matrix {
    template <size_t template_block_rows, size_t template_block_cols>
    class sparse_block final {
    public:
        constexpr static const size_t block_rows = template_block_rows;
        constexpr static const size_t block_cols = template_block_cols;
        using block_type = matrix<double, block_rows, block_cols>;

    public:
        template <class type_lhs, class type_rhs>
        class block_key {
        public:
            type_lhs first;
            type_rhs second;

            bool operator==(const block_key& other) const {
                return (this->first == other.first && this->second == other.second);
            }

            std::size_t operator()(const block_key& k) const {
                return ((std::hash<type_lhs>()(k.first) ^ (std::hash<type_rhs>()(k.second) << 1)) >> 1);
            }
        };

    private:
        template <size_t friend_rows, size_t friend_cols>
        friend class sparse_block;

    private:
        size_t matrix_rows;
        size_t matrix_cols;
        std::unordered_map<block_key<size_t, size_t>, block_type, block_key<size_t, size_t>> sparse_blocks;

    public:
        sparse_block(size_t input_rows = 0, size_t input_cols = 0)
            : matrix_rows(input_rows)
            , matrix_cols(input_cols) {
            ASSERT(input_rows % sparse_block::block_rows == 0, "Matrix rows size must be a multiple of the block rows.");
            ASSERT(input_cols % sparse_block::block_cols == 0, "Matrix cols size must be a multiple of the block cols.");
        }

    public:
        size_t rows() const {
            return this->matrix_rows;
        }

        size_t cols() const {
            return this->matrix_cols;
        }

    public:
        double operator()(size_t index_row, size_t index_col) const {
            ASSERT(index_row < this->rows(), "Row index must be less than the height of the matrix.");
            ASSERT(index_col < this->cols(), "Column index must be less than the width of the matrix.");
            const size_t block_row = index_row / sparse_block::block_rows;
            const size_t block_col = index_col / sparse_block::block_cols;
            typename std::unordered_map<block_key<size_t, size_t>, block_type>::const_iterator block_iterator = this->sparse_blocks.find({ block_row, block_col });
            if (block_iterator == this->sparse_blocks.end()) {
                return 0;
            }
            const size_t offset_row = index_row % sparse_block::block_rows;
            const size_t offset_col = index_col % sparse_block::block_cols;
            return block_iterator->second(offset_row, offset_col);
        }

    public:
        void set_size(size_t input_rows = 0, size_t input_cols = 0) {
            ASSERT(input_rows % sparse_block::block_rows == 0, "Matrix rows size must be a multiple of the block rows.");
            ASSERT(input_cols % sparse_block::block_cols == 0, "Matrix cols size must be a multiple of the block cols.");
            this->matrix_rows = input_rows;
            this->matrix_cols = input_cols;
        }

        std::unordered_map<block_key<size_t, size_t>, block_type, block_key<size_t, size_t>>& blocks() {
            return this->sparse_blocks;
        }

        const std::unordered_map<block_key<size_t, size_t>, block_type, block_key<size_t, size_t>>& blocks() const {
            return this->sparse_blocks;
        }

    public:
        bool validate() const {
            for (typename std::unordered_map<block_key<size_t, size_t>, block_type>::const_iterator block_iterator = this->sparse_blocks.begin(); block_iterator != this->sparse_blocks.end(); ++block_iterator) {
                if (block_iterator->first.first >= this->rows() / sparse_block::block_rows) {
                    return false;
                }
                if (block_iterator->first.second >= this->cols() / sparse_block::block_cols) {
                    return false;
                }
            }
            return true;
        }

    public:
        sparse_block<block_cols, block_rows> get_transpose() const {
            sparse_block<block_cols, block_rows> result(this->cols(), this->rows());
            result.sparse_blocks.reserve(this->sparse_blocks.size());
            for (typename std::unordered_map<block_key<size_t, size_t>, block_type>::const_iterator block_iterator = this->sparse_blocks.begin(); block_iterator != this->sparse_blocks.end(); ++block_iterator) {
                result.sparse_blocks[{ block_iterator->first.second, block_iterator->first.first }] = transpose(block_iterator->second);
            }
            return result;
        }

    public:
        sparse_block add(const sparse_block& rhs) const {
            ASSERT(this->rows() == rhs.rows(), "LHS rows must equal RHS rows.");
            ASSERT(this->cols() == rhs.cols(), "LHS cols must equal RHS cols.");
            sparse_block result = *this;
            for (typename std::unordered_map<block_key<size_t, size_t>, block_type>::const_iterator rhs_iterator = rhs.sparse_blocks.begin(); rhs_iterator != rhs.sparse_blocks.end(); ++rhs_iterator) {
                typename std::unordered_map<block_key<size_t, size_t>, block_type>::iterator result_iterator = result.sparse_blocks.find(rhs_iterator->first);
                if (result_iterator == result.sparse_blocks.end()) {
                    result.sparse_blocks[rhs_iterator->first] = rhs_iterator->second;
                }
                else {
                    result_iterator->second = result_iterator->second + rhs_iterator->second;
                }
            }
            return result;
        }

        sparse_block<block_rows, block_rows> multiply(const sparse_block<block_cols, block_rows>& rhs) const {
            ASSERT(this->cols() == rhs.rows(), "LHS cols must equal RHS rows.");
            sparse_block<block_rows, block_rows> result(this->rows(), rhs.cols());
            result.sparse_blocks.reserve(this->sparse_blocks.size());
            // For each block in the lhs matrix.
            for (auto lhs_iterator = this->sparse_blocks.cbegin(); lhs_iterator != this->sparse_blocks.cend(); ++lhs_iterator) {
                // Scan each column of the rhs matrix.
                for (size_t rhs_col = 0; rhs_col < rhs.cols(); ++rhs_col) {
                    // If there if a block in the rhs matrix with rhs_row == lhs_col.
                    auto rhs_iterator = rhs.sparse_blocks.find({ lhs_iterator->first.second, rhs_col });
                    if (rhs_iterator != rhs.sparse_blocks.end()) {
                        // Add the value of the multiplication of the blocks to the result block at location {lhs_row, rhs_col}.
                        auto result_iterator = result.sparse_blocks.find({ lhs_iterator->first.first, rhs_iterator->first.second });
                        if (result_iterator == result.sparse_blocks.end()) {
                            result.blocks()[{ lhs_iterator->first.first, rhs_iterator->first.second }] = lhs_iterator->second * rhs_iterator->second;
                        }
                        else {
                            result_iterator->second = result_iterator->second + lhs_iterator->second * rhs_iterator->second;
                        }
                    }
                }
            }
            return result;
        }

        matrix<double, 0, 0> multiply(const matrix<double, 0, 0>& rhs_vector) const {
            ASSERT(rhs_vector.cols() == 1, "RHS Vector must be of width 1.");
            ASSERT(rhs_vector.rows() == this->cols(), "RHS Vector must be same height as sparse matrix.");
            matrix<double, 0, 0> result = matrix<double, 0, 0>::zero(this->rows(), 1);
            for (auto block_iterator = this->sparse_blocks.cbegin(); block_iterator != this->sparse_blocks.cend(); ++block_iterator) {
                const size_t offset_row = block_iterator->first.first * sparse_block::block_rows;
                const size_t offset_col = block_iterator->first.second * sparse_block::block_cols;
                const matrix<double, 0, 0> result_block = get_block(result, offset_row, 0, sparse_block::block_rows, 1);
                const matrix<double, 0, 0> vector_block = get_block(rhs_vector, offset_col, 0, sparse_block::block_cols, 1);
                set_block(result, offset_row, 0, result_block + block_iterator->second * vector_block);
            }
            return result;
        }

        matrix<double, 0, 0> multiply_right(const matrix<double, 0, 0>& lhs_vector) const {
            ASSERT(lhs_vector.cols() == this->rows(), "LHS Vector must be same height as sparse matrix.");
            ASSERT(lhs_vector.rows() == 1, "LHS Vector must be of width 1.");
            matrix<double, 0, 0> result = matrix<double, 0, 0>::zero(1, this->cols());
            for (auto block_iterator = this->sparse_blocks.cbegin(); block_iterator != this->sparse_blocks.cend(); ++block_iterator) {
                const size_t offset_row = block_iterator->first.first * sparse_block::block_rows;
                const size_t offset_col = block_iterator->first.second * sparse_block::block_cols;
                const matrix<double, 0, 0> result_block = get_block(result, 0, offset_col, 1, sparse_block::block_cols);
                const matrix<double, 0, 0> vector_block = get_block(lhs_vector, 0, offset_row, 1, sparse_block::block_rows);
                set_block(result, 0, offset_col, result_block + vector_block * block_iterator->second);
            }
            return result;
        }
    };
}

#endif // MATRIX_SPARSE_BLOCK_HPP