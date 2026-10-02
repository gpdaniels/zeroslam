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
#ifndef ZEROSLAM_OPTIMISATION_BLOCK_CHOLESKY_HPP
#define ZEROSLAM_OPTIMISATION_BLOCK_CHOLESKY_HPP

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

namespace optimisation {
    // Sparse Cholesky factorisation L * L^T of a symmetric positive definite matrix made of dense blocks.
    // The blocks are eliminated in a minimum degree order, so the factor is stored by position in that order,
    // each column holding its diagonal block and then its nonzero blocks below the diagonal, all row major.
    class block_cholesky final {
    public:
        // The smallest fraction of its diagonal a pivot may keep after the elimination, below which the matrix is singular.
        constexpr static const double pivot_fraction_minimum = 1.0e-13;

    private:
        std::vector<int> dimensions;
        std::vector<double> original_diagonal;
        std::vector<size_t> original_offsets;
        std::vector<int> order;
        std::vector<int> positions;
        std::vector<size_t> diagonal_offsets;
        std::vector<size_t> column_begin;
        std::vector<int> entry_rows;
        std::vector<size_t> entry_offsets;
        std::vector<size_t> row_begin;
        std::vector<int> row_columns;
        std::vector<size_t> row_entries;
        std::vector<size_t> vector_offsets;
        std::vector<size_t> scatter;
        std::vector<double> values;
        std::vector<double> workspace;
        size_t total_dimensions = 0;

    public:
        // Sets up the elimination order and the factor pattern for blocks of the given dimensions, where neighbours
        // lists the other blocks that share a nonzero block with each block (either direction, duplicates allowed).
        void analyse(const std::vector<int>& block_dimensions, const std::vector<std::vector<int>>& neighbours);

        int block_count() const;
        size_t dimension_count() const;
        size_t factor_blocks() const;
        int position_of(const int block) const;
        int block_at(const int position) const;
        int dimensions_at(const int position) const;

        // The storage of the diagonal block at a position, and of the block with rows at row_position below it.
        size_t diagonal_offset(const int position) const;
        bool find_offset(const int row_position, const int column_position, size_t& offset) const;
        size_t column_entries_begin(const int position) const;
        size_t column_entries_end(const int position) const;
        int entry_row(const size_t entry) const;
        size_t entry_offset(const size_t entry) const;

        double* get_values();
        const double* get_values() const;

        // Factorises the lower triangle of the assembled blocks in place, failing on a pivot that keeps less than
        // pivot_fraction_minimum of its row's diagonal entry.
        bool factorise();

        // Solves for a right hand side in the original block order, both vectors laid out block after block.
        bool solve(const double* right_hand_side, double* solution);
    };
}

#endif // ZEROSLAM_OPTIMISATION_BLOCK_CHOLESKY_HPP
