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

#include "optimisation/block_cholesky.hpp"

#include "core/assert.hpp"
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <set>
#include <utility>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace optimisation {
    void block_cholesky::analyse(const std::vector<int>& block_dimensions, const std::vector<std::vector<int>>& neighbours) {
        ASSERT(neighbours.size() == block_dimensions.size(), "Every block needs a list of neighbours.");
        const size_t count = block_dimensions.size();
        this->dimensions = block_dimensions;
        this->original_offsets.assign(count + 1, 0);
        for (size_t block = 0; block < count; ++block) {
            ASSERT(block_dimensions[block] > 0, "A block needs a positive dimension.");
            this->original_offsets[block + 1] = this->original_offsets[block] + static_cast<size_t>(block_dimensions[block]);
        }
        this->total_dimensions = this->original_offsets[count];

        // The elimination graph, as sorted lists of the neighbours that are not yet eliminated.
        std::vector<std::vector<int>> graph(count);
        for (size_t block = 0; block < count; ++block) {
            for (const int other : neighbours[block]) {
                ASSERT((other >= 0) && (static_cast<size_t>(other) < count), "A neighbour must be one of the blocks.");
                if (static_cast<size_t>(other) == block) {
                    continue;
                }
                graph[block].push_back(other);
                graph[static_cast<size_t>(other)].push_back(static_cast<int>(block));
            }
        }
        for (std::vector<int>& list : graph) {
            std::sort(list.begin(), list.end());
            list.erase(std::unique(list.begin(), list.end()), list.end());
        }

        // Minimum degree order, ties going to the lowest block so the order only depends on the pattern.
        // Eliminating a block joins its remaining neighbours into a clique, which is its column of the factor.
        std::set<std::pair<size_t, int>> queue;
        for (size_t block = 0; block < count; ++block) {
            queue.insert({ graph[block].size(), static_cast<int>(block) });
        }
        std::vector<std::vector<int>> patterns(count);
        std::vector<int> merged;
        this->order.clear();
        this->order.reserve(count);
        this->positions.assign(count, -1);
        while (!queue.empty()) {
            const int eliminated = queue.begin()->second;
            queue.erase(queue.begin());
            this->positions[static_cast<size_t>(eliminated)] = static_cast<int>(this->order.size());
            this->order.push_back(eliminated);
            const std::vector<int>& clique = graph[static_cast<size_t>(eliminated)];
            for (const int member : clique) {
                std::vector<int>& list = graph[static_cast<size_t>(member)];
                queue.erase({ list.size(), member });
                merged.clear();
                size_t i = 0;
                size_t j = 0;
                while ((i < list.size()) || (j < clique.size())) {
                    int next = 0;
                    if ((j >= clique.size()) || ((i < list.size()) && (list[i] < clique[j]))) {
                        next = list[i++];
                    }
                    else if ((i >= list.size()) || (clique[j] < list[i])) {
                        next = clique[j++];
                    }
                    else {
                        next = list[i];
                        ++i;
                        ++j;
                    }
                    if ((next != member) && (next != eliminated)) {
                        merged.push_back(next);
                    }
                }
                list.swap(merged);
                queue.insert({ list.size(), member });
            }
            patterns[static_cast<size_t>(eliminated)].swap(graph[static_cast<size_t>(eliminated)]);
        }

        // The factor pattern by position, each column's rows ascending.
        this->column_begin.assign(count + 1, 0);
        for (size_t position = 0; position < count; ++position) {
            this->column_begin[position + 1] = this->column_begin[position] + patterns[static_cast<size_t>(this->order[position])].size();
        }
        this->entry_rows.assign(this->column_begin[count], 0);
        this->entry_offsets.assign(this->column_begin[count], 0);
        this->diagonal_offsets.assign(count, 0);
        size_t cursor = 0;
        for (size_t position = 0; position < count; ++position) {
            const size_t block = static_cast<size_t>(this->order[position]);
            const size_t columns = static_cast<size_t>(this->dimensions[block]);
            this->diagonal_offsets[position] = cursor;
            cursor += columns * columns;
            size_t entry = this->column_begin[position];
            for (const int row_block : patterns[block]) {
                this->entry_rows[entry] = this->positions[static_cast<size_t>(row_block)];
                ASSERT(this->entry_rows[entry] > static_cast<int>(position), "A factor column only holds rows below its diagonal.");
                ++entry;
            }
            std::sort(this->entry_rows.data() + this->column_begin[position], this->entry_rows.data() + this->column_begin[position + 1]);
            for (entry = this->column_begin[position]; entry < this->column_begin[position + 1]; ++entry) {
                this->entry_offsets[entry] = cursor;
                cursor += static_cast<size_t>(this->dimensions[static_cast<size_t>(this->order[static_cast<size_t>(this->entry_rows[entry])])]) * columns;
            }
        }
        this->values.assign(cursor, 0.0);

        // The transposed pattern, each row's columns ascending, for the left looking factorisation.
        this->row_begin.assign(count + 1, 0);
        for (size_t entry = 0; entry < this->entry_rows.size(); ++entry) {
            ++this->row_begin[static_cast<size_t>(this->entry_rows[entry]) + 1];
        }
        for (size_t position = 0; position < count; ++position) {
            this->row_begin[position + 1] += this->row_begin[position];
        }
        this->row_columns.assign(this->entry_rows.size(), 0);
        this->row_entries.assign(this->entry_rows.size(), 0);
        std::vector<size_t> fill(this->row_begin.begin(), this->row_begin.end() - 1);
        for (size_t position = 0; position < count; ++position) {
            for (size_t entry = this->column_begin[position]; entry < this->column_begin[position + 1]; ++entry) {
                const size_t row = static_cast<size_t>(this->entry_rows[entry]);
                this->row_columns[fill[row]] = static_cast<int>(position);
                this->row_entries[fill[row]] = entry;
                ++fill[row];
            }
        }

        this->vector_offsets.assign(count + 1, 0);
        for (size_t position = 0; position < count; ++position) {
            this->vector_offsets[position + 1] = this->vector_offsets[position] + static_cast<size_t>(this->dimensions[static_cast<size_t>(this->order[position])]);
        }
        this->scatter.assign(count, 0);
        this->workspace.assign(this->total_dimensions, 0.0);
    }

    int block_cholesky::block_count() const {
        return static_cast<int>(this->order.size());
    }

    size_t block_cholesky::dimension_count() const {
        return this->total_dimensions;
    }

    size_t block_cholesky::factor_blocks() const {
        return this->order.size() + this->entry_rows.size();
    }

    int block_cholesky::position_of(const int block) const {
        return this->positions[static_cast<size_t>(block)];
    }

    int block_cholesky::block_at(const int position) const {
        return this->order[static_cast<size_t>(position)];
    }

    int block_cholesky::dimensions_at(const int position) const {
        return this->dimensions[static_cast<size_t>(this->order[static_cast<size_t>(position)])];
    }

    size_t block_cholesky::diagonal_offset(const int position) const {
        return this->diagonal_offsets[static_cast<size_t>(position)];
    }

    bool block_cholesky::find_offset(const int row_position, const int column_position, size_t& offset) const {
        const int* const begin = this->entry_rows.data() + this->column_begin[static_cast<size_t>(column_position)];
        const int* const end = this->entry_rows.data() + this->column_begin[static_cast<size_t>(column_position) + 1];
        const int* const found = std::lower_bound(begin, end, row_position);
        if ((found == end) || (*found != row_position)) {
            return false;
        }
        offset = this->entry_offsets[this->column_begin[static_cast<size_t>(column_position)] + static_cast<size_t>(found - begin)];
        return true;
    }

    size_t block_cholesky::column_entries_begin(const int position) const {
        return this->column_begin[static_cast<size_t>(position)];
    }

    size_t block_cholesky::column_entries_end(const int position) const {
        return this->column_begin[static_cast<size_t>(position) + 1];
    }

    int block_cholesky::entry_row(const size_t entry) const {
        return this->entry_rows[entry];
    }

    size_t block_cholesky::entry_offset(const size_t entry) const {
        return this->entry_offsets[entry];
    }

    double* block_cholesky::get_values() {
        return this->values.data();
    }

    const double* block_cholesky::get_values() const {
        return this->values.data();
    }

    bool block_cholesky::factorise() {
        const size_t count = this->order.size();
        double* const data = this->values.data();
        double scale = 0.0;
        for (size_t position = 0; position < count; ++position) {
            const size_t size = static_cast<size_t>(this->dimensions_at(static_cast<int>(position)));
            const double* const diagonal = data + this->diagonal_offsets[position];
            for (size_t i = 0; i < size; ++i) {
                scale = math::max(scale, math::abs(diagonal[(i * size) + i]));
            }
        }
        const double epsilon = scale * 1e-14;
        for (size_t position = 0; position < count; ++position) {
            const size_t size = static_cast<size_t>(this->dimensions_at(static_cast<int>(position)));
            double* const diagonal = data + this->diagonal_offsets[position];
            for (size_t entry = this->column_begin[position]; entry < this->column_begin[position + 1]; ++entry) {
                this->scatter[static_cast<size_t>(this->entry_rows[entry])] = this->entry_offsets[entry];
            }
            // Subtract the columns to the left that have a block in this row.
            for (size_t index = this->row_begin[position]; index < this->row_begin[position + 1]; ++index) {
                const size_t column = static_cast<size_t>(this->row_columns[index]);
                const size_t inner = static_cast<size_t>(this->dimensions_at(static_cast<int>(column)));
                const size_t first = this->row_entries[index];
                const double* const pivot = data + this->entry_offsets[first];
                for (size_t a = 0; a < size; ++a) {
                    for (size_t b = 0; b <= a; ++b) {
                        double sum = 0.0;
                        for (size_t c = 0; c < inner; ++c) {
                            sum += pivot[(a * inner) + c] * pivot[(b * inner) + c];
                        }
                        diagonal[(a * size) + b] -= sum;
                    }
                }
                for (size_t entry = first + 1; entry < this->column_begin[column + 1]; ++entry) {
                    const size_t row = static_cast<size_t>(this->entry_rows[entry]);
                    const size_t rows = static_cast<size_t>(this->dimensions_at(static_cast<int>(row)));
                    const double* const source = data + this->entry_offsets[entry];
                    double* const target = data + this->scatter[row];
                    for (size_t a = 0; a < rows; ++a) {
                        for (size_t b = 0; b < size; ++b) {
                            double sum = 0.0;
                            for (size_t c = 0; c < inner; ++c) {
                                sum += source[(a * inner) + c] * pivot[(b * inner) + c];
                            }
                            target[(a * size) + b] -= sum;
                        }
                    }
                }
            }
            // Factorise the diagonal block.
            for (size_t i = 0; i < size; ++i) {
                for (size_t k = 0; k < i; ++k) {
                    double value = diagonal[(i * size) + k];
                    for (size_t j = 0; j < k; ++j) {
                        value -= diagonal[(i * size) + j] * diagonal[(k * size) + j];
                    }
                    diagonal[(i * size) + k] = value / diagonal[(k * size) + k];
                }
                double value = diagonal[(i * size) + i];
                for (size_t j = 0; j < i; ++j) {
                    value -= diagonal[(i * size) + j] * diagonal[(i * size) + j];
                }
                if (!(value > epsilon)) {
                    return false;
                }
                diagonal[(i * size) + i] = math::sqrt(value);
            }
            // The blocks below the diagonal become B * L^-T.
            for (size_t entry = this->column_begin[position]; entry < this->column_begin[position + 1]; ++entry) {
                const size_t rows = static_cast<size_t>(this->dimensions_at(this->entry_rows[entry]));
                double* const target = data + this->entry_offsets[entry];
                for (size_t a = 0; a < rows; ++a) {
                    double* const row = target + (a * size);
                    for (size_t j = 0; j < size; ++j) {
                        double value = row[j];
                        for (size_t m = 0; m < j; ++m) {
                            value -= row[m] * diagonal[(j * size) + m];
                        }
                        row[j] = value / diagonal[(j * size) + j];
                    }
                }
            }
        }
        return true;
    }

    bool block_cholesky::solve(const double* right_hand_side, double* solution) {
        const size_t count = this->order.size();
        const double* const data = this->values.data();
        double scale = 0.0;
        for (size_t position = 0; position < count; ++position) {
            const size_t size = static_cast<size_t>(this->dimensions_at(static_cast<int>(position)));
            const double* const diagonal = data + this->diagonal_offsets[position];
            for (size_t i = 0; i < size; ++i) {
                scale = math::max(scale, math::abs(diagonal[(i * size) + i]));
            }
        }
        const double epsilon = scale * 1e-7;
        for (size_t position = 0; position < count; ++position) {
            const size_t size = static_cast<size_t>(this->dimensions_at(static_cast<int>(position)));
            const double* const diagonal = data + this->diagonal_offsets[position];
            for (size_t i = 0; i < size; ++i) {
                if (!(math::abs(diagonal[(i * size) + i]) > epsilon)) {
                    return false;
                }
            }
        }
        double* const work = this->workspace.data();
        for (size_t position = 0; position < count; ++position) {
            const size_t size = static_cast<size_t>(this->dimensions_at(static_cast<int>(position)));
            const double* const source = right_hand_side + this->original_offsets[static_cast<size_t>(this->order[position])];
            for (size_t i = 0; i < size; ++i) {
                work[this->vector_offsets[position] + i] = source[i];
            }
        }
        for (size_t position = 0; position < count; ++position) {
            const size_t size = static_cast<size_t>(this->dimensions_at(static_cast<int>(position)));
            const double* const diagonal = data + this->diagonal_offsets[position];
            double* const current = work + this->vector_offsets[position];
            for (size_t i = 0; i < size; ++i) {
                double value = current[i];
                for (size_t j = 0; j < i; ++j) {
                    value -= diagonal[(i * size) + j] * current[j];
                }
                current[i] = value / diagonal[(i * size) + i];
            }
            for (size_t entry = this->column_begin[position]; entry < this->column_begin[position + 1]; ++entry) {
                const size_t row = static_cast<size_t>(this->entry_rows[entry]);
                const size_t rows = static_cast<size_t>(this->dimensions_at(static_cast<int>(row)));
                const double* const block = data + this->entry_offsets[entry];
                double* const target = work + this->vector_offsets[row];
                for (size_t a = 0; a < rows; ++a) {
                    double sum = 0.0;
                    for (size_t b = 0; b < size; ++b) {
                        sum += block[(a * size) + b] * current[b];
                    }
                    target[a] -= sum;
                }
            }
        }
        for (size_t position = count; position-- > 0;) {
            const size_t size = static_cast<size_t>(this->dimensions_at(static_cast<int>(position)));
            const double* const diagonal = data + this->diagonal_offsets[position];
            double* const current = work + this->vector_offsets[position];
            for (size_t entry = this->column_begin[position]; entry < this->column_begin[position + 1]; ++entry) {
                const size_t row = static_cast<size_t>(this->entry_rows[entry]);
                const size_t rows = static_cast<size_t>(this->dimensions_at(static_cast<int>(row)));
                const double* const block = data + this->entry_offsets[entry];
                const double* const source = work + this->vector_offsets[row];
                for (size_t b = 0; b < size; ++b) {
                    double sum = 0.0;
                    for (size_t a = 0; a < rows; ++a) {
                        sum += block[(a * size) + b] * source[a];
                    }
                    current[b] -= sum;
                }
            }
            for (size_t i = size; i-- > 0;) {
                double value = current[i];
                for (size_t j = i + 1; j < size; ++j) {
                    value -= diagonal[(j * size) + i] * current[j];
                }
                current[i] = value / diagonal[(i * size) + i];
            }
        }
        for (size_t position = 0; position < count; ++position) {
            const size_t size = static_cast<size_t>(this->dimensions_at(static_cast<int>(position)));
            double* const target = solution + this->original_offsets[static_cast<size_t>(this->order[position])];
            for (size_t i = 0; i < size; ++i) {
                target[i] = work[this->vector_offsets[position] + i];
            }
        }
        return true;
    }
}
