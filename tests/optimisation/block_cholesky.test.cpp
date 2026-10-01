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

#include "core/random_pcg.hpp"
#include "math/matrix_decomposition_cholesky.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

namespace {
    // A symmetric positive definite matrix of blocks with a given pattern, held dense for reference.
    class block_problem final {
    public:
        std::vector<int> dimensions;
        std::vector<size_t> offsets;
        std::vector<std::vector<int>> neighbours;
        std::vector<double> dense;
        size_t size = 0;
    };

    block_problem make_problem(core::random_pcg& rng, const std::vector<int>& dimensions, const std::vector<std::pair<int, int>>& pattern) {
        block_problem problem;
        problem.dimensions = dimensions;
        problem.offsets.assign(dimensions.size() + 1, 0);
        for (size_t block = 0; block < dimensions.size(); ++block) {
            problem.offsets[block + 1] = problem.offsets[block] + static_cast<size_t>(dimensions[block]);
        }
        problem.size = problem.offsets[dimensions.size()];
        problem.neighbours.assign(dimensions.size(), {});
        // A random J^T J over the pattern, plus a diagonal, so the matrix is positive definite with exactly that pattern.
        problem.dense.assign(problem.size * problem.size, 0.0);
        const auto add_term = [&problem, &rng](const int first, const int second) {
            const size_t rows = static_cast<size_t>(problem.dimensions[static_cast<size_t>(first)] + ((first != second) ? problem.dimensions[static_cast<size_t>(second)] : 0));
            std::vector<size_t> columns;
            for (size_t c = 0; c < static_cast<size_t>(problem.dimensions[static_cast<size_t>(first)]); ++c) {
                columns.push_back(problem.offsets[static_cast<size_t>(first)] + c);
            }
            if (first != second) {
                for (size_t c = 0; c < static_cast<size_t>(problem.dimensions[static_cast<size_t>(second)]); ++c) {
                    columns.push_back(problem.offsets[static_cast<size_t>(second)] + c);
                }
            }
            for (size_t r = 0; r < rows; ++r) {
                std::vector<double> row(columns.size());
                for (double& value : row) {
                    value = rng.get_random(-1.0, 1.0);
                }
                for (size_t a = 0; a < columns.size(); ++a) {
                    for (size_t b = 0; b < columns.size(); ++b) {
                        problem.dense[(columns[a] * problem.size) + columns[b]] += row[a] * row[b];
                    }
                }
            }
        };
        for (size_t block = 0; block < dimensions.size(); ++block) {
            add_term(static_cast<int>(block), static_cast<int>(block));
        }
        for (const std::pair<int, int>& link : pattern) {
            add_term(link.first, link.second);
            problem.neighbours[static_cast<size_t>(link.first)].push_back(link.second);
        }
        for (size_t i = 0; i < problem.size; ++i) {
            problem.dense[(i * problem.size) + i] += 0.5;
        }
        return problem;
    }

    // Copies the dense matrix into the factor storage, each block of the pattern once below the diagonal.
    void assemble(const block_problem& problem, optimisation::block_cholesky& factor) {
        double* const values = factor.get_values();
        for (int position = 0; position < factor.block_count(); ++position) {
            const size_t column_block = static_cast<size_t>(factor.block_at(position));
            const size_t cols = static_cast<size_t>(problem.dimensions[column_block]);
            double* const diagonal = values + factor.diagonal_offset(position);
            for (size_t a = 0; a < cols; ++a) {
                for (size_t b = 0; b < cols; ++b) {
                    diagonal[(a * cols) + b] = problem.dense[((problem.offsets[column_block] + a) * problem.size) + problem.offsets[column_block] + b];
                }
            }
            for (size_t entry = factor.column_entries_begin(position); entry < factor.column_entries_end(position); ++entry) {
                const size_t row_block = static_cast<size_t>(factor.block_at(factor.entry_row(entry)));
                const size_t rows = static_cast<size_t>(problem.dimensions[row_block]);
                double* const target = values + factor.entry_offset(entry);
                for (size_t a = 0; a < rows; ++a) {
                    for (size_t b = 0; b < cols; ++b) {
                        target[(a * cols) + b] = problem.dense[((problem.offsets[row_block] + a) * problem.size) + problem.offsets[column_block] + b];
                    }
                }
            }
        }
    }

    // Solves the problem both ways and returns the largest difference relative to the largest solution entry.
    double compare_with_dense(const block_problem& problem, optimisation::block_cholesky& factor, core::random_pcg& rng) {
        std::vector<double> right_hand_side(problem.size);
        for (double& value : right_hand_side) {
            value = rng.get_random(-1.0, 1.0);
        }
        std::vector<double> lower(problem.size * problem.size);
        std::vector<double> expected(problem.size);
        REQUIRE(math::decompose_cholesky(problem.dense.data(), static_cast<int>(problem.size), static_cast<int>(problem.size), lower.data()));
        REQUIRE(math::solve_cholesky(lower.data(), right_hand_side.data(), static_cast<int>(problem.size), static_cast<int>(problem.size), expected.data()));
        assemble(problem, factor);
        REQUIRE(factor.factorise());
        std::vector<double> solution(problem.size);
        REQUIRE(factor.solve(right_hand_side.data(), solution.data()));
        double largest = 0.0;
        double difference = 0.0;
        for (size_t i = 0; i < problem.size; ++i) {
            largest = std::max(largest, std::abs(expected[i]));
            difference = std::max(difference, std::abs(expected[i] - solution[i]));
        }
        return difference / largest;
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    core::random_pcg rng(0x5eedb10cull);

    {
        // A chain of pose sized blocks, as keyframes in time order, factors without fill.
        std::vector<std::pair<int, int>> pattern;
        for (int i = 0; i + 1 < 40; ++i) {
            pattern.push_back({ i, i + 1 });
        }
        const block_problem problem = make_problem(rng, std::vector<int>(40, 6), pattern);
        optimisation::block_cholesky factor;
        factor.analyse(problem.dimensions, problem.neighbours);
        REQUIRE(factor.block_count() == 40);
        REQUIRE(factor.dimension_count() == 240);
        REQUIRE(factor.factor_blocks() == 40 + 39);
        REQUIRE(compare_with_dense(problem, factor, rng) < 1e-10);
    }

    {
        // A star, one block seen with all others: the leaves go first and the hub last of all but one, so there is no fill.
        std::vector<std::pair<int, int>> pattern;
        for (int i = 1; i < 30; ++i) {
            pattern.push_back({ 0, i });
        }
        const block_problem problem = make_problem(rng, std::vector<int>(30, 3), pattern);
        optimisation::block_cholesky factor;
        factor.analyse(problem.dimensions, problem.neighbours);
        REQUIRE(factor.factor_blocks() == 30 + 29);
        REQUIRE(factor.position_of(0) >= 28);
        REQUIRE(compare_with_dense(problem, factor, rng) < 1e-10);
    }

    for (int trial = 0; trial < 20; ++trial) {
        // Random patterns and block sizes, the order found twice must be the same, and every block visited once.
        const int blocks = 5 + static_cast<int>(rng.get_random_raw() % 40u);
        std::vector<int> dimensions;
        for (int i = 0; i < blocks; ++i) {
            dimensions.push_back(1 + static_cast<int>(rng.get_random_raw() % 8u));
        }
        std::vector<std::pair<int, int>> pattern;
        for (int i = 0; i < blocks; ++i) {
            for (int j = i + 1; j < blocks; ++j) {
                if (rng.get_random(0.0, 1.0) < 0.12) {
                    pattern.push_back({ ((trial % 2) == 0) ? i : j, ((trial % 2) == 0) ? j : i });
                }
            }
        }
        const block_problem problem = make_problem(rng, dimensions, pattern);
        optimisation::block_cholesky factor;
        factor.analyse(problem.dimensions, problem.neighbours);
        optimisation::block_cholesky again;
        again.analyse(problem.dimensions, problem.neighbours);
        std::vector<int> seen(static_cast<size_t>(blocks), 0);
        for (int position = 0; position < blocks; ++position) {
            REQUIRE(factor.block_at(position) == again.block_at(position));
            REQUIRE(factor.position_of(factor.block_at(position)) == position);
            ++seen[static_cast<size_t>(factor.block_at(position))];
            for (size_t entry = factor.column_entries_begin(position); entry < factor.column_entries_end(position); ++entry) {
                REQUIRE(factor.entry_row(entry) > position);
                size_t offset = 0;
                REQUIRE(factor.find_offset(factor.entry_row(entry), position, offset));
                REQUIRE(offset == factor.entry_offset(entry));
            }
        }
        for (const int count : seen) {
            REQUIRE(count == 1);
        }
        REQUIRE(compare_with_dense(problem, factor, rng) < 1e-9);
    }

    {
        // Every pair of blocks sharing a nonzero block has one in the factor, and a block outside the pattern is not found.
        const block_problem problem = make_problem(rng, { 2, 3, 4 }, { { 0, 1 } });
        optimisation::block_cholesky factor;
        factor.analyse(problem.dimensions, problem.neighbours);
        const int first = factor.position_of(0);
        const int second = factor.position_of(1);
        const int third = factor.position_of(2);
        size_t offset = 0;
        REQUIRE(factor.find_offset(std::max(first, second), std::min(first, second), offset));
        REQUIRE(!factor.find_offset(std::max(first, third), std::min(first, third), offset));
        REQUIRE(!factor.find_offset(std::max(second, third), std::min(second, third), offset));
    }

    {
        // A matrix that is not positive definite fails the factorisation, as the dense one does.
        const block_problem base = make_problem(rng, { 3, 3 }, { { 0, 1 } });
        block_problem indefinite = base;
        for (size_t i = 0; i < indefinite.size; ++i) {
            indefinite.dense[(i * indefinite.size) + i] -= 1000.0;
        }
        optimisation::block_cholesky factor;
        factor.analyse(indefinite.dimensions, indefinite.neighbours);
        assemble(indefinite, factor);
        REQUIRE(!factor.factorise());
    }

    {
        // An empty system factorises and solves trivially.
        optimisation::block_cholesky factor;
        factor.analyse({}, {});
        REQUIRE(factor.block_count() == 0);
        REQUIRE(factor.factorise());
        REQUIRE(factor.solve(nullptr, nullptr));
    }

    return EXIT_SUCCESS;
}
