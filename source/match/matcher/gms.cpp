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

#include "match/matcher/gms.hpp"

#include "core/arena.hpp"
#include "core/arena_allocator.hpp"
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace match::matcher {
    namespace {
        constexpr static const int rotation_count = 8;
        constexpr static const int scale_count = 5;

        // The cell itself, then its eight neighbours in order around it, so one step along that ring turns the neighbourhood by 45 degrees.
        constexpr static const int neighbour_x[9] = { 0, -1, 0, 1, 1, 1, 0, -1, -1 };
        constexpr static const int neighbour_y[9] = { 0, -1, -1, -1, 0, 1, 1, 1, 0 };

        // The right to left resolution ratios 1, 1/2, 1/sqrt(2), sqrt(2) and 2 of the reference implementation. A ratio below one refines
        // the left grid rather than coarsening the right, as a right grid of a few cells passes most false matches at small match counts.
        constexpr static const float scale_factors[scale_count] = { 1.0f, 2.0f, 1.41421356f, 1.41421356f, 2.0f };
        constexpr static const bool scale_refines_lhs[scale_count] = { false, true, true, false, false };

        // The right-hand neighbour paired with a left-hand neighbour when the neighbourhood is turned by rotation steps of 45 degrees.
        int turned(const int neighbour, const int rotation) {
            return (neighbour == 0) ? 0 : (1 + ((neighbour - 1 + rotation) % rotation_count));
        }
    }

    size_t gms::filter(const feature::point* const lhs_points, const float lhs_width, const float lhs_height, const feature::point* const rhs_points, const float rhs_width, const float rhs_height, match::pair* const matches, const size_t matches_size) {
        return gms::filter(lhs_points, lhs_width, lhs_height, rhs_points, rhs_width, rhs_height, matches, matches_size, options());
    }

    size_t gms::filter(const feature::point* const lhs_points, const float lhs_width, const float lhs_height, const feature::point* const rhs_points, const float rhs_width, const float rhs_height, match::pair* const matches, const size_t matches_size, const options& settings) {
        if (matches_size == 0) {
            return 0;
        }
        core::arena::scope scratch;
        int grid = settings.grid_size;
        if (grid < 1) {
            grid = static_cast<int>(math::sqrt(static_cast<float>(matches_size) / math::max(settings.grid_matches_per_cell, 1.0f)));
            grid = math::max(settings.grid_size_minimum, math::min(settings.grid_size_maximum, grid));
        }
        grid = math::max(grid, 1);
        const int scales = settings.scale ? scale_count : 1;
        const int rotations = settings.rotation ? rotation_count : 1;
        int lhs_grids[scale_count] = {};
        int rhs_grids[scale_count] = {};
        size_t lhs_cells_maximum = 1;
        size_t support_maximum = 1;
        for (int scale = 0; scale < scales; ++scale) {
            const int refined = math::max(1, static_cast<int>(static_cast<float>(grid) * scale_factors[scale]));
            lhs_grids[scale] = scale_refines_lhs[scale] ? refined : grid;
            rhs_grids[scale] = scale_refines_lhs[scale] ? grid : refined;
            const size_t lhs_cells = static_cast<size_t>(lhs_grids[scale] * lhs_grids[scale]);
            lhs_cells_maximum = math::max(lhs_cells_maximum, lhs_cells);
            support_maximum = math::max(support_maximum, lhs_cells * static_cast<size_t>(rhs_grids[scale] * rhs_grids[scale]));
        }

        struct cell_set final {
            int cells[4];
            int count;
        };

        const auto cells_of = [&settings](const float x, const float y, const float width, const float height, const int cells_across, const float shift_x, const float shift_y, cell_set& out) {
            const float cell_width = width / static_cast<float>(cells_across);
            const float cell_height = height / static_cast<float>(cells_across);
            const float gx = (x / cell_width) + shift_x;
            const float gy = (y / cell_height) + shift_y;
            const int cx = math::max(0, math::min(cells_across - 1, static_cast<int>(math::floor(gx))));
            const int cy = math::max(0, math::min(cells_across - 1, static_cast<int>(math::floor(gy))));
            out.count = 0;
            out.cells[out.count++] = (cy * cells_across) + cx;
            if (settings.margin > 0.0f) {
                const float fx = gx - static_cast<float>(cx);
                const float fy = gy - static_cast<float>(cy);
                const int nx = (fx < settings.margin) ? (cx - 1) : ((fx > 1.0f - settings.margin) ? (cx + 1) : cx);
                const int ny = (fy < settings.margin) ? (cy - 1) : ((fy > 1.0f - settings.margin) ? (cy + 1) : cy);
                if ((nx != cx) && (nx >= 0) && (nx < cells_across)) {
                    out.cells[out.count++] = (cy * cells_across) + nx;
                }
                if ((ny != cy) && (ny >= 0) && (ny < cells_across)) {
                    out.cells[out.count++] = (ny * cells_across) + cx;
                }
                if ((nx != cx) && (ny != cy) && (nx >= 0) && (nx < cells_across) && (ny >= 0) && (ny < cells_across)) {
                    out.cells[out.count++] = (ny * cells_across) + nx;
                }
            }
        };
        std::vector<cell_set, core::arena_allocator<cell_set>> lhs_sets(matches_size);
        std::vector<cell_set, core::arena_allocator<cell_set>> rhs_sets(matches_size);
        // Zeroed once here, afterwards only the entries a hypothesis incremented are cleared again.
        std::vector<int, core::arena_allocator<int>> support(support_maximum);
        std::vector<int, core::arena_allocator<int>> lhs_count(lhs_cells_maximum);
        std::vector<int, core::arena_allocator<int>> best_count(lhs_cells_maximum);
        std::vector<int, core::arena_allocator<int>> best_rhs(lhs_cells_maximum);
        std::vector<unsigned char, core::arena_allocator<unsigned char>> cell_passes(lhs_cells_maximum);
        // Bit (scale * rotations + rotation) of a match is set once any pass marks it an inlier under that hypothesis.
        std::vector<unsigned long long int, core::arena_allocator<unsigned long long int>> hypotheses(matches_size, 0ull);
        size_t inliers[scale_count * rotation_count] = {};
        const float shifts[4][2] = { { 0.0f, 0.0f }, { 0.5f, 0.0f }, { 0.0f, 0.5f }, { 0.5f, 0.5f } };
        for (int pass = 0; pass < 4; ++pass) {
            for (int scale = 0; scale < scales; ++scale) {
                const int lhs_grid = lhs_grids[scale];
                const int rhs_grid = rhs_grids[scale];
                const size_t lhs_cells = static_cast<size_t>(lhs_grid * lhs_grid);
                const size_t rhs_cells = static_cast<size_t>(rhs_grid * rhs_grid);
                for (size_t c = 0; c < lhs_cells; ++c) {
                    lhs_count[c] = 0;
                    best_count[c] = 0;
                    best_rhs[c] = -1;
                }
                for (size_t m = 0; m < matches_size; ++m) {
                    const feature::point& lhs = lhs_points[matches[m].lhs_index];
                    const feature::point& rhs = rhs_points[matches[m].rhs_index];
                    cells_of(lhs.x, lhs.y, lhs_width, lhs_height, lhs_grid, shifts[pass][0], shifts[pass][1], lhs_sets[m]);
                    cells_of(rhs.x, rhs.y, rhs_width, rhs_height, rhs_grid, shifts[pass][0], shifts[pass][1], rhs_sets[m]);
                    ++lhs_count[static_cast<size_t>(lhs_sets[m].cells[0])];
                    for (int a = 0; a < lhs_sets[m].count; ++a) {
                        const int lhs_cell = lhs_sets[m].cells[a];
                        for (int b = 0; b < rhs_sets[m].count; ++b) {
                            const int rhs_cell = rhs_sets[m].cells[b];
                            const int count = ++support[(static_cast<size_t>(lhs_cell) * rhs_cells) + static_cast<size_t>(rhs_cell)];
                            // The running maximum of each row, a tie going to the lower right cell as a scan of the finished row would.
                            if ((count > best_count[static_cast<size_t>(lhs_cell)]) || ((count == best_count[static_cast<size_t>(lhs_cell)]) && (rhs_cell < best_rhs[static_cast<size_t>(lhs_cell)]))) {
                                best_count[static_cast<size_t>(lhs_cell)] = count;
                                best_rhs[static_cast<size_t>(lhs_cell)] = rhs_cell;
                            }
                        }
                    }
                }
                for (int rotation = 0; rotation < rotations; ++rotation) {
                    for (size_t i = 0; i < lhs_cells; ++i) {
                        cell_passes[i] = static_cast<unsigned char>(0);
                        if (best_rhs[i] < 0) {
                            continue;
                        }
                        const int ix = static_cast<int>(i) % lhs_grid;
                        const int iy = static_cast<int>(i) / lhs_grid;
                        const int jx = best_rhs[i] % rhs_grid;
                        const int jy = best_rhs[i] / rhs_grid;
                        int score = 0;
                        int matches_in_neighbourhood = 0;
                        int neighbourhood_cells = 0;
                        for (int neighbour = 0; neighbour < 9; ++neighbour) {
                            const int ax = ix + neighbour_x[neighbour];
                            const int ay = iy + neighbour_y[neighbour];
                            const int bx = jx + neighbour_x[turned(neighbour, rotation)];
                            const int by = jy + neighbour_y[turned(neighbour, rotation)];
                            if ((ax < 0) || (ax >= lhs_grid) || (ay < 0) || (ay >= lhs_grid) || (bx < 0) || (bx >= rhs_grid) || (by < 0) || (by >= rhs_grid)) {
                                continue;
                            }
                            const size_t a = static_cast<size_t>((ay * lhs_grid) + ax);
                            const size_t b = static_cast<size_t>((by * rhs_grid) + bx);
                            score += support[(a * rhs_cells) + b];
                            matches_in_neighbourhood += lhs_count[a];
                            ++neighbourhood_cells;
                        }
                        const float threshold = settings.alpha * math::sqrt(static_cast<float>(matches_in_neighbourhood) / static_cast<float>(neighbourhood_cells));
                        if (static_cast<float>(score) > threshold) {
                            cell_passes[i] = static_cast<unsigned char>(1);
                        }
                    }
                    const size_t hypothesis = static_cast<size_t>((scale * rotations) + rotation);
                    const unsigned long long int bit = 1ull << hypothesis;
                    for (size_t m = 0; m < matches_size; ++m) {
                        const size_t cell = static_cast<size_t>(lhs_sets[m].cells[0]);
                        if ((cell_passes[cell] != 0) && (rhs_sets[m].cells[0] == best_rhs[cell]) && ((hypotheses[m] & bit) == 0)) {
                            hypotheses[m] |= bit;
                            ++inliers[hypothesis];
                        }
                    }
                }
                for (size_t m = 0; m < matches_size; ++m) {
                    for (int a = 0; a < lhs_sets[m].count; ++a) {
                        for (int b = 0; b < rhs_sets[m].count; ++b) {
                            support[(static_cast<size_t>(lhs_sets[m].cells[a]) * rhs_cells) + static_cast<size_t>(rhs_sets[m].cells[b])] = 0;
                        }
                    }
                }
            }
        }
        size_t chosen = 0;
        for (size_t hypothesis = 1; hypothesis < static_cast<size_t>(scales * rotations); ++hypothesis) {
            if (inliers[hypothesis] > inliers[chosen]) {
                chosen = hypothesis;
            }
        }
        const unsigned long long int chosen_bit = 1ull << chosen;
        size_t kept = 0;
        for (size_t m = 0; m < matches_size; ++m) {
            if ((hypotheses[m] & chosen_bit) != 0) {
                matches[kept++] = matches[m];
            }
        }
        return kept;
    }
}
