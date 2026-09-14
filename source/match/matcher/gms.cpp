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
        const size_t cells = static_cast<size_t>(grid * grid);
        std::vector<unsigned char, core::arena_allocator<unsigned char>> inlier(matches_size, static_cast<unsigned char>(0));

        struct cell_set final {
            int cells[4];
            int count;
        };

        const auto cells_of = [&](const float x, const float y, const float width, const float height, const float shift_x, const float shift_y, cell_set& out) {
            const float cell_width = width / static_cast<float>(grid);
            const float cell_height = height / static_cast<float>(grid);
            const float gx = (x / cell_width) + shift_x;
            const float gy = (y / cell_height) + shift_y;
            const int cx = math::max(0, math::min(grid - 1, static_cast<int>(math::floor(gx))));
            const int cy = math::max(0, math::min(grid - 1, static_cast<int>(math::floor(gy))));
            out.count = 0;
            out.cells[out.count++] = (cy * grid) + cx;
            if (settings.margin > 0.0f) {
                const float fx = gx - static_cast<float>(cx);
                const float fy = gy - static_cast<float>(cy);
                const int nx = (fx < settings.margin) ? (cx - 1) : ((fx > 1.0f - settings.margin) ? (cx + 1) : cx);
                const int ny = (fy < settings.margin) ? (cy - 1) : ((fy > 1.0f - settings.margin) ? (cy + 1) : cy);
                if ((nx != cx) && (nx >= 0) && (nx < grid)) {
                    out.cells[out.count++] = (cy * grid) + nx;
                }
                if ((ny != cy) && (ny >= 0) && (ny < grid)) {
                    out.cells[out.count++] = (ny * grid) + cx;
                }
                if ((nx != cx) && (ny != cy) && (nx >= 0) && (nx < grid) && (ny >= 0) && (ny < grid)) {
                    out.cells[out.count++] = (ny * grid) + nx;
                }
            }
        };
        std::vector<int, core::arena_allocator<int>> lhs_cell(matches_size);
        std::vector<int, core::arena_allocator<int>> rhs_cell(matches_size);
        std::vector<int, core::arena_allocator<int>> support(cells * cells);
        std::vector<int, core::arena_allocator<int>> lhs_count(cells);
        std::vector<int, core::arena_allocator<int>> best_rhs(cells);
        const float shifts[4][2] = { { 0.0f, 0.0f }, { 0.5f, 0.0f }, { 0.0f, 0.5f }, { 0.5f, 0.5f } };
        for (int pass = 0; pass < 4; ++pass) {
            for (size_t c = 0; c < support.size(); ++c) {
                support[c] = 0;
            }
            for (size_t c = 0; c < cells; ++c) {
                lhs_count[c] = 0;
                best_rhs[c] = -1;
            }
            for (size_t m = 0; m < matches_size; ++m) {
                const feature::point& lhs = lhs_points[matches[m].lhs_index];
                const feature::point& rhs = rhs_points[matches[m].rhs_index];
                cell_set lhs_set;
                cell_set rhs_set;
                cells_of(lhs.x, lhs.y, lhs_width, lhs_height, shifts[pass][0], shifts[pass][1], lhs_set);
                cells_of(rhs.x, rhs.y, rhs_width, rhs_height, shifts[pass][0], shifts[pass][1], rhs_set);
                lhs_cell[m] = lhs_set.cells[0];
                rhs_cell[m] = rhs_set.cells[0];
                for (int a = 0; a < lhs_set.count; ++a) {
                    for (int b = 0; b < rhs_set.count; ++b) {
                        ++support[(static_cast<size_t>(lhs_set.cells[a]) * cells) + static_cast<size_t>(rhs_set.cells[b])];
                    }
                }
                ++lhs_count[static_cast<size_t>(lhs_set.cells[0])];
            }
            for (size_t i = 0; i < cells; ++i) {
                int best = 0;
                for (size_t j = 0; j < cells; ++j) {
                    const int count = support[(i * cells) + j];
                    if (count > best) {
                        best = count;
                        best_rhs[i] = static_cast<int>(j);
                    }
                }
            }
            for (size_t i = 0; i < cells; ++i) {
                if (best_rhs[i] < 0) {
                    continue;
                }
                const int ix = static_cast<int>(i) % grid;
                const int iy = static_cast<int>(i) / grid;
                const int jx = best_rhs[i] % grid;
                const int jy = best_rhs[i] / grid;
                int score = 0;
                int matches_in_neighbourhood = 0;
                int neighbourhood_cells = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int ax = ix + dx;
                        const int ay = iy + dy;
                        const int bx = jx + dx;
                        const int by = jy + dy;
                        if ((ax < 0) || (ax >= grid) || (ay < 0) || (ay >= grid) || (bx < 0) || (bx >= grid) || (by < 0) || (by >= grid)) {
                            continue;
                        }
                        const size_t a = static_cast<size_t>((ay * grid) + ax);
                        const size_t b = static_cast<size_t>((by * grid) + bx);
                        score += support[(a * cells) + b];
                        matches_in_neighbourhood += lhs_count[a];
                        ++neighbourhood_cells;
                    }
                }
                if (neighbourhood_cells == 0) {
                    continue;
                }
                const float threshold = settings.alpha * math::sqrt(static_cast<float>(matches_in_neighbourhood) / static_cast<float>(neighbourhood_cells));
                if (static_cast<float>(score) <= threshold) {
                    continue;
                }
                for (size_t m = 0; m < matches_size; ++m) {
                    if ((lhs_cell[m] == static_cast<int>(i)) && (rhs_cell[m] == best_rhs[i])) {
                        inlier[m] = static_cast<unsigned char>(1);
                    }
                }
            }
        }
        size_t kept = 0;
        for (size_t m = 0; m < matches_size; ++m) {
            if (inlier[m]) {
                matches[kept++] = matches[m];
            }
        }
        return kept;
    }
}
