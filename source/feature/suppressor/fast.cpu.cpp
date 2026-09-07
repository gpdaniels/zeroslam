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

#include "feature/point.hpp"

namespace feature::suppressor {
    size_t suppress_cpu(const point* __restrict const features, const size_t features_count, const size_t max_row, point* __restrict const features_suppressed);

    size_t suppress_cpu(
        const point* __restrict const features,
        const size_t features_count,
        const size_t max_row,
        point* __restrict const features_suppressed
    ) {
        if (features_count == 0) {
            return 0;
        }
        size_t* const index_rows = new size_t[max_row + 1];
        for (size_t i = 0; i < max_row + 1; ++i) {
            index_rows[i] = static_cast<size_t>(-1);
        }
        size_t index_row = static_cast<size_t>(-1);
        for (size_t i = 0; i < features_count; ++i) {
            if (static_cast<size_t>(features[i].y) != index_row) {
                if (static_cast<size_t>(features[i].y) > max_row) {
                    delete[] index_rows;
                    return 0;
                }
                index_rows[static_cast<size_t>(features[i].y)] = i;
                index_row = static_cast<size_t>(features[i].y);
            }
        }
        size_t index_row_above = 0;
        size_t index_row_below = 0;
        size_t suppressed_count = 0;
        for (size_t i = 0; i < features_count; ++i) {
            if (i > 0) {
                if ((features[i - 1].x == features[i].x - 1) && (features[i - 1].y == features[i].y) && (features[i - 1].response > features[i].response)) {
                    continue;
                }
            }
            if (i < (features_count - 1)) {
                if ((features[i + 1].x == features[i].x + 1) && (features[i + 1].y == features[i].y) && (features[i + 1].response > features[i].response)) {
                    continue;
                }
            }
            if ((features[i].y != 0) && (index_rows[static_cast<size_t>(features[i].y - 1)] != static_cast<size_t>(-1))) {
                if (features[index_row_above].y < (features[i].y - 1)) {
                    index_row_above = index_rows[static_cast<size_t>(features[i].y - 1)];
                }
                while ((features[index_row_above].y < features[i].y) && (features[index_row_above].x < (features[i].x - 1))) {
                    ++index_row_above;
                }
                bool skip = false;
                for (size_t j = index_row_above; (features[j].y < features[i].y) && (features[j].x <= (features[i].x + 1)); ++j) {
                    if (((features[j].x == (features[i].x - 1)) || (features[j].x == (features[i].x + 0)) || (features[j].x == (features[i].x + 1))) && (features[j].response > features[i].response)) {
                        skip = true;
                        break;
                    }
                }
                if (skip) {
                    continue;
                }
            }
            if ((static_cast<size_t>(features[i].y) != max_row) && (index_rows[static_cast<size_t>(features[i].y + 1)] != static_cast<size_t>(-1)) && (index_row_below < features_count)) {
                if (features[index_row_below].y < (features[i].y + 1)) {
                    index_row_below = index_rows[static_cast<size_t>(features[i].y + 1)];
                }
                while ((index_row_below < features_count) && (features[index_row_below].y == (features[i].y + 1)) && (features[index_row_below].x < (features[i].x - 1))) {
                    ++index_row_below;
                }
                bool skip = false;
                for (size_t j = index_row_below; (j < features_count) && (features[j].y == (features[i].y + 1)) && (features[j].x <= (features[i].x + 1)); ++j) {
                    if (((features[j].x == (features[i].x - 1)) || (features[j].x == (features[i].x + 0)) || (features[j].x == (features[i].x + 1))) && (features[j].response > features[i].response)) {
                        skip = true;
                        break;
                    }
                }
                if (skip) {
                    continue;
                }
            }
            features_suppressed[suppressed_count++] = features[i];
        }
        delete[] index_rows;
        return suppressed_count;
    }
}
