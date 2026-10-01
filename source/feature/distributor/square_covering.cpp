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

#include "feature/distributor/square_covering.hpp"

#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::distributor {
    int square_covering::distribute(
        const point* __restrict const features_detected_sorted,
        const int features_detected_sorted_size,
        const int max_width,
        const int max_height,
        const int min_features,
        const int max_features,
        point* __restrict features_distributed
    ) {
        constexpr static const int square_covering_radius = 2;
        if (max_features >= features_detected_sorted_size) {
            for (int i = 0; i < features_detected_sorted_size; ++i) {
                features_distributed[i] = features_detected_sorted[i];
            }
            return features_detected_sorted_size;
        }
        if (max_features <= 1) {
            const int count = math::max(0, max_features);
            for (int i = 0; i < count; ++i) {
                features_distributed[i] = features_detected_sorted[i];
            }
            return count;
        }
        const long long int delta =
            4ll * static_cast<long long int>(max_width) +
            4ll * static_cast<long long int>(max_features) +
            4ll * static_cast<long long int>(max_height) * static_cast<long long int>(max_features) +
            1ll * static_cast<long long int>(max_height) * static_cast<long long int>(max_height) +
            1ll * static_cast<long long int>(max_width) * static_cast<long long int>(max_width) -
            2ll * static_cast<long long int>(max_height) * static_cast<long long int>(max_width) +
            4ll * static_cast<long long int>(max_height) * static_cast<long long int>(max_width) * static_cast<long long int>(max_features);
        const int delta_sqrt = static_cast<int>(math::sqrt(static_cast<double>(delta)));
        const int numerator = delta_sqrt - (max_width + max_height + 2 * max_features);
        const int denominator = 2 * (max_features - 1);
        int square_size_max = numerator / denominator;
        int square_size_min = math::max(1, static_cast<int>(math::sqrt(static_cast<double>(features_detected_sorted_size) / static_cast<double>(2 * max_features))));
        if (square_size_max <= square_size_min) {
            // Note: The size estimate fails when nearly every pixel may hold a feature, so search every size, the largest covers the image with one square.
            square_size_min = 1;
            square_size_max = math::max(max_width, max_height) + 1;
        }
        const size_t covered_squares_size = static_cast<size_t>(max_width + 1) * static_cast<size_t>(max_height + 1);
        std::vector<unsigned char> covered_squares(covered_squares_size);
        std::vector<int> indexes(static_cast<size_t>(features_detected_sorted_size));
        std::vector<int> indexes_best(static_cast<size_t>(features_detected_sorted_size));
        int indexes_size = 0;
        int indexes_best_size = 0;
        bool best_too_many = false;
        int square_size_previous = 0;
        while ((indexes_size < min_features) || (indexes_size > max_features)) {
            const int square_size = (square_size_max + square_size_min) / 2;
            if (square_size == square_size_previous) {
                // Note: The smallest size estimated can already be too sparse for clustered features, so search the sizes below it before settling for too few.
                if (!best_too_many && (square_size_min > 1)) {
                    square_size_max = square_size_min;
                    square_size_min = 1;
                    continue;
                }
                break;
            }
            square_size_previous = square_size;
            indexes_size = 0;
            const int grid_width = max_width / square_size;
            const int grid_height = max_height / square_size;
            const int grid_stride = grid_width + 1;
            for (int i = 0; i < (grid_width + 1) * (grid_height + 1); ++i) {
                covered_squares[static_cast<size_t>(i)] = 0;
            }
            for (int i = 0; i < features_detected_sorted_size; ++i) {
                const int cell_x = static_cast<int>(features_detected_sorted[i].x) / square_size;
                const int cell_y = static_cast<int>(features_detected_sorted[i].y) / square_size;
                if (covered_squares[static_cast<size_t>(cell_y * grid_stride + cell_x)] == 0) {
                    indexes[static_cast<size_t>(indexes_size++)] = i;
                    const int cell_x_min = math::max(0, cell_x - square_covering_radius);
                    const int cell_x_max = math::min(grid_width, cell_x + square_covering_radius);
                    const int cell_y_min = math::max(0, cell_y - square_covering_radius);
                    const int cell_y_max = math::min(grid_height, cell_y + square_covering_radius);
                    // Mark all squares within as covered.
                    for (int y = cell_y_min; y <= cell_y_max; ++y) {
                        for (int x = cell_x_min; x <= cell_x_max; ++x) {
                            covered_squares[static_cast<size_t>(y * grid_stride + x)] = 1;
                        }
                    }
                }
            }
            const bool too_few = (indexes_size < min_features);
            const bool too_many = (indexes_size > max_features);
            if (too_few) {
                square_size_max = square_size;
            }
            else if (too_many) {
                square_size_min = square_size;
            }
            // Note: The count can jump over the range between two sizes, so keep the sparsest covering with too many features, truncated to its strongest features it beats every covering with too few.
            if (!too_few || !best_too_many) {
                indexes.swap(indexes_best);
                indexes_best_size = indexes_size;
                best_too_many = too_many;
            }
        }
        const int features_distributed_size = math::min(indexes_best_size, max_features);
        for (int i = 0; i < features_distributed_size; ++i) {
            features_distributed[i] = features_detected_sorted[indexes_best[static_cast<size_t>(i)]];
        }
        return features_distributed_size;
    }
}
