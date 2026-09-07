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

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-8) {
    if (std::isnan(lhs) && std::isnan(rhs))
        return true;
    if (std::isnan(lhs) != std::isnan(rhs))
        return false;
    if (std::isinf(lhs) != std::isinf(rhs))
        return false;
    if (std::signbit(lhs + epsilon) != std::signbit(rhs + epsilon))
        return false;
    if (std::isinf(lhs) && std::isinf(rhs))
        return true;
    return (std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon);
}

static inline bool is_value_approx(float lhs, float rhs, double epsilon = 1e-8) {
    return is_value_approx(static_cast<double>(lhs), static_cast<double>(rhs), epsilon);
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        const feature::point features_detected_sorted_by_response[10] = {
            { 1, 21, 89, 0.2f, 0 },
            { 42, 42, 65, 0.6f, 0 },
            { 26, 1, 35, 0.3f, 0 },
            { 21, 15, 21, 0.4f, 0 },
            { 48, 72, 10, 0.7f, 0 },
            { 72, 2, 3, 0.9f, 0 },
            { 7, 3, 2, 0.1f, 0 },
            { 63, 16, 2, 0.8f, 0 },
            { 79, 68, 2, 1.0f, 0 },
            { 24, 29, 1, 0.5f, 0 }
        };
        const int features_detected_sorted_size = 10;
        const int max_width = 80;
        const int max_height = 80;
        feature::point features_distributed[20] = {};
        const int max_sizes[10] = {
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            20
        };
        const int distributed_sizes[10] = {
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            10
        };
        const int distributed_indexes[10][10] = {
            { 0, 8, 0, 0, 0, 0, 0, 0, 0, 0 },
            { 0, 1, 5, 0, 0, 0, 0, 0, 0, 0 },
            { 0, 1, 5, 8, 0, 0, 0, 0, 0, 0 },
            { 0, 1, 4, 5, 8, 0, 0, 0, 0, 0 },
            { 0, 1, 2, 4, 5, 8, 0, 0, 0, 0 },
            { 0, 1, 2, 4, 5, 8, 9, 0, 0, 0 },
            { 0, 1, 2, 4, 5, 6, 8, 9, 0, 0 },
            { 0, 1, 2, 3, 4, 5, 6, 7, 8, 0 },
            { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 },
            { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }
        };
        for (int i = 0; i < 10; ++i) {
            const int min_features = max_sizes[i];
            const int max_features = max_sizes[i];
            const int features = feature::distributor::square_covering::distribute(&features_detected_sorted_by_response[0], features_detected_sorted_size, max_width, max_height, min_features, max_features, &features_distributed[0]);
            REQUIRE(features == distributed_sizes[i]);
            for (int j = 0; j < features; ++j) {
                const int distributed_index = distributed_indexes[i][j];
                REQUIRE(features_distributed[j].x == features_detected_sorted_by_response[distributed_index].x);
                REQUIRE(features_distributed[j].y == features_detected_sorted_by_response[distributed_index].y);
                REQUIRE(is_value_approx(features_distributed[j].response, features_detected_sorted_by_response[distributed_index].response));
                REQUIRE(is_value_approx(features_distributed[j].angle, features_detected_sorted_by_response[distributed_index].angle));
            }
        }
    }

    return EXIT_SUCCESS;
}
