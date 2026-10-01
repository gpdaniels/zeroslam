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
#include <vector>

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

    {
        // Clusters of 3x3 detections 3 px apart: square size 2 keeps only the 165 centres and size 1 keeps all 1485, so no size lands inside [200, 800].
        std::vector<feature::point> features_detected_sorted;
        for (int cluster = 0; cluster < 165; ++cluster) {
            feature::point centre = {};
            centre.x = static_cast<float>(20 + 40 * (cluster % 15));
            centre.y = static_cast<float>(20 + 40 * (cluster / 15));
            centre.response = 100.0f;
            features_detected_sorted.push_back(centre);
        }
        for (int cluster = 0; cluster < 165; ++cluster) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if ((dx != 0) || (dy != 0)) {
                        feature::point neighbour = features_detected_sorted[static_cast<size_t>(cluster)];
                        neighbour.x += static_cast<float>(3 * dx);
                        neighbour.y += static_cast<float>(3 * dy);
                        neighbour.response = 10.0f;
                        features_detected_sorted.push_back(neighbour);
                    }
                }
            }
        }
        const int min_features = 200;
        const int max_features = 800;
        const int guard = 64;
        std::vector<feature::point> features_distributed(static_cast<size_t>(max_features + guard), feature::point{ -1.0f, -1.0f, -1.0f, -1.0f, -1 });
        const int features = feature::distributor::square_covering::distribute(features_detected_sorted.data(), static_cast<int>(features_detected_sorted.size()), 640, 480, min_features, max_features, features_distributed.data());
        REQUIRE(features == max_features);
        for (int i = 0; i < 165; ++i) {
            REQUIRE(features_distributed[static_cast<size_t>(i)].response == 100.0f);
        }
        for (int i = max_features; i < max_features + guard; ++i) {
            REQUIRE(features_distributed[static_cast<size_t>(i)].response == -1.0f);
        }
    }

    {
        // Forty clusters with a feature on every pixel: the smallest size the estimate allows, 2, keeps 16 per cluster, too few, while size 1 keeps every third pixel.
        std::vector<feature::point> features_detected_sorted;
        for (int cluster = 0; cluster < 40; ++cluster) {
            const int centre_x = 40 + 80 * (cluster % 8);
            const int centre_y = 48 + 96 * (cluster / 8);
            for (int y = centre_y - 10; y <= centre_y + 10; ++y) {
                for (int x = centre_x - 10; x <= centre_x + 10; ++x) {
                    feature::point detection = {};
                    detection.x = static_cast<float>(x);
                    detection.y = static_cast<float>(y);
                    detection.response = static_cast<float>(100000 - static_cast<int>(features_detected_sorted.size()));
                    features_detected_sorted.push_back(detection);
                }
            }
        }
        const int min_features = 700;
        const int max_features = 2000;
        std::vector<feature::point> features_distributed(static_cast<size_t>(max_features));
        const int features = feature::distributor::square_covering::distribute(features_detected_sorted.data(), static_cast<int>(features_detected_sorted.size()), 640, 480, min_features, max_features, features_distributed.data());
        REQUIRE(features == 40 * 49);
        for (int i = 0; i < features; ++i) {
            for (int j = i + 1; j < features; ++j) {
                const float dx = std::abs(features_distributed[static_cast<size_t>(i)].x - features_distributed[static_cast<size_t>(j)].x);
                const float dy = std::abs(features_distributed[static_cast<size_t>(i)].y - features_distributed[static_cast<size_t>(j)].y);
                REQUIRE((dx >= 3.0f) || (dy >= 3.0f));
            }
        }
    }

    {
        std::vector<feature::point> features_detected_sorted;
        for (int i = 0; i < 400; ++i) {
            features_detected_sorted.push_back(feature::point{ static_cast<float>(i % 20), static_cast<float>(i / 20), static_cast<float>(400 - i), 0.0f, 0 });
        }
        std::vector<feature::point> features_distributed(features_detected_sorted.size() + 1, feature::point{ -1.0f, -1.0f, -1.0f, -1.0f, -1 });

        // A budget of one keeps the strongest feature and a budget of zero writes nothing.
        REQUIRE(feature::distributor::square_covering::distribute(features_detected_sorted.data(), 400, 20, 20, 1, 1, features_distributed.data()) == 1);
        REQUIRE(features_distributed[0].response == 400.0f);
        REQUIRE(features_distributed[1].response == -1.0f);
        features_distributed[0].response = -1.0f;
        REQUIRE(feature::distributor::square_covering::distribute(features_detected_sorted.data(), 400, 20, 20, 0, 0, features_distributed.data()) == 0);
        REQUIRE(features_distributed[0].response == -1.0f);

        // A feature on every pixel defeats the square size estimate, the densest covering keeps every third pixel in each direction.
        const int features = feature::distributor::square_covering::distribute(features_detected_sorted.data(), 400, 20, 20, 100, 300, features_distributed.data());
        REQUIRE(features == 49);
        for (int i = 0; i < features; ++i) {
            for (int j = i + 1; j < features; ++j) {
                const float dx = std::abs(features_distributed[static_cast<size_t>(i)].x - features_distributed[static_cast<size_t>(j)].x);
                const float dy = std::abs(features_distributed[static_cast<size_t>(i)].y - features_distributed[static_cast<size_t>(j)].y);
                REQUIRE((dx >= 3.0f) || (dy >= 3.0f));
            }
        }
        REQUIRE(features_distributed[static_cast<size_t>(features)].response == -1.0f);
    }

    return EXIT_SUCCESS;
}
