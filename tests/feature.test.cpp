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

#include "feature.hpp"

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

template <typename array_type>
static inline bool are_values_approx(const array_type& lhs, const array_type& rhs, unsigned long long int length, double epsilon = 1e-8) {
    for (size_t index = 0; index < length; ++index) {
        if (!is_value_approx(lhs[index], rhs[index], epsilon)) {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        feature::point point{ 123, 456, 0.123f, 0.456f };
        REQUIRE(point.x == 123);
        REQUIRE(point.y == 456);
        REQUIRE(point.response == 0.123f);
        REQUIRE(point.angle == 0.456f);
    }
    {
        feature::descriptor descriptor{
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31
        };
        for (size_t i = 0; i < 32; ++i) {
            REQUIRE(descriptor.data[i] == i);
        }
    }
    {
        feature::match match{ 1, 2, 0.123f };
        REQUIRE(match.lhs_index == 1);
        REQUIRE(match.rhs_index == 2);
        REQUIRE(match.score == 0.123f);
    }

    {
        constexpr static const size_t data_width = 128;
        constexpr static const size_t data_height = 64;
        unsigned char data[data_height][data_width] = {};
        for (size_t y = 0; y < data_height; ++y) {
            for (size_t x = 0; x < data_width; ++x) {
                data[y][x] = static_cast<unsigned char>(((x > data_width / 4) && (x < (data_width * 3) / 4) && (y > data_height / 4) && (y < (data_height * 3) / 4)) * 255);
            }
        }
        feature::point features_opencv[24] = {
            { 33, 17, 0, 0 },
            { 34, 17, 0, 0 },
            { 35, 17, 0, 0 },
            { 93, 17, 0, 0 },
            { 94, 17, 0, 0 },
            { 95, 17, 0, 0 },
            { 33, 18, 0, 0 },
            { 34, 18, 0, 0 },
            { 94, 18, 0, 0 },
            { 95, 18, 0, 0 },
            { 33, 19, 0, 0 },
            { 95, 19, 0, 0 },
            { 33, 45, 0, 0 },
            { 95, 45, 0, 0 },
            { 33, 46, 0, 0 },
            { 34, 46, 0, 0 },
            { 94, 46, 0, 0 },
            { 95, 46, 0, 0 },
            { 33, 47, 0, 0 },
            { 34, 47, 0, 0 },
            { 35, 47, 0, 0 },
            { 93, 47, 0, 0 },
            { 94, 47, 0, 0 },
            { 95, 47, 0, 0 }
        };
        feature::point features[data_width * data_height];
        const size_t features_10_count = feature::detect(&data[0][0], data_width, data_height, data_width, 10, data_width * data_height, features);
        REQUIRE(features_10_count == 24);
        for (size_t i = 0; i < 24; ++i) {
            if (i >= features_10_count)
                continue;
            REQUIRE(features[i].x == features_opencv[i].x);
            REQUIRE(features[i].y == features_opencv[i].y);
        }
        const size_t features_255_count = feature::detect(&data[0][0], data_width, data_height, data_width, 255, data_width * data_height, features);
        REQUIRE(features_255_count == 0);
    }

    {
        feature::point features[25] = {
            { -2, -2, 0, 0 },
            { -1, -2, 0, 0 },
            { 0, -2, 0, 0 },
            { +1, -2, 0, 0 },
            { +2, -2, 0, 0 },
            { -2, -1, 0, 0 },
            { -1, -1, 0, 0 },
            { 0, -1, 0, 0 },
            { +1, -1, 0, 0 },
            { +2, -1, 0, 0 },
            { -2, 0, 0, 0 },
            { -1, 0, 0, 0 },
            { 0, 0, 0, 0 },
            { +1, 0, 0, 0 },
            { +2, 0, 0, 0 },
            { -2, +1, 0, 0 },
            { -1, +1, 0, 0 },
            { 0, +1, 0, 0 },
            { +1, +1, 0, 0 },
            { +2, +1, 0, 0 },
            { -2, +2, 0, 0 },
            { -1, +2, 0, 0 },
            { 0, +2, 0, 0 },
            { +1, +2, 0, 0 },
            { +2, +2, 0, 0 }
        };
        size_t prune_count = 25;
        feature::prune(&features[0], prune_count, [](const feature::point& feature) {
            return (feature.x < -1) || (feature.x > 1) || (feature.y < -1) || (feature.y > 1);
        });
        REQUIRE(prune_count == 9);
        const feature::point should_still_exist[9] = {
            { -1, -1, 0, 0 },
            { 0, -1, 0, 0 },
            { +1, -1, 0, 0 },
            { -1, 0, 0, 0 },
            { 0, 0, 0, 0 },
            { +1, 0, 0, 0 },
            { -1, +1, 0, 0 },
            { 0, +1, 0, 0 },
            { +1, +1, 0, 0 }
        };
        for (size_t i = 0; i < prune_count; ++i) {
            bool found = false;
            for (size_t j = 0; j < 9; ++j) {
                if (features[i].x == should_still_exist[j].x && features[i].y == should_still_exist[j].y) {
                    REQUIRE(found == false);
                    found = true;
                }
            }
            REQUIRE(found);
        }
    }

    {
        constexpr static const size_t data_width = 128;
        constexpr static const size_t data_height = 64;
        unsigned char data[data_height][data_width] = {};
        for (size_t y = 0; y < data_height; ++y) {
            for (size_t x = 0; x < data_width; ++x) {
                data[y][x] = static_cast<unsigned char>(x + y);
                if ((x > data_width / 4) && (x < (data_width * 3) / 4) && (y > data_height / 4) && (y < (data_height * 3) / 4)) {
                    data[y][x] = 0;
                }
            }
        }
        constexpr static const unsigned int features_count = 3;
        feature::point features[features_count] = {
            { 95, 19, 109, 0 },
            { 35, 47, 77, 0 },
            { 95, 47, 141, 0 }
        };
        for (size_t i = 0; i < features_count; ++i) {
            const float response = feature::score(&data[static_cast<size_t>(features[i].y)][static_cast<size_t>(features[i].x)], data_width);
            REQUIRE(is_value_approx(features[i].response, response));
        }
    }

    {
        {
            const feature::point features_detected_sorted_by_y[12] = {
                { 0, 0, 0, 0.0f },
                { 1, 0, 1, 0.1f },
                { 2, 0, 2, 0.2f },
                { 3, 0, 3, 0.3f },
                { 4, 0, 4, 0.4f },
                { 1, 1, 2, 0.5f },
                { 2, 1, 4, 0.6f },
                { 3, 1, 3, 0.7f },
                { 1, 2, 1, 0.8f },
                { 2, 2, 1, 0.9f },
                { 3, 2, 8, 1.0f },
                { 2, 20, 0, 1.1f },
            };
            feature::point features_suppressed[12] = {};
            const int features = feature::suppress(&features_detected_sorted_by_y[0], 12, 21, &features_suppressed[0]);
            REQUIRE(features == 3);
            REQUIRE(features_suppressed[0].x == features_detected_sorted_by_y[4].x);
            REQUIRE(features_suppressed[0].y == features_detected_sorted_by_y[4].y);
            REQUIRE(features_suppressed[0].response == features_detected_sorted_by_y[4].response);
            REQUIRE(features_suppressed[1].x == features_detected_sorted_by_y[10].x);
            REQUIRE(features_suppressed[1].y == features_detected_sorted_by_y[10].y);
            REQUIRE(features_suppressed[1].response == features_detected_sorted_by_y[10].response);
            REQUIRE(features_suppressed[2].x == features_detected_sorted_by_y[11].x);
            REQUIRE(features_suppressed[2].y == features_detected_sorted_by_y[11].y);
            REQUIRE(features_suppressed[2].response == features_detected_sorted_by_y[11].response);
        }

        {
            const feature::point features_detected_sorted_by_y[12] = {
                { 0, 0, 0, 0.0f },
                { 2, 2, 0, 0.1f },
                { 4, 4, 0, 0.2f },
                { 8, 8, 0, 0.3f },
                { 10, 10, 0, 0.4f },
                { 12, 12, 0, 0.5f },
                { 14, 14, 0, 0.6f },
                { 16, 16, 0, 0.7f },
                { 18, 18, 0, 0.8f },
                { 20, 20, 0, 0.9f },
                { 22, 22, 0, 1.0f },
                { 24, 24, 0, 1.1f },
            };
            feature::point features_suppressed[12] = {};
            const int features = feature::suppress(&features_detected_sorted_by_y[0], 12, 25, &features_suppressed[0]);
            REQUIRE(features == 12);
        }
    }

    {
        feature::point features[9] = {
            { -1, -1, 5, 0 },
            { 0, -1, 6, 0 },
            { +1, -1, 7, 0 },
            { -1, 0, 4, 0 },
            { 0, 0, 1, 0 },
            { +1, 0, 8, 0 },
            { -1, +1, 3, 0 },
            { 0, +1, 2, 0 },
            { +1, +1, 9, 0 }
        };
        feature::sort(&features[0], 9, [](const feature::point& lhs, const feature::point& rhs) {
            return lhs.response > rhs.response;
        });
        for (unsigned int i = 0; i < 9; ++i) {
            REQUIRE(is_value_approx(features[i].response, (9 - i)));
        }
    }

    {
        const feature::point features_detected_sorted_by_response[10] = {
            { 1, 21, 89, 0.2f },
            { 42, 42, 65, 0.6f },
            { 26, 1, 35, 0.3f },
            { 21, 15, 21, 0.4f },
            { 48, 72, 10, 0.7f },
            { 72, 2, 3, 0.9f },
            { 7, 3, 2, 0.1f },
            { 63, 16, 2, 0.8f },
            { 79, 68, 2, 1.0f },
            { 24, 29, 1, 0.5f }
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
            { 0, 4, 0, 0, 0, 0, 0, 0, 0, 0 },
            { 0, 1, 4, 0, 0, 0, 0, 0, 0, 0 },
            { 0, 1, 5, 8, 0, 0, 0, 0, 0, 0 },
            { 0, 1, 4, 7, 8, 0, 0, 0, 0, 0 },
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
            const int features = feature::distribute(&features_detected_sorted_by_response[0], features_detected_sorted_size, max_width, max_height, min_features, max_features, &features_distributed[0]);
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
        constexpr static const size_t data_width = 128;
        constexpr static const size_t data_height = 64;
        unsigned char data[data_height][data_width] = {};
        for (size_t y = 0; y < data_height; ++y) {
            for (size_t x = 0; x < data_width; ++x) {
                if ((x >= data_width / 2) || (y >= data_height / 2)) {
                    if ((x == data_width / 2) || (y == data_height / 2)) {
                        data[y][x] = 128;
                    }
                    else {
                        data[y][x] = 255;
                    }
                }
                else {
                    data[y][x] = 0;
                }
            }
        }
        for (int y = -2; y <= 2; ++y) {
            for (int x = -2; x <= 2; ++x) {
                float offset_x = 0;
                float offset_y = 0;
                REQUIRE(feature::refine(&data[data_height / 2 + y][data_width / 2 + x], data_width, offset_x, offset_y));
                REQUIRE(is_value_approx(offset_x, -x, 1e-2));
                REQUIRE(is_value_approx(offset_y, -y, 1e-2));
            }
        }
    }

    {
        constexpr static const size_t data_width = 128;
        constexpr static const size_t data_height = 64;
        unsigned char data[data_height][data_width] = {};
        for (size_t y = 0; y < data_height; ++y) {
            for (size_t x = 0; x < data_width; ++x) {
                if ((x >= data_width / 2) || (y >= data_height / 2)) {
                    if ((x == data_width / 2) || (y == data_height / 2)) {
                        data[y][x] = 128;
                    }
                    else {
                        data[y][x] = 255;
                    }
                }
                else {
                    data[y][x] = 0;
                }
            }
        }
        for (int y = -2; y <= 2; ++y) {
            for (int x = -2; x <= 2; ++x) {
                float offset_x = 0;
                float offset_y = 0;
                REQUIRE(feature::refine_bilinear(&data[data_height / 2 + y][data_width / 2 + x], data_width, offset_x, offset_y));
                REQUIRE(is_value_approx(offset_x, -x, 1e-2));
                REQUIRE(is_value_approx(offset_y, -y, 1e-2));
            }
        }
    }

    {
        constexpr static const size_t data_width = 128;
        constexpr static const size_t data_height = 64;
        unsigned char data[data_height][data_width] = {};
        for (size_t y = 0; y < data_height; ++y) {
            for (size_t x = 0; x < data_width; ++x) {
                data[y][x] = static_cast<unsigned char>(x + y);
                if ((x > data_width / 4) && (x < (data_width * 3) / 4) && (y > data_height / 4) && (y < (data_height * 3) / 4)) {
                    data[y][x] = 0;
                }
            }
        }
        constexpr static const size_t features_count = 12;
        const size_t features[features_count][2] = {
            { 93, 17 },
            { 94, 17 },
            { 95, 17 },
            { 94, 18 },
            { 95, 18 },
            { 95, 19 },
            { 95, 45 },
            { 94, 46 },
            { 95, 46 },
            { 93, 47 },
            { 94, 47 },
            { 95, 47 }
        };
        const float angles_radians[features_count] = {
            -0.640070f,
            -0.576437f,
            -0.514927f,
            -0.543202f,
            -0.483588f,
            -0.448683f,
            +0.698690f,
            +0.785398f,
            +0.743421f,
            +0.872107f,
            +0.827376f,
            +0.785398f
        };
        for (size_t i = 0; i < features_count; ++i) {
            REQUIRE(is_value_approx(feature::dominant_angle(&data[features[i][1]][features[i][0]], data_width), angles_radians[i], 1e-5));
        }
    }

    {
        constexpr static const size_t data_width = 128;
        constexpr static const size_t data_height = 64;
        unsigned char data[data_height][data_width] = {};
        for (size_t y = 0; y < data_height; ++y) {
            for (size_t x = 0; x < data_width; ++x) {
                data[y][x] = static_cast<unsigned char>(x + y);
            }
        }
        feature::descriptor descriptor_expected = {
            0x06,
            0x41,
            0x10,
            0x05,
            0x51,
            0x4E,
            0xB2,
            0x28,
            0x2C,
            0xA4,
            0x4B,
            0x01,
            0xCC,
            0x12,
            0x0D,
            0xC4,
            0x90,
            0x1D,
            0x82,
            0x63,
            0x44,
            0x90,
            0x51,
            0x4C,
            0x25,
            0x34,
            0x0A,
            0x07,
            0x00,
            0x37,
            0x0D,
            0x86
        };
        feature::descriptor descriptor;
        feature::describe(&data[data_height / 2][data_width / 2], data_width, 190.0f, descriptor);
        REQUIRE(are_values_approx(&descriptor.data[0], &descriptor_expected.data[0], 256 / 8));
    }

    {
        unsigned long long int lhs[4] = { 0, 0, 0, 1 };
        unsigned long long int rhs[4] = { 0, 0, 0, 0 };
        while ((lhs[0] != 0) || (lhs[1] != 0) || (lhs[2] != 0) || (lhs[3] != 0)) {
            REQUIRE(feature::distance<256>(reinterpret_cast<unsigned char*>(&lhs[0]), reinterpret_cast<unsigned char*>(&rhs[0])) == 1);
            lhs[3] <<= 1;
            for (int i = 2; i >= 0; --i) {
                if ((lhs[0] == 0) && (lhs[1] == 0) && (lhs[2] == 0) && (lhs[3] == 0)) {
                    lhs[i] = 0b0000000000000000000000000000000000000000000000000000000000000001;
                }
                else {
                    lhs[i] <<= 1;
                }
            }
        }
    }

    {
        feature::descriptor lhs[10] = {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 0, 0, 0, 0 },
            { 4, 4, 4, 4 },
            { 1, 1, 2, 2 },
            { 4, 3, 2, 1 },
            { 8, 7, 6, 5 },
            { 8, 8, 8, 8 },
            { 4, 4, 4, 4 },
            { 3, 3, 4, 4 }
        };
        feature::descriptor rhs[10] = {
            { 4, 3, 2, 1 },
            { 8, 7, 6, 5 },
            { 8, 8, 8, 8 },
            { 4, 4, 4, 4 },
            { 3, 3, 4, 4 },
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 0, 0, 0, 0 },
            { 4, 4, 4, 4 },
            { 1, 1, 2, 2 }
        };
        feature::match matches[20];
        size_t matches_count = feature::find_matches(&lhs[0], 10, &rhs[0], 10, 1, 2, &matches[0], 20);
        REQUIRE(matches_count == 12);
    }

    return EXIT_SUCCESS;
}