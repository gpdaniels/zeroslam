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

#include "feature/detector/structure_tensor.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    using measure = feature::score::structure_tensor::measure;

    {
        constexpr static const size_t data_width = 128;
        constexpr static const size_t data_height = 64;
        unsigned char data[data_height][data_width] = {};
        for (size_t y = 0; y < data_height; ++y) {
            for (size_t x = 0; x < data_width; ++x) {
                data[y][x] = static_cast<unsigned char>(((x > data_width / 4) && (x < (data_width * 3) / 4) && (y > data_height / 4) && (y < (data_height * 3) / 4)) * 255);
            }
        }
        const float corners[4][2] = { { 33.0f, 17.0f }, { 95.0f, 17.0f }, { 33.0f, 47.0f }, { 95.0f, 47.0f } };
        feature::point features[64];
        for (measure kind : { measure::klt, measure::forstner, measure::harris, measure::rohr, measure::kenney }) {
            const size_t count = feature::detector::structure_tensor::detect(&data[0][0], data_width, data_height, data_width, kind, 1.5f, 10.0f, 64, features);
            REQUIRE(count == 4);
            for (size_t i = 0; i < count; ++i) {
                REQUIRE(features[i].response > 0.0f);
                float nearest = 1e9f;
                for (const float* corner : corners) {
                    const float distance = std::sqrt((features[i].x - corner[0]) * (features[i].x - corner[0]) + (features[i].y - corner[1]) * (features[i].y - corner[1]));
                    nearest = (distance < nearest) ? distance : nearest;
                }
                REQUIRE(nearest < 2.0f);
            }
            REQUIRE((features[0].y <= features[1].y) && (features[2].y <= features[3].y));
            REQUIRE(feature::detector::structure_tensor::detect(&data[0][0], data_width, data_height, data_width, kind, 1.5f, 10.0f, 2, features) == 2);
            unsigned char flat[data_height][data_width] = {};
            REQUIRE(feature::detector::structure_tensor::detect(&flat[0][0], data_width, data_height, data_width, kind, 1.5f, 10.0f, 64, features) == 0);
        }
    }

    return EXIT_SUCCESS;
}
