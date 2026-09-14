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

#include "feature/refiner/structure_tensor.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
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
    using refiner = feature::refiner::structure_tensor;

    REQUIRE(refiner::footprint(refiner::sigma_maximum) <= 25);

    {
        constexpr static const size_t data_width = 96;
        constexpr static const size_t data_height = 96;
        constexpr static const double centre_x = 48.3;
        constexpr static const double centre_y = 47.8;
        unsigned char data[data_height][data_width] = {};
        for (size_t y = 0; y < data_height; ++y) {
            for (size_t x = 0; x < data_width; ++x) {
                const double coverage_x = std::min(1.0, std::max(0.0, centre_x - (static_cast<double>(x) - 0.5)));
                const double coverage_y = std::min(1.0, std::max(0.0, centre_y - (static_cast<double>(y) - 0.5)));
                data[y][x] = static_cast<unsigned char>(std::floor(255.0 * ((coverage_x * coverage_y) + ((1.0 - coverage_x) * (1.0 - coverage_y))) + 0.5));
            }
        }
        for (measure kind : { measure::klt, measure::forstner, measure::harris, measure::rohr, measure::kenney }) {
            for (size_t start = 46; start <= 50; ++start) {
                float offset_x = 0;
                float offset_y = 0;
                REQUIRE(refiner::refine(&data[start][start], data_width, kind, 1.5f, offset_x, offset_y));
                REQUIRE(std::abs((static_cast<double>(start) + static_cast<double>(offset_x)) - centre_x) < 0.2);
                REQUIRE(std::abs((static_cast<double>(start) + static_cast<double>(offset_y)) - centre_y) < 0.2);
            }
            float offset_x = 0;
            float offset_y = 0;
            REQUIRE(!refiner::refine(&data[44][44], data_width, kind, 1.5f, offset_x, offset_y));
            REQUIRE((offset_x == 0.0f) && (offset_y == 0.0f));
            REQUIRE(!refiner::refine(&data[20][70], data_width, kind, 1.5f, offset_x, offset_y));
        }
    }

    return EXIT_SUCCESS;
}
