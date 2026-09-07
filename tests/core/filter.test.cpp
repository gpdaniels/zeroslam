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

#include "core/filter.hpp"

#include "feature/point.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>

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

    {
        feature::point features[25] = {
            { -2, -2, 0, 0, 0 },
            { -1, -2, 0, 0, 0 },
            { 0, -2, 0, 0, 0 },
            { +1, -2, 0, 0, 0 },
            { +2, -2, 0, 0, 0 },
            { -2, -1, 0, 0, 0 },
            { -1, -1, 0, 0, 0 },
            { 0, -1, 0, 0, 0 },
            { +1, -1, 0, 0, 0 },
            { +2, -1, 0, 0, 0 },
            { -2, 0, 0, 0, 0 },
            { -1, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0 },
            { +1, 0, 0, 0, 0 },
            { +2, 0, 0, 0, 0 },
            { -2, +1, 0, 0, 0 },
            { -1, +1, 0, 0, 0 },
            { 0, +1, 0, 0, 0 },
            { +1, +1, 0, 0, 0 },
            { +2, +1, 0, 0, 0 },
            { -2, +2, 0, 0, 0 },
            { -1, +2, 0, 0, 0 },
            { 0, +2, 0, 0, 0 },
            { +1, +2, 0, 0, 0 },
            { +2, +2, 0, 0, 0 }
        };
        size_t prune_count = 25;
        core::filter::remove_if(&features[0], prune_count, [](const feature::point& feature) {
            return (feature.x < -1) || (feature.x > 1) || (feature.y < -1) || (feature.y > 1);
        });
        REQUIRE(prune_count == 9);
        const feature::point should_still_exist[9] = {
            { -1, -1, 0, 0, 0 },
            { 0, -1, 0, 0, 0 },
            { +1, -1, 0, 0, 0 },
            { -1, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0 },
            { +1, 0, 0, 0, 0 },
            { -1, +1, 0, 0, 0 },
            { 0, +1, 0, 0, 0 },
            { +1, +1, 0, 0, 0 }
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

    return EXIT_SUCCESS;
}
