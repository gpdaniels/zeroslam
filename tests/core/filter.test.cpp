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
        // Each removed element is replaced by the last kept one, so the order is fixed.
        const feature::point expected_order[9] = {
            { +1, +1, 0, 0, 0 },
            { 0, +1, 0, 0, 0 },
            { -1, +1, 0, 0, 0 },
            { +1, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0 },
            { -1, 0, 0, 0, 0 },
            { -1, -1, 0, 0, 0 },
            { 0, -1, 0, 0, 0 },
            { +1, -1, 0, 0, 0 }
        };
        for (size_t i = 0; i < prune_count; ++i) {
            REQUIRE(features[i].x == expected_order[i].x);
            REQUIRE(features[i].y == expected_order[i].y);
        }
    }

    {
        int values[1] = { 7 };
        size_t count = 0;
        int calls = 0;
        core::filter::remove_if(&values[0], count, [&calls](const int) {
            ++calls;
            return true;
        });
        REQUIRE(count == 0);
        REQUIRE(calls == 0);
        REQUIRE(values[0] == 7);
    }

    {
        int values[1] = { 7 };
        size_t count = 1;
        int calls = 0;
        core::filter::remove_if(&values[0], count, [&calls](const int) {
            ++calls;
            return true;
        });
        REQUIRE(count == 0);
        REQUIRE(calls == 1);
    }

    {
        int values[1] = { 7 };
        size_t count = 1;
        int calls = 0;
        core::filter::remove_if(&values[0], count, [&calls](const int) {
            ++calls;
            return false;
        });
        REQUIRE(count == 1);
        REQUIRE(calls == 1);
        REQUIRE(values[0] == 7);
    }

    {
        int values[6] = { 1, 3, 5, 7, 9, 11 };
        size_t count = 6;
        int calls = 0;
        core::filter::remove_if(&values[0], count, [&calls](const int) {
            ++calls;
            return true;
        });
        REQUIRE(count == 0);
        REQUIRE(calls == 6);
    }

    {
        int values[6] = { 2, 4, 6, 8, 10, 12 };
        size_t count = 6;
        int calls = 0;
        core::filter::remove_if(&values[0], count, [&calls](const int) {
            ++calls;
            return false;
        });
        REQUIRE(count == 6);
        REQUIRE(calls == 6);
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(values[i] == static_cast<int>(2 * (i + 1)));
        }
    }

    {
        int values[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        size_t count = 10;
        int calls = 0;
        core::filter::remove_if(&values[0], count, [&calls](const int value) {
            ++calls;
            return (value % 2) != 0;
        });
        const int expected[5] = { 10, 2, 8, 4, 6 };
        REQUIRE(count == 5);
        REQUIRE(calls == 10);
        for (size_t i = 0; i < count; ++i) {
            REQUIRE(values[i] == expected[i]);
        }
    }

    {
        int values[5] = { 1, 1, 2, 1, 1 };
        size_t count = 5;
        core::filter::remove_if(&values[0], count, [](const int value) {
            return value == 1;
        });
        REQUIRE(count == 1);
        REQUIRE(values[0] == 2);
    }

    return EXIT_SUCCESS;
}
