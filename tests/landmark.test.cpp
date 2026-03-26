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

#include "landmark.hpp"

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
        landmark::point p;
    }
    {
        landmark::point p0({ { 1, 2, 3 } }, { { 4, 5, 6 } });
        REQUIRE(p0.id == 0);
        landmark::point p1({ { 4, 5, 6 } }, { { 7, 8, 9 } });
        REQUIRE(p1.id == 1);
    }
    {
        landmark::line l;
    }
    {
        landmark::line l0({ { 1, 2, 3 } }, { { 1, 2, 3 } });
        REQUIRE(l0.id == 0);
        landmark::line l1({ { 4, 5, 6 } }, { { 4, 5, 6 } });
        REQUIRE(l1.id == 1);
    }

    return EXIT_SUCCESS;
}