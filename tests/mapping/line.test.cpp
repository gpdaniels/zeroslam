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

#include "mapping/line.hpp"

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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        mapping::line l;
        REQUIRE(l.id == -1);
    }
    {
        geometry::plucker line;
        REQUIRE(geometry::plucker::from_points({ { 1.0, 2.0, 0.0 } }, { { 1.0, 2.0, 1.0 } }, line));
        mapping::line l(3, line, { { 1.0, 2.0, 0.0 } }, { { 1.0, 2.0, 1.0 } });
        REQUIRE(l.id == 3);
        l.locations[0] = { { 1.5, 2.5, 0.0 } };
        l.locations[1] = { { 0.5, 1.5, 4.0 } };
        l.resynchronise_endpoints();
        REQUIRE(std::abs(l.locations[0][0] - 1.0) < 1.0e-12);
        REQUIRE(std::abs(l.locations[0][1] - 2.0) < 1.0e-12);
        REQUIRE(std::abs(l.locations[0][2] - 0.0) < 1.0e-12);
        REQUIRE(std::abs(l.locations[1][0] - 1.0) < 1.0e-12);
        REQUIRE(std::abs(l.locations[1][1] - 2.0) < 1.0e-12);
        REQUIRE(std::abs(l.locations[1][2] - 4.0) < 1.0e-12);
    }

    return EXIT_SUCCESS;
}
