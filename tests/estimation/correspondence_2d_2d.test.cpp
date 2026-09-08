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

#include "estimation/correspondence_2d_2d.hpp"

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

    static_assert(sizeof(estimation::correspondence_2d_2d<double>) == 4 * sizeof(double));
    static_assert(sizeof(estimation::correspondence_2d_2d<float>) == 4 * sizeof(float));

    {
        estimation::correspondence_2d_2d<double> correspondence;
        correspondence.lhs = math::matrix<double, 2, 1>{ { 0.25, -0.5 } };
        correspondence.rhs = math::matrix<double, 2, 1>{ { 1.5, 2.0 } };
        const estimation::correspondence_2d_2d<double> copy = correspondence;
        REQUIRE(copy.lhs[0] == 0.25);
        REQUIRE(copy.lhs[1] == -0.5);
        REQUIRE(copy.rhs[0] == 1.5);
        REQUIRE(copy.rhs[1] == 2.0);
    }

    return EXIT_SUCCESS;
}
