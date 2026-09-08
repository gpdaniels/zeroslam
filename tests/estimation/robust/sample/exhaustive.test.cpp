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

#include "estimation/robust/sample/exhaustive.hpp"

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

    static_assert(estimation::robust::sample::exhaustive<4>::sample_size == 4);

    // Every combination of two from five in order, then wrapping.
    {
        estimation::robust::sample::exhaustive<2> sampler;
        sampler.prepare(5);
        const size_t expected[10][2] = { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 0, 4 }, { 1, 2 }, { 1, 3 }, { 1, 4 }, { 2, 3 }, { 2, 4 }, { 3, 4 } };
        for (int pass = 0; pass < 2; ++pass) {
            for (size_t draw = 0; draw < 10; ++draw) {
                size_t indices[2];
                sampler.sample(indices);
                REQUIRE(indices[0] == expected[draw][0]);
                REQUIRE(indices[1] == expected[draw][1]);
            }
        }
    }

    // Three from six visits each combination exactly once per cycle.
    {
        estimation::robust::sample::exhaustive<3> sampler;
        sampler.prepare(6);
        bool seen[6][6][6] = {};
        for (size_t draw = 0; draw < 20; ++draw) {
            size_t indices[3];
            sampler.sample(indices);
            REQUIRE(indices[0] < indices[1]);
            REQUIRE(indices[1] < indices[2]);
            REQUIRE(indices[2] < 6);
            REQUIRE(!seen[indices[0]][indices[1]][indices[2]]);
            seen[indices[0]][indices[1]][indices[2]] = true;
        }
        size_t indices[3];
        sampler.sample(indices);
        REQUIRE(indices[0] == 0);
        REQUIRE(indices[1] == 1);
        REQUIRE(indices[2] == 2);
    }

    // A sample the size of the data always returns the identity.
    {
        estimation::robust::sample::exhaustive<3> sampler;
        sampler.prepare(3);
        for (int draw = 0; draw < 4; ++draw) {
            size_t indices[3];
            sampler.sample(indices);
            REQUIRE(indices[0] == 0);
            REQUIRE(indices[1] == 1);
            REQUIRE(indices[2] == 2);
        }
    }

    return EXIT_SUCCESS;
}
