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

#include "estimation/robust/sample/random.hpp"

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

    static_assert(estimation::robust::sample::random<5>::sample_size == 5);

    // Indices are distinct and in range.
    {
        estimation::robust::sample::random<4> sampler;
        sampler.prepare(10);
        size_t histogram[10] = {};
        for (int draw = 0; draw < 1000; ++draw) {
            size_t indices[4];
            sampler.sample(indices);
            for (size_t i = 0; i < 4; ++i) {
                REQUIRE(indices[i] < 10);
                ++histogram[indices[i]];
                for (size_t j = 0; j < i; ++j) {
                    REQUIRE(indices[j] != indices[i]);
                }
            }
        }
        for (size_t i = 0; i < 10; ++i) {
            REQUIRE(histogram[i] > 200);
        }
    }

    // A sample the size of the data is a permutation.
    {
        estimation::robust::sample::random<3> sampler;
        sampler.prepare(3);
        for (int draw = 0; draw < 16; ++draw) {
            size_t indices[3];
            sampler.sample(indices);
            REQUIRE(indices[0] + indices[1] + indices[2] == 3);
            REQUIRE(indices[0] != indices[1]);
            REQUIRE(indices[1] != indices[2]);
        }
    }

    // Two default constructed samplers draw the same sequence.
    {
        estimation::robust::sample::random<2> lhs;
        estimation::robust::sample::random<2> rhs;
        lhs.prepare(100);
        rhs.prepare(100);
        for (int draw = 0; draw < 64; ++draw) {
            size_t lhs_indices[2];
            size_t rhs_indices[2];
            lhs.sample(lhs_indices);
            rhs.sample(rhs_indices);
            REQUIRE(lhs_indices[0] == rhs_indices[0]);
            REQUIRE(lhs_indices[1] == rhs_indices[1]);
        }
    }

    return EXIT_SUCCESS;
}
