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

    // Samplers with equal seeds draw the same sequence, different seeds a different one, and seeded draws are still distinct and in range.
    {
        estimation::robust::sample::random<3> lhs(0x5eed0200ull);
        estimation::robust::sample::random<3> rhs(0x5eed0200ull);
        estimation::robust::sample::random<3> other(0x5eed0201ull);
        lhs.prepare(50);
        rhs.prepare(50);
        other.prepare(50);
        size_t differing = 0;
        for (int draw = 0; draw < 64; ++draw) {
            size_t lhs_indices[3];
            size_t rhs_indices[3];
            size_t other_indices[3];
            lhs.sample(lhs_indices);
            rhs.sample(rhs_indices);
            other.sample(other_indices);
            for (size_t i = 0; i < 3; ++i) {
                REQUIRE(lhs_indices[i] == rhs_indices[i]);
                REQUIRE(other_indices[i] < 50);
                differing += (lhs_indices[i] != other_indices[i]) ? size_t(1) : size_t(0);
            }
            REQUIRE(other_indices[0] != other_indices[1]);
            REQUIRE(other_indices[1] != other_indices[2]);
            REQUIRE(other_indices[0] != other_indices[2]);
        }
        REQUIRE(differing > 100);
    }

    // The data seed is a function of the bytes alone: equal data gives equal seeds, and changing one value or the size changes it.
    {
        double data[12];
        double copy[12];
        for (size_t i = 0; i < 12; ++i) {
            data[i] = 0.25 * static_cast<double>(i) - 1.0;
            copy[i] = data[i];
        }
        const unsigned long long int seed = estimation::robust::sample::random<2>::seed_from(data, 12);
        REQUIRE(estimation::robust::sample::random<2>::seed_from(copy, 12) == seed);
        REQUIRE(estimation::robust::sample::random<5>::seed_from(copy, 12) == seed);
        REQUIRE(estimation::robust::sample::random<2>::seed_from(data, 11) != seed);
        copy[7] = 0.5000000001;
        REQUIRE(estimation::robust::sample::random<2>::seed_from(copy, 12) != seed);
        const float odd_sized[3] = { 1.0f, 2.0f, 3.0f };
        const float odd_sized_changed[3] = { 1.0f, 2.0f, 3.5f };
        REQUIRE(estimation::robust::sample::random<2>::seed_from(odd_sized, 3) != estimation::robust::sample::random<2>::seed_from(odd_sized_changed, 3));
    }

    return EXIT_SUCCESS;
}
