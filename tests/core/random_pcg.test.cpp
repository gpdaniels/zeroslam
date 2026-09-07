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

#include "core/random_pcg.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline double round_to_millionths(const double value) {
    return std::round(value * 1000000.0) / 1000000.0;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        REQUIRE(std::is_pod<core::random_pcg>::value == false);
        REQUIRE(std::is_trivial<core::random_pcg>::value == false);
        REQUIRE(std::is_trivially_copyable<core::random_pcg>::value == true);
        REQUIRE(std::is_standard_layout<core::random_pcg>::value == true);
    }

    {
        core::random_pcg random_pcg;
        static_cast<void>(random_pcg);
    }

    {
        core::random_pcg random_pcg(0x01234567);
        static_cast<void>(random_pcg);
    }

    {
        core::random_pcg random_pcg;
        random_pcg.seed(0x01234567);
    }

    {
        core::random_pcg random_pcg;
        random_pcg.seed(0x01234567);
        REQUIRE(random_pcg.get_random_raw() == 1350580380u);
        REQUIRE(random_pcg.get_random_raw() == 206545039u);
        REQUIRE(random_pcg.get_random_raw() == 167867113u);
        REQUIRE(random_pcg.get_random_raw() == 3789127393u);
        REQUIRE(random_pcg.get_random_raw() == 236503499u);
    }

    {
        core::random_pcg random_pcg;
        random_pcg.seed(0x01234567);
        REQUIRE(round_to_millionths(random_pcg.get_random_exclusive()) == 0.314456);
        REQUIRE(round_to_millionths(random_pcg.get_random_exclusive()) == 0.048090);
        REQUIRE(round_to_millionths(random_pcg.get_random_exclusive()) == 0.039085);
        REQUIRE(round_to_millionths(random_pcg.get_random_exclusive()) == 0.882225);
        REQUIRE(round_to_millionths(random_pcg.get_random_exclusive()) == 0.055065);
    }

    {
        core::random_pcg random_pcg;
        random_pcg.seed(0x01234567);
        REQUIRE(round_to_millionths(random_pcg.get_random_exclusive_top()) == 0.314456);
        REQUIRE(round_to_millionths(random_pcg.get_random_exclusive_top()) == 0.048090);
        REQUIRE(round_to_millionths(random_pcg.get_random_exclusive_top()) == 0.039085);
        REQUIRE(round_to_millionths(random_pcg.get_random_exclusive_top()) == 0.882225);
        REQUIRE(round_to_millionths(random_pcg.get_random_exclusive_top()) == 0.055065);
    }

    {
        core::random_pcg random_pcg;
        random_pcg.seed(0x01234567);
        REQUIRE(round_to_millionths(random_pcg.get_random_inclusive()) == 0.314456);
        REQUIRE(round_to_millionths(random_pcg.get_random_inclusive()) == 0.048090);
        REQUIRE(round_to_millionths(random_pcg.get_random_inclusive()) == 0.039085);
        REQUIRE(round_to_millionths(random_pcg.get_random_inclusive()) == 0.882225);
        REQUIRE(round_to_millionths(random_pcg.get_random_inclusive()) == 0.055065);
    }

    {
        core::random_pcg random_pcg;
        random_pcg.seed(0x01234567);
        REQUIRE(random_pcg.get_random(0u, 1u) == 0u);
        REQUIRE(random_pcg.get_random(0u, 1u) == 0u);
        REQUIRE(random_pcg.get_random(0u, 1u) == 0u);
        REQUIRE(random_pcg.get_random(0u, 1u) == 1u);
        REQUIRE(random_pcg.get_random(0u, 1u) == 0u);
    }

    {
        core::random_pcg random_pcg;
        random_pcg.seed(0x01234567);
        REQUIRE(round_to_millionths(random_pcg.get_random(-1.0, 0.0)) == -0.685544);
        REQUIRE(round_to_millionths(random_pcg.get_random(-1.0, 0.0)) == -0.951910);
        REQUIRE(round_to_millionths(random_pcg.get_random(-1.0, 0.0)) == -0.960915);
        REQUIRE(round_to_millionths(random_pcg.get_random(-1.0, 0.0)) == -0.117775);
        REQUIRE(round_to_millionths(random_pcg.get_random(-1.0, 0.0)) == -0.944935);
    }

    {
        core::random_pcg random_pcg;
        random_pcg.seed(0x01234567);
        int random_bias = 0;
        for (unsigned int iteration = 0; iteration < 1000000; ++iteration) {
            random_bias += (random_pcg.get_random(0u, 1u) == 0) ? +1 : -1;
        }
        REQUIRE(random_bias == 1040);
    }

    return EXIT_SUCCESS;
}
