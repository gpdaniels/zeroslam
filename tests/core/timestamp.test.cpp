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

#include "core/timestamp.hpp"

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
        REQUIRE(core::timestamp() > 0);
    }

    {
        long long int previous = core::timestamp();
        for (int i = 0; i < 100000; ++i) {
            const long long int current = core::timestamp();
            REQUIRE(current >= previous);
            previous = current;
        }
    }

    {
        const long long int start = core::timestamp();
        long long int current = start;
        int reads = 0;
        while ((current == start) && (reads < 10000000)) {
            current = core::timestamp();
            ++reads;
        }
        REQUIRE(current > start);
        REQUIRE((current - start) < 1000000ll);
    }

    {
        const long long int wait = 10000000ll;
        const long long int start = core::timestamp();
        while ((core::timestamp() - start) < wait) {
        }
        const long long int elapsed = core::timestamp() - start;
        REQUIRE(elapsed >= wait);
        REQUIRE(elapsed < 1000000000ll);
    }

    return EXIT_SUCCESS;
}
