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

#include "match/matcher/bruteforce.hpp"

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
        feature::descriptor::binary<256> lhs[10] = {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 0, 0, 0, 0 },
            { 4, 4, 4, 4 },
            { 1, 1, 2, 2 },
            { 4, 3, 2, 1 },
            { 8, 7, 6, 5 },
            { 8, 8, 8, 8 },
            { 4, 4, 4, 4 },
            { 3, 3, 4, 4 }
        };
        feature::descriptor::binary<256> rhs[10] = {
            { 4, 3, 2, 1 },
            { 8, 7, 6, 5 },
            { 8, 8, 8, 8 },
            { 4, 4, 4, 4 },
            { 3, 3, 4, 4 },
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 0, 0, 0, 0 },
            { 4, 4, 4, 4 },
            { 1, 1, 2, 2 }
        };
        match::pair matches[20];
        size_t matches_count = match::matcher::bruteforce::find_matches(&lhs[0], 10, &rhs[0], 10, 1, 2, &matches[0], 20);
        REQUIRE(matches_count == 12);
    }

    return EXIT_SUCCESS;
}
