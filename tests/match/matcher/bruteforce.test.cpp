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
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

// A descriptor with its first count bits set, so its distance from the empty descriptor is count.
static feature::descriptor::binary<256> with_bits(const size_t count) {
    feature::descriptor::binary<256> descriptor{};
    for (size_t bit = 0; bit < count; ++bit) {
        descriptor.data[bit / 8] = static_cast<unsigned char>(descriptor.data[bit / 8] | (1u << (bit % 8)));
    }
    return descriptor;
}

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
        // Every left descriptor has an exact match, so each reports its runner up as well.
        size_t matches_count = match::matcher::bruteforce::find_matches(&lhs[0], 10, &rhs[0], 10, 1, 2, &matches[0], 20);
        REQUIRE(matches_count == 20);
        size_t exact = 0;
        for (size_t i = 0; i < matches_count; ++i) {
            exact += (matches[i].score == 0.0f) ? 1u : 0u;
        }
        REQUIRE(exact == 12);
    }

    {
        // A runner up at the threshold is still reported, so a ratio test sees 49 against 50 and can reject it.
        const feature::descriptor::binary<256> lhs[1] = { with_bits(0) };
        const feature::descriptor::binary<256> rhs[3] = { with_bits(50), with_bits(49), with_bits(120) };
        match::pair matches[3];
        REQUIRE(match::matcher::bruteforce::find_matches(&lhs[0], 1, &rhs[0], 3, 50.0f, 2, &matches[0], 3) == 2);
        REQUIRE((matches[0].lhs_index == 0) && (matches[0].rhs_index == 1) && (matches[0].score == 49.0f));
        REQUIRE((matches[1].lhs_index == 0) && (matches[1].rhs_index == 0) && (matches[1].score == 50.0f));
        REQUIRE(!(matches[0].score < 0.8f * matches[1].score));
        REQUIRE(match::matcher::bruteforce::find_matches(&lhs[0], 1, &rhs[0], 3, 50.0f, 1, &matches[0], 3) == 1);
        REQUIRE((matches[0].rhs_index == 1) && (matches[0].score == 49.0f));
        REQUIRE(match::matcher::bruteforce::find_matches(&lhs[0], 1, &rhs[0], 3, 50.0f, 3, &matches[0], 3) == 3);
        REQUIRE((matches[2].rhs_index == 2) && (matches[2].score == 120.0f));
        REQUIRE(match::matcher::bruteforce::find_matches(&lhs[0], 1, &rhs[0], 3, 49.0f, 2, &matches[0], 3) == 0);
        // Fewer right descriptors than matches_count, none at all, and no room for matches_count slots.
        REQUIRE(match::matcher::bruteforce::find_matches(&lhs[0], 1, &rhs[1], 1, 50.0f, 2, &matches[0], 3) == 1);
        REQUIRE((matches[0].rhs_index == 0) && (matches[0].score == 49.0f));
        REQUIRE(match::matcher::bruteforce::find_matches(&lhs[0], 1, &rhs[0], 0, 50.0f, 2, &matches[0], 3) == 0);
        REQUIRE(match::matcher::bruteforce::find_matches(&lhs[0], 1, &rhs[0], 3, 50.0f, 2, &matches[0], 1) == 0);
    }

    {
        // Across the blocks the distances are computed in, a tie goes to the lower right index.
        const feature::descriptor::binary<256> lhs[1] = { with_bits(0) };
        std::vector<feature::descriptor::binary<256>> rhs(600, with_bits(100));
        rhs[520] = with_bits(40);
        rhs[300] = with_bits(20);
        rhs[5] = with_bits(40);
        match::pair matches[3];
        REQUIRE(match::matcher::bruteforce::find_matches(&lhs[0], 1, rhs.data(), rhs.size(), 50.0f, 3, &matches[0], 3) == 3);
        REQUIRE((matches[0].rhs_index == 300) && (matches[0].score == 20.0f));
        REQUIRE((matches[1].rhs_index == 5) && (matches[1].score == 40.0f));
        REQUIRE((matches[2].rhs_index == 520) && (matches[2].score == 40.0f));
    }

    return EXIT_SUCCESS;
}
