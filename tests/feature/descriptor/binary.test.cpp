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

#include "feature/descriptor/binary.hpp"

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
        feature::descriptor::binary<256> descriptor{
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31
        };
        for (size_t i = 0; i < 32; ++i) {
            REQUIRE(descriptor.data[i] == i);
        }
    }

    {
        REQUIRE(feature::descriptor::binary<64>::size_bytes == 8);
        REQUIRE(sizeof(feature::descriptor::binary<64>) == 8);
        REQUIRE(feature::descriptor::binary<128>::size_bytes == 16);
        REQUIRE(feature::descriptor::binary<256>::size_bytes == 32);
        REQUIRE(sizeof(feature::descriptor::binary<256>) == 32);
        feature::descriptor::binary<64> small{ 1, 2, 3, 4, 5, 6, 7, 8 };
        for (size_t i = 0; i < 8; ++i) {
            REQUIRE(small[i] == i + 1);
        }
    }

    return EXIT_SUCCESS;
}
