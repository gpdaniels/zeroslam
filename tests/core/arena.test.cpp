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

#include "core/arena.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

    core::arena& arena = core::arena::instance();
    REQUIRE(&arena == &core::arena::instance());
    REQUIRE(core::arena::depth() == 0);

    {
        void* first = nullptr;
        {
            core::arena::scope scope;
            REQUIRE(core::arena::depth() == 1);
            first = arena.allocate(100, 8);
            void* second = arena.allocate(100, 64);
            REQUIRE(first != nullptr);
            REQUIRE(second != nullptr);
            REQUIRE((reinterpret_cast<std::uintptr_t>(first) % 8) == 0);
            REQUIRE((reinterpret_cast<std::uintptr_t>(second) % 64) == 0);
            REQUIRE(static_cast<unsigned char*>(second) >= static_cast<unsigned char*>(first) + 100);
            std::memset(first, 0xAB, 100);
            std::memset(second, 0xCD, 100);
            REQUIRE(static_cast<unsigned char*>(first)[99] == 0xAB);
            REQUIRE(static_cast<unsigned char*>(second)[0] == 0xCD);
        }
        REQUIRE(core::arena::depth() == 0);
        const size_t chunks = arena.chunk_count();
        {
            core::arena::scope scope;
            void* again = arena.allocate(100, 8);
            REQUIRE(again == first);
        }
        REQUIRE(arena.chunk_count() == chunks);
    }

    {
        core::arena::scope outer;
        void* outer_block = arena.allocate(64, 8);
        std::memset(outer_block, 1, 64);
        void* inner_block = nullptr;
        {
            core::arena::scope inner;
            REQUIRE(core::arena::depth() == 2);
            inner_block = arena.allocate(64, 8);
            REQUIRE(inner_block != outer_block);
        }
        void* reused = arena.allocate(64, 8);
        REQUIRE(reused == inner_block);
        REQUIRE(static_cast<unsigned char*>(outer_block)[63] == 1);
    }

    {
        core::arena::scope scope;
        const size_t chunks_before = arena.chunk_count();
        unsigned char* large = static_cast<unsigned char*>(arena.allocate(3 * core::arena::chunk_size, 16));
        REQUIRE(large != nullptr);
        large[3 * core::arena::chunk_size - 1] = 7;
        REQUIRE(arena.chunk_count() > chunks_before);
        std::vector<unsigned char*> blocks;
        for (int i = 0; i < 4096; ++i) {
            blocks.push_back(static_cast<unsigned char*>(arena.allocate(1000, 8)));
            blocks.back()[999] = static_cast<unsigned char>(i);
        }
        for (int i = 0; i < 4096; ++i) {
            REQUIRE(blocks[static_cast<size_t>(i)][999] == static_cast<unsigned char>(i));
        }
    }

    return EXIT_SUCCESS;
}
