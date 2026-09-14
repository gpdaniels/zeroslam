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

#include "core/arena_allocator.hpp"

#include "core/thread_pool.hpp"

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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        core::arena::scope scope;
        std::vector<double, core::arena_allocator<double>> values;
        for (int i = 0; i < 10000; ++i) {
            values.push_back(static_cast<double>(i));
        }
        for (int i = 0; i < 10000; ++i) {
            REQUIRE(values[static_cast<size_t>(i)] == static_cast<double>(i));
        }
        std::vector<int, core::arena_allocator<int>> other(1000, 3);
        REQUIRE(other[999] == 3);
        REQUIRE(core::arena_allocator<double>() == core::arena_allocator<int>());
    }

    {
        std::vector<int> sums(64, 0);
        core::thread_pool::instance().parallel_for(64, 1, [&sums](const size_t index) {
            core::arena::scope scope;
            std::vector<int, core::arena_allocator<int>> scratch(1000, static_cast<int>(index));
            int sum = 0;
            for (const int value : scratch) {
                sum += value;
            }
            sums[index] = sum;
        });
        for (size_t index = 0; index < sums.size(); ++index) {
            REQUIRE(sums[index] == static_cast<int>(index) * 1000);
        }
        REQUIRE(core::arena::depth() == 0);
    }

    return EXIT_SUCCESS;
}
