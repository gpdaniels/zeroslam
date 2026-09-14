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

#include "core/thread_pool.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <atomic>
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

    core::thread_pool& pool = core::thread_pool::instance();
    REQUIRE(&pool == &core::thread_pool::instance());

    for (size_t count : { size_t(0), size_t(1), size_t(7), size_t(1000), size_t(12345) }) {
        for (size_t grain : { size_t(1), size_t(16), size_t(100000) }) {
            std::vector<std::atomic<int>> hits(count);
            for (std::atomic<int>& hit : hits) {
                hit = 0;
            }
            pool.parallel_for(count, grain, [&hits](const size_t index) {
                ++hits[index];
            });
            for (size_t index = 0; index < count; ++index) {
                REQUIRE(hits[index] == 1);
            }
        }
    }

    {
        std::vector<double> values(5000, 0.0);
        pool.parallel_for(values.size(), 1, [&values](const size_t index) {
            values[index] = static_cast<double>(index) * 0.5;
        });
        for (size_t index = 0; index < values.size(); ++index) {
            REQUIRE(values[index] == static_cast<double>(index) * 0.5);
        }
    }

    {
        std::atomic<int> total(0);
        core::thread_pool::queue tasks(pool, 1);
        for (int i = 0; i < 200; ++i) {
            tasks.push([&total, i]() {
                total += i;
            });
        }
        tasks.drain();
        REQUIRE(tasks.empty());
        REQUIRE(tasks.finished());
        REQUIRE(total == 199 * 200 / 2);
    }

    {
        std::atomic<int> first_total(0);
        std::atomic<int> second_total(0);
        core::thread_pool::queue first(pool, 0);
        core::thread_pool::queue second(pool, 2);
        for (int i = 0; i < 100; ++i) {
            first.push([&first_total]() {
                ++first_total;
            });
            second.push([&second_total]() {
                second_total += 2;
            });
        }
        first.drain();
        second.drain();
        REQUIRE(first_total == 100);
        REQUIRE(second_total == 200);
    }

    {
        std::atomic<int> total(0);
        for (int repeat = 0; repeat < 20000; ++repeat) {
            pool.parallel_for(8, 1, [&total](const size_t) {
                ++total;
            });
        }
        REQUIRE(total == 20000 * 8);
    }

    {
        std::atomic<int> inner_total(0);
        pool.parallel_for(8, 1, [&pool, &inner_total](const size_t) {
            pool.parallel_for(64, 8, [&inner_total](const size_t) {
                ++inner_total;
            });
        });
        REQUIRE(inner_total == 8 * 64);
    }

    return EXIT_SUCCESS;
}
