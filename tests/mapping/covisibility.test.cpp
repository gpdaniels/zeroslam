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

#include "mapping/covisibility.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstddef>
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
        mapping::covisibility graph;
        REQUIRE(graph.num_frames() == 0);
        REQUIRE(graph.weight(0, 1) == 0);
        REQUIRE(graph.neighbours(0, 1).empty());
    }

    {
        mapping::covisibility graph;
        const int first[3] = { 0, 3, 5 };
        const int second[2] = { 3, 5 };
        const int third[2] = { 5, 9 };
        const int repeated[3] = { 9, 9, 12 };
        graph.add(&first[0], 3);
        graph.add(&second[0], 2);
        graph.add(&third[0], 2);
        graph.add(&repeated[0], 3);
        REQUIRE(graph.num_frames() == 5);
        REQUIRE(graph.weight(0, 3) == 1);
        REQUIRE(graph.weight(3, 0) == 1);
        REQUIRE(graph.weight(3, 5) == 2);
        REQUIRE(graph.weight(5, 9) == 1);
        REQUIRE(graph.weight(0, 9) == 0);
        REQUIRE(graph.weight(9, 9) == 0);
        REQUIRE(graph.weight(9, 12) == 1);
        REQUIRE(graph.weight(42, 0) == 0);
    }

    {
        mapping::covisibility graph;
        for (int landmark = 0; landmark < 20; ++landmark) {
            const int frames[3] = { 7, 2, 11 };
            graph.add(&frames[0], (landmark < 15) ? 3 : 2);
        }
        REQUIRE(graph.weight(7, 2) == 20);
        REQUIRE(graph.weight(7, 11) == 15);
        REQUIRE(graph.weight(2, 11) == 15);
        const std::vector<int> all = graph.neighbours(7, 1);
        REQUIRE(all.size() == 2);
        REQUIRE(all[0] == 2);
        REQUIRE(all[1] == 11);
        const std::vector<int> strong = graph.neighbours(7, 16);
        REQUIRE(strong.size() == 1);
        REQUIRE(strong[0] == 2);
        REQUIRE(graph.neighbours(7, 21).empty());
        REQUIRE(graph.neighbours(99, 1).empty());
    }

    {
        mapping::covisibility graph;
        const int frames[2] = { 1, 2 };
        graph.add(&frames[0], 2);
        REQUIRE(graph.weight(1, 2) == 1);
        graph.clear();
        REQUIRE(graph.num_frames() == 0);
        REQUIRE(graph.weight(1, 2) == 0);
        graph.add(&frames[0], 2);
        graph.add(&frames[0], 2);
        REQUIRE(graph.weight(2, 1) == 2);
    }

    // Updating landmarks incrementally, as frames join and leave them and landmarks come and go, gives the graph a rebuild from scratch gives.
    {
        unsigned int state = 2463534242u;
        const auto next = [&state](const unsigned int bound) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return state % bound;
        };
        std::vector<std::vector<int>> observers(60);
        std::vector<bool> alive(observers.size(), false);
        mapping::covisibility incremental;
        for (int round = 0; round < 40; ++round) {
            for (size_t landmark = 0; landmark < observers.size(); ++landmark) {
                const unsigned int action = next(10);
                if (!alive[landmark]) {
                    if (action < 4) {
                        alive[landmark] = true;
                        observers[landmark].clear();
                        const unsigned int count = 1 + next(4);
                        for (unsigned int i = 0; i < count; ++i) {
                            observers[landmark].push_back(static_cast<int>(next(12)));
                        }
                    }
                }
                else if (action == 0) {
                    alive[landmark] = false;
                }
                else if ((action < 4) && !observers[landmark].empty()) {
                    observers[landmark].erase(observers[landmark].begin() + static_cast<std::ptrdiff_t>(next(static_cast<unsigned int>(observers[landmark].size()))));
                }
                else if (action < 7) {
                    // Frames can repeat and arrive out of order.
                    observers[landmark].push_back(static_cast<int>(next(12)));
                }
            }
            incremental.begin_update();
            for (size_t landmark = 0; landmark < observers.size(); ++landmark) {
                if (alive[landmark]) {
                    incremental.update(static_cast<int>(landmark), observers[landmark].data(), observers[landmark].size());
                }
            }
            incremental.end_update();
            mapping::covisibility rebuilt;
            for (size_t landmark = 0; landmark < observers.size(); ++landmark) {
                if (alive[landmark]) {
                    rebuilt.add(observers[landmark].data(), observers[landmark].size());
                }
            }
            const std::vector<mapping::covisibility::edge> expected = rebuilt.edges();
            const std::vector<mapping::covisibility::edge> found = incremental.edges();
            REQUIRE(found.size() == expected.size());
            for (size_t i = 0; (i < found.size()) && (i < expected.size()); ++i) {
                REQUIRE(found[i].frame_a == expected[i].frame_a);
                REQUIRE(found[i].frame_b == expected[i].frame_b);
                REQUIRE(found[i].weight == expected[i].weight);
            }
            REQUIRE(incremental.num_frames() == rebuilt.num_frames());
        }
        incremental.remove(0);
        incremental.remove(1000);
        incremental.begin_update();
        incremental.end_update();
        REQUIRE(incremental.num_frames() == 0);
        REQUIRE(incremental.edges().empty());
    }

    return EXIT_SUCCESS;
}
