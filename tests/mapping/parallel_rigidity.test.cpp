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

#include "mapping/parallel_rigidity.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

namespace {
    using rigidity = mapping::parallel_rigidity;
    using observations = std::vector<std::pair<int, int>>;

    rigidity::track seen_by(const int point, const std::vector<int>& frames) {
        rigidity::track result;
        result.point = point;
        result.frames = frames;
        return result;
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // Nothing has no components.
    {
        const rigidity::result result = rigidity::analyse({});
        REQUIRE(result.components.empty());
        REQUIRE(result.unlooped.empty());
    }

    // Two frames that both see two points are a four-loop, and rigid.
    {
        const rigidity::result result = rigidity::analyse({ seen_by(10, { 1, 2 }), seen_by(11, { 2, 1 }) });
        REQUIRE(result.components.size() == 1);
        REQUIRE(result.components[0].frames == std::vector<int>({ 1, 2 }));
        REQUIRE(result.components[0].points == std::vector<int>({ 10, 11 }));
        REQUIRE(result.unlooped.empty());
    }

    // A point two frames share alone is in no four-loop, and neither is a single observation of a point.
    {
        const rigidity::result result = rigidity::analyse({ seen_by(10, { 1, 2 }), seen_by(11, { 1, 2, 3 }), seen_by(12, { 4 }), seen_by(13, { 3, 5 }) });
        REQUIRE(result.components.size() == 1);
        REQUIRE(result.components[0].frames == std::vector<int>({ 1, 2 }));
        REQUIRE(result.components[0].points == std::vector<int>({ 10, 11 }));
        REQUIRE(result.unlooped == observations({ { 11, 3 }, { 12, 4 }, { 13, 3 }, { 13, 5 } }));
    }

    // Type 1 (the paper's figure 3a): loops sharing a frame and a point merge.
    {
        const rigidity::result result = rigidity::analyse({ seen_by(1, { 1, 2 }), seen_by(2, { 1, 2, 3 }), seen_by(3, { 2, 3 }) });
        REQUIRE(result.components.size() == 1);
        REQUIRE(result.components[0].frames == std::vector<int>({ 1, 2, 3 }));
        REQUIRE(result.components[0].points == std::vector<int>({ 1, 2, 3 }));
        REQUIRE(result.unlooped.empty());
    }

    // Type 2 (figure 3c): loops sharing two frames are one edge.
    {
        const rigidity::result result = rigidity::analyse({ seen_by(1, { 1, 2 }), seen_by(2, { 1, 2 }), seen_by(3, { 1, 2 }), seen_by(4, { 1, 2 }) });
        REQUIRE(result.components.size() == 1);
        REQUIRE(result.components[0].points == std::vector<int>({ 1, 2, 3, 4 }));
    }

    // Loops sharing only a frame can be scaled about it independently, so they stay apart, the frame in both.
    {
        const rigidity::result result = rigidity::analyse({ seen_by(1, { 1, 2 }), seen_by(2, { 1, 2 }), seen_by(3, { 2, 3 }), seen_by(4, { 2, 3 }) });
        REQUIRE(result.components.size() == 2);
        REQUIRE(result.components[0].frames == std::vector<int>({ 1, 2 }));
        REQUIRE(result.components[1].frames == std::vector<int>({ 2, 3 }));
        REQUIRE(result.components[0].points == std::vector<int>({ 1, 2 }));
        REQUIRE(result.components[1].points == std::vector<int>({ 3, 4 }));
    }

    // Loops joined only through a point a frame of each sees stay apart, and that point is in no loop.
    {
        const rigidity::result result = rigidity::analyse({ seen_by(1, { 1, 2 }), seen_by(2, { 1, 2 }), seen_by(3, { 3, 4 }), seen_by(4, { 3, 4 }), seen_by(5, { 2, 3 }) });
        REQUIRE(result.components.size() == 2);
        REQUIRE(result.unlooped == observations({ { 5, 2 }, { 5, 3 } }));
    }

    // Type 3: subgraphs whose frames share no two points, but which share two points between them, merge.
    {
        // Frames 1, 2, 5 and 6 are one subgraph through points 20 and 21, and frames 3 and 4 another; points 10 and
        // 11 are seen by both, by different frames of the first.
        std::vector<rigidity::track> tracks = {
            seen_by(10, { 1, 2, 3, 4 }),
            seen_by(11, { 5, 6, 3, 4 }),
            seen_by(12, { 1, 2 }),
            seen_by(13, { 5, 6 }),
            seen_by(20, { 1, 2, 5 }),
            seen_by(21, { 2, 5, 6 }),
        };
        const rigidity::result joined = rigidity::analyse(tracks);
        REQUIRE(joined.components.size() == 1);
        REQUIRE(joined.components[0].frames == std::vector<int>({ 1, 2, 3, 4, 5, 6 }));
        REQUIRE(joined.components[0].points == std::vector<int>({ 10, 11, 12, 13, 20, 21 }));
        REQUIRE(joined.unlooped.empty());
        // With one point in common they stay apart: frames 3 and 4 see point 14 instead of 11.
        tracks[1] = seen_by(11, { 5, 6 });
        tracks.push_back(seen_by(14, { 3, 4 }));
        const rigidity::result apart = rigidity::analyse(tracks);
        REQUIRE(apart.components.size() == 2);
        REQUIRE(apart.components[0].frames == std::vector<int>({ 1, 2, 5, 6 }));
        REQUIRE(apart.components[1].frames == std::vector<int>({ 3, 4 }));
        REQUIRE(apart.components[1].points == std::vector<int>({ 10, 14 }));
    }

    // The result does not depend on the order of the tracks or of their frames.
    {
        std::vector<rigidity::track> tracks = {
            seen_by(1, { 1, 2 }),
            seen_by(2, { 1, 2, 3 }),
            seen_by(3, { 2, 3 }),
            seen_by(4, { 7, 8 }),
            seen_by(5, { 8, 7 }),
            seen_by(6, { 3, 9 }),
        };
        const rigidity::result forward = rigidity::analyse(tracks);
        std::reverse(tracks.begin(), tracks.end());
        for (rigidity::track& seen : tracks) {
            std::reverse(seen.frames.begin(), seen.frames.end());
        }
        const rigidity::result backward = rigidity::analyse(tracks);
        REQUIRE(forward.components.size() == 2);
        REQUIRE(forward.components.size() == backward.components.size());
        for (size_t index = 0; index < forward.components.size(); ++index) {
            REQUIRE(forward.components[index].frames == backward.components[index].frames);
            REQUIRE(forward.components[index].points == backward.components[index].points);
        }
        REQUIRE(forward.unlooped == backward.unlooped);
        REQUIRE(forward.unlooped == observations({ { 6, 3 }, { 6, 9 } }));
    }

    return EXIT_SUCCESS;
}
