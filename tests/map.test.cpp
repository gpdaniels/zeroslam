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

#include "map.hpp"

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

    // Basic insert / lookup tests.
    {
        map::map m;

        frame::frame f;
        f.id = 1;
        REQUIRE(m.frames.find(f.id) == m.frames.end());
        m.add_frame(f);
        auto itf = m.frames.find(f.id);
        REQUIRE(itf != m.frames.end());
        REQUIRE(itf->second.id == f.id);

        landmark::point p;
        p.id = 10;
        REQUIRE(m.landmarks.find(p.id) == m.landmarks.end());
        m.add_landmark(p);
        auto itp = m.landmarks.find(p.id);
        REQUIRE(itp != m.landmarks.end());
        REQUIRE(itp->second.id == p.id);

        const size_t kp_index = 3;
        m.add_observation(f, p, kp_index);
        auto ito = m.observations.find(p.id);
        REQUIRE(ito != m.observations.end());
        REQUIRE(ito->second.size() == 1);
        REQUIRE(ito->second[0].first == f.id);
        REQUIRE(ito->second[0].second == kp_index);

        frame::frame f2;
        f2.id = 2;
        m.add_frame(f2);

        const size_t kp_index2 = 5;
        m.add_observation(f2, p, kp_index2);

        auto ito2 = m.observations.find(p.id);
        REQUIRE(ito2 != m.observations.end());
        REQUIRE(ito2->second.size() == 2);
        REQUIRE(ito2->second[0].first == f.id);
        REQUIRE(ito2->second[0].second == kp_index);
        REQUIRE(ito2->second[1].first == f2.id);
        REQUIRE(ito2->second[1].second == kp_index2);

        REQUIRE(m.frames.size() >= 2);
        REQUIRE(m.landmarks.find(p.id) != m.landmarks.end());
    }

    // Edge cases: repeated add_frame / add_landmark with same id should overwrite existing entry.
    {
        map::map m;
        frame::frame f;
        f.id = 100;
        m.add_frame(f);

        frame::frame f_modified;
        f_modified.id = 100;
        m.add_frame(f_modified);
        REQUIRE(m.frames.find(100) != m.frames.end());
        REQUIRE(m.frames.at(100).id == 100);

        landmark::point p;
        p.id = 200;
        m.add_landmark(p);
        landmark::point p_modified;
        p_modified.id = 200;
        m.add_landmark(p_modified);
        REQUIRE(m.landmarks.find(200) != m.landmarks.end());
        REQUIRE(m.landmarks.at(200).id == 200);
    }

    // Verify optimise() runs safely even with trivial data.
    {
        map::map m;
        frame::frame f1;
        f1.id = 0;
        f1.keypoint_pyramid = { { { feature::point{ 0.0f, 0.0f, 0.0f, 0.0f } } } };
        frame::frame f2;
        f2.id = 1;
        f2.keypoint_pyramid = { { { feature::point{ 0.0f, 0.0f, 0.0f, 0.0f } } } };
        landmark::point l1;
        l1.id = 0;
        m.add_frame(f1);
        m.add_frame(f2);
        m.add_landmark(l1);
        m.add_observation(f1, l1, 0);
        m.add_observation(f2, l1, 0);
        m.optimise(2, false, 3);
    }

    // Verify cull() removes bad landmarks.
    {
        map::map m;
        frame::frame f1;
        f1.id = 0;
        f1.keypoint_pyramid = { { { feature::point{ 0.0f, 0.0f, 0.0f, 0.0f } } } };
        landmark::point l1;
        l1.id = 0;
        m.add_frame(f1);
        m.add_landmark(l1);
        m.add_observation(f1, l1, 0);
        const size_t before = m.landmarks.size();
        m.cull();
        const size_t after = m.landmarks.size();
        REQUIRE(before > after);
    }

    return EXIT_SUCCESS;
}