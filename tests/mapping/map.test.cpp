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

#include "mapping/map.hpp"

#include "math/lie.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
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

    // Basic insert / lookup tests.
    {
        mapping::map m;

        mapping::frame f;
        f.id = 1;
        {
            std::vector<feature::point> kps;
            for (int k = 0; k < 6; ++k) {
                kps.push_back(feature::point{ static_cast<float>(10 * k), static_cast<float>(10 * k + 1), 0.0f, 0.0f, 0 });
            }
            f.keypoints = kps;
        }
        REQUIRE(m.frames.find(f.id) == m.frames.end());
        m.add_frame(f);
        auto itf = m.frames.find(f.id);
        REQUIRE(itf != m.frames.end());
        REQUIRE(itf->second.id == f.id);

        mapping::point p;
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
        REQUIRE(ito->second[0].frame_id == f.id);
        REQUIRE(ito->second[0].kp_index == kp_index);
        REQUIRE(ito->second[0].point[0] == 30.0);
        REQUIRE(ito->second[0].point[1] == 31.0);

        mapping::frame f2;
        f2.id = 2;
        {
            std::vector<feature::point> kps;
            for (int k = 0; k < 6; ++k) {
                kps.push_back(feature::point{ static_cast<float>(10 * k), static_cast<float>(10 * k + 1), 0.0f, 0.0f, 0 });
            }
            f2.keypoints = kps;
        }
        m.add_frame(f2);

        const size_t kp_index2 = 5;
        m.add_observation(f2, p, kp_index2);
        m.add_observation(99, p, 1.0, 2.0);
        m.add_observation(98, p, 1.0, 2.0);
        m.observations.at(p.id).resize(2);

        auto ito2 = m.observations.find(p.id);
        REQUIRE(ito2 != m.observations.end());
        REQUIRE(ito2->second.size() == 2);
        REQUIRE(ito2->second[0].frame_id == f.id);
        REQUIRE(ito2->second[0].kp_index == kp_index);
        REQUIRE(ito2->second[1].frame_id == f2.id);
        REQUIRE(ito2->second[1].kp_index == kp_index2);
        REQUIRE(ito2->second[1].point[0] == 50.0);
        REQUIRE(ito2->second[1].point[1] == 51.0);

        REQUIRE(m.frames.size() >= 2);
        REQUIRE(m.landmarks.find(p.id) != m.landmarks.end());
    }
    {
        mapping::map m;
        mapping::frame f;
        f.id = 100;
        m.add_frame(f);

        mapping::frame f_modified;
        f_modified.id = 100;
        m.add_frame(f_modified);
        REQUIRE(m.frames.find(100) != m.frames.end());
        REQUIRE(m.frames.at(100).id == 100);

        mapping::point p;
        p.id = 200;
        m.add_landmark(p);
        mapping::point p_modified;
        p_modified.id = 200;
        m.add_landmark(p_modified);
        REQUIRE(m.landmarks.find(200) != m.landmarks.end());
        REQUIRE(m.landmarks.at(200).id == 200);
    }
    {
        mapping::map m;
        mapping::frame f1;
        f1.id = 0;
        f1.keypoints = { feature::point{ 0.0f, 0.0f, 0.0f, 0.0f, 0 } };
        mapping::frame f2;
        f2.id = 1;
        f2.keypoints = { feature::point{ 0.0f, 0.0f, 0.0f, 0.0f, 0 } };
        mapping::point l1;
        l1.id = 0;
        m.add_frame(f1);
        m.add_frame(f2);
        m.add_landmark(l1);
        m.add_observation(f1, l1, 0);
        m.add_observation(f2, l1, 0);
        m.optimise(2, false, 3);
    }
    {
        mapping::map m;
        mapping::frame f1;
        f1.id = 0;
        f1.keypoints = { feature::point{ 0.0f, 0.0f, 0.0f, 0.0f, 0 } };
        mapping::point l1;
        l1.id = 0;
        m.add_frame(f1);
        m.add_landmark(l1);
        m.add_observation(f1, l1, 0);
        const size_t before = m.landmarks.size();
        m.cull();
        const size_t after = m.landmarks.size();
        REQUIRE(before > after);
    }
    {
        mapping::map m;
        REQUIRE(m.frames.empty());
        m.optimise(10, false, 50);
        m.optimise(1, true, 50);
        REQUIRE(m.frames.empty());
    }
    {
        mapping::map m;
        mapping::point p;
        p.id = 42;
        m.add_landmark(p);
        REQUIRE(m.observations.find(p.id) == m.observations.end());
        REQUIRE(m.landmarks.find(p.id) != m.landmarks.end());
        m.cull();
        REQUIRE(m.landmarks.find(p.id) == m.landmarks.end());
    }

    return EXIT_SUCCESS;
}
