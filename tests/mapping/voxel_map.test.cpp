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

#include "mapping/voxel_map.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

namespace {
    // Unit rays over a square field of view of half_angle radians about +z, count by count.
    std::vector<math::matrix<double, 3, 1>> fan(const double half_angle, const int count) {
        std::vector<math::matrix<double, 3, 1>> rays;
        for (int v = 0; v < count; ++v) {
            for (int u = 0; u < count; ++u) {
                const double x = std::tan(half_angle) * ((2.0 * (static_cast<double>(u) + 0.5) / static_cast<double>(count)) - 1.0);
                const double y = std::tan(half_angle) * ((2.0 * (static_cast<double>(v) + 0.5) / static_cast<double>(count)) - 1.0);
                const double length = std::sqrt((x * x) + (y * y) + 1.0);
                rays.push_back(math::matrix<double, 3, 1>({ x / length, y / length, 1.0 / length }));
            }
        }
        return rays;
    }

    bool contains(const std::vector<int>& ids, const int id) {
        return std::find(ids.begin(), ids.end(), id) != ids.end();
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        // Keys floor every axis, so negative coordinates fall in their own voxels, and a location beyond the grid is refused.
        mapping::voxel_map map(2.0);
        long long key_a = 0;
        long long key_b = 0;
        int x = 0;
        int y = 0;
        int z = 0;
        REQUIRE(map.key_of(math::matrix<double, 3, 1>({ -0.5, 0.5, 3.9 }), key_a, x, y, z));
        REQUIRE((x == -1) && (y == 0) && (z == 1));
        REQUIRE(map.key_of(math::matrix<double, 3, 1>({ 0.5, 0.5, 3.9 }), key_b, x, y, z));
        REQUIRE((x == 0) && (key_a != key_b));
        REQUIRE(!map.key_of(math::matrix<double, 3, 1>({ 1.0e9, 0.0, 0.0 }), key_a, x, y, z));
        REQUIRE(!map.key_of(math::matrix<double, 3, 1>({ std::nan(""), 0.0, 0.0 }), key_a, x, y, z));
        REQUIRE(!map.insert(7, math::matrix<double, 3, 1>({ 1.0e9, 0.0, 0.0 })));

        REQUIRE(map.insert(1, math::matrix<double, 3, 1>({ 0.5, 0.5, 0.5 })));
        REQUIRE(map.insert(2, math::matrix<double, 3, 1>({ 1.5, 1.5, 1.5 })));
        REQUIRE(map.insert(3, math::matrix<double, 3, 1>({ -1.5, 0.5, 0.5 })));
        REQUIRE((map.voxel_count() == 2) && (map.point_count() == 3));
        REQUIRE(!map.remove(9, math::matrix<double, 3, 1>({ 0.5, 0.5, 0.5 })));
        REQUIRE(map.remove(3, math::matrix<double, 3, 1>({ -1.5, 0.5, 0.5 })));
        REQUIRE((map.voxel_count() == 1) && (map.point_count() == 2));
        map.clear(0.5);
        REQUIRE((map.voxel_count() == 0) && (map.point_count() == 0) && (map.voxel_size() == 0.5));
    }

    {
        // A wall of points 10 ahead, a dense block in front of its middle at 5, a lone point in front of its left at 4,
        // points behind the camera, off to the side, and beyond the range.
        mapping::voxel_map map(1.0);
        int id = 0;
        std::vector<int> wall_middle;
        std::vector<int> wall_left;
        for (int u = -40; u <= 40; ++u) {
            for (int v = -40; v <= 40; ++v) {
                const double x = 0.1 * static_cast<double>(u) + 0.05;
                const double y = 0.1 * static_cast<double>(v) + 0.05;
                REQUIRE(map.insert(id, math::matrix<double, 3, 1>({ x, y, 10.5 })));
                if ((std::abs(x) < 0.5) && (std::abs(y) < 0.5)) {
                    wall_middle.push_back(id);
                }
                if ((x < -2.5) && (x > -3.5) && (std::abs(y) < 0.5)) {
                    wall_left.push_back(id);
                }
                ++id;
            }
        }
        std::vector<int> block;
        for (int u = -6; u < 6; ++u) {
            for (int v = -6; v < 6; ++v) {
                REQUIRE(map.insert(id, math::matrix<double, 3, 1>({ 0.1 * static_cast<double>(u) + 0.05, 0.1 * static_cast<double>(v) + 0.05, 5.5 })));
                block.push_back(id++);
            }
        }
        const int lone = id++;
        REQUIRE(map.insert(lone, math::matrix<double, 3, 1>({ -2.9, 0.1, 4.5 })));
        const int behind = id++;
        REQUIRE(map.insert(behind, math::matrix<double, 3, 1>({ 0.0, 0.0, -5.0 })));
        const int aside = id++;
        REQUIRE(map.insert(aside, math::matrix<double, 3, 1>({ 30.0, 0.0, 5.0 })));
        const int distant = id++;
        REQUIRE(map.insert(distant, math::matrix<double, 3, 1>({ 0.0, 2.0, 60.0 })));

        const std::vector<math::matrix<double, 3, 1>> rays = fan(0.4, 48);
        const math::matrix<double, 3, 1> centre({ 0.0, 0.0, 0.0 });
        const math::matrix<double, 3, 3> identity = math::matrix<double, 3, 3>::identity();

        std::vector<int> seen;
        map.cast(centre, identity, rays.data(), rays.size(), 50.0, 10, seen);
        const std::unordered_set<int> unique(seen.begin(), seen.end());
        REQUIRE(unique.size() == seen.size());
        for (const int b : block) {
            REQUIRE(contains(seen, b));
        }
        // The block hides the wall behind it, while the lone point does not hide the wall behind it.
        for (const int w : wall_middle) {
            REQUIRE(!contains(seen, w));
        }
        for (const int w : wall_left) {
            REQUIRE(contains(seen, w));
        }
        REQUIRE(contains(seen, lone));
        REQUIRE(!contains(seen, behind));
        REQUIRE(!contains(seen, aside));
        REQUIRE(!contains(seen, distant));

        // Without occlusion the wall behind the block is reached too, and the range limit stops short of the wall.
        seen.clear();
        map.cast(centre, identity, rays.data(), rays.size(), 50.0, static_cast<size_t>(-1), seen);
        for (const int w : wall_middle) {
            REQUIRE(contains(seen, w));
        }
        seen.clear();
        map.cast(centre, identity, rays.data(), rays.size(), 8.0, static_cast<size_t>(-1), seen);
        REQUIRE(contains(seen, block.front()));
        REQUIRE(!contains(seen, wall_left.front()));

        // Turned to look along +x, the camera sees the point to the side and none of the wall.
        const math::matrix<double, 3, 3> look_x({ { 0.0, 0.0, 1.0 }, { 0.0, 1.0, 0.0 }, { -1.0, 0.0, 0.0 } });
        seen.clear();
        map.cast(centre, look_x, rays.data(), rays.size(), 50.0, 10, seen);
        REQUIRE(contains(seen, aside));
        REQUIRE(!contains(seen, wall_left.front()));
        REQUIRE(!contains(seen, block.front()));

        // An empty map and a zero range return nothing.
        seen.clear();
        map.cast(centre, identity, rays.data(), rays.size(), 0.0, 10, seen);
        REQUIRE(seen.empty());
        mapping::voxel_map empty(1.0);
        empty.cast(centre, identity, rays.data(), rays.size(), 50.0, 10, seen);
        REQUIRE(seen.empty());
    }
    return EXIT_SUCCESS;
}
