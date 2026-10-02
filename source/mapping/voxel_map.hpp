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

#pragma once
#ifndef ZEROSLAM_MAPPING_VOXEL_MAP_HPP
#define ZEROSLAM_MAPPING_VOXEL_MAP_HPP

#include "math/matrix.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace mapping {
    // Landmarks held in a hashed regular grid of voxels (Muglikar, Zhang and Scaramuzza, "Voxel Map for Visual SLAM",
    // ICRA 2020). The landmarks a camera can see are those in the voxels its rays reach, found by marching a fixed set of
    // rays through the grid voxel by voxel, at a cost that depends on the view and not on the size of the map; a ray stops
    // once the voxels it has crossed hold enough landmarks to hide what lies behind them.
    class voxel_map final {
    public:
        // Each voxel coordinate is held in 21 bits, so a location further than this many voxels from the origin along an axis
        // is outside the grid.
        constexpr static const long long coordinate_limit = (1LL << 20) - 1;

        struct voxel final {
            int x;
            int y;
            int z;
            std::vector<int> ids;
        };

    private:
        double size;
        std::unordered_map<long long, voxel> voxels;
        size_t points;

    public:
        voxel_map();

        explicit voxel_map(const double voxel_size);

        // Empties the map and sets the voxel size.
        void clear(const double voxel_size);

        double voxel_size() const;

        size_t voxel_count() const;

        size_t point_count() const;

        const std::unordered_map<long long, voxel>& occupied() const;

        // The key of the voxel holding a location, or false when the location is not finite or lies outside the grid.
        bool key_of(const math::matrix<double, 3, 1>& location, long long& key, int& x, int& y, int& z) const;

        // Adds an id at a location, returning false when the location is outside the grid.
        bool insert(const int id, const math::matrix<double, 3, 1>& location);

        // Removes an id from the voxel holding a location, returning false when it is not there.
        bool remove(const int id, const math::matrix<double, 3, 1>& location);

        // Marches each ray (a unit direction in the camera frame, turned into the world by camera_to_world) from the camera
        // centre through the voxels it crosses, up to max_distance, and appends to ids the ids of every voxel a ray reaches,
        // each voxel once. A ray stops after crossing voxels that hold occluding_points ids or more between them.
        void cast(const math::matrix<double, 3, 1>& centre, const math::matrix<double, 3, 3>& camera_to_world, const math::matrix<double, 3, 1>* const rays, const size_t ray_count, const double max_distance, const size_t occluding_points, std::vector<int>& ids) const;
    };
}

#endif // ZEROSLAM_MAPPING_VOXEL_MAP_HPP
