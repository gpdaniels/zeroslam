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

#include "math/math.hpp"

namespace mapping {
    namespace {
        // Three 21-bit coordinates, each offset to be non-negative, in one key.
        long long pack(const long long x, const long long y, const long long z) {
            constexpr static const long long offset = voxel_map::coordinate_limit;
            return ((x + offset) << 42) | ((y + offset) << 21) | (z + offset);
        }
    }

    voxel_map::voxel_map()
        : voxel_map(1.0) {
    }

    voxel_map::voxel_map(const double voxel_size)
        : size(voxel_size)
        , voxels()
        , points(0) {
    }

    void voxel_map::clear(const double voxel_size) {
        this->size = voxel_size;
        this->voxels.clear();
        this->points = 0;
    }

    double voxel_map::voxel_size() const {
        return this->size;
    }

    size_t voxel_map::voxel_count() const {
        return this->voxels.size();
    }

    size_t voxel_map::point_count() const {
        return this->points;
    }

    const std::unordered_map<long long, voxel_map::voxel>& voxel_map::occupied() const {
        return this->voxels;
    }

    bool voxel_map::key_of(const math::matrix<double, 3, 1>& location, long long& key, int& x, int& y, int& z) const {
        if (!(this->size > 0.0)) {
            return false;
        }
        long long cell[3] = { 0, 0, 0 };
        for (size_t axis = 0; axis < 3; ++axis) {
            const double scaled = math::floor(location[axis] / this->size);
            if (!(scaled >= -static_cast<double>(voxel_map::coordinate_limit)) || !(scaled <= static_cast<double>(voxel_map::coordinate_limit))) {
                return false;
            }
            cell[axis] = static_cast<long long>(scaled);
        }
        key = pack(cell[0], cell[1], cell[2]);
        x = static_cast<int>(cell[0]);
        y = static_cast<int>(cell[1]);
        z = static_cast<int>(cell[2]);
        return true;
    }

    bool voxel_map::insert(const int id, const math::matrix<double, 3, 1>& location) {
        long long key = 0;
        int x = 0;
        int y = 0;
        int z = 0;
        if (!this->key_of(location, key, x, y, z)) {
            return false;
        }
        voxel& cell = this->voxels[key];
        if (cell.ids.empty()) {
            cell.x = x;
            cell.y = y;
            cell.z = z;
        }
        cell.ids.push_back(id);
        ++this->points;
        return true;
    }

    bool voxel_map::remove(const int id, const math::matrix<double, 3, 1>& location) {
        long long key = 0;
        int x = 0;
        int y = 0;
        int z = 0;
        if (!this->key_of(location, key, x, y, z)) {
            return false;
        }
        const std::unordered_map<long long, voxel>::iterator found = this->voxels.find(key);
        if (found == this->voxels.end()) {
            return false;
        }
        std::vector<int>& ids = found->second.ids;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] != id) {
                continue;
            }
            ids[i] = ids.back();
            ids.pop_back();
            --this->points;
            if (ids.empty()) {
                this->voxels.erase(found);
            }
            return true;
        }
        return false;
    }

    void voxel_map::cast(const math::matrix<double, 3, 1>& centre, const math::matrix<double, 3, 3>& camera_to_world, const math::matrix<double, 3, 1>* const rays, const size_t ray_count, const double max_distance, const size_t occluding_points, std::vector<int>& ids) const {
        long long start_key = 0;
        int start[3] = { 0, 0, 0 };
        if (this->voxels.empty() || !(max_distance > 0.0) || !this->key_of(centre, start_key, start[0], start[1], start[2])) {
            return;
        }
        std::unordered_set<long long> collected;
        for (size_t r = 0; r < ray_count; ++r) {
            const math::matrix<double, 3, 1> direction = camera_to_world * rays[r];
            // Amanatides and Woo's traversal: the distance along the ray to the next boundary on each axis, and between boundaries.
            int cell[3] = { start[0], start[1], start[2] };
            int step[3] = { 0, 0, 0 };
            double next[3] = { 0.0, 0.0, 0.0 };
            double delta[3] = { 0.0, 0.0, 0.0 };
            for (size_t axis = 0; axis < 3; ++axis) {
                if (direction[axis] > 0.0) {
                    step[axis] = 1;
                    next[axis] = ((static_cast<double>(cell[axis] + 1) * this->size) - centre[axis]) / direction[axis];
                    delta[axis] = this->size / direction[axis];
                }
                else if (direction[axis] < 0.0) {
                    step[axis] = -1;
                    next[axis] = ((static_cast<double>(cell[axis]) * this->size) - centre[axis]) / direction[axis];
                    delta[axis] = -this->size / direction[axis];
                }
                else {
                    next[axis] = 1.0e300;
                    delta[axis] = 1.0e300;
                }
            }
            size_t crossed = 0;
            double travelled = 0.0;
            while (travelled <= max_distance) {
                if ((cell[0] < -voxel_map::coordinate_limit) || (cell[0] > voxel_map::coordinate_limit) || (cell[1] < -voxel_map::coordinate_limit) || (cell[1] > voxel_map::coordinate_limit) || (cell[2] < -voxel_map::coordinate_limit) || (cell[2] > voxel_map::coordinate_limit)) {
                    break;
                }
                const long long key = pack(cell[0], cell[1], cell[2]);
                const std::unordered_map<long long, voxel>::const_iterator found = this->voxels.find(key);
                if (found != this->voxels.end()) {
                    if (collected.insert(key).second) {
                        ids.insert(ids.end(), found->second.ids.begin(), found->second.ids.end());
                    }
                    crossed += found->second.ids.size();
                    if (crossed >= occluding_points) {
                        break;
                    }
                }
                const size_t axis = (next[0] < next[1]) ? ((next[0] < next[2]) ? 0 : 2) : ((next[1] < next[2]) ? 1 : 2);
                travelled = next[axis];
                next[axis] += delta[axis];
                cell[axis] += step[axis];
            }
        }
    }
}
