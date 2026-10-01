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
#ifndef ZEROSLAM_MAPPING_COVISIBILITY_HPP
#define ZEROSLAM_MAPPING_COVISIBILITY_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace mapping {
    class covisibility final {
    private:
        struct contribution final {
            std::vector<int> frame_ids;
            bool updated = false;
        };

        std::unordered_map<int, std::unordered_map<int, int>> weights;
        std::unordered_map<int, contribution> contributions;
        std::vector<int> scratch;

        void change(const int frame_a, const int frame_b, const int delta);

    public:
        covisibility();

    public:
        void clear();

        // Count one co-observation for every pair of distinct frames, from a landmark that is not tracked by id.
        void add(const int* const frame_ids, const size_t frame_ids_size);

        // Bring one landmark's contribution in line with the frames observing it now, changing only the pairs that differ.
        void update(const int landmark_id, const int* const frame_ids, const size_t frame_ids_size);

        // Withdraw a landmark's contribution.
        void remove(const int landmark_id);

        // Between the two, update every landmark that still exists; end_update removes the contributions of the others.
        void begin_update();
        void end_update();

        int weight(const int frame_a, const int frame_b) const;

        std::vector<int> neighbours(const int frame_id, const int minimum_weight) const;

        struct edge final {
            int frame_a;
            int frame_b;
            int weight;
        };

        std::vector<edge> edges() const;

        size_t num_frames() const;
    };
}

#endif // ZEROSLAM_MAPPING_COVISIBILITY_HPP
