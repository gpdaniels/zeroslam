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
#ifndef ZEROSLAM_MAPPING_PLACE_RECOGNITION_HPP
#define ZEROSLAM_MAPPING_PLACE_RECOGNITION_HPP

#include "feature/descriptor/binary.hpp"
#include "match/index/hbst.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace mapping {
    class place_recognition final {
    public:
        class candidate final {
        public:
            int keyframe_id;
            size_t votes;
            float score;
            std::vector<size_t> query_indices;
        };

    private:
        match::index::hbst index;
        std::vector<int> keyframe_ids;
        unsigned int max_distance;

    public:
        explicit place_recognition(const unsigned int distance_threshold = 40);

    public:
        void add_keyframe(const int keyframe_id, const feature::descriptor::binary<256>* const descriptors, const size_t descriptors_size);
        void remove_keyframe(const int keyframe_id);
        void clear();
        size_t num_keyframes() const;

        std::vector<candidate> get_candidates(
            const feature::descriptor::binary<256>* const query_descriptors,
            const size_t query_descriptors_size,
            const int current_keyframe_id,
            const size_t max_candidates
        ) const;
    };
}

#endif // ZEROSLAM_MAPPING_PLACE_RECOGNITION_HPP
