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

#include "mapping/place_recognition.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mapping {
    place_recognition::place_recognition(const unsigned int distance_threshold)
        : index()
        , keyframe_ids()
        , max_distance(distance_threshold) {
    }

    void place_recognition::add_keyframe(const int keyframe_id, const feature::descriptor::binary<256>* const descriptors, const size_t descriptors_size) {
        if (descriptors_size == 0) {
            return;
        }
        this->keyframe_ids.push_back(keyframe_id);
        this->index.insert(keyframe_id, descriptors, descriptors_size);
    }

    void place_recognition::remove_keyframe(const int keyframe_id) {
        const std::vector<int>::iterator end = std::remove(this->keyframe_ids.begin(), this->keyframe_ids.end(), keyframe_id);
        if (end == this->keyframe_ids.end()) {
            return;
        }
        this->keyframe_ids.erase(end, this->keyframe_ids.end());
        this->index.remove(keyframe_id);
    }

    void place_recognition::clear() {
        this->keyframe_ids.clear();
        this->index.clear();
    }

    size_t place_recognition::num_keyframes() const {
        return this->keyframe_ids.size();
    }

    std::vector<place_recognition::candidate> place_recognition::get_candidates(
        const feature::descriptor::binary<256>* const query_descriptors,
        const size_t query_descriptors_size,
        const int current_keyframe_id,
        const size_t max_candidates
    ) const {
        std::vector<candidate> candidates;
        for (size_t query_index = 0; query_index < query_descriptors_size; ++query_index) {
            for (const match::index::hbst::hit& found : this->index.search(query_descriptors[query_index], this->max_distance)) {
                if (found.keyframe_id == current_keyframe_id) {
                    continue;
                }
                size_t candidate_index = 0;
                while ((candidate_index < candidates.size()) && (candidates[candidate_index].keyframe_id != found.keyframe_id)) {
                    ++candidate_index;
                }
                if (candidate_index == candidates.size()) {
                    candidates.push_back(candidate{ found.keyframe_id, 0, 0.0f, {} });
                }
                ++candidates[candidate_index].votes;
                candidates[candidate_index].query_indices.push_back(query_index);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const candidate& lhs, const candidate& rhs) {
            if (lhs.votes != rhs.votes) {
                return lhs.votes > rhs.votes;
            }
            return lhs.keyframe_id < rhs.keyframe_id;
        });
        if (candidates.size() > max_candidates) {
            candidates.resize(max_candidates);
        }
        for (candidate& ranked : candidates) {
            ranked.score = static_cast<float>(ranked.votes) / static_cast<float>(query_descriptors_size);
        }
        return candidates;
    }
}
