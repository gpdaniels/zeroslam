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

#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mapping {
    covisibility::covisibility()
        : weights() {
    }

    void covisibility::clear() {
        this->weights.clear();
    }

    void covisibility::add(const int* const frame_ids, const size_t frame_ids_size) {
        const auto is_repeat = [frame_ids](const size_t index) {
            for (size_t earlier = 0; earlier < index; ++earlier) {
                if (frame_ids[earlier] == frame_ids[index]) {
                    return true;
                }
            }
            return false;
        };
        for (size_t i = 0; i < frame_ids_size; ++i) {
            if (is_repeat(i)) {
                continue;
            }
            for (size_t j = i + 1; j < frame_ids_size; ++j) {
                if ((frame_ids[i] == frame_ids[j]) || is_repeat(j)) {
                    continue;
                }
                ++this->weights[frame_ids[i]][frame_ids[j]];
                ++this->weights[frame_ids[j]][frame_ids[i]];
            }
        }
    }

    int covisibility::weight(const int frame_a, const int frame_b) const {
        const std::unordered_map<int, std::unordered_map<int, int>>::const_iterator row = this->weights.find(frame_a);
        if (row == this->weights.end()) {
            return 0;
        }
        const std::unordered_map<int, int>::const_iterator entry = row->second.find(frame_b);
        return (entry == row->second.end()) ? 0 : entry->second;
    }

    std::vector<int> covisibility::neighbours(const int frame_id, const int minimum_weight) const {
        std::vector<int> found;
        const std::unordered_map<int, std::unordered_map<int, int>>::const_iterator row = this->weights.find(frame_id);
        if (row == this->weights.end()) {
            return found;
        }
        for (const auto& [other_id, shared] : row->second) {
            if (shared >= minimum_weight) {
                found.push_back(other_id);
            }
        }
        std::sort(found.begin(), found.end());
        return found;
    }

    std::vector<covisibility::edge> covisibility::edges() const {
        std::vector<edge> found;
        for (const auto& [frame_a, row] : this->weights) {
            for (const auto& [frame_b, shared] : row) {
                if (frame_a < frame_b) {
                    found.push_back(edge{ frame_a, frame_b, shared });
                }
            }
        }
        std::sort(found.begin(), found.end(), [](const edge& lhs, const edge& rhs) {
            return (lhs.frame_a != rhs.frame_a) ? (lhs.frame_a < rhs.frame_a) : (lhs.frame_b < rhs.frame_b);
        });
        return found;
    }

    size_t covisibility::num_frames() const {
        return this->weights.size();
    }
}
