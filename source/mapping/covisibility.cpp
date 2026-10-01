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
        : weights()
        , contributions()
        , scratch() {
    }

    void covisibility::clear() {
        this->weights.clear();
        this->contributions.clear();
    }

    void covisibility::change(const int frame_a, const int frame_b, const int delta) {
        for (int direction = 0; direction < 2; ++direction) {
            const int from = (direction == 0) ? frame_a : frame_b;
            const int to = (direction == 0) ? frame_b : frame_a;
            std::unordered_map<int, int>& row = this->weights[from];
            const int weight = (row[to] += delta);
            if (weight == 0) {
                row.erase(to);
                if (row.empty()) {
                    this->weights.erase(from);
                }
            }
        }
    }

    void covisibility::update(const int landmark_id, const int* const frame_ids, const size_t frame_ids_size) {
        std::vector<int>& current = this->scratch;
        current.assign(frame_ids, frame_ids + frame_ids_size);
        if (!std::is_sorted(current.begin(), current.end())) {
            std::sort(current.begin(), current.end());
        }
        current.erase(std::unique(current.begin(), current.end()), current.end());
        contribution& recorded = this->contributions[landmark_id];
        recorded.updated = true;
        if (recorded.frame_ids == current) {
            return;
        }
        // Pairs with a frame that left lose this landmark, pairs with a frame that joined gain it, and the rest are unchanged.
        const std::vector<int>& previous = recorded.frame_ids;
        const auto contains = [](const std::vector<int>& sorted, const int frame_id) {
            return std::binary_search(sorted.begin(), sorted.end(), frame_id);
        };
        for (const int left : previous) {
            if (contains(current, left)) {
                continue;
            }
            for (const int other : previous) {
                if ((other != left) && (contains(current, other) || (other > left))) {
                    this->change(left, other, -1);
                }
            }
        }
        for (const int joined : current) {
            if (contains(previous, joined)) {
                continue;
            }
            for (const int other : current) {
                if ((other != joined) && (contains(previous, other) || (other > joined))) {
                    this->change(joined, other, +1);
                }
            }
        }
        recorded.frame_ids = current;
    }

    void covisibility::remove(const int landmark_id) {
        const std::unordered_map<int, contribution>::iterator recorded = this->contributions.find(landmark_id);
        if (recorded == this->contributions.end()) {
            return;
        }
        const std::vector<int>& frame_ids = recorded->second.frame_ids;
        for (size_t i = 0; i < frame_ids.size(); ++i) {
            for (size_t j = i + 1; j < frame_ids.size(); ++j) {
                this->change(frame_ids[i], frame_ids[j], -1);
            }
        }
        this->contributions.erase(recorded);
    }

    void covisibility::begin_update() {
        for (auto& [landmark_id, recorded] : this->contributions) {
            static_cast<void>(landmark_id);
            recorded.updated = false;
        }
    }

    void covisibility::end_update() {
        std::vector<int> stale;
        for (const auto& [landmark_id, recorded] : this->contributions) {
            if (!recorded.updated) {
                stale.push_back(landmark_id);
            }
        }
        for (const int landmark_id : stale) {
            this->remove(landmark_id);
        }
    }

    void covisibility::add(const int* const frame_ids, const size_t frame_ids_size) {
        std::vector<int>& unique = this->scratch;
        unique.assign(frame_ids, frame_ids + frame_ids_size);
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        for (size_t i = 0; i < unique.size(); ++i) {
            for (size_t j = i + 1; j < unique.size(); ++j) {
                this->change(unique[i], unique[j], +1);
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
