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

#include "match/index/hbst.hpp"

#include "match/distance/hamming.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace match::index {
    hbst::hbst()
        : nodes(1)
        , entry_count(0) {
    }

    bool hbst::get_bit(const feature::descriptor::binary<descriptor_bits>& descriptor, const size_t bit) {
        return ((static_cast<unsigned int>(descriptor[bit / 8]) >> (bit % 8)) & 1u) != 0;
    }

    void hbst::clear() {
        this->nodes.assign(1, node());
        this->entry_count = 0;
    }

    size_t hbst::size() const {
        return this->entry_count;
    }

    bool hbst::empty() const {
        return this->entry_count == 0;
    }

    void hbst::insert(const int keyframe_id, const feature::descriptor::binary<descriptor_bits>* const descriptors, const size_t descriptors_size) {
        for (size_t descriptor_index = 0; descriptor_index < descriptors_size; ++descriptor_index) {
            this->insert(entry{ descriptors[descriptor_index], keyframe_id, descriptor_index });
        }
    }

    void hbst::insert(const entry& stored) {
        const size_t leaf_index = this->find_leaf(stored.descriptor);
        this->nodes[leaf_index].entries.push_back(stored);
        ++this->entry_count;
        if (this->nodes[leaf_index].entries.size() > hbst::leaf_capacity) {
            this->split(leaf_index);
        }
    }

    void hbst::remove(const int keyframe_id) {
        const auto belongs_to_keyframe = [keyframe_id](const entry& stored) {
            return stored.keyframe_id == keyframe_id;
        };
        for (node& current : this->nodes) {
            const size_t before = current.entries.size();
            current.entries.erase(std::remove_if(current.entries.begin(), current.entries.end(), belongs_to_keyframe), current.entries.end());
            this->entry_count -= before - current.entries.size();
        }
    }

    std::vector<hbst::hit> hbst::search(const feature::descriptor::binary<descriptor_bits>& query, const unsigned int max_distance) const {
        std::vector<hit> hits;
        const node& leaf = this->nodes[this->find_leaf(query)];
        for (const entry& stored : leaf.entries) {
            const unsigned int distance = distance::hamming::distance(query, stored.descriptor);
            if (distance >= max_distance) {
                continue;
            }
            bool merged = false;
            for (hit& existing : hits) {
                if (existing.keyframe_id != stored.keyframe_id) {
                    continue;
                }
                if ((distance < existing.distance) || ((distance == existing.distance) && (stored.descriptor_index < existing.descriptor_index))) {
                    existing.descriptor_index = stored.descriptor_index;
                    existing.distance = distance;
                }
                merged = true;
                break;
            }
            if (!merged) {
                hits.push_back(hit{ stored.keyframe_id, stored.descriptor_index, distance });
            }
        }
        std::sort(hits.begin(), hits.end(), [](const hit& lhs, const hit& rhs) {
            if (lhs.distance != rhs.distance) {
                return lhs.distance < rhs.distance;
            }
            return lhs.keyframe_id < rhs.keyframe_id;
        });
        return hits;
    }

    size_t hbst::find_leaf(const feature::descriptor::binary<descriptor_bits>& descriptor) const {
        size_t index = 0;
        while (this->nodes[index].split_bit >= 0) {
            index = this->nodes[index].child[hbst::get_bit(descriptor, static_cast<size_t>(this->nodes[index].split_bit)) ? 1 : 0];
        }
        return index;
    }

    void hbst::split(const size_t leaf_index) {
        const size_t count = this->nodes[leaf_index].entries.size();
        size_t ones[hbst::descriptor_bits] = {};
        for (const entry& stored : this->nodes[leaf_index].entries) {
            for (size_t bit = 0; bit < hbst::descriptor_bits; ++bit) {
                ones[bit] += hbst::get_bit(stored.descriptor, bit) ? 1u : 0u;
            }
        }
        int best_bit = -1;
        size_t best_imbalance = count;
        for (size_t bit = 0; bit < hbst::descriptor_bits; ++bit) {
            const size_t imbalance = (2 * ones[bit] > count) ? (2 * ones[bit] - count) : (count - 2 * ones[bit]);
            if (imbalance < best_imbalance) {
                best_imbalance = imbalance;
                best_bit = static_cast<int>(bit);
            }
        }
        if ((best_bit < 0) || (best_imbalance == count)) {
            return;
        }
        this->nodes.push_back(node());
        this->nodes.push_back(node());
        const size_t child_zero = this->nodes.size() - 2;
        const size_t child_one = this->nodes.size() - 1;
        node& leaf = this->nodes[leaf_index];
        for (const entry& stored : leaf.entries) {
            this->nodes[hbst::get_bit(stored.descriptor, static_cast<size_t>(best_bit)) ? child_one : child_zero].entries.push_back(stored);
        }
        leaf.entries.clear();
        leaf.entries.shrink_to_fit();
        leaf.split_bit = best_bit;
        leaf.child[0] = child_zero;
        leaf.child[1] = child_one;
    }
}
