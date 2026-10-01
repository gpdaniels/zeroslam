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
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace match::index {
    namespace {
        bool identical(const feature::descriptor::binary<hbst::descriptor_bits>& lhs, const feature::descriptor::binary<hbst::descriptor_bits>& rhs) {
            for (size_t byte = 0; byte < feature::descriptor::binary<hbst::descriptor_bits>::size_bytes; ++byte) {
                if (lhs.data[byte] != rhs.data[byte]) {
                    return false;
                }
            }
            return true;
        }
    }

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
        node& leaf = this->nodes[leaf_index];
        if (leaf.unsplittable && !leaf.descriptors.empty() && !identical(leaf.descriptors[0], stored.descriptor)) {
            leaf.unsplittable = false;
        }
        leaf.descriptors.push_back(stored.descriptor);
        leaf.keyframe_ids.push_back(stored.keyframe_id);
        leaf.descriptor_indices.push_back(stored.descriptor_index);
        ++this->entry_count;
        if (!leaf.unsplittable && (leaf.descriptors.size() > hbst::leaf_capacity)) {
            this->split(leaf_index);
        }
    }

    void hbst::remove(const int keyframe_id) {
        for (node& current : this->nodes) {
            size_t kept = 0;
            for (size_t i = 0; i < current.keyframe_ids.size(); ++i) {
                if (current.keyframe_ids[i] == keyframe_id) {
                    continue;
                }
                current.descriptors[kept] = current.descriptors[i];
                current.keyframe_ids[kept] = current.keyframe_ids[i];
                current.descriptor_indices[kept] = current.descriptor_indices[i];
                ++kept;
            }
            this->entry_count -= current.keyframe_ids.size() - kept;
            current.descriptors.resize(kept);
            current.keyframe_ids.resize(kept);
            current.descriptor_indices.resize(kept);
        }
    }

    std::vector<hbst::hit> hbst::search(const feature::descriptor::binary<descriptor_bits>& query, const unsigned int max_distance) const {
        std::vector<hit> hits;
        this->search(query, max_distance, hits);
        return hits;
    }

    void hbst::search(const feature::descriptor::binary<descriptor_bits>& query, const unsigned int max_distance, std::vector<hit>& hits) const {
        hits.clear();
        const node& leaf = this->nodes[this->find_leaf(query)];
        constexpr static const size_t block_size = 128;
        unsigned int distances[block_size];
        for (size_t block_begin = 0; block_begin < leaf.descriptors.size(); block_begin += block_size) {
            const size_t block_count = math::min(block_size, leaf.descriptors.size() - block_begin);
            distance::hamming::distances(query, &leaf.descriptors[block_begin], block_count, &distances[0]);
            for (size_t block_index = 0; block_index < block_count; ++block_index) {
                const unsigned int found = distances[block_index];
                if (found >= max_distance) {
                    continue;
                }
                const int keyframe_id = leaf.keyframe_ids[block_begin + block_index];
                const size_t descriptor_index = leaf.descriptor_indices[block_begin + block_index];
                bool merged = false;
                for (hit& existing : hits) {
                    if (existing.keyframe_id != keyframe_id) {
                        continue;
                    }
                    if ((found < existing.distance) || ((found == existing.distance) && (descriptor_index < existing.descriptor_index))) {
                        existing.descriptor_index = descriptor_index;
                        existing.distance = found;
                    }
                    merged = true;
                    break;
                }
                if (!merged) {
                    hits.push_back(hit{ keyframe_id, descriptor_index, found });
                }
            }
        }
        std::sort(hits.begin(), hits.end(), [](const hit& lhs, const hit& rhs) {
            if (lhs.distance != rhs.distance) {
                return lhs.distance < rhs.distance;
            }
            return lhs.keyframe_id < rhs.keyframe_id;
        });
    }

    size_t hbst::find_leaf(const feature::descriptor::binary<descriptor_bits>& descriptor) const {
        size_t index = 0;
        while (this->nodes[index].split_bit >= 0) {
            index = this->nodes[index].child[hbst::get_bit(descriptor, static_cast<size_t>(this->nodes[index].split_bit)) ? 1 : 0];
        }
        return index;
    }

    void hbst::split(const size_t leaf_index) {
        const size_t count = this->nodes[leaf_index].descriptors.size();
        size_t ones[hbst::descriptor_bits] = {};
        for (const feature::descriptor::binary<descriptor_bits>& stored : this->nodes[leaf_index].descriptors) {
            for (size_t bit = 0; bit < hbst::descriptor_bits; ++bit) {
                ones[bit] += hbst::get_bit(stored, bit) ? 1u : 0u;
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
            // No bit differs between the entries, so they are all one descriptor and no split can separate them.
            this->nodes[leaf_index].unsplittable = true;
            return;
        }
        this->nodes.push_back(node());
        this->nodes.push_back(node());
        const size_t child_zero = this->nodes.size() - 2;
        const size_t child_one = this->nodes.size() - 1;
        node& leaf = this->nodes[leaf_index];
        for (size_t i = 0; i < count; ++i) {
            node& child = this->nodes[hbst::get_bit(leaf.descriptors[i], static_cast<size_t>(best_bit)) ? child_one : child_zero];
            child.descriptors.push_back(leaf.descriptors[i]);
            child.keyframe_ids.push_back(leaf.keyframe_ids[i]);
            child.descriptor_indices.push_back(leaf.descriptor_indices[i]);
        }
        leaf.descriptors.clear();
        leaf.descriptors.shrink_to_fit();
        leaf.keyframe_ids.clear();
        leaf.keyframe_ids.shrink_to_fit();
        leaf.descriptor_indices.clear();
        leaf.descriptor_indices.shrink_to_fit();
        leaf.split_bit = best_bit;
        leaf.child[0] = child_zero;
        leaf.child[1] = child_one;
    }
}
