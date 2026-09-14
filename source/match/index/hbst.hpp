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
#ifndef ZEROSLAM_MATCH_INDEX_HBST_HPP
#define ZEROSLAM_MATCH_INDEX_HBST_HPP

#include "feature/descriptor/binary.hpp"

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

namespace match::index {
    class hbst final {
    public:
        constexpr static const size_t descriptor_bits = 256;
        constexpr static const size_t leaf_capacity = 100;

        class entry final {
        public:
            feature::descriptor::binary<descriptor_bits> descriptor;
            int keyframe_id;
            size_t descriptor_index;
        };

        class hit final {
        public:
            int keyframe_id;
            size_t descriptor_index;
            unsigned int distance;
        };

    private:
        class node final {
        public:
            int split_bit = -1;
            size_t child[2] = { 0, 0 };
            std::vector<entry> entries;
        };

    private:
        std::vector<node> nodes;
        size_t entry_count;

    public:
        hbst();

    public:
        static bool get_bit(const feature::descriptor::binary<descriptor_bits>& descriptor, const size_t bit);

    public:
        void clear();
        size_t size() const;
        bool empty() const;

        void insert(const int keyframe_id, const feature::descriptor::binary<descriptor_bits>* const descriptors, const size_t descriptors_size);
        void insert(const entry& stored);

        void remove(const int keyframe_id);

        std::vector<hit> search(const feature::descriptor::binary<descriptor_bits>& query, const unsigned int max_distance) const;

    private:
        size_t find_leaf(const feature::descriptor::binary<descriptor_bits>& descriptor) const;

        void split(const size_t leaf_index);
    };
}

#endif // ZEROSLAM_MATCH_INDEX_HBST_HPP
