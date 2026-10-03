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
#ifndef ZEROSLAM_MATCH_INDEX_BINARY_VOCABULARY_HPP
#define ZEROSLAM_MATCH_INDEX_BINARY_VOCABULARY_HPP

#include "core/random_pcg.hpp"
#include "feature/descriptor/binary.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <functional>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace match::index {
    // An incremental vocabulary of binary words, after Garcia-Fidalgo and Ortiz's OBIndex2 ("iBoW-LCD", RA-L 2018): no
    // training, the words are the descriptors seen so far, held in a forest of hierarchical clustering trees (Muja and Lowe,
    // CRV 2012) that a query searches best bin first. A descriptor matched to a word is merged into it (a bitwise and) and
    // posts its document (a keyframe) to the word; an unmatched one becomes a new word, which is deleted again if it has
    // too few postings once a document after the one that made it has been added.
    class binary_vocabulary final {
    public:
        using descriptor = feature::descriptor::stored;
        constexpr static const size_t no_word = static_cast<size_t>(-1);

        class settings final {
        public:
            // Children of each split node, and the words a leaf holds before it splits.
            size_t branching = 16;
            size_t leaf_size = 150;
            size_t trees = 4;
            // The words a search compares before it stops, and with an admissibility test the most it compares in all, as
            // words it may not return can fill whole leaves.
            size_t checks = 64;
            size_t examined_limit = 1024;
            bool merge = true;
            bool purge = true;
            size_t minimum_postings = 2;
        };

        class posting final {
        public:
            int document;
            size_t feature;
        };

        class neighbour final {
        public:
            size_t word;
            unsigned int distance;
        };

    private:
        constexpr static const size_t no_node = static_cast<size_t>(-1);

        class word final {
        public:
            descriptor value;
            std::vector<posting> postings;
            size_t made_at;
            bool alive;
        };

        class node final {
        public:
            bool leaf = true;
            size_t centre = binary_vocabulary::no_word;
            size_t parent = binary_vocabulary::no_node;
            std::vector<size_t> children;
            std::vector<size_t> members;
        };

        class tree final {
        public:
            std::vector<node> nodes;
            std::vector<size_t> free_nodes;
            std::vector<size_t> leaf_of;
            core::random_pcg random;
        };

        class document final {
        public:
            size_t sequence;
            std::vector<size_t> words;
        };

        settings options;
        std::vector<word> words;
        std::vector<tree> forest;
        std::unordered_map<int, document> documents;
        std::vector<size_t> recent;
        size_t alive_words;
        size_t sequence;
        // The search a word was last seen in, so a search marks the words it has compared without a set (searches are not
        // run concurrently).
        mutable std::vector<unsigned int> seen_in;
        mutable unsigned int search_count;

    public:
        binary_vocabulary();

        explicit binary_vocabulary(const settings& vocabulary_settings);

    public:
        void clear();

        size_t word_count() const;

        // One more than the largest word id, live or not.
        size_t word_capacity() const;

        size_t document_count() const;

        bool has_document(const int document_id) const;

        // The k nearest live words to a descriptor, nearest first (ties by word), from a search that compares at least
        // settings::checks words, every tree's nearest leaf first; with admissible given only the words it accepts count.
        void search(const descriptor& query, const size_t k, std::vector<neighbour>& nearest, const std::function<bool(const size_t)>& admissible = nullptr) const;

        // The same over every live word, for measuring the search against.
        void search_exhaustive(const descriptor& query, const size_t k, std::vector<neighbour>& nearest, const std::function<bool(const size_t)>& admissible = nullptr) const;

        // Adds a document of descriptors, word_of[i] the word descriptor i matched or no_word for a new one, then purges.
        void add_document(const int document_id, const descriptor* const descriptors, const size_t descriptors_size, const size_t* const word_of);

        // Removes a document's postings, deleting the words it leaves with none.
        void remove_document(const int document_id);

        bool alive(const size_t word_id) const;

        const descriptor& value(const size_t word_id) const;

        const std::vector<posting>& postings(const size_t word_id) const;

        // The distinct documents a word was posted from.
        size_t document_frequency(const size_t word_id) const;

    private:
        size_t make_node(tree& owner, const size_t parent);

        void release_node(tree& owner, const size_t node_index);

        void build(tree& owner, const size_t node_index, std::vector<size_t>& members);

        void insert(tree& owner, const size_t word_id);

        void erase(tree& owner, const size_t word_id);

        void delete_word(const size_t word_id);

        void purge();
    };
}

#endif // ZEROSLAM_MATCH_INDEX_BINARY_VOCABULARY_HPP
