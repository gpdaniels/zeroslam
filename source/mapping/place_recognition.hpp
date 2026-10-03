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
#include "match/index/binary_vocabulary.hpp"
#include "match/index/hbst.hpp"

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

namespace mapping {
    // Ranks the keyframes a query's descriptors were seen in. The hbst engine counts the keyframes of the descriptors in the
    // query's leaf of a Hamming binary search tree (Schlegel and Grisetti, RA-L 2018). The ibow engine is iBoW-LCD
    // (Garcia-Fidalgo and Ortiz, RA-L 2018): an incremental vocabulary of binary words scores the keyframes by tf-idf, and
    // a loop query groups the best into islands of consecutive keyframes, preferring the island next to the one the last
    // query chose.
    class place_recognition final {
    public:
        enum class engine {
            hbst,
            ibow
        };

        class candidate final {
        public:
            int keyframe_id;
            size_t votes;
            float score;
            std::vector<size_t> query_indices;
        };

    private:
        engine kind;
        match::index::hbst index;
        match::index::binary_vocabulary vocabulary;
        std::vector<int> keyframe_ids;
        unsigned int max_distance;
        // Each keyframe's place in the order the keyframes were added, which the islands span.
        std::unordered_map<int, size_t> sequence_of;
        size_t next_sequence;
        bool last_island_valid;
        size_t last_island_first;
        size_t last_island_last;
        // The last descriptors matched against the whole vocabulary and their words: a keyframe is queried and then added
        // with the same descriptors, and the vocabulary does not change in between.
        mutable std::vector<feature::descriptor::stored> matched_descriptors;
        mutable std::vector<size_t> matched_words;

        class island final {
        public:
            size_t first;
            size_t last;
            int keyframe_id;
            float score;
        };

        // The words each query descriptor matches under the ratio test, or no_word, among the admissible ones when given.
        std::vector<size_t> match_words(const feature::descriptor::stored* const descriptors, const size_t descriptors_size, const std::function<bool(const size_t)>& admissible = nullptr) const;

        // The tf-idf score, matched descriptors and their indices of every keyframe the query shares a word with.
        std::vector<candidate> score_keyframes(const feature::descriptor::stored* const query_descriptors, const size_t query_descriptors_size, const std::function<bool(const size_t)>& admissible = nullptr) const;

    public:
        constexpr static const unsigned int default_distance_threshold = 40;
        // iBoW-LCD's nearest neighbour ratio, normalised score threshold and island half width (its island_size of 7).
        constexpr static const float ibow_ratio = 0.8f;
        constexpr static const float ibow_minimum_score = 0.3f;
        constexpr static const size_t ibow_island_offset = 3;

        explicit place_recognition(const unsigned int distance_threshold = place_recognition::default_distance_threshold);

    public:
        // Switching engines empties the index.
        void set_engine(const engine chosen);
        engine get_engine() const;
        void set_distance_threshold(const unsigned int distance_threshold);
        void add_keyframe(const int keyframe_id, const feature::descriptor::stored* const descriptors, const size_t descriptors_size);
        void remove_keyframe(const int keyframe_id);
        void clear();
        size_t num_keyframes() const;

        // The keyframes most voted for by the query's descriptors, among those before before_keyframe_id when one is given.
        std::vector<candidate> get_candidates(
            const feature::descriptor::stored* const query_descriptors,
            const size_t query_descriptors_size,
            const int current_keyframe_id,
            const size_t max_candidates,
            const int before_keyframe_id = -1
        ) const;

        // The candidates a loop query verifies: the keyframes ranked as get_candidates ranks them with every keyframe
        // excluded() rejects left out before the list is cut, and for the ibow engine the best keyframe of each island,
        // the island beside the last query's first.
        std::vector<candidate> get_loop_candidates(
            const feature::descriptor::stored* const query_descriptors,
            const size_t query_descriptors_size,
            const int current_keyframe_id,
            const size_t max_candidates,
            const std::function<bool(const int)>& excluded
        );
    };
}

#endif // ZEROSLAM_MAPPING_PLACE_RECOGNITION_HPP
