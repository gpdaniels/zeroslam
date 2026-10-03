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

#include "core/logger.hpp"
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mapping {
    place_recognition::place_recognition(const unsigned int distance_threshold)
        : kind(engine::hbst)
        , index()
        , vocabulary()
        , keyframe_ids()
        , max_distance(distance_threshold)
        , sequence_of()
        , next_sequence(0)
        , last_island_valid(false)
        , last_island_first(0)
        , last_island_last(0)
        , matched_descriptors()
        , matched_words() {
    }

    void place_recognition::set_engine(const engine chosen) {
        if (chosen == this->kind) {
            return;
        }
        this->clear();
        this->kind = chosen;
    }

    place_recognition::engine place_recognition::get_engine() const {
        return this->kind;
    }

    void place_recognition::set_distance_threshold(const unsigned int distance_threshold) {
        this->max_distance = distance_threshold;
    }

    std::vector<size_t> place_recognition::match_words(const feature::descriptor::stored* const descriptors, const size_t descriptors_size, const std::function<bool(const size_t)>& admissible) const {
        const auto same_descriptors = [&]() {
            if (this->matched_descriptors.size() != descriptors_size) {
                return false;
            }
            for (size_t i = 0; i < descriptors_size; ++i) {
                for (size_t byte = 0; byte < feature::descriptor::stored::size_bytes; ++byte) {
                    if (this->matched_descriptors[i].data[byte] != descriptors[i].data[byte]) {
                        return false;
                    }
                }
            }
            return true;
        };
        if (!admissible && same_descriptors()) {
            return this->matched_words;
        }
        std::vector<size_t> word_of(descriptors_size, match::index::binary_vocabulary::no_word);
        std::vector<match::index::binary_vocabulary::neighbour> nearest;
        for (size_t i = 0; i < descriptors_size; ++i) {
            this->vocabulary.search(descriptors[i], 2, nearest, admissible);
            if ((nearest.size() == 2) && (static_cast<float>(nearest[0].distance) <= place_recognition::ibow_ratio * static_cast<float>(nearest[1].distance))) {
                word_of[i] = nearest[0].word;
            }
        }
        if (!admissible) {
            this->matched_descriptors.assign(descriptors, descriptors + descriptors_size);
            this->matched_words = word_of;
        }
        return word_of;
    }

    void place_recognition::add_keyframe(const int keyframe_id, const feature::descriptor::stored* const descriptors, const size_t descriptors_size) {
        if (descriptors_size == 0) {
            return;
        }
        this->keyframe_ids.push_back(keyframe_id);
        this->sequence_of[keyframe_id] = this->next_sequence++;
        if (this->kind == engine::ibow) {
            const std::vector<size_t> word_of = this->match_words(descriptors, descriptors_size);
            this->vocabulary.add_document(keyframe_id, descriptors, descriptors_size, word_of.data());
            this->matched_descriptors.clear();
            return;
        }
        this->index.insert(keyframe_id, descriptors, descriptors_size);
    }

    void place_recognition::remove_keyframe(const int keyframe_id) {
        const std::vector<int>::iterator end = std::remove(this->keyframe_ids.begin(), this->keyframe_ids.end(), keyframe_id);
        if (end == this->keyframe_ids.end()) {
            return;
        }
        this->keyframe_ids.erase(end, this->keyframe_ids.end());
        this->sequence_of.erase(keyframe_id);
        if (this->kind == engine::ibow) {
            this->vocabulary.remove_document(keyframe_id);
            this->matched_descriptors.clear();
            return;
        }
        this->index.remove(keyframe_id);
    }

    void place_recognition::clear() {
        this->keyframe_ids.clear();
        this->index.clear();
        this->vocabulary.clear();
        this->matched_descriptors.clear();
        this->sequence_of.clear();
        this->next_sequence = 0;
        this->last_island_valid = false;
    }

    size_t place_recognition::num_keyframes() const {
        return this->keyframe_ids.size();
    }

    std::vector<place_recognition::candidate> place_recognition::score_keyframes(const feature::descriptor::stored* const query_descriptors, const size_t query_descriptors_size, const std::function<bool(const size_t)>& admissible) const {
        std::vector<candidate> scored;
        if ((query_descriptors_size == 0) || (this->vocabulary.document_count() == 0)) {
            return scored;
        }
        const std::vector<size_t> word_of = this->match_words(query_descriptors, query_descriptors_size, admissible);
        std::unordered_map<size_t, size_t> matches_of_word;
        for (const size_t word_id : word_of) {
            if (word_id != match::index::binary_vocabulary::no_word) {
                ++matches_of_word[word_id];
            }
        }
        if (core::logger::enabled(core::logger::level::debug)) {
            size_t matched = 0;
            for (const size_t word_id : word_of) {
                matched += (word_id != match::index::binary_vocabulary::no_word) ? 1 : 0;
            }
            core::logger::log(core::logger::level::debug, "Vocabulary query: %zu of %zu descriptors matched %zu words, of %zu words over %zu keyframes.", matched, query_descriptors_size, matches_of_word.size(), this->vocabulary.word_count(), this->vocabulary.document_count());
        }
        // As iBoW-LCD scores: each matched descriptor adds its word's tf-idf (the word's share of the query's descriptors
        // times the log of the keyframes over those the word was seen in) to every keyframe the word was posted from.
        const double documents = static_cast<double>(this->vocabulary.document_count());
        std::unordered_map<int, size_t> slot_of_keyframe;
        std::unordered_map<size_t, double> idf_of_word;
        for (size_t query_index = 0; query_index < query_descriptors_size; ++query_index) {
            const size_t word_id = word_of[query_index];
            if (word_id == match::index::binary_vocabulary::no_word) {
                continue;
            }
            std::unordered_map<size_t, double>::iterator idf = idf_of_word.find(word_id);
            if (idf == idf_of_word.end()) {
                const size_t frequency = this->vocabulary.document_frequency(word_id);
                idf = idf_of_word.emplace(word_id, (frequency > 0) ? math::log(documents / static_cast<double>(frequency)) : 0.0).first;
            }
            const double weight = (static_cast<double>(matches_of_word.at(word_id)) / static_cast<double>(query_descriptors_size)) * idf->second;
            for (const match::index::binary_vocabulary::posting& posted : this->vocabulary.postings(word_id)) {
                std::unordered_map<int, size_t>::iterator slot = slot_of_keyframe.find(posted.document);
                if (slot == slot_of_keyframe.end()) {
                    slot = slot_of_keyframe.emplace(posted.document, scored.size()).first;
                    scored.push_back(candidate{ posted.document, 0, 0.0f, {} });
                }
                candidate& entry = scored[slot->second];
                entry.score += static_cast<float>(weight);
                ++entry.votes;
                entry.query_indices.push_back(query_index);
            }
        }
        return scored;
    }

    std::vector<place_recognition::candidate> place_recognition::get_candidates(
        const feature::descriptor::stored* const query_descriptors,
        const size_t query_descriptors_size,
        const int current_keyframe_id,
        const size_t max_candidates,
        const int before_keyframe_id
    ) const {
        std::vector<candidate> candidates;
        if (this->kind == engine::ibow) {
            for (candidate& scored : this->score_keyframes(query_descriptors, query_descriptors_size)) {
                if ((scored.keyframe_id == current_keyframe_id) || ((before_keyframe_id >= 0) && (scored.keyframe_id >= before_keyframe_id))) {
                    continue;
                }
                candidates.push_back(static_cast<candidate&&>(scored));
            }
            std::sort(candidates.begin(), candidates.end(), [](const candidate& lhs, const candidate& rhs) {
                if (lhs.score != rhs.score) {
                    return lhs.score > rhs.score;
                }
                return lhs.keyframe_id < rhs.keyframe_id;
            });
            if (candidates.size() > max_candidates) {
                candidates.resize(max_candidates);
            }
            return candidates;
        }
        std::vector<match::index::hbst::hit> hits;
        for (size_t query_index = 0; query_index < query_descriptors_size; ++query_index) {
            this->index.search(query_descriptors[query_index], this->max_distance, hits);
            for (const match::index::hbst::hit& found : hits) {
                if ((found.keyframe_id == current_keyframe_id) || ((before_keyframe_id >= 0) && (found.keyframe_id >= before_keyframe_id))) {
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

    std::vector<place_recognition::candidate> place_recognition::get_loop_candidates(
        const feature::descriptor::stored* const query_descriptors,
        const size_t query_descriptors_size,
        const int current_keyframe_id,
        const size_t max_candidates,
        const std::function<bool(const int)>& excluded
    ) {
        std::vector<candidate> kept;
        if (this->kind == engine::hbst) {
            for (candidate& ranked : this->get_candidates(query_descriptors, query_descriptors_size, current_keyframe_id, this->keyframe_ids.size())) {
                if (excluded(ranked.keyframe_id)) {
                    continue;
                }
                kept.push_back(static_cast<candidate&&>(ranked));
                if (kept.size() >= max_candidates) {
                    break;
                }
            }
            return kept;
        }

        // The query only sees the words posted from a keyframe it may close a loop with, as iBoW-LCD's index holds no recent
        // image: otherwise the words of the recent keyframes, which track the same points, would be every descriptor's
        // nearest and the earlier visit's words would never be matched.
        std::unordered_map<int, bool> excluded_keyframe;
        for (const int keyframe_id : this->keyframe_ids) {
            excluded_keyframe[keyframe_id] = (keyframe_id == current_keyframe_id) || excluded(keyframe_id);
        }
        // Per word: 0 not yet known, 1 admissible, 2 not.
        std::vector<unsigned char> admissible_word(this->vocabulary.word_capacity(), static_cast<unsigned char>(0));
        const auto admissible = [this, &excluded_keyframe, &admissible_word](const size_t word_id) {
            if (admissible_word[word_id] != 0) {
                return admissible_word[word_id] == 1;
            }
            bool accepted = false;
            for (const match::index::binary_vocabulary::posting& posted : this->vocabulary.postings(word_id)) {
                const std::unordered_map<int, bool>::const_iterator found = excluded_keyframe.find(posted.document);
                if ((found != excluded_keyframe.end()) && !found->second) {
                    accepted = true;
                    break;
                }
            }
            admissible_word[word_id] = static_cast<unsigned char>(accepted ? 1 : 2);
            return accepted;
        };

        // The scores of the keyframes left, normalised between the least and the most of them (a keyframe sharing no word
        // scores zero), and those above ibow_minimum_score in descending order.
        std::vector<candidate> scored = this->score_keyframes(query_descriptors, query_descriptors_size, admissible);
        std::unordered_map<int, size_t> scored_slot;
        for (size_t i = 0; i < scored.size(); ++i) {
            scored_slot[scored[i].keyframe_id] = i;
        }
        float lowest = 0.0f;
        float highest = 0.0f;
        bool any = false;
        for (const int keyframe_id : this->keyframe_ids) {
            if ((keyframe_id == current_keyframe_id) || excluded(keyframe_id)) {
                continue;
            }
            const std::unordered_map<int, size_t>::const_iterator slot = scored_slot.find(keyframe_id);
            const float score = (slot == scored_slot.end()) ? 0.0f : scored[slot->second].score;
            lowest = any ? math::min(lowest, score) : score;
            highest = any ? math::max(highest, score) : score;
            if (slot != scored_slot.end()) {
                kept.push_back(scored[slot->second]);
            }
            any = true;
        }
        if (!any || !(highest > lowest)) {
            return {};
        }
        for (candidate& entry : kept) {
            entry.score = (entry.score - lowest) / (highest - lowest);
        }
        const auto below_threshold = [](const candidate& entry) {
            return !(entry.score > place_recognition::ibow_minimum_score);
        };
        kept.erase(std::remove_if(kept.begin(), kept.end(), below_threshold), kept.end());
        std::sort(kept.begin(), kept.end(), [](const candidate& lhs, const candidate& rhs) {
            if (lhs.score != rhs.score) {
                return lhs.score > rhs.score;
            }
            return lhs.keyframe_id < rhs.keyframe_id;
        });

        // Dynamic islands (iBoW-LCD's Algorithm 3): each keyframe joins the island whose span holds its place in the
        // sequence, or starts one ibow_island_offset either side of it, trimmed so islands never overlap; an island scores
        // the mean over its span and is represented by the keyframe that started it, its best.
        std::vector<island> islands;
        std::vector<size_t> member_of(kept.size(), static_cast<size_t>(-1));
        for (size_t i = 0; i < kept.size(); ++i) {
            const size_t place = this->sequence_of.at(kept[i].keyframe_id);
            size_t first = (place > place_recognition::ibow_island_offset) ? (place - place_recognition::ibow_island_offset) : 0;
            size_t last = place + place_recognition::ibow_island_offset;
            bool joined = false;
            for (size_t j = 0; j < islands.size(); ++j) {
                if ((place >= islands[j].first) && (place <= islands[j].last)) {
                    islands[j].score += kept[i].score;
                    member_of[i] = j;
                    joined = true;
                    break;
                }
                if (place > islands[j].last) {
                    first = math::max(first, islands[j].last + 1);
                }
                else {
                    last = math::min(last, islands[j].first - 1);
                }
            }
            if (!joined) {
                member_of[i] = islands.size();
                islands.push_back(island{ first, last, kept[i].keyframe_id, kept[i].score });
            }
        }
        if (islands.empty()) {
            return {};
        }
        for (island& grouped : islands) {
            grouped.score /= static_cast<float>(grouped.last - grouped.first + 1);
        }
        std::vector<size_t> order(islands.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&islands](const size_t lhs, const size_t rhs) {
            if (islands[lhs].score != islands[rhs].score) {
                return islands[lhs].score > islands[rhs].score;
            }
            return islands[lhs].keyframe_id < islands[rhs].keyframe_id;
        });
        // The best island overlapping the last query's choice goes first, as consecutive keyframes close loops with the same
        // region; it becomes this query's choice.
        if (this->last_island_valid) {
            for (size_t rank = 0; rank < order.size(); ++rank) {
                const island& candidate_island = islands[order[rank]];
                if ((candidate_island.first <= this->last_island_last) && (this->last_island_first <= candidate_island.last)) {
                    std::rotate(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(rank), order.begin() + static_cast<std::ptrdiff_t>(rank) + 1);
                    break;
                }
            }
        }
        this->last_island_valid = true;
        this->last_island_first = islands[order[0]].first;
        this->last_island_last = islands[order[0]].last;

        std::vector<candidate> representatives;
        for (const size_t island_index : order) {
            for (size_t i = 0; i < kept.size(); ++i) {
                if ((member_of[i] == island_index) && (kept[i].keyframe_id == islands[island_index].keyframe_id)) {
                    candidate representative = kept[i];
                    representative.score = islands[island_index].score;
                    representatives.push_back(static_cast<candidate&&>(representative));
                    break;
                }
            }
            if (representatives.size() >= max_candidates) {
                break;
            }
        }
        return representatives;
    }
}
