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

#include "match/index/binary_vocabulary.hpp"

#include "match/distance/hamming.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace match::index {
    binary_vocabulary::binary_vocabulary()
        : binary_vocabulary(settings()) {
    }

    binary_vocabulary::binary_vocabulary(const settings& vocabulary_settings)
        : options(vocabulary_settings)
        , words()
        , forest()
        , documents()
        , recent()
        , alive_words(0)
        , sequence(0)
        , seen_in()
        , search_count(0) {
        this->options.branching = (this->options.branching < 2) ? 2 : this->options.branching;
        this->options.leaf_size = (this->options.leaf_size <= this->options.branching) ? (this->options.branching + 1) : this->options.leaf_size;
        this->options.trees = (this->options.trees < 1) ? 1 : this->options.trees;
        this->clear();
    }

    void binary_vocabulary::clear() {
        this->words.clear();
        this->documents.clear();
        this->recent.clear();
        this->alive_words = 0;
        this->sequence = 0;
        this->seen_in.clear();
        this->search_count = 0;
        this->forest.assign(this->options.trees, tree());
        for (size_t index = 0; index < this->forest.size(); ++index) {
            this->forest[index].nodes.assign(1, node());
            // Each tree splits its nodes with its own random centres, the same from run to run.
            this->forest[index].random.seed(0x9E3779B97F4A7C15ull + index);
        }
    }

    size_t binary_vocabulary::word_count() const {
        return this->alive_words;
    }

    size_t binary_vocabulary::word_capacity() const {
        return this->words.size();
    }

    size_t binary_vocabulary::document_count() const {
        return this->documents.size();
    }

    bool binary_vocabulary::has_document(const int document_id) const {
        return this->documents.count(document_id) != 0;
    }

    bool binary_vocabulary::alive(const size_t word_id) const {
        return (word_id < this->words.size()) && this->words[word_id].alive;
    }

    const binary_vocabulary::descriptor& binary_vocabulary::value(const size_t word_id) const {
        return this->words[word_id].value;
    }

    const std::vector<binary_vocabulary::posting>& binary_vocabulary::postings(const size_t word_id) const {
        return this->words[word_id].postings;
    }

    size_t binary_vocabulary::document_frequency(const size_t word_id) const {
        const std::vector<posting>& posted = this->words[word_id].postings;
        size_t distinct = 0;
        for (size_t i = 0; i < posted.size(); ++i) {
            bool repeated = false;
            for (size_t j = 0; (j < i) && !repeated; ++j) {
                repeated = posted[j].document == posted[i].document;
            }
            distinct += repeated ? 0 : 1;
        }
        return distinct;
    }

    size_t binary_vocabulary::make_node(tree& owner, const size_t parent) {
        size_t index = 0;
        if (!owner.free_nodes.empty()) {
            index = owner.free_nodes.back();
            owner.free_nodes.pop_back();
            owner.nodes[index] = node();
        }
        else {
            index = owner.nodes.size();
            owner.nodes.push_back(node());
        }
        owner.nodes[index].parent = parent;
        return index;
    }

    void binary_vocabulary::release_node(tree& owner, const size_t node_index) {
        owner.nodes[node_index] = node();
        owner.free_nodes.push_back(node_index);
    }

    void binary_vocabulary::build(tree& owner, const size_t node_index, std::vector<size_t>& members) {
        if (members.size() < this->options.leaf_size) {
            owner.nodes[node_index].leaf = true;
            owner.nodes[node_index].children.clear();
            owner.nodes[node_index].members = members;
            for (const size_t member : members) {
                owner.leaf_of[member] = node_index;
            }
            return;
        }
        // The branching factor's worth of distinct random members become the centres, and every other member joins its
        // nearest centre.
        const size_t branching = this->options.branching;
        for (size_t i = 0; i < branching; ++i) {
            const size_t pick = i + static_cast<size_t>(owner.random.get_random(0u, static_cast<unsigned int>(members.size() - 1 - i)));
            std::swap(members[i], members[pick]);
        }
        std::vector<std::vector<size_t>> clusters(branching);
        for (size_t i = 0; i < branching; ++i) {
            clusters[i].push_back(members[i]);
        }
        for (size_t m = branching; m < members.size(); ++m) {
            size_t best = 0;
            unsigned int best_distance = static_cast<unsigned int>(-1);
            for (size_t i = 0; i < branching; ++i) {
                const unsigned int distance = match::distance::hamming::distance(this->words[members[m]].value, this->words[members[i]].value);
                if (distance < best_distance) {
                    best_distance = distance;
                    best = i;
                }
            }
            clusters[best].push_back(members[m]);
        }
        owner.nodes[node_index].leaf = false;
        owner.nodes[node_index].members.clear();
        owner.nodes[node_index].children.clear();
        for (size_t i = 0; i < branching; ++i) {
            const size_t child = this->make_node(owner, node_index);
            owner.nodes[child].centre = members[i];
            owner.nodes[node_index].children.push_back(child);
            this->build(owner, child, clusters[i]);
        }
    }

    void binary_vocabulary::insert(tree& owner, const size_t word_id) {
        if (owner.leaf_of.size() <= word_id) {
            owner.leaf_of.resize(word_id + 1, binary_vocabulary::no_node);
        }
        size_t current = 0;
        while (!owner.nodes[current].leaf) {
            size_t best = binary_vocabulary::no_node;
            unsigned int best_distance = static_cast<unsigned int>(-1);
            for (const size_t child : owner.nodes[current].children) {
                const unsigned int distance = match::distance::hamming::distance(this->words[word_id].value, this->words[owner.nodes[child].centre].value);
                if (distance < best_distance) {
                    best_distance = distance;
                    best = child;
                }
            }
            current = best;
        }
        if (owner.nodes[current].members.size() + 1 < this->options.leaf_size) {
            owner.nodes[current].members.push_back(word_id);
            owner.leaf_of[word_id] = current;
            return;
        }
        std::vector<size_t> members = owner.nodes[current].members;
        members.push_back(word_id);
        this->build(owner, current, members);
    }

    void binary_vocabulary::erase(tree& owner, const size_t word_id) {
        if ((word_id >= owner.leaf_of.size()) || (owner.leaf_of[word_id] == binary_vocabulary::no_node)) {
            return;
        }
        const size_t leaf_index = owner.leaf_of[word_id];
        owner.leaf_of[word_id] = binary_vocabulary::no_node;
        std::vector<size_t>& members = owner.nodes[leaf_index].members;
        members.erase(std::remove(members.begin(), members.end(), word_id), members.end());
        if (!members.empty()) {
            if (owner.nodes[leaf_index].centre == word_id) {
                owner.nodes[leaf_index].centre = members[static_cast<size_t>(owner.random.get_random(0u, static_cast<unsigned int>(members.size() - 1)))];
            }
            return;
        }
        // An empty leaf leaves its parent, and so does every ancestor left without children; an emptied root is a leaf again.
        size_t current = leaf_index;
        while (current != 0) {
            const size_t parent = owner.nodes[current].parent;
            std::vector<size_t>& siblings = owner.nodes[parent].children;
            siblings.erase(std::remove(siblings.begin(), siblings.end(), current), siblings.end());
            this->release_node(owner, current);
            if (!owner.nodes[parent].children.empty()) {
                return;
            }
            current = parent;
        }
        owner.nodes[0].leaf = true;
        owner.nodes[0].children.clear();
        owner.nodes[0].members.clear();
    }

    void binary_vocabulary::delete_word(const size_t word_id) {
        if (!this->alive(word_id)) {
            return;
        }
        for (tree& owner : this->forest) {
            this->erase(owner, word_id);
        }
        this->words[word_id].alive = false;
        this->words[word_id].postings.clear();
        this->words[word_id].postings.shrink_to_fit();
        --this->alive_words;
    }

    void binary_vocabulary::purge() {
        if (this->sequence == 0) {
            return;
        }
        const size_t current = this->sequence - 1;
        std::vector<size_t> waiting;
        waiting.reserve(this->recent.size());
        for (const size_t word_id : this->recent) {
            if (!this->words[word_id].alive) {
                continue;
            }
            // A word is judged once two documents have followed the one that made it.
            if (current - this->words[word_id].made_at > 1) {
                if (this->words[word_id].postings.size() < this->options.minimum_postings) {
                    this->delete_word(word_id);
                }
                continue;
            }
            waiting.push_back(word_id);
        }
        this->recent.swap(waiting);
    }

    void binary_vocabulary::add_document(const int document_id, const descriptor* const descriptors, const size_t descriptors_size, const size_t* const word_of) {
        if (this->documents.count(document_id) != 0) {
            this->remove_document(document_id);
        }
        document& added = this->documents[document_id];
        added.sequence = this->sequence++;
        added.words.reserve(descriptors_size);
        for (size_t i = 0; i < descriptors_size; ++i) {
            size_t word_id = (word_of != nullptr) ? word_of[i] : binary_vocabulary::no_word;
            if (this->alive(word_id)) {
                if (this->options.merge) {
                    for (size_t byte = 0; byte < descriptor::size_bytes; ++byte) {
                        this->words[word_id].value.data[byte] = static_cast<unsigned char>(this->words[word_id].value.data[byte] & descriptors[i].data[byte]);
                    }
                }
                this->words[word_id].postings.push_back(posting{ document_id, i });
            }
            else {
                word_id = this->words.size();
                this->words.push_back(word{ descriptors[i], { posting{ document_id, i } }, added.sequence, true });
                ++this->alive_words;
                for (tree& owner : this->forest) {
                    this->insert(owner, word_id);
                }
                this->recent.push_back(word_id);
            }
            added.words.push_back(word_id);
        }
        if (this->options.purge) {
            this->purge();
        }
    }

    void binary_vocabulary::remove_document(const int document_id) {
        const std::unordered_map<int, document>::iterator found = this->documents.find(document_id);
        if (found == this->documents.end()) {
            return;
        }
        for (const size_t word_id : found->second.words) {
            if (!this->alive(word_id)) {
                continue;
            }
            std::vector<posting>& posted = this->words[word_id].postings;
            const auto from_document = [document_id](const posting& entry) {
                return entry.document == document_id;
            };
            posted.erase(std::remove_if(posted.begin(), posted.end(), from_document), posted.end());
            if (posted.empty()) {
                this->delete_word(word_id);
            }
        }
        this->documents.erase(found);
    }

    void binary_vocabulary::search_exhaustive(const descriptor& query, const size_t k, std::vector<neighbour>& nearest, const std::function<bool(const size_t)>& admissible) const {
        nearest.clear();
        for (size_t word_id = 0; word_id < this->words.size(); ++word_id) {
            if (this->words[word_id].alive && (!admissible || admissible(word_id))) {
                nearest.push_back(neighbour{ word_id, match::distance::hamming::distance(query, this->words[word_id].value) });
            }
        }
        std::sort(nearest.begin(), nearest.end(), [](const neighbour& lhs, const neighbour& rhs) {
            if (lhs.distance != rhs.distance) {
                return lhs.distance < rhs.distance;
            }
            return lhs.word < rhs.word;
        });
        if (nearest.size() > k) {
            nearest.resize(k);
        }
    }

    void binary_vocabulary::search(const descriptor& query, const size_t k, std::vector<neighbour>& nearest, const std::function<bool(const size_t)>& admissible) const {
        nearest.clear();
        if ((this->alive_words == 0) || (k == 0)) {
            return;
        }

        class pending final {
        public:
            unsigned int distance;
            size_t tree_index;
            size_t node_index;
        };

        // A min-heap on distance, ties broken by tree and node so a search is the same from run to run.
        const auto later = [](const pending& lhs, const pending& rhs) {
            if (lhs.distance != rhs.distance) {
                return lhs.distance > rhs.distance;
            }
            if (lhs.tree_index != rhs.tree_index) {
                return lhs.tree_index > rhs.tree_index;
            }
            return lhs.node_index > rhs.node_index;
        };
        std::vector<pending> heap;
        std::vector<neighbour> found;
        size_t examined = 0;
        if (this->seen_in.size() < this->words.size()) {
            this->seen_in.resize(this->words.size(), 0u);
        }
        if (++this->search_count == 0) {
            std::fill(this->seen_in.begin(), this->seen_in.end(), 0u);
            this->search_count = 1;
        }
        const unsigned int stamp = this->search_count;
        std::vector<unsigned int> child_distances;
        const auto descend = [&](const size_t tree_index, size_t node_index) {
            const tree& owner = this->forest[tree_index];
            while (!owner.nodes[node_index].leaf) {
                const std::vector<size_t>& children = owner.nodes[node_index].children;
                child_distances.resize(children.size());
                size_t best = 0;
                for (size_t c = 0; c < children.size(); ++c) {
                    child_distances[c] = match::distance::hamming::distance(query, this->words[owner.nodes[children[c]].centre].value);
                    if (child_distances[c] < child_distances[best]) {
                        best = c;
                    }
                }
                for (size_t c = 0; c < children.size(); ++c) {
                    if (c != best) {
                        heap.push_back(pending{ child_distances[c], tree_index, children[c] });
                        std::push_heap(heap.begin(), heap.end(), later);
                    }
                }
                node_index = children[best];
            }
            for (const size_t member : owner.nodes[node_index].members) {
                if (!this->words[member].alive || (this->seen_in[member] == stamp)) {
                    continue;
                }
                this->seen_in[member] = stamp;
                ++examined;
                if (!admissible || admissible(member)) {
                    found.push_back(neighbour{ member, match::distance::hamming::distance(query, this->words[member].value) });
                }
            }
        };
        for (size_t tree_index = 0; tree_index < this->forest.size(); ++tree_index) {
            descend(tree_index, 0);
        }
        while ((found.size() < this->options.checks) && (examined < this->options.examined_limit) && !heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), later);
            const pending next = heap.back();
            heap.pop_back();
            descend(next.tree_index, next.node_index);
        }
        std::sort(found.begin(), found.end(), [](const neighbour& lhs, const neighbour& rhs) {
            if (lhs.distance != rhs.distance) {
                return lhs.distance < rhs.distance;
            }
            return lhs.word < rhs.word;
        });
        if (found.size() > k) {
            found.resize(k);
        }
        nearest.swap(found);
    }
}
