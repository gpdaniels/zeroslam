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

#include "core/random_pcg.hpp"
#include "feature/descriptor/binary.hpp"
#include "match/distance/hamming.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

using vocabulary = match::index::binary_vocabulary;

static feature::descriptor::stored random_descriptor(core::random_pcg& random) {
    feature::descriptor::stored descriptor = {};
    for (size_t j = 0; j < 32; ++j) {
        descriptor[j] = static_cast<unsigned char>(random.get_random_raw() % 256);
    }
    return descriptor;
}

static std::vector<feature::descriptor::stored> random_descriptors(core::random_pcg& random, const size_t count) {
    std::vector<feature::descriptor::stored> descriptors;
    for (size_t i = 0; i < count; ++i) {
        descriptors.push_back(random_descriptor(random));
    }
    return descriptors;
}

// A copy with the given number of distinct bits of the first 256 flipped.
static feature::descriptor::stored flipped(const feature::descriptor::stored& original, const size_t bits, core::random_pcg& random) {
    feature::descriptor::stored copy = original;
    std::vector<unsigned char> chosen(256, 0);
    size_t done = 0;
    while (done < bits) {
        const size_t bit = random.get_random(0u, 255u);
        if (chosen[bit] != 0) {
            continue;
        }
        chosen[bit] = 1;
        copy[bit / 8] = static_cast<unsigned char>(copy[bit / 8] ^ (1u << (bit % 8)));
        ++done;
    }
    return copy;
}

// The word id of each live word's descriptor, found by searching for it.
static size_t nearest_word(const vocabulary& words, const feature::descriptor::stored& query, unsigned int& distance) {
    std::vector<vocabulary::neighbour> nearest;
    words.search(query, 1, nearest);
    distance = nearest.empty() ? static_cast<unsigned int>(-1) : nearest[0].distance;
    return nearest.empty() ? vocabulary::no_word : nearest[0].word;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // An empty vocabulary finds nothing.
    {
        const vocabulary words;
        REQUIRE(words.word_count() == 0);
        REQUIRE(words.document_count() == 0);
        std::vector<vocabulary::neighbour> nearest;
        core::random_pcg random;
        words.search(random_descriptor(random), 2, nearest);
        REQUIRE(nearest.empty());
    }

    // Unmatched descriptors become words, and every word is its own nearest at distance zero once the trees have split many
    // times over.
    {
        vocabulary::settings options;
        options.purge = false;
        vocabulary words(options);
        core::random_pcg random(7);
        const std::vector<feature::descriptor::stored> first = random_descriptors(random, 1200);
        words.add_document(10, first.data(), first.size(), nullptr);
        const std::vector<feature::descriptor::stored> second = random_descriptors(random, 900);
        words.add_document(20, second.data(), second.size(), nullptr);
        REQUIRE(words.word_count() == 2100);
        REQUIRE(words.document_count() == 2);
        REQUIRE(words.has_document(10) && words.has_document(20) && !words.has_document(30));
        size_t exact = 0;
        for (const feature::descriptor::stored& descriptor : first) {
            unsigned int distance = 0;
            const size_t word_id = nearest_word(words, descriptor, distance);
            exact += ((word_id != vocabulary::no_word) && (distance == 0) && (words.postings(word_id).size() == 1) && (words.postings(word_id)[0].document == 10)) ? 1u : 0u;
        }
        REQUIRE(exact == first.size());
        // The two nearest come back nearest first.
        std::vector<vocabulary::neighbour> nearest;
        words.search(second[5], 2, nearest);
        REQUIRE(nearest.size() == 2);
        REQUIRE(nearest[0].distance == 0);
        REQUIRE(nearest[0].distance <= nearest[1].distance);
        REQUIRE(words.postings(nearest[0].word)[0].document == 20);
        REQUIRE(words.postings(nearest[0].word)[0].feature == static_cast<size_t>(5));

        // With an admissibility test only the words it accepts come back, and the exhaustive search agrees on the nearest
        // whenever the tree search finds one.
        const auto from_second = [&words](const size_t word_id) {
            return words.postings(word_id)[0].document == 20;
        };
        size_t agreed = 0;
        for (size_t i = 0; i < 100; ++i) {
            std::vector<vocabulary::neighbour> tree;
            std::vector<vocabulary::neighbour> exhaustive;
            words.search(first[i], 2, tree, from_second);
            words.search_exhaustive(first[i], 2, exhaustive, from_second);
            REQUIRE(exhaustive.size() == 2);
            for (const vocabulary::neighbour& found : tree) {
                REQUIRE(words.postings(found.word)[0].document == 20);
            }
            REQUIRE(tree.empty() || (tree[0].distance >= exhaustive[0].distance));
            agreed += (!tree.empty() && (tree[0].distance == exhaustive[0].distance)) ? 1u : 0u;
        }
        REQUIRE(agreed >= 10);
        std::vector<vocabulary::neighbour> everything;
        words.search_exhaustive(second[7], 3, everything);
        REQUIRE((everything.size() == 3) && (everything[0].distance == 0) && (everything[0].distance <= everything[1].distance) && (everything[1].distance <= everything[2].distance));

        // Noisy copies (16 of 256 bits flipped) still find their word nearly always.
        size_t recovered = 0;
        for (size_t i = 0; i < 300; ++i) {
            unsigned int distance = 0;
            const size_t word_id = nearest_word(words, flipped(first[i], 16, random), distance);
            recovered += ((word_id != vocabulary::no_word) && (distance == 16) && (words.postings(word_id)[0].document == 10) && (words.postings(word_id)[0].feature == i)) ? 1u : 0u;
        }
        REQUIRE(recovered >= 285);
    }

    // A matched descriptor is merged into its word with a bitwise and, and posts its document to it.
    {
        vocabulary::settings options;
        options.purge = false;
        vocabulary words(options);
        feature::descriptor::stored a = {};
        feature::descriptor::stored b = {};
        a[0] = 0xF0;
        a[1] = 0x0F;
        b[0] = 0x3C;
        b[1] = 0xFF;
        words.add_document(1, &a, 1, nullptr);
        unsigned int distance = 0;
        const size_t word_id = nearest_word(words, a, distance);
        REQUIRE(word_id != vocabulary::no_word);
        words.add_document(2, &b, 1, &word_id);
        REQUIRE(words.word_count() == 1);
        REQUIRE(words.value(word_id)[0] == 0x30);
        REQUIRE(words.value(word_id)[1] == 0x0F);
        REQUIRE(words.postings(word_id).size() == 2);
        REQUIRE(words.document_frequency(word_id) == 2);
        // Without merging the word keeps its first descriptor.
        vocabulary::settings unmerged = options;
        unmerged.merge = false;
        vocabulary kept(unmerged);
        kept.add_document(1, &a, 1, nullptr);
        const size_t kept_id = nearest_word(kept, a, distance);
        kept.add_document(2, &b, 1, &kept_id);
        REQUIRE(kept.value(kept_id)[0] == 0xF0);
        // Two features of one document count it once.
        const feature::descriptor::stored pair[2] = { a, a };
        const size_t both[2] = { kept_id, kept_id };
        kept.add_document(3, &pair[0], 2, &both[0]);
        REQUIRE(kept.postings(kept_id).size() == 4);
        REQUIRE(kept.document_frequency(kept_id) == 3);
    }

    // A new word survives only if it gains a second posting before two more documents have arrived.
    {
        vocabulary words;
        core::random_pcg random(11);
        const std::vector<feature::descriptor::stored> first = random_descriptors(random, 400);
        words.add_document(100, first.data(), first.size(), nullptr);
        std::vector<size_t> first_words(first.size(), vocabulary::no_word);
        for (size_t i = 0; i < first.size(); ++i) {
            unsigned int distance = 0;
            first_words[i] = nearest_word(words, first[i], distance);
            REQUIRE(distance == 0);
        }
        // The second document sees the first half again.
        const std::vector<feature::descriptor::stored> fresh = random_descriptors(random, 300);
        std::vector<feature::descriptor::stored> second(first.begin(), first.begin() + 200);
        second.insert(second.end(), fresh.begin(), fresh.end());
        std::vector<size_t> second_words(second.size(), vocabulary::no_word);
        for (size_t i = 0; i < 200; ++i) {
            second_words[i] = first_words[i];
        }
        words.add_document(101, second.data(), second.size(), second_words.data());
        // Nothing is judged until a second document has followed.
        REQUIRE(words.word_count() == 700);
        const std::vector<feature::descriptor::stored> third = random_descriptors(random, 50);
        words.add_document(102, third.data(), third.size(), nullptr);
        // The unseen half of the first document is gone; the second's new words wait for one more document.
        REQUIRE(words.word_count() == 200 + 300 + 50);
        for (size_t i = 0; i < first.size(); ++i) {
            REQUIRE(words.alive(first_words[i]) == (i < 200));
        }
        words.add_document(103, nullptr, 0, nullptr);
        REQUIRE(words.word_count() == 200 + 50);
        unsigned int distance = 0;
        const size_t survivor = nearest_word(words, first[10], distance);
        REQUIRE((survivor == first_words[10]) && (distance == 0));
        const size_t gone = nearest_word(words, first[300], distance);
        REQUIRE((gone != first_words[300]) && (distance > 0));
    }

    // Removing a document removes its postings and the words it leaves without any, and searches skip them.
    {
        vocabulary::settings options;
        options.purge = false;
        vocabulary words(options);
        core::random_pcg random(13);
        const std::vector<feature::descriptor::stored> first = random_descriptors(random, 500);
        const std::vector<feature::descriptor::stored> second = random_descriptors(random, 500);
        words.add_document(1, first.data(), first.size(), nullptr);
        std::vector<size_t> shared(second.size(), vocabulary::no_word);
        for (size_t i = 0; i < 100; ++i) {
            unsigned int distance = 0;
            shared[i] = nearest_word(words, first[i], distance);
        }
        words.add_document(2, second.data(), second.size(), shared.data());
        REQUIRE(words.word_count() == 500 + 400);
        words.remove_document(1);
        REQUIRE(words.document_count() == 1);
        REQUIRE(!words.has_document(1));
        REQUIRE(words.word_count() == 500);
        for (size_t i = 0; i < 100; ++i) {
            REQUIRE(words.alive(shared[i]));
            REQUIRE(words.postings(shared[i]).size() == 1);
            REQUIRE(words.postings(shared[i])[0].document == 2);
        }
        for (size_t i = 100; i < first.size(); ++i) {
            unsigned int distance = 0;
            const size_t found = nearest_word(words, first[i], distance);
            REQUIRE((found == vocabulary::no_word) || (words.postings(found)[0].document == 2));
        }
        words.remove_document(2);
        REQUIRE(words.word_count() == 0);
        REQUIRE(words.document_count() == 0);
        std::vector<vocabulary::neighbour> nearest;
        words.search(first[0], 2, nearest);
        REQUIRE(nearest.empty());
        // The emptied trees take words again.
        words.add_document(3, second.data(), second.size(), nullptr);
        REQUIRE(words.word_count() == second.size());
        unsigned int distance = 0;
        const size_t found = nearest_word(words, second[42], distance);
        REQUIRE((found != vocabulary::no_word) && (distance == 0) && (words.postings(found)[0].feature == static_cast<size_t>(42)));
    }

    // The same documents build the same vocabulary, and searches agree.
    {
        core::random_pcg random(17);
        const std::vector<feature::descriptor::stored> first = random_descriptors(random, 700);
        const std::vector<feature::descriptor::stored> queries = random_descriptors(random, 50);
        vocabulary one;
        vocabulary two;
        one.add_document(1, first.data(), first.size(), nullptr);
        two.add_document(1, first.data(), first.size(), nullptr);
        for (const feature::descriptor::stored& query : queries) {
            std::vector<vocabulary::neighbour> lhs;
            std::vector<vocabulary::neighbour> rhs;
            one.search(query, 2, lhs);
            two.search(query, 2, rhs);
            REQUIRE(lhs.size() == rhs.size());
            for (size_t i = 0; i < lhs.size(); ++i) {
                REQUIRE((lhs[i].word == rhs[i].word) && (lhs[i].distance == rhs[i].distance));
                REQUIRE(lhs[i].distance == match::distance::hamming::distance(query, one.value(lhs[i].word)));
            }
        }
    }

    return EXIT_SUCCESS;
}
