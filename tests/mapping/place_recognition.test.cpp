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

#include "core/random_pcg.hpp"
#include "feature/descriptor/binary.hpp"

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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        mapping::place_recognition recognition;
        core::random_pcg random;
        for (int keyframe_id = 0; keyframe_id < 5; ++keyframe_id) {
            const std::vector<feature::descriptor::stored> descriptors = random_descriptors(random, 50);
            recognition.add_keyframe(keyframe_id, descriptors.data(), descriptors.size());
        }
        REQUIRE(recognition.num_keyframes() == 5);
        const std::vector<feature::descriptor::stored> query = random_descriptors(random, 20);
        REQUIRE(recognition.get_candidates(query.data(), query.size(), 100, 5).empty());
    }

    {
        mapping::place_recognition recognition;
        core::random_pcg random;
        const std::vector<feature::descriptor::stored> base = random_descriptors(random, 50);
        recognition.add_keyframe(1, base.data(), base.size());
        for (int keyframe_id = 2; keyframe_id <= 5; ++keyframe_id) {
            std::vector<feature::descriptor::stored> noisy = base;
            for (feature::descriptor::stored& descriptor : noisy) {
                for (int flip = 0; flip < 3; ++flip) {
                    descriptor[random.get_random_raw() % 32] ^= static_cast<unsigned char>(1u << (random.get_random_raw() % 8));
                }
            }
            recognition.add_keyframe(keyframe_id, noisy.data(), noisy.size());
        }

        const std::vector<mapping::place_recognition::candidate> candidates = recognition.get_candidates(base.data(), 30, 100, 3);
        REQUIRE(candidates.size() == 3);
        REQUIRE(candidates[0].keyframe_id == 1);
        REQUIRE(candidates[0].votes == 30);
        REQUIRE(candidates[0].query_indices.size() == 30);
        REQUIRE(candidates[0].score == 1.0f);
        REQUIRE(candidates[1].votes <= candidates[0].votes);
        REQUIRE(candidates[2].votes <= candidates[1].votes);
        REQUIRE((candidates[1].votes != candidates[2].votes) || (candidates[1].keyframe_id < candidates[2].keyframe_id));
        REQUIRE(recognition.get_candidates(base.data(), 30, 100, 10).size() == 5);

        const std::vector<mapping::place_recognition::candidate> others = recognition.get_candidates(base.data(), 30, 1, 10);
        REQUIRE(others.size() == 4);
        for (const mapping::place_recognition::candidate& other : others) {
            REQUIRE(other.keyframe_id != 1);
        }

        // Only the keyframes before a given one are ranked when it is given, as for a submap looking for the map it lost.
        const std::vector<mapping::place_recognition::candidate> earlier = recognition.get_candidates(base.data(), 30, 100, 10, 3);
        REQUIRE(earlier.size() == 2);
        for (const mapping::place_recognition::candidate& candidate : earlier) {
            REQUIRE(candidate.keyframe_id < 3);
        }
        REQUIRE(earlier[0].keyframe_id == 1);
    }

    {
        mapping::place_recognition recognition;
        core::random_pcg random;
        const std::vector<feature::descriptor::stored> base = random_descriptors(random, 50);
        recognition.add_keyframe(10, base.data(), base.size());
        std::vector<feature::descriptor::stored> noisy = base;
        noisy[0][0] ^= 0x01;
        recognition.add_keyframe(20, noisy.data(), noisy.size());
        REQUIRE(recognition.get_candidates(base.data(), base.size(), 100, 1)[0].keyframe_id == 10);
        recognition.remove_keyframe(10);
        REQUIRE(recognition.num_keyframes() == 1);
        const std::vector<mapping::place_recognition::candidate> remaining = recognition.get_candidates(base.data(), base.size(), 100, 1);
        REQUIRE(remaining.size() == 1);
        REQUIRE(remaining[0].keyframe_id == 20);
        REQUIRE(remaining[0].votes == 50);
        recognition.add_keyframe(30, nullptr, 0);
        REQUIRE(recognition.num_keyframes() == 1);
        recognition.clear();
        REQUIRE(recognition.num_keyframes() == 0);
        REQUIRE(recognition.get_candidates(base.data(), base.size(), 100, 1).empty());
    }

    {
        mapping::place_recognition recognition(1);
        core::random_pcg random;
        const std::vector<feature::descriptor::stored> base = random_descriptors(random, 50);
        recognition.add_keyframe(10, base.data(), base.size());
        std::vector<feature::descriptor::stored> noisy = base;
        for (feature::descriptor::stored& descriptor : noisy) {
            descriptor[1] ^= 0x01;
        }
        recognition.add_keyframe(20, noisy.data(), noisy.size());
        const std::vector<mapping::place_recognition::candidate> candidates = recognition.get_candidates(base.data(), base.size(), 100, 5);
        REQUIRE(candidates.size() == 1);
        REQUIRE(candidates[0].keyframe_id == 10);
    }

    // The ibow engine: places seen by three consecutive keyframes each, as tracked landmarks are, so their words survive
    // the purge of words seen only once.
    {
        mapping::place_recognition recognition;
        recognition.set_engine(mapping::place_recognition::engine::ibow);
        REQUIRE(recognition.get_engine() == mapping::place_recognition::engine::ibow);
        core::random_pcg random(0x1b0eull);
        const auto noisy_copy = [&random](const std::vector<feature::descriptor::stored>& base, const int flips) {
            std::vector<feature::descriptor::stored> noisy = base;
            for (feature::descriptor::stored& descriptor : noisy) {
                for (int flip = 0; flip < flips; ++flip) {
                    descriptor[random.get_random_raw() % 32] ^= static_cast<unsigned char>(1u << (random.get_random_raw() % 8));
                }
            }
            return noisy;
        };
        std::vector<std::vector<feature::descriptor::stored>> places;
        int keyframe_id = 0;
        for (int place = 0; place < 6; ++place) {
            places.push_back(random_descriptors(random, 150));
            for (int view = 0; view < 3; ++view) {
                const std::vector<feature::descriptor::stored> seen = noisy_copy(places.back(), 2);
                recognition.add_keyframe(keyframe_id++, seen.data(), seen.size());
            }
        }
        REQUIRE(recognition.num_keyframes() == 18);
        const auto place_of = [](const int id) {
            return id / 3;
        };

        // A view of place 2 ranks one of its keyframes first, by tf-idf.
        const std::vector<feature::descriptor::stored> query = noisy_copy(places[2], 6);
        const std::vector<mapping::place_recognition::candidate> ranked = recognition.get_candidates(query.data(), query.size(), 100, 5);
        REQUIRE(!ranked.empty());
        REQUIRE(place_of(ranked[0].keyframe_id) == 2);
        REQUIRE(ranked[0].votes > 50);
        for (size_t i = 1; i < ranked.size(); ++i) {
            REQUIRE(ranked[i].score <= ranked[i - 1].score);
        }

        // A loop query returns one keyframe per island, the place first; keyframes it may not close with never come back.
        const auto none = [](const int) {
            return false;
        };
        const std::vector<mapping::place_recognition::candidate> islands = recognition.get_loop_candidates(query.data(), query.size(), 100, 5, none);
        REQUIRE(!islands.empty());
        REQUIRE(place_of(islands[0].keyframe_id) == 2);
        for (size_t i = 1; i < islands.size(); ++i) {
            REQUIRE(place_of(islands[i].keyframe_id) != 2);
        }
        const auto not_place_2 = [&place_of](const int id) {
            return place_of(id) == 2;
        };
        for (const mapping::place_recognition::candidate& candidate : recognition.get_loop_candidates(query.data(), query.size(), 100, 5, not_place_2)) {
            REQUIRE(place_of(candidate.keyframe_id) != 2);
        }

        // The island next to the last query's choice goes first, even when another island scores higher.
        const std::vector<feature::descriptor::stored> first_visit = noisy_copy(places[4], 6);
        REQUIRE(place_of(recognition.get_loop_candidates(first_visit.data(), first_visit.size(), 100, 5, none)[0].keyframe_id) == 4);
        std::vector<feature::descriptor::stored> mixed = noisy_copy(places[1], 6);
        mixed.resize(100);
        const std::vector<feature::descriptor::stored> some_of_4 = noisy_copy(places[4], 6);
        mixed.insert(mixed.end(), some_of_4.begin(), some_of_4.begin() + 50);
        const std::vector<mapping::place_recognition::candidate> preferred = recognition.get_loop_candidates(mixed.data(), mixed.size(), 100, 5, none);
        REQUIRE(preferred.size() >= 2);
        REQUIRE(place_of(preferred[0].keyframe_id) == 4);
        REQUIRE(place_of(preferred[1].keyframe_id) == 1);

        // Removing a place's keyframes removes it from the rankings.
        for (int removed = 6; removed < 9; ++removed) {
            recognition.remove_keyframe(removed);
        }
        REQUIRE(recognition.num_keyframes() == 15);
        for (const mapping::place_recognition::candidate& candidate : recognition.get_candidates(query.data(), query.size(), 100, 15)) {
            REQUIRE(place_of(candidate.keyframe_id) != 2);
        }
        // Switching engines empties the index.
        recognition.set_engine(mapping::place_recognition::engine::hbst);
        REQUIRE(recognition.num_keyframes() == 0);
        REQUIRE(recognition.get_candidates(query.data(), query.size(), 100, 5).empty());
    }

    return EXIT_SUCCESS;
}
