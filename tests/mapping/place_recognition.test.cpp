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

static feature::descriptor::binary<256> random_descriptor(core::random_pcg& random) {
    feature::descriptor::binary<256> descriptor;
    for (size_t j = 0; j < 32; ++j) {
        descriptor[j] = static_cast<unsigned char>(random.get_random_raw() % 256);
    }
    return descriptor;
}

static std::vector<feature::descriptor::binary<256>> random_descriptors(core::random_pcg& random, const size_t count) {
    std::vector<feature::descriptor::binary<256>> descriptors;
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
            const std::vector<feature::descriptor::binary<256>> descriptors = random_descriptors(random, 50);
            recognition.add_keyframe(keyframe_id, descriptors.data(), descriptors.size());
        }
        REQUIRE(recognition.num_keyframes() == 5);
        const std::vector<feature::descriptor::binary<256>> query = random_descriptors(random, 20);
        REQUIRE(recognition.get_candidates(query.data(), query.size(), 100, 5).empty());
    }

    {
        mapping::place_recognition recognition;
        core::random_pcg random;
        const std::vector<feature::descriptor::binary<256>> base = random_descriptors(random, 50);
        recognition.add_keyframe(1, base.data(), base.size());
        for (int keyframe_id = 2; keyframe_id <= 5; ++keyframe_id) {
            std::vector<feature::descriptor::binary<256>> noisy = base;
            for (feature::descriptor::binary<256>& descriptor : noisy) {
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
    }

    {
        mapping::place_recognition recognition;
        core::random_pcg random;
        const std::vector<feature::descriptor::binary<256>> base = random_descriptors(random, 50);
        recognition.add_keyframe(10, base.data(), base.size());
        std::vector<feature::descriptor::binary<256>> noisy = base;
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
        const std::vector<feature::descriptor::binary<256>> base = random_descriptors(random, 50);
        recognition.add_keyframe(10, base.data(), base.size());
        std::vector<feature::descriptor::binary<256>> noisy = base;
        for (feature::descriptor::binary<256>& descriptor : noisy) {
            descriptor[1] ^= 0x01;
        }
        recognition.add_keyframe(20, noisy.data(), noisy.size());
        const std::vector<mapping::place_recognition::candidate> candidates = recognition.get_candidates(base.data(), base.size(), 100, 5);
        REQUIRE(candidates.size() == 1);
        REQUIRE(candidates[0].keyframe_id == 10);
    }

    return EXIT_SUCCESS;
}
