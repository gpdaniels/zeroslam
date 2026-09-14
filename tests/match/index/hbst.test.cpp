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
        feature::descriptor::binary<256> descriptor = {};
        descriptor[3] = 0x10;
        REQUIRE(match::index::hbst::get_bit(descriptor, 3 * 8 + 4));
        REQUIRE(!match::index::hbst::get_bit(descriptor, 3 * 8 + 3));
        REQUIRE(!match::index::hbst::get_bit(descriptor, 0));
    }

    {
        match::index::hbst tree;
        REQUIRE(tree.empty());
        core::random_pcg random;
        const std::vector<feature::descriptor::binary<256>> descriptors = random_descriptors(random, 100);
        tree.insert(1, descriptors.data(), descriptors.size());
        REQUIRE(tree.size() == 100);
        for (size_t i = 0; i < descriptors.size(); ++i) {
            const std::vector<match::index::hbst::hit> hits = tree.search(descriptors[i], 40);
            REQUIRE(hits.size() == 1);
            REQUIRE(hits[0].keyframe_id == 1);
            REQUIRE(hits[0].descriptor_index == i);
            REQUIRE(hits[0].distance == 0);
        }
        REQUIRE(tree.search(random_descriptor(random), 40).empty());
        tree.clear();
        REQUIRE(tree.empty());
        REQUIRE(tree.search(descriptors[0], 40).empty());
    }

    {
        match::index::hbst tree;
        core::random_pcg random;
        std::vector<std::vector<feature::descriptor::binary<256>>> keyframes;
        for (int keyframe_id = 0; keyframe_id < 40; ++keyframe_id) {
            keyframes.push_back(random_descriptors(random, 100));
            tree.insert(keyframe_id, keyframes.back().data(), keyframes.back().size());
        }
        REQUIRE(tree.size() == 4000);
        for (size_t keyframe_id = 0; keyframe_id < keyframes.size(); ++keyframe_id) {
            for (size_t i = 0; i < keyframes[keyframe_id].size(); ++i) {
                const std::vector<match::index::hbst::hit> hits = tree.search(keyframes[keyframe_id][i], 40);
                REQUIRE(!hits.empty());
                REQUIRE(hits[0].keyframe_id == static_cast<int>(keyframe_id));
                REQUIRE(hits[0].descriptor_index == i);
                REQUIRE(hits[0].distance == 0);
            }
        }
        tree.remove(17);
        REQUIRE(tree.size() == 3900);
        REQUIRE(tree.search(keyframes[17][0], 40).empty());
        REQUIRE(tree.search(keyframes[18][0], 40)[0].keyframe_id == 18);
    }

    {
        match::index::hbst tree;
        core::random_pcg random;
        const feature::descriptor::binary<256> base = random_descriptor(random);
        feature::descriptor::binary<256> neighbours[2] = { base, base };
        neighbours[0][5] ^= 0x03;
        neighbours[1][9] ^= 0x01;
        tree.insert(30, &neighbours[0], 2);
        tree.insert(20, &base, 1);
        tree.insert(25, &base, 1);
        const std::vector<match::index::hbst::hit> hits = tree.search(base, 40);
        REQUIRE(hits.size() == 3);
        REQUIRE((hits[0].keyframe_id == 20) && (hits[0].distance == 0));
        REQUIRE((hits[1].keyframe_id == 25) && (hits[1].distance == 0));
        REQUIRE((hits[2].keyframe_id == 30) && (hits[2].distance == 1) && (hits[2].descriptor_index == 1));
        REQUIRE(tree.search(base, 1).size() == 2);
        REQUIRE(tree.search(base, 0).empty());
    }

    {
        match::index::hbst tree_a;
        match::index::hbst tree_b;
        core::random_pcg random;
        std::vector<feature::descriptor::binary<256>> all;
        for (int keyframe_id = 0; keyframe_id < 10; ++keyframe_id) {
            const std::vector<feature::descriptor::binary<256>> descriptors = random_descriptors(random, 100);
            tree_a.insert(keyframe_id, descriptors.data(), descriptors.size());
            tree_b.insert(keyframe_id, descriptors.data(), descriptors.size());
            all.insert(all.end(), descriptors.begin(), descriptors.end());
        }
        for (const feature::descriptor::binary<256>& query : all) {
            const std::vector<match::index::hbst::hit> hits_a = tree_a.search(query, 256);
            const std::vector<match::index::hbst::hit> hits_b = tree_b.search(query, 256);
            REQUIRE(hits_a.size() == hits_b.size());
            for (size_t i = 0; i < hits_a.size(); ++i) {
                REQUIRE(hits_a[i].keyframe_id == hits_b[i].keyframe_id);
                REQUIRE(hits_a[i].descriptor_index == hits_b[i].descriptor_index);
                REQUIRE(hits_a[i].distance == hits_b[i].distance);
            }
        }
    }

    return EXIT_SUCCESS;
}
