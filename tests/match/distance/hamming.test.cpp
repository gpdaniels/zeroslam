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

#include "match/distance/hamming.hpp"

#include "core/cpu.hpp"
#include "core/random_pcg.hpp"
#include "core/timestamp.hpp"

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

namespace match::distance {
    unsigned int distance_cpu(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
    void distances_cpu(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results);
    void distances_indexed_cpu(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results);
#if defined(ZEROSLAM_SIMD_AVX)
    unsigned int distance_avx(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
    void distances_avx(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results);
    void distances_indexed_avx(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results);
#endif
#if defined(ZEROSLAM_SIMD_AVX2)
    unsigned int distance_avx2(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
    void distances_avx2(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results);
    void distances_indexed_avx2(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results);
#endif
#if defined(ZEROSLAM_SIMD_NEON)
    unsigned int distance_neon(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
    void distances_neon(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results);
    void distances_indexed_neon(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results);
#endif
}

using distance_kernel = unsigned int (*)(const unsigned char* const data_lhs, const unsigned char* const data_rhs);
using distances_kernel = void (*)(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* const descriptors, const size_t descriptors_size, unsigned int* const results);
using distances_indexed_kernel = void (*)(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* const descriptors, const size_t* const indices, const size_t indices_size, unsigned int* const results);

static unsigned int bit_count_distance(const feature::descriptor::binary<256>& lhs, const feature::descriptor::binary<256>& rhs) {
    unsigned int count = 0;
    for (size_t byte = 0; byte < 32; ++byte) {
        for (unsigned int bit = 0; bit < 8; ++bit) {
            count += ((static_cast<unsigned int>(lhs.data[byte] ^ rhs.data[byte]) >> bit) & 1u);
        }
    }
    return count;
}

static std::vector<feature::descriptor::binary<256>> test_descriptors(const size_t count) {
    core::random_pcg random(0x6d61u);
    std::vector<feature::descriptor::binary<256>> descriptors(count);
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < 32; ++j) {
            const unsigned int value = random.get_random_raw();
            // Mix in all-clear and all-set bytes so full and empty lanes reach the kernels' widest partial sums.
            descriptors[i].data[j] = static_cast<unsigned char>(((i % 7) == 3) ? 0x00u : (((i % 7) == 5) ? 0xFFu : (value >> 13u)));
        }
    }
    return descriptors;
}

// Every kernel against the bit by bit count, for every count up to 17 so the four at a time loops and their tails are covered.
static void test_kernels(const distance_kernel distance, const distances_kernel distances, const distances_indexed_kernel distances_indexed) {
    const std::vector<feature::descriptor::binary<256>> descriptors = test_descriptors(257);
    std::vector<size_t> indices(descriptors.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = (i * 101u + 7u) % descriptors.size();
    }
    std::vector<unsigned int> results(descriptors.size() + 1);
    for (size_t query = 0; query < 16; ++query) {
        for (size_t i = 0; i < descriptors.size(); ++i) {
            REQUIRE(distance(descriptors[query].data, descriptors[i].data) == bit_count_distance(descriptors[query], descriptors[i]));
        }
        for (size_t count = 0; count <= 17; ++count) {
            for (size_t offset = 0; offset < 3; ++offset) {
                results.assign(results.size(), 1000u);
                distances(descriptors[query], descriptors.data() + offset, count, results.data());
                for (size_t i = 0; i < count; ++i) {
                    REQUIRE(results[i] == bit_count_distance(descriptors[query], descriptors[offset + i]));
                }
                REQUIRE(results[count] == 1000u);
                results.assign(results.size(), 1000u);
                distances_indexed(descriptors[query], descriptors.data(), indices.data() + offset, count, results.data());
                for (size_t i = 0; i < count; ++i) {
                    REQUIRE(results[i] == bit_count_distance(descriptors[query], descriptors[indices[offset + i]]));
                }
                REQUIRE(results[count] == 1000u);
            }
        }
        distances(descriptors[query], descriptors.data(), descriptors.size(), results.data());
        for (size_t i = 0; i < descriptors.size(); ++i) {
            REQUIRE(results[i] == bit_count_distance(descriptors[query], descriptors[i]));
        }
        distances_indexed(descriptors[query], descriptors.data(), indices.data(), indices.size(), results.data());
        for (size_t i = 0; i < indices.size(); ++i) {
            REQUIRE(results[i] == bit_count_distance(descriptors[query], descriptors[indices[i]]));
        }
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        feature::descriptor::binary<256> zero{};
        feature::descriptor::binary<256> ones;
        for (size_t i = 0; i < 32; ++i) {
            ones.data[i] = 0xFF;
        }
        REQUIRE(match::distance::hamming::distance(zero, zero) == 0);
        REQUIRE(match::distance::hamming::distance(ones, ones) == 0);
        REQUIRE(match::distance::hamming::distance(zero, ones) == 256);
        REQUIRE(match::distance::hamming::distance(ones, zero) == 256);
        feature::descriptor::binary<256> one{};
        one.data[5] = 0x11;
        REQUIRE(match::distance::hamming::distance(zero, one) == 2);
        REQUIRE(match::distance::hamming::distance(one, ones) == 254);

        const feature::descriptor::binary<256> many[5] = { zero, ones, one, ones, zero };
        unsigned int results[5] = { 0, 0, 0, 0, 0 };
        match::distance::hamming::distances(zero, &many[0], 5, &results[0]);
        REQUIRE((results[0] == 0) && (results[1] == 256) && (results[2] == 2) && (results[3] == 256) && (results[4] == 0));
        const size_t indices[3] = { 2, 2, 1 };
        match::distance::hamming::distances(ones, &many[0], &indices[0], 3, &results[0]);
        REQUIRE((results[0] == 254) && (results[1] == 254) && (results[2] == 0));
        match::distance::hamming::distances(ones, nullptr, 0, nullptr);
        match::distance::hamming::distances(ones, nullptr, nullptr, 0, nullptr);
    }

    {
        unsigned long long int lhs[4] = { 0, 0, 0, 1 };
        unsigned long long int rhs[4] = { 0, 0, 0, 0 };
        while ((lhs[0] != 0) || (lhs[1] != 0) || (lhs[2] != 0) || (lhs[3] != 0)) {
            REQUIRE(match::distance::distance_cpu(reinterpret_cast<unsigned char*>(&lhs[0]), reinterpret_cast<unsigned char*>(&rhs[0])) == 1);
            lhs[3] <<= 1;
            for (int i = 2; i >= 0; --i) {
                if ((lhs[0] == 0) && (lhs[1] == 0) && (lhs[2] == 0) && (lhs[3] == 0)) {
                    lhs[i] = 0b0000000000000000000000000000000000000000000000000000000000000001;
                }
                else {
                    lhs[i] <<= 1;
                }
            }
        }
    }

    test_kernels(match::distance::distance_cpu, match::distance::distances_cpu, match::distance::distances_indexed_cpu);

#if defined(ZEROSLAM_SIMD_AVX)
    {
        if (core::cpu::has_avx() && core::cpu::has_popcnt()) {
            std::vector<feature::descriptor::binary<256>> lhs(256);
            std::vector<feature::descriptor::binary<256>> rhs(256);
            for (size_t i = 0; i < 256; ++i) {
                for (size_t j = 0; j < 32; ++j) {
                    lhs[i].data[j] = static_cast<unsigned char>((i * 31 + j * 17) % 256);
                    rhs[i].data[j] = static_cast<unsigned char>((i * 13 + j * 29 + 7) % 256);
                }
            }
            for (size_t i = 0; i < 256; ++i) {
                REQUIRE(match::distance::distance_cpu(lhs[i].data, rhs[i].data) == match::distance::distance_avx(lhs[i].data, rhs[i].data));
            }
            REQUIRE(match::distance::hamming::distance(lhs[0], rhs[0]) == match::distance::distance_cpu(lhs[0].data, rhs[0].data));
            test_kernels(match::distance::distance_avx, match::distance::distances_avx, match::distance::distances_indexed_avx);
        }
    }
#endif

#if defined(ZEROSLAM_SIMD_AVX2)
    {
        if (core::cpu::has_avx2() && core::cpu::has_popcnt()) {
            std::vector<feature::descriptor::binary<256>> lhs(256);
            std::vector<feature::descriptor::binary<256>> rhs(256);
            for (size_t i = 0; i < 256; ++i) {
                for (size_t j = 0; j < 32; ++j) {
                    lhs[i].data[j] = static_cast<unsigned char>((i * 31 + j * 17) % 256);
                    rhs[i].data[j] = static_cast<unsigned char>((i * 13 + j * 29 + 7) % 256);
                }
            }
            for (size_t i = 0; i < 256; ++i) {
                REQUIRE(match::distance::distance_cpu(lhs[i].data, rhs[i].data) == match::distance::distance_avx2(lhs[i].data, rhs[i].data));
            }
            REQUIRE(match::distance::hamming::distance(lhs[0], rhs[0]) == match::distance::distance_cpu(lhs[0].data, rhs[0].data));
            test_kernels(match::distance::distance_avx2, match::distance::distances_avx2, match::distance::distances_indexed_avx2);
        }
    }
#endif

#if defined(ZEROSLAM_SIMD_NEON)
    {
        if (core::cpu::has_neon()) {
            std::vector<feature::descriptor::binary<256>> lhs(256);
            std::vector<feature::descriptor::binary<256>> rhs(256);
            for (size_t i = 0; i < 256; ++i) {
                for (size_t j = 0; j < 32; ++j) {
                    lhs[i].data[j] = static_cast<unsigned char>((i * 31 + j * 17) % 256);
                    rhs[i].data[j] = static_cast<unsigned char>((i * 13 + j * 29 + 7) % 256);
                }
            }
            for (size_t i = 0; i < 256; ++i) {
                REQUIRE(match::distance::distance_cpu(lhs[i].data, rhs[i].data) == match::distance::distance_neon(lhs[i].data, rhs[i].data));
            }
            REQUIRE(match::distance::hamming::distance(lhs[0], rhs[0]) == match::distance::distance_cpu(lhs[0].data, rhs[0].data));
            test_kernels(match::distance::distance_neon, match::distance::distances_neon, match::distance::distances_indexed_neon);
        }
    }
#endif

    // The dispatched one to many distances match the pairwise ones, and their timing.
    {
        const size_t count = 2000;
        const std::vector<feature::descriptor::binary<256>> descriptors = test_descriptors(2 * count);
        std::vector<unsigned int> pairwise(count);
        std::vector<unsigned int> batched(count);
        unsigned long long int pairwise_sum = 0;
        unsigned long long int batched_sum = 0;
        const long long int pairwise_start = core::timestamp();
        for (size_t i = 0; i < count; ++i) {
            for (size_t j = 0; j < count; ++j) {
                pairwise[j] = match::distance::hamming::distance(descriptors[i], descriptors[count + j]);
            }
            pairwise_sum += pairwise[i];
        }
        const long long int pairwise_end = core::timestamp();
        const long long int batched_start = core::timestamp();
        for (size_t i = 0; i < count; ++i) {
            match::distance::hamming::distances(descriptors[i], descriptors.data() + count, count, batched.data());
            batched_sum += batched[i];
        }
        const long long int batched_end = core::timestamp();
        REQUIRE(pairwise_sum == batched_sum);
        for (size_t j = 0; j < count; ++j) {
            REQUIRE(pairwise[j] == batched[j]);
        }
        std::printf("Hamming distances (%zu x %zu descriptors):\n", count, count);
        std::printf("  One to one: %lld us\n", (pairwise_end - pairwise_start) / 1000ll);
        std::printf("  One to many: %lld us\n", (batched_end - batched_start) / 1000ll);
    }

    return EXIT_SUCCESS;
}
