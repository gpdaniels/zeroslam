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

#include "feature/descriptor/binary.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <immintrin.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace match::distance {
    unsigned int distance_avx(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
    void distances_avx(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results);
    void distances_indexed_avx(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results);
    unsigned int distance_512_avx(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
    void distances_512_avx(const feature::descriptor::binary<512>& query, const feature::descriptor::binary<512>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results);
    void distances_indexed_512_avx(const feature::descriptor::binary<512>& query, const feature::descriptor::binary<512>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results);

    namespace {
        template <size_t size_bits>
        class contiguous final {
        public:
            const feature::descriptor::binary<size_bits>* descriptors;

            const unsigned char* operator()(const size_t index) const {
                return this->descriptors[index].data;
            }
        };

        template <size_t size_bits>
        class indexed final {
        public:
            const feature::descriptor::binary<size_bits>* descriptors;
            const size_t* indices;

            const unsigned char* operator()(const size_t index) const {
                return this->descriptors[this->indices[index]].data;
            }
        };

        __m128i byte_counts(const __m128i value) {
            const __m128i lookup = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
            const __m128i nibble = _mm_set1_epi8(0x0F);
            return _mm_add_epi8(_mm_shuffle_epi8(lookup, _mm_and_si128(value, nibble)), _mm_shuffle_epi8(lookup, _mm_and_si128(_mm_srli_epi16(value, 4), nibble)));
        }

        // Bit counts by nibble lookup, summed over each 64-bit lane, so no 64-bit general purpose register (absent on 32-bit targets) is needed.
        __m128i lane_counts(const __m128i query_low, const __m128i query_high, const unsigned char* __restrict const data) {
            const __m128i low = _mm_xor_si128(query_low, _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 0)));
            const __m128i high = _mm_xor_si128(query_high, _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 16)));
            return _mm_sad_epu8(_mm_add_epi8(byte_counts(low), byte_counts(high)), _mm_setzero_si128());
        }

        unsigned int total(const __m128i lanes) {
            return static_cast<unsigned int>(_mm_cvtsi128_si32(_mm_add_epi64(lanes, _mm_unpackhi_epi64(lanes, lanes))));
        }

        // The counts of every 32 byte block of the descriptor, summed lane by lane (at most 128 per block and lane).
        template <size_t blocks>
        __m128i descriptor_counts(const __m128i (&query_bits)[2 * blocks], const unsigned char* __restrict const data) {
            __m128i lanes = lane_counts(query_bits[0], query_bits[1], data);
            for (size_t block = 1; block < blocks; ++block) {
                lanes = _mm_add_epi64(lanes, lane_counts(query_bits[2 * block], query_bits[(2 * block) + 1], data + (32 * block)));
            }
            return lanes;
        }

        template <size_t size_bits, typename descriptor_source>
        void distances_of(const feature::descriptor::binary<size_bits>& query, const descriptor_source& descriptor_of, const size_t count, unsigned int* __restrict const results) {
            constexpr static const size_t blocks = size_bits / 256;
            static_assert((blocks * 256) == size_bits, "The kernel steps through descriptors 32 bytes at a time.");
            __m128i query_bits[2 * blocks];
            for (size_t half = 0; half < 2 * blocks; ++half) {
                query_bits[half] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(query.data + (16 * half)));
            }
            size_t index = 0;
            for (; index + 4 <= count; index += 4) {
                const __m128i lanes_0 = descriptor_counts<blocks>(query_bits, descriptor_of(index + 0));
                const __m128i lanes_1 = descriptor_counts<blocks>(query_bits, descriptor_of(index + 1));
                const __m128i lanes_2 = descriptor_counts<blocks>(query_bits, descriptor_of(index + 2));
                const __m128i lanes_3 = descriptor_counts<blocks>(query_bits, descriptor_of(index + 3));
                // Reduce the four descriptors together, then gather the low half of each 64-bit sum.
                const __m128i sums_01 = _mm_add_epi64(_mm_unpacklo_epi64(lanes_0, lanes_1), _mm_unpackhi_epi64(lanes_0, lanes_1));
                const __m128i sums_23 = _mm_add_epi64(_mm_unpacklo_epi64(lanes_2, lanes_3), _mm_unpackhi_epi64(lanes_2, lanes_3));
                const __m128i sums = _mm_unpacklo_epi64(_mm_shuffle_epi32(sums_01, 0x08), _mm_shuffle_epi32(sums_23, 0x08));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(results + index), sums);
            }
            for (; index < count; ++index) {
                results[index] = total(descriptor_counts<blocks>(query_bits, descriptor_of(index)));
            }
        }
    }

    unsigned int distance_avx(
        const unsigned char* __restrict const data_lhs,
        const unsigned char* __restrict const data_rhs
    ) {
        return total(lane_counts(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data_lhs + 0)), _mm_loadu_si128(reinterpret_cast<const __m128i*>(data_lhs + 16)), data_rhs));
    }

    void distances_avx(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results) {
        distances_of(query, contiguous<256>{ descriptors }, descriptors_size, results);
    }

    void distances_indexed_avx(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results) {
        distances_of(query, indexed<256>{ descriptors, indices }, indices_size, results);
    }

    unsigned int distance_512_avx(
        const unsigned char* __restrict const data_lhs,
        const unsigned char* __restrict const data_rhs
    ) {
        return distance_avx(data_lhs, data_rhs) + distance_avx(data_lhs + 32, data_rhs + 32);
    }

    void distances_512_avx(const feature::descriptor::binary<512>& query, const feature::descriptor::binary<512>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results) {
        distances_of(query, contiguous<512>{ descriptors }, descriptors_size, results);
    }

    void distances_indexed_512_avx(const feature::descriptor::binary<512>& query, const feature::descriptor::binary<512>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results) {
        distances_of(query, indexed<512>{ descriptors, indices }, indices_size, results);
    }
}
