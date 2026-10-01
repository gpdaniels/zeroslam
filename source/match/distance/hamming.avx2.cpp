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
    unsigned int distance_avx2(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
    void distances_avx2(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results);
    void distances_indexed_avx2(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results);

    namespace {
        class contiguous final {
        public:
            const feature::descriptor::binary<256>* descriptors;

            const unsigned char* operator()(const size_t index) const {
                return this->descriptors[index].data;
            }
        };

        class indexed final {
        public:
            const feature::descriptor::binary<256>* descriptors;
            const size_t* indices;

            const unsigned char* operator()(const size_t index) const {
                return this->descriptors[this->indices[index]].data;
            }
        };

        // Bit counts by nibble lookup, summed over each 64-bit lane, so no 64-bit general purpose register (absent on 32-bit targets) is needed.
        __m256i lane_counts(const __m256i query, const unsigned char* __restrict const data) {
            const __m256i lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
            const __m256i nibble = _mm256_set1_epi8(0x0F);
            const __m256i difference = _mm256_xor_si256(query, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data)));
            const __m256i low = _mm256_shuffle_epi8(lookup, _mm256_and_si256(difference, nibble));
            const __m256i high = _mm256_shuffle_epi8(lookup, _mm256_and_si256(_mm256_srli_epi16(difference, 4), nibble));
            return _mm256_sad_epu8(_mm256_add_epi8(low, high), _mm256_setzero_si256());
        }

        unsigned int total(const __m256i lanes) {
            const __m128i sum = _mm_add_epi64(_mm256_castsi256_si128(lanes), _mm256_extracti128_si256(lanes, 1));
            return static_cast<unsigned int>(_mm_cvtsi128_si32(_mm_add_epi64(sum, _mm_unpackhi_epi64(sum, sum))));
        }

        template <typename descriptor_source>
        void distances_of(const feature::descriptor::binary<256>& query, const descriptor_source& descriptor_of, const size_t count, unsigned int* __restrict const results) {
            const __m256i query_bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query.data));
            const __m256i low_halves = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            size_t index = 0;
            for (; index + 4 <= count; index += 4) {
                const __m256i lanes_0 = lane_counts(query_bits, descriptor_of(index + 0));
                const __m256i lanes_1 = lane_counts(query_bits, descriptor_of(index + 1));
                const __m256i lanes_2 = lane_counts(query_bits, descriptor_of(index + 2));
                const __m256i lanes_3 = lane_counts(query_bits, descriptor_of(index + 3));
                // Reduce the four descriptors together, ending with one sum per 64-bit lane in order.
                const __m256i pairs_01 = _mm256_add_epi64(_mm256_unpacklo_epi64(lanes_0, lanes_1), _mm256_unpackhi_epi64(lanes_0, lanes_1));
                const __m256i pairs_23 = _mm256_add_epi64(_mm256_unpacklo_epi64(lanes_2, lanes_3), _mm256_unpackhi_epi64(lanes_2, lanes_3));
                const __m256i sums = _mm256_add_epi64(_mm256_permute2x128_si256(pairs_01, pairs_23, 0x20), _mm256_permute2x128_si256(pairs_01, pairs_23, 0x31));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(results + index), _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(sums, low_halves)));
            }
            for (; index < count; ++index) {
                results[index] = total(lane_counts(query_bits, descriptor_of(index)));
            }
        }
    }

    unsigned int distance_avx2(
        const unsigned char* __restrict const data_lhs,
        const unsigned char* __restrict const data_rhs
    ) {
        return total(lane_counts(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data_lhs)), data_rhs));
    }

    void distances_avx2(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results) {
        distances_of(query, contiguous{ descriptors }, descriptors_size, results);
    }

    void distances_indexed_avx2(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results) {
        distances_of(query, indexed{ descriptors, indices }, indices_size, results);
    }
}
