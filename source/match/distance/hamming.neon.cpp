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

#include <arm_neon.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace match::distance {
    unsigned int distance_neon(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
    void distances_neon(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results);
    void distances_indexed_neon(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results);
    unsigned int distance_512_neon(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
    void distances_512_neon(const feature::descriptor::binary<512>& query, const feature::descriptor::binary<512>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results);
    void distances_indexed_512_neon(const feature::descriptor::binary<512>& query, const feature::descriptor::binary<512>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results);

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

        // The bit counts of the two 16 byte halves added bytewise, at most 16 per byte.
        uint8x16_t byte_counts(const uint8x16_t query_low, const uint8x16_t query_high, const unsigned char* __restrict const data) {
            return vaddq_u8(vcntq_u8(veorq_u8(query_low, vld1q_u8(data + 0))), vcntq_u8(veorq_u8(query_high, vld1q_u8(data + 16))));
        }

        // The bit counts of every 32 byte block added bytewise, at most 16 per byte and block.
        template <size_t blocks>
        uint8x16_t descriptor_counts(const uint8x16_t (&query_bits)[2 * blocks], const unsigned char* __restrict const data) {
            uint8x16_t counts = byte_counts(query_bits[0], query_bits[1], data);
            for (size_t block = 1; block < blocks; ++block) {
                counts = vaddq_u8(counts, byte_counts(query_bits[2 * block], query_bits[(2 * block) + 1], data + (32 * block)));
            }
            return counts;
        }

        template <size_t size_bits, typename descriptor_source>
        void distances_of(const feature::descriptor::binary<size_bits>& query, const descriptor_source& descriptor_of, const size_t count, unsigned int* __restrict const results) {
            constexpr static const size_t blocks = size_bits / 256;
            static_assert(((blocks * 256) == size_bits) && (blocks <= 3), "The kernel steps through descriptors 32 bytes at a time, and four blocks would overflow the pairwise byte sums.");
            uint8x16_t query_bits[2 * blocks];
            for (size_t half = 0; half < 2 * blocks; ++half) {
                query_bits[half] = vld1q_u8(query.data + (16 * half));
            }
            size_t index = 0;
            for (; index + 4 <= count; index += 4) {
                const uint8x16_t counts_0 = descriptor_counts<blocks>(query_bits, descriptor_of(index + 0));
                const uint8x16_t counts_1 = descriptor_counts<blocks>(query_bits, descriptor_of(index + 1));
                const uint8x16_t counts_2 = descriptor_counts<blocks>(query_bits, descriptor_of(index + 2));
                const uint8x16_t counts_3 = descriptor_counts<blocks>(query_bits, descriptor_of(index + 3));
                // Pairwise adds leave four bytes per descriptor in order (at most 64 per block each), then two widening adds give one sum each.
                const uint8x16_t quarters = vpaddq_u8(vpaddq_u8(counts_0, counts_1), vpaddq_u8(counts_2, counts_3));
                vst1q_u32(results + index, vpaddlq_u16(vpaddlq_u8(quarters)));
            }
            for (; index < count; ++index) {
                results[index] = static_cast<unsigned int>(vaddlvq_u8(descriptor_counts<blocks>(query_bits, descriptor_of(index))));
            }
        }
    }

    unsigned int distance_neon(
        const unsigned char* __restrict const data_lhs,
        const unsigned char* __restrict const data_rhs
    ) {
        const uint8x16_t low = veorq_u8(vld1q_u8(data_lhs + 0), vld1q_u8(data_rhs + 0));
        const uint8x16_t high = veorq_u8(vld1q_u8(data_lhs + 16), vld1q_u8(data_rhs + 16));
        return static_cast<unsigned int>(vaddlvq_u8(vcntq_u8(low))) + static_cast<unsigned int>(vaddlvq_u8(vcntq_u8(high)));
    }

    void distances_neon(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results) {
        distances_of(query, contiguous<256>{ descriptors }, descriptors_size, results);
    }

    void distances_indexed_neon(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results) {
        distances_of(query, indexed<256>{ descriptors, indices }, indices_size, results);
    }

    unsigned int distance_512_neon(
        const unsigned char* __restrict const data_lhs,
        const unsigned char* __restrict const data_rhs
    ) {
        return distance_neon(data_lhs, data_rhs) + distance_neon(data_lhs + 32, data_rhs + 32);
    }

    void distances_512_neon(const feature::descriptor::binary<512>& query, const feature::descriptor::binary<512>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results) {
        distances_of(query, contiguous<512>{ descriptors }, descriptors_size, results);
    }

    void distances_indexed_512_neon(const feature::descriptor::binary<512>& query, const feature::descriptor::binary<512>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results) {
        distances_of(query, indexed<512>{ descriptors, indices }, indices_size, results);
    }
}
