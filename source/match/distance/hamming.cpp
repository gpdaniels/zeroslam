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

namespace match::distance {
#if defined(ZEROSLAM_SIMD_NEON)
    unsigned int distance_neon(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
    void distances_neon(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results);
    void distances_indexed_neon(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results);
#endif
#if defined(ZEROSLAM_SIMD_AVX2)
    unsigned int distance_avx2(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
    void distances_avx2(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results);
    void distances_indexed_avx2(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results);
#endif
#if defined(ZEROSLAM_SIMD_AVX)
    unsigned int distance_avx(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
    void distances_avx(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results);
    void distances_indexed_avx(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results);
#endif
    unsigned int distance_cpu(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
    void distances_cpu(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t descriptors_size, unsigned int* __restrict const results);
    void distances_indexed_cpu(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* __restrict const descriptors, const size_t* __restrict const indices, const size_t indices_size, unsigned int* __restrict const results);

    namespace {
        static_assert(sizeof(feature::descriptor::binary<256>) == 32, "The kernels step through descriptor arrays 32 bytes at a time.");

        class tier final {
        public:
            unsigned int (*distance)(const unsigned char* const data_lhs, const unsigned char* const data_rhs);
            void (*distances)(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* const descriptors, const size_t descriptors_size, unsigned int* const results);
            void (*distances_indexed)(const feature::descriptor::binary<256>& query, const feature::descriptor::binary<256>* const descriptors, const size_t* const indices, const size_t indices_size, unsigned int* const results);
        };

        tier select_tier() {
#if defined(ZEROSLAM_SIMD_NEON)
            if (core::cpu::has_neon()) {
                return tier{ distance_neon, distances_neon, distances_indexed_neon };
            }
#endif
#if defined(ZEROSLAM_SIMD_AVX2)
            if (core::cpu::has_avx2() && core::cpu::has_popcnt()) {
                return tier{ distance_avx2, distances_avx2, distances_indexed_avx2 };
            }
#endif
#if defined(ZEROSLAM_SIMD_AVX)
            if (core::cpu::has_avx() && core::cpu::has_popcnt()) {
                return tier{ distance_avx, distances_avx, distances_indexed_avx };
            }
#endif
            return tier{ distance_cpu, distances_cpu, distances_indexed_cpu };
        }

        const tier& selected_tier() {
            // A function local static is initialised exactly once even when first reached by several threads at once.
            static const tier selected = select_tier();
            return selected;
        }
    }

    unsigned int hamming::distance(
        const feature::descriptor::binary<256>& lhs,
        const feature::descriptor::binary<256>& rhs
    ) {
        return selected_tier().distance(lhs.data, rhs.data);
    }

    void hamming::distances(
        const feature::descriptor::binary<256>& query,
        const feature::descriptor::binary<256>* __restrict const descriptors,
        const size_t descriptors_size,
        unsigned int* __restrict const results
    ) {
        if (descriptors_size == 0) {
            return;
        }
        selected_tier().distances(query, descriptors, descriptors_size, results);
    }

    void hamming::distances(
        const feature::descriptor::binary<256>& query,
        const feature::descriptor::binary<256>* __restrict const descriptors,
        const size_t* __restrict const indices,
        const size_t indices_size,
        unsigned int* __restrict const results
    ) {
        if (indices_size == 0) {
            return;
        }
        selected_tier().distances_indexed(query, descriptors, indices, indices_size, results);
    }
}
