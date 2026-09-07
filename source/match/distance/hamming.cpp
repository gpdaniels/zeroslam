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
#endif
#if defined(ZEROSLAM_SIMD_AVX2)
    unsigned int distance_avx2(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
#endif
#if defined(ZEROSLAM_SIMD_AVX)
    unsigned int distance_avx(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
#endif
    unsigned int distance_cpu(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);

    unsigned int hamming::distance(
        const feature::descriptor::binary<256>& lhs,
        const feature::descriptor::binary<256>& rhs
    ) {
#if defined(ZEROSLAM_SIMD_NEON)
        if (core::cpu::has_neon()) {
            return distance_neon(lhs.data, rhs.data);
        }
#endif
#if defined(ZEROSLAM_SIMD_AVX2)
        if (core::cpu::has_avx2() && core::cpu::has_popcnt()) {
            return distance_avx2(lhs.data, rhs.data);
        }
#endif
#if defined(ZEROSLAM_SIMD_AVX)
        if (core::cpu::has_avx() && core::cpu::has_popcnt()) {
            return distance_avx(lhs.data, rhs.data);
        }
#endif
        return distance_cpu(lhs.data, rhs.data);
    }
}
