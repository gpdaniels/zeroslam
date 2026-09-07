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

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <immintrin.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace match::distance {
    unsigned int distance_avx2(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);

    unsigned int distance_avx2(
        const unsigned char* __restrict const data_lhs,
        const unsigned char* __restrict const data_rhs
    ) {
        const __m256i lhs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data_lhs));
        const __m256i rhs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data_rhs));
        const __m256i xor_result = _mm256_xor_si256(lhs, rhs);
        const __m128i lo = _mm256_castsi256_si128(xor_result);
        const __m128i hi = _mm256_extracti128_si256(xor_result, 1);
        const long long sum0 = _mm_popcnt_u64(static_cast<unsigned long long>(_mm_extract_epi64(lo, 0)));
        const long long sum1 = _mm_popcnt_u64(static_cast<unsigned long long>(_mm_extract_epi64(lo, 1)));
        const long long sum2 = _mm_popcnt_u64(static_cast<unsigned long long>(_mm_extract_epi64(hi, 0)));
        const long long sum3 = _mm_popcnt_u64(static_cast<unsigned long long>(_mm_extract_epi64(hi, 1)));
        return static_cast<unsigned int>((sum0 + sum1) + (sum2 + sum3));
    }
}
