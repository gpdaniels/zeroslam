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
    unsigned int distance_avx(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);

    unsigned int distance_avx(
        const unsigned char* __restrict const data_lhs,
        const unsigned char* __restrict const data_rhs
    ) {
        const __m128i low = _mm_xor_si128(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(data_lhs + 0)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(data_rhs + 0))
        );
        const __m128i high = _mm_xor_si128(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(data_lhs + 16)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(data_rhs + 16))
        );
        const long long sum0 = _mm_popcnt_u64(static_cast<unsigned long long>(_mm_extract_epi64(low, 0)));
        const long long sum1 = _mm_popcnt_u64(static_cast<unsigned long long>(_mm_extract_epi64(low, 1)));
        const long long sum2 = _mm_popcnt_u64(static_cast<unsigned long long>(_mm_extract_epi64(high, 0)));
        const long long sum3 = _mm_popcnt_u64(static_cast<unsigned long long>(_mm_extract_epi64(high, 1)));
        return static_cast<unsigned int>((sum0 + sum1) + (sum2 + sum3));
    }
}
