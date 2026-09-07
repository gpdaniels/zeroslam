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

#include <arm_neon.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace match::distance {
    unsigned int distance_neon(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);

    unsigned int distance_neon(
        const unsigned char* __restrict const data_lhs,
        const unsigned char* __restrict const data_rhs
    ) {
        const uint8x16_t low = veorq_u8(vld1q_u8(data_lhs + 0), vld1q_u8(data_rhs + 0));
        const uint8x16_t high = veorq_u8(vld1q_u8(data_lhs + 16), vld1q_u8(data_rhs + 16));
        return static_cast<unsigned int>(vaddlvq_u8(vcntq_u8(low))) + static_cast<unsigned int>(vaddlvq_u8(vcntq_u8(high)));
    }
}
