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

#include "feature/descriptor/orb.hpp"

#include "core/cpu.hpp"

namespace feature::descriptor {
#if defined(ZEROSLAM_SIMD_NEON)
    void describe_neon(const unsigned char* __restrict const data, const int stride, const float angle_radians, binary<256>& descriptor);
#endif
#if defined(ZEROSLAM_SIMD_AVX2)
    void describe_avx2(const unsigned char* __restrict const data, const int stride, const float angle_radians, binary<256>& descriptor);
#endif
    void describe_cpu(const unsigned char* __restrict const data, const int stride, const float angle_radians, binary<256>& descriptor);

    void orb::describe(
        const unsigned char* __restrict const data,
        const int stride,
        const float angle_radians,
        binary<256>& descriptor
    ) {
#if defined(ZEROSLAM_SIMD_NEON)
        if (core::cpu::has_neon()) {
            describe_neon(data, stride, angle_radians, descriptor);
            return;
        }
#endif
#if defined(ZEROSLAM_SIMD_AVX2)
        if (core::cpu::has_avx2() && core::cpu::has_popcnt()) {
            describe_avx2(data, stride, angle_radians, descriptor);
            return;
        }
#endif
        describe_cpu(data, stride, angle_radians, descriptor);
    }
}
