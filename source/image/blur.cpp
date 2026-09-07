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

#include "image/blur.hpp"

#include "core/cpu.hpp"

namespace image {
#if defined(ZEROSLAM_SIMD_NEON)
    void gaussian_5x5_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
    void gaussian_7x7_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
#endif
#if defined(ZEROSLAM_SIMD_AVX2)
    void gaussian_5x5_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
    void gaussian_7x7_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
#endif
    void gaussian_5x5_cpu(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
    void gaussian_7x7_cpu(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);

    void blur::gaussian_5x5(
        const unsigned char* __restrict const source_data,
        const int source_width,
        const int source_height,
        const int source_stride,
        unsigned char* __restrict const target_data
    ) {
#if defined(ZEROSLAM_SIMD_NEON)
        if (core::cpu::has_neon()) {
            gaussian_5x5_neon(source_data, source_width, source_height, source_stride, target_data);
            return;
        }
#endif
#if defined(ZEROSLAM_SIMD_AVX2)
        if (core::cpu::has_avx2()) {
            gaussian_5x5_avx2(source_data, source_width, source_height, source_stride, target_data);
            return;
        }
#endif
        gaussian_5x5_cpu(source_data, source_width, source_height, source_stride, target_data);
    }

    void blur::gaussian_7x7(
        const unsigned char* __restrict const source_data,
        const int source_width,
        const int source_height,
        const int source_stride,
        unsigned char* __restrict const target_data
    ) {
#if defined(ZEROSLAM_SIMD_NEON)
        if (core::cpu::has_neon()) {
            gaussian_7x7_neon(source_data, source_width, source_height, source_stride, target_data);
            return;
        }
#endif
#if defined(ZEROSLAM_SIMD_AVX2)
        if (core::cpu::has_avx2()) {
            gaussian_7x7_avx2(source_data, source_width, source_height, source_stride, target_data);
            return;
        }
#endif
        gaussian_7x7_cpu(source_data, source_width, source_height, source_stride, target_data);
    }
}
