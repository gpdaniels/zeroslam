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

#include "feature/detector/fast.hpp"

#include "core/cpu.hpp"

namespace feature::detector {
#if defined(ZEROSLAM_SIMD_NEON)
    size_t detect_neon(const unsigned char* __restrict const data, const int width, const int height, const int stride, const int threshold, const size_t feature_point_buffer_size, point* __restrict feature_point_buffer);
#endif
#if defined(ZEROSLAM_SIMD_AVX2)
    size_t detect_avx2(const unsigned char* __restrict const data, const int width, const int height, const int stride, const int threshold, const size_t feature_point_buffer_size, point* __restrict feature_point_buffer);
#endif
    size_t detect_cpu(const unsigned char* __restrict const data, const int width, const int height, const int stride, const int threshold, const size_t feature_point_buffer_size, point* __restrict feature_point_buffer);

    size_t fast::detect(
        const unsigned char* __restrict const data,
        const int width,
        const int height,
        const int stride,
        const int threshold,
        const size_t feature_point_buffer_size,
        point* __restrict feature_point_buffer
    ) {
#if defined(ZEROSLAM_SIMD_NEON)
        if (core::cpu::has_neon()) {
            return detect_neon(data, width, height, stride, threshold, feature_point_buffer_size, feature_point_buffer);
        }
#endif
#if defined(ZEROSLAM_SIMD_AVX2)
        if (core::cpu::has_avx2()) {
            return detect_avx2(data, width, height, stride, threshold, feature_point_buffer_size, feature_point_buffer);
        }
#endif
        return detect_cpu(data, width, height, stride, threshold, feature_point_buffer_size, feature_point_buffer);
    }
}
