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

#include "feature/suppressor/fast.hpp"

#include "core/cpu.hpp"

namespace feature::suppressor {
#if defined(ZEROSLAM_SIMD_NEON)
    size_t suppress_neon(const point* __restrict const features, const size_t features_count, const size_t max_row, point* __restrict const features_suppressed);
#endif
#if defined(ZEROSLAM_SIMD_AVX2)
    size_t suppress_avx2(const point* __restrict const features, const size_t features_count, const size_t max_row, point* __restrict const features_suppressed);
#endif
    size_t suppress_cpu(const point* __restrict const features, const size_t features_count, const size_t max_row, point* __restrict const features_suppressed);

    size_t fast::suppress(
        const point* __restrict const features,
        const size_t features_count,
        const size_t max_row,
        point* __restrict const features_suppressed
    ) {
#if defined(ZEROSLAM_SIMD_NEON)
        if (core::cpu::has_neon()) {
            return suppress_neon(features, features_count, max_row, features_suppressed);
        }
#endif
#if defined(ZEROSLAM_SIMD_AVX2)
        if (core::cpu::has_avx2()) {
            return suppress_avx2(features, features_count, max_row, features_suppressed);
        }
#endif
        return suppress_cpu(features, features_count, max_row, features_suppressed);
    }
}
