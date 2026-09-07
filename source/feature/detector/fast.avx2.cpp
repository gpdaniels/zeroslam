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

#include "feature/point.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <immintrin.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::detector {
    size_t detect_avx2(const unsigned char* __restrict const data, const int width, const int height, const int stride, const int threshold, const size_t feature_point_buffer_size, point* __restrict feature_point_buffer);

    namespace {
        static inline unsigned int count_trailing_zeros(const unsigned int mask) {
#if defined(_MSC_VER)
            return _tzcnt_u32(mask);
#else
            return static_cast<unsigned int>(__builtin_ctz(mask));
#endif
        }
    }

    size_t detect_avx2(
        const unsigned char* __restrict const data,
        const int width,
        const int height,
        const int stride,
        const int threshold,
        const size_t feature_point_buffer_size,
        point* __restrict feature_point_buffer
    ) {
        constexpr static const int patch_size = 7;
        constexpr static const int patch_radius = patch_size / 2;

        const int pattern_offset[24] = {
            +patch_radius * stride + 0,
            +patch_radius * stride + 1,
            +2 * stride + 2,
            +patch_radius + stride + 0,
            +patch_radius,
            +patch_radius - stride + 0,
            -2 * stride + 2,
            -patch_radius * stride + 1,
            -patch_radius * stride + 0,
            -patch_radius * stride - 1,
            -2 * stride - 2,
            -patch_radius - stride + 0,
            -patch_radius,
            -patch_radius + stride + 0,
            +2 * stride - 2,
            +patch_radius * stride - 1,
            +patch_radius * stride + 0,
            +patch_radius * stride + 1,
            +2 * stride + 2,
            +patch_radius + stride + 0,
            +patch_radius,
            +patch_radius - stride + 0,
            -2 * stride + 2,
            -patch_radius * stride + 1
        };

        // Unsigned shift: XOR with 0x80 to convert unsigned comparisons to signed.
        const __m256i ushft = _mm256_set1_epi8(-128);

        // Threshold broadcast to all 32 bytes.
        const __m256i t = _mm256_set1_epi8(static_cast<char>(threshold));

        // Consecutive count threshold: > 8 means corner (i.e. 9 consecutive).
        const __m256i consec = _mm256_set1_epi8(8);

        // Bounded memcpy lambda for handling the last partial chunk of columns.
        const auto bounded_load = [&ushft](const unsigned char* src, int remaining) -> __m256i {
            alignas(32) unsigned char buf[32] = {};
            const int count = (remaining > 32) ? 32 : ((remaining > 0) ? remaining : 0);
            for (int i = 0; i < count; ++i) {
                buf[i] = src[i];
            }
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(buf));
        };

        size_t feature_count = 0;

        for (int y = patch_radius; y < height - patch_radius; ++y) {
            const unsigned char* ptr = data + y * stride + patch_radius;

            int x = patch_radius;
            for (; x <= width - patch_radius - 32; x += 32, ptr += 32) {
                if (feature_count >= feature_point_buffer_size) {
                    return feature_count;
                }

                // Load 32 center pixels and 4 cardinal circle pixels.
                const __m256i ppt_orig = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
                const __m256i p0 = _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + pattern_offset[0])), ushft);
                const __m256i p4 = _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + pattern_offset[4])), ushft);
                const __m256i p8 = _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + pattern_offset[8])), ushft);
                const __m256i p12 = _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + pattern_offset[12])), ushft);

                // p - threshold and p + threshold (in shifted space).
                const __m256i pmt = _mm256_xor_si256(_mm256_subs_epu8(ppt_orig, t), ushft);
                const __m256i ppt = _mm256_xor_si256(_mm256_adds_epu8(ppt_orig, t), ushft);

                // Quick bail: check if at least 2 of 4 adjacent cardinal points are brighter or darker.
                __m256i ppt_accum = _mm256_and_si256(_mm256_cmpgt_epi8(p0, ppt), _mm256_cmpgt_epi8(p4, ppt));
                __m256i pmt_accum = _mm256_and_si256(_mm256_cmpgt_epi8(pmt, p0), _mm256_cmpgt_epi8(pmt, p4));

                ppt_accum = _mm256_or_si256(ppt_accum, _mm256_and_si256(_mm256_cmpgt_epi8(p4, ppt), _mm256_cmpgt_epi8(p8, ppt)));
                pmt_accum = _mm256_or_si256(pmt_accum, _mm256_and_si256(_mm256_cmpgt_epi8(pmt, p4), _mm256_cmpgt_epi8(pmt, p8)));

                ppt_accum = _mm256_or_si256(ppt_accum, _mm256_and_si256(_mm256_cmpgt_epi8(p8, ppt), _mm256_cmpgt_epi8(p12, ppt)));
                pmt_accum = _mm256_or_si256(pmt_accum, _mm256_and_si256(_mm256_cmpgt_epi8(pmt, p8), _mm256_cmpgt_epi8(pmt, p12)));

                ppt_accum = _mm256_or_si256(ppt_accum, _mm256_and_si256(_mm256_cmpgt_epi8(p12, ppt), _mm256_cmpgt_epi8(p0, ppt)));
                pmt_accum = _mm256_or_si256(pmt_accum, _mm256_and_si256(_mm256_cmpgt_epi8(pmt, p12), _mm256_cmpgt_epi8(pmt, p0)));

                unsigned int m = static_cast<unsigned int>(_mm256_movemask_epi8(_mm256_or_si256(ppt_accum, pmt_accum)));
                if (m == 0) {
                    continue;
                }

                // Full circle test: count consecutive salient pixels.
                __m256i ppt_cnt = _mm256_setzero_si256();
                __m256i pmt_cnt = _mm256_setzero_si256();
                __m256i ppt_max = _mm256_setzero_si256();
                __m256i pmt_max = _mm256_setzero_si256();

                for (int k = 0; k < 24; ++k) {
                    const __m256i pk = _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + pattern_offset[k])), ushft);

                    const __m256i bright = _mm256_cmpgt_epi8(pk, ppt);
                    const __m256i dark = _mm256_cmpgt_epi8(pmt, pk);

                    ppt_cnt = _mm256_and_si256(_mm256_sub_epi8(ppt_cnt, bright), bright);
                    ppt_max = _mm256_max_epu8(ppt_max, ppt_cnt);

                    pmt_cnt = _mm256_and_si256(_mm256_sub_epi8(pmt_cnt, dark), dark);
                    pmt_max = _mm256_max_epu8(pmt_max, pmt_cnt);
                }

                m = static_cast<unsigned int>(_mm256_movemask_epi8(
                    _mm256_cmpgt_epi8(_mm256_max_epu8(ppt_max, pmt_max), consec)
                ));

                while (m) {
                    const unsigned int bit = count_trailing_zeros(m);
                    m &= (m - 1);

                    if (feature_count < feature_point_buffer_size) {
                        feature_point_buffer[feature_count++] = {
                            static_cast<float>(x + static_cast<int>(bit)),
                            static_cast<float>(y),
                            0,
                            0,
                            0
                        };
                    }
                }
            }

            // Handle remaining columns.
            if (x < width - patch_radius) {
                if (feature_count >= feature_point_buffer_size) {
                    return feature_count;
                }

                const int remaining = width - patch_radius - x;

                const __m256i ppt_orig = bounded_load(ptr, remaining);
                const __m256i p0 = _mm256_xor_si256(bounded_load(ptr + pattern_offset[0], remaining), ushft);
                const __m256i p4 = _mm256_xor_si256(bounded_load(ptr + pattern_offset[4], remaining), ushft);
                const __m256i p8 = _mm256_xor_si256(bounded_load(ptr + pattern_offset[8], remaining), ushft);
                const __m256i p12 = _mm256_xor_si256(bounded_load(ptr + pattern_offset[12], remaining), ushft);

                const __m256i pmt = _mm256_xor_si256(_mm256_subs_epu8(ppt_orig, t), ushft);
                const __m256i ppt = _mm256_xor_si256(_mm256_adds_epu8(ppt_orig, t), ushft);

                __m256i ppt_accum = _mm256_and_si256(_mm256_cmpgt_epi8(p0, ppt), _mm256_cmpgt_epi8(p4, ppt));
                __m256i pmt_accum = _mm256_and_si256(_mm256_cmpgt_epi8(pmt, p0), _mm256_cmpgt_epi8(pmt, p4));

                ppt_accum = _mm256_or_si256(ppt_accum, _mm256_and_si256(_mm256_cmpgt_epi8(p4, ppt), _mm256_cmpgt_epi8(p8, ppt)));
                pmt_accum = _mm256_or_si256(pmt_accum, _mm256_and_si256(_mm256_cmpgt_epi8(pmt, p4), _mm256_cmpgt_epi8(pmt, p8)));

                ppt_accum = _mm256_or_si256(ppt_accum, _mm256_and_si256(_mm256_cmpgt_epi8(p8, ppt), _mm256_cmpgt_epi8(p12, ppt)));
                pmt_accum = _mm256_or_si256(pmt_accum, _mm256_and_si256(_mm256_cmpgt_epi8(pmt, p8), _mm256_cmpgt_epi8(pmt, p12)));

                ppt_accum = _mm256_or_si256(ppt_accum, _mm256_and_si256(_mm256_cmpgt_epi8(p12, ppt), _mm256_cmpgt_epi8(p0, ppt)));
                pmt_accum = _mm256_or_si256(pmt_accum, _mm256_and_si256(_mm256_cmpgt_epi8(pmt, p12), _mm256_cmpgt_epi8(pmt, p0)));

                const unsigned int last_cols_mask = static_cast<unsigned int>((1ull << remaining) - 1);
                unsigned int m = static_cast<unsigned int>(_mm256_movemask_epi8(_mm256_or_si256(ppt_accum, pmt_accum))) & last_cols_mask;

                if (m != 0) {
                    __m256i ppt_cnt = _mm256_setzero_si256();
                    __m256i pmt_cnt = _mm256_setzero_si256();
                    __m256i ppt_max = _mm256_setzero_si256();
                    __m256i pmt_max = _mm256_setzero_si256();

                    for (int k = 0; k < 24; ++k) {
                        const __m256i pk = _mm256_xor_si256(bounded_load(ptr + pattern_offset[k], remaining), ushft);

                        const __m256i bright = _mm256_cmpgt_epi8(pk, ppt);
                        const __m256i dark = _mm256_cmpgt_epi8(pmt, pk);

                        ppt_cnt = _mm256_and_si256(_mm256_sub_epi8(ppt_cnt, bright), bright);
                        ppt_max = _mm256_max_epu8(ppt_max, ppt_cnt);

                        pmt_cnt = _mm256_and_si256(_mm256_sub_epi8(pmt_cnt, dark), dark);
                        pmt_max = _mm256_max_epu8(pmt_max, pmt_cnt);
                    }

                    m = static_cast<unsigned int>(_mm256_movemask_epi8(
                            _mm256_cmpgt_epi8(_mm256_max_epu8(ppt_max, pmt_max), consec)
                        )) &
                        last_cols_mask;

                    while (m) {
                        const unsigned int bit = count_trailing_zeros(m);
                        m &= (m - 1);

                        if (feature_count < feature_point_buffer_size) {
                            feature_point_buffer[feature_count++] = {
                                static_cast<float>(x + static_cast<int>(bit)),
                                static_cast<float>(y),
                                0,
                                0,
                                0
                            };
                        }
                    }
                }
            }
        }

        return feature_count;
    }
}
