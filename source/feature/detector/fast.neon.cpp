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

#include <arm_neon.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::detector {
    size_t detect_neon(const unsigned char* __restrict const data, const int width, const int height, const int stride, const int threshold, const size_t feature_point_buffer_size, point* __restrict feature_point_buffer);

    namespace {
        // Neon has no movemask, so narrow each 16-bit pair of lane masks with a shift by four: that packs
        // the upper nibble of the even lane and the lower nibble of the odd lane into one byte, which gives
        // four bits per lane in a 64-bit integer. Keeping only the lowest bit of every nibble leaves lane i
        // set at bit 4 * i.
        static inline unsigned long long lane_mask(const uint8x16_t mask) {
            const uint8x8_t packed = vshrn_n_u16(vreinterpretq_u16_u8(mask), 4);
            return vget_lane_u64(vreinterpret_u64_u8(packed), 0) & 0x1111111111111111ull;
        }

    }

    size_t detect_neon(
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

        // Threshold broadcast to all 16 bytes.
        const uint8x16_t t = vdupq_n_u8(static_cast<uint8_t>(threshold));

        // Consecutive count threshold: > 8 means corner (i.e. 9 consecutive).
        const uint8x16_t consec = vdupq_n_u8(8);

        // Bounded load for handling the last partial chunk of columns.
        const auto bounded_load = [](const unsigned char* src, int remaining) -> uint8x16_t {
            alignas(16) unsigned char buf[16] = {};
            const int count = (remaining > 16) ? 16 : ((remaining > 0) ? remaining : 0);
            for (int i = 0; i < count; ++i) {
                buf[i] = src[i];
            }
            return vld1q_u8(buf);
        };

        size_t feature_count = 0;

        for (int y = patch_radius; y < height - patch_radius; ++y) {
            const unsigned char* ptr = data + y * stride + patch_radius;

            int x = patch_radius;
            for (; x <= width - patch_radius - 16; x += 16, ptr += 16) {
                if (feature_count >= feature_point_buffer_size) {
                    return feature_count;
                }

                // Load 16 center pixels and 4 cardinal circle pixels.
                const uint8x16_t p = vld1q_u8(ptr);
                const uint8x16_t p0 = vld1q_u8(ptr + pattern_offset[0]);
                const uint8x16_t p4 = vld1q_u8(ptr + pattern_offset[4]);
                const uint8x16_t p8 = vld1q_u8(ptr + pattern_offset[8]);
                const uint8x16_t p12 = vld1q_u8(ptr + pattern_offset[12]);

                // p - threshold and p + threshold, saturated.
                const uint8x16_t pmt = vqsubq_u8(p, t);
                const uint8x16_t ppt = vqaddq_u8(p, t);

                // Quick bail: check if at least 2 of 4 adjacent cardinal points are brighter or darker.
                uint8x16_t ppt_accum = vandq_u8(vcgtq_u8(p0, ppt), vcgtq_u8(p4, ppt));
                uint8x16_t pmt_accum = vandq_u8(vcgtq_u8(pmt, p0), vcgtq_u8(pmt, p4));

                ppt_accum = vorrq_u8(ppt_accum, vandq_u8(vcgtq_u8(p4, ppt), vcgtq_u8(p8, ppt)));
                pmt_accum = vorrq_u8(pmt_accum, vandq_u8(vcgtq_u8(pmt, p4), vcgtq_u8(pmt, p8)));

                ppt_accum = vorrq_u8(ppt_accum, vandq_u8(vcgtq_u8(p8, ppt), vcgtq_u8(p12, ppt)));
                pmt_accum = vorrq_u8(pmt_accum, vandq_u8(vcgtq_u8(pmt, p8), vcgtq_u8(pmt, p12)));

                ppt_accum = vorrq_u8(ppt_accum, vandq_u8(vcgtq_u8(p12, ppt), vcgtq_u8(p0, ppt)));
                pmt_accum = vorrq_u8(pmt_accum, vandq_u8(vcgtq_u8(pmt, p12), vcgtq_u8(pmt, p0)));

                if (vmaxvq_u8(vorrq_u8(ppt_accum, pmt_accum)) == 0) {
                    continue;
                }

                // Full circle test: count consecutive salient pixels.
                uint8x16_t ppt_cnt = vdupq_n_u8(0);
                uint8x16_t pmt_cnt = vdupq_n_u8(0);
                uint8x16_t ppt_max = vdupq_n_u8(0);
                uint8x16_t pmt_max = vdupq_n_u8(0);

                for (int k = 0; k < 24; ++k) {
                    const uint8x16_t pk = vld1q_u8(ptr + pattern_offset[k]);

                    const uint8x16_t bright = vcgtq_u8(pk, ppt);
                    const uint8x16_t dark = vcgtq_u8(pmt, pk);

                    ppt_cnt = vandq_u8(vsubq_u8(ppt_cnt, bright), bright);
                    ppt_max = vmaxq_u8(ppt_max, ppt_cnt);

                    pmt_cnt = vandq_u8(vsubq_u8(pmt_cnt, dark), dark);
                    pmt_max = vmaxq_u8(pmt_max, pmt_cnt);
                }

                unsigned long long m = lane_mask(vcgtq_u8(vmaxq_u8(ppt_max, pmt_max), consec));

                while (m) {
                    const int lane = __builtin_ctzll(m) / 4;
                    m &= (m - 1);

                    if (feature_count < feature_point_buffer_size) {
                        feature_point_buffer[feature_count++] = {
                            static_cast<float>(x + lane),
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

                const uint8x16_t p = bounded_load(ptr, remaining);
                const uint8x16_t p0 = bounded_load(ptr + pattern_offset[0], remaining);
                const uint8x16_t p4 = bounded_load(ptr + pattern_offset[4], remaining);
                const uint8x16_t p8 = bounded_load(ptr + pattern_offset[8], remaining);
                const uint8x16_t p12 = bounded_load(ptr + pattern_offset[12], remaining);

                const uint8x16_t pmt = vqsubq_u8(p, t);
                const uint8x16_t ppt = vqaddq_u8(p, t);

                uint8x16_t ppt_accum = vandq_u8(vcgtq_u8(p0, ppt), vcgtq_u8(p4, ppt));
                uint8x16_t pmt_accum = vandq_u8(vcgtq_u8(pmt, p0), vcgtq_u8(pmt, p4));

                ppt_accum = vorrq_u8(ppt_accum, vandq_u8(vcgtq_u8(p4, ppt), vcgtq_u8(p8, ppt)));
                pmt_accum = vorrq_u8(pmt_accum, vandq_u8(vcgtq_u8(pmt, p4), vcgtq_u8(pmt, p8)));

                ppt_accum = vorrq_u8(ppt_accum, vandq_u8(vcgtq_u8(p8, ppt), vcgtq_u8(p12, ppt)));
                pmt_accum = vorrq_u8(pmt_accum, vandq_u8(vcgtq_u8(pmt, p8), vcgtq_u8(pmt, p12)));

                ppt_accum = vorrq_u8(ppt_accum, vandq_u8(vcgtq_u8(p12, ppt), vcgtq_u8(p0, ppt)));
                pmt_accum = vorrq_u8(pmt_accum, vandq_u8(vcgtq_u8(pmt, p12), vcgtq_u8(pmt, p0)));

                // Four bits per lane, so the mask of the valid lanes covers 4 * remaining bits.
                const unsigned long long last_cols_mask = (1ull << (4 * remaining)) - 1;
                unsigned long long m = lane_mask(vorrq_u8(ppt_accum, pmt_accum)) & last_cols_mask;

                if (m != 0) {
                    uint8x16_t ppt_cnt = vdupq_n_u8(0);
                    uint8x16_t pmt_cnt = vdupq_n_u8(0);
                    uint8x16_t ppt_max = vdupq_n_u8(0);
                    uint8x16_t pmt_max = vdupq_n_u8(0);

                    for (int k = 0; k < 24; ++k) {
                        const uint8x16_t pk = bounded_load(ptr + pattern_offset[k], remaining);

                        const uint8x16_t bright = vcgtq_u8(pk, ppt);
                        const uint8x16_t dark = vcgtq_u8(pmt, pk);

                        ppt_cnt = vandq_u8(vsubq_u8(ppt_cnt, bright), bright);
                        ppt_max = vmaxq_u8(ppt_max, ppt_cnt);

                        pmt_cnt = vandq_u8(vsubq_u8(pmt_cnt, dark), dark);
                        pmt_max = vmaxq_u8(pmt_max, pmt_cnt);
                    }

                    m = lane_mask(vcgtq_u8(vmaxq_u8(ppt_max, pmt_max), consec)) & last_cols_mask;

                    while (m) {
                        const int lane = __builtin_ctzll(m) / 4;
                        m &= (m - 1);

                        if (feature_count < feature_point_buffer_size) {
                            feature_point_buffer[feature_count++] = {
                                static_cast<float>(x + lane),
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
