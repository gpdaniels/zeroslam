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

namespace {
    using size_t = decltype(sizeof(0));
}

namespace image {
    void gaussian_5x5_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
    void gaussian_5x5_decimate_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data, const int target_stride);
    void gaussian_7x7_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);

    namespace {
        // Source columns per strip, a strip of vertical sums and its reflected borders is a small stack buffer instead of a whole image.
        constexpr static const int strip_width = 1024;
        constexpr static const int strip_capacity = strip_width + 8;

        int reflect(int index, const int size) {
            if (size == 1) {
                return 0;
            }
            while ((index < 0) || (index >= size)) {
                if (index < 0) {
                    index = -index;
                }
                if (index >= size) {
                    index = (2 * size) - 2 - index;
                }
            }
            return index;
        }

        template <int kernel_size>
        struct kernel_traits;

        template <>
        struct kernel_traits<5> {
            constexpr static const int taps[5] = { 1, 4, 6, 4, 1 };
            constexpr static const int shift = 8;
        };

        template <>
        struct kernel_traits<7> {
            constexpr static const int taps[7] = { 1, 6, 15, 20, 15, 6, 1 };
            constexpr static const int shift = 12;
        };

        // Vertical sums of the kernel rows for the columns [first, last) into strip[column - first], sixteen columns at a time, the columns outside the image reflected.
        template <int kernel_size>
        void vertical_strip(const unsigned char* const* const rows, const int width, const int first, const int last, unsigned short* __restrict const strip) {
            const int begin = (first < 0) ? 0 : first;
            const int end = (last < width) ? last : width;
            int x = begin;
            for (; x + 16 <= end; x += 16) {
                const uint8x16_t pixels = vld1q_u8(rows[0] + x);
                uint16x8_t sum_low = vmulq_n_u16(vmovl_u8(vget_low_u8(pixels)), static_cast<uint16_t>(kernel_traits<kernel_size>::taps[0]));
                uint16x8_t sum_high = vmulq_n_u16(vmovl_u8(vget_high_u8(pixels)), static_cast<uint16_t>(kernel_traits<kernel_size>::taps[0]));
                for (int k = 1; k < kernel_size; ++k) {
                    const uint8x16_t next = vld1q_u8(rows[k] + x);
                    sum_low = vmlaq_n_u16(sum_low, vmovl_u8(vget_low_u8(next)), static_cast<uint16_t>(kernel_traits<kernel_size>::taps[k]));
                    sum_high = vmlaq_n_u16(sum_high, vmovl_u8(vget_high_u8(next)), static_cast<uint16_t>(kernel_traits<kernel_size>::taps[k]));
                }
                vst1q_u16(strip + (x - first), sum_low);
                vst1q_u16(strip + (x - first) + 8, sum_high);
            }
            for (; x < end; ++x) {
                int sum = 0;
                for (int k = 0; k < kernel_size; ++k) {
                    sum += rows[k][x] * kernel_traits<kernel_size>::taps[k];
                }
                strip[x - first] = static_cast<unsigned short>(sum);
            }
            for (x = first; x < begin; ++x) {
                strip[x - first] = strip[reflect(x, width) - first];
            }
            for (x = end; x < last; ++x) {
                strip[x - first] = strip[reflect(x, width) - first];
            }
        }

        template <int kernel_size>
        unsigned char horizontal_one(const unsigned short* const window) {
            constexpr static const int shift = kernel_traits<kernel_size>::shift;
            unsigned int sum = 0;
            for (int k = 0; k < kernel_size; ++k) {
                sum += static_cast<unsigned int>(window[k]) * static_cast<unsigned int>(kernel_traits<kernel_size>::taps[k]);
            }
            return static_cast<unsigned char>((sum + (1u << (shift - 1))) >> shift);
        }

        // Eight outputs from strip[x + k], accumulated in 32 bits.
        template <int kernel_size>
        uint8x8_t horizontal_eight(const unsigned short* const window) {
            constexpr static const int shift = kernel_traits<kernel_size>::shift;
            uint32x4_t sum_low = vdupq_n_u32(1u << (shift - 1));
            uint32x4_t sum_high = sum_low;
            for (int k = 0; k < kernel_size; ++k) {
                const uint16x8_t values = vld1q_u16(window + k);
                sum_low = vmlal_n_u16(sum_low, vget_low_u16(values), static_cast<uint16_t>(kernel_traits<kernel_size>::taps[k]));
                sum_high = vmlal_n_u16(sum_high, vget_high_u16(values), static_cast<uint16_t>(kernel_traits<kernel_size>::taps[k]));
            }
            return vmovn_u16(vcombine_u16(vshrn_n_u32(sum_low, shift), vshrn_n_u32(sum_high, shift)));
        }

        // The 5 tap sums are at most 16 * 4080, so they fit 16 bits and the rounding shift narrows them exactly.
        template <>
        uint8x8_t horizontal_eight<5>(const unsigned short* const window) {
            const uint16x8_t centre = vld1q_u16(window + 2);
            const uint16x8_t outer = vaddq_u16(vld1q_u16(window), vld1q_u16(window + 4));
            const uint16x8_t inner = vshlq_n_u16(vaddq_u16(vaddq_u16(vld1q_u16(window + 1), vld1q_u16(window + 3)), centre), 2);
            return vrshrn_n_u16(vaddq_u16(vaddq_u16(outer, inner), vshlq_n_u16(centre, 1)), kernel_traits<5>::shift);
        }

        // Eight outputs of the 5 tap kernel at every second column, strip[2 i + k], from the even and odd columns loaded apart, in 16 bits as above.
        uint8x8_t decimated_eight(const unsigned short* const window) {
            const uint16x8x2_t centre = vld2q_u16(window);
            const uint16x8x2_t next = vld2q_u16(window + 2);
            const uint16x8_t outer = vaddq_u16(centre.val[0], vld2q_u16(window + 4).val[0]);
            const uint16x8_t inner = vshlq_n_u16(vaddq_u16(centre.val[1], next.val[1]), 2);
            const uint16x8_t middle = vmulq_n_u16(next.val[0], 6);
            return vrshrn_n_u16(vaddq_u16(vaddq_u16(outer, inner), middle), kernel_traits<5>::shift);
        }

        template <int kernel_size>
        void gaussian_kernel(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data) {
            constexpr static const int kernel_radius = kernel_size / 2;
            if ((source_width <= 0) || (source_height <= 0)) {
                return;
            }
            unsigned short strip[strip_capacity] = {};
            for (int y = 0; y < source_height; ++y) {
                const unsigned char* rows[static_cast<size_t>(kernel_size)];
                for (int k = 0; k < kernel_size; ++k) {
                    rows[k] = source_data + reflect(y + k - kernel_radius, source_height) * source_stride;
                }
                unsigned char* const out = target_data + y * source_stride;
                for (int x_begin = 0; x_begin < source_width; x_begin += strip_width) {
                    const int x_end = (x_begin + strip_width < source_width) ? (x_begin + strip_width) : source_width;
                    vertical_strip<kernel_size>(rows, source_width, x_begin - kernel_radius, x_end + kernel_radius, strip);
                    int x = x_begin;
                    for (; x + 16 <= x_end; x += 16) {
                        const unsigned short* const window = strip + (x - x_begin);
                        vst1q_u8(out + x, vcombine_u8(horizontal_eight<kernel_size>(window), horizontal_eight<kernel_size>(window + 8)));
                    }
                    for (; x < x_end; ++x) {
                        out[x] = horizontal_one<kernel_size>(strip + (x - x_begin));
                    }
                }
            }
        }
    }

    void gaussian_5x5_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data) {
        gaussian_kernel<5>(source_data, source_width, source_height, source_stride, target_data);
    }

    void gaussian_5x5_decimate_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data, const int target_stride) {
        const int target_width = source_width / 2;
        const int target_height = source_height / 2;
        if ((target_width <= 0) || (target_height <= 0)) {
            return;
        }
        unsigned short strip[strip_capacity] = {};
        for (int y = 0; y < target_height; ++y) {
            const unsigned char* rows[5];
            for (int k = 0; k < 5; ++k) {
                rows[k] = source_data + reflect((2 * y) + k - 2, source_height) * source_stride;
            }
            unsigned char* const out = target_data + y * target_stride;
            for (int x_begin = 0; x_begin < target_width; x_begin += strip_width / 2) {
                const int x_end = (x_begin + (strip_width / 2) < target_width) ? (x_begin + (strip_width / 2)) : target_width;
                vertical_strip<5>(rows, source_width, (2 * x_begin) - 2, (2 * x_end) + 1, strip);
                int x = x_begin;
                for (; x + 16 <= x_end; x += 16) {
                    const unsigned short* const window = strip + (2 * (x - x_begin));
                    vst1q_u8(out + x, vcombine_u8(decimated_eight(window), decimated_eight(window + 16)));
                }
                for (; x < x_end; ++x) {
                    out[x] = horizontal_one<5>(strip + (2 * (x - x_begin)));
                }
            }
        }
    }

    void gaussian_7x7_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data) {
        gaussian_kernel<7>(source_data, source_width, source_height, source_stride, target_data);
    }
}
