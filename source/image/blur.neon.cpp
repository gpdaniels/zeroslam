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
    void gaussian_7x7_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);

    namespace {
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

        template <int kernel_size>
        void vertical_scalar(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, const int x_begin, const int x_end, unsigned short* __restrict const convolved) {
            constexpr static const int kernel_radius = kernel_size / 2;
            for (int y = 0; y < source_height; ++y) {
                const unsigned char* rows[static_cast<size_t>(kernel_size)];
                for (int k = 0; k < kernel_size; ++k) {
                    rows[k] = source_data + reflect(y + k - kernel_radius, source_height) * source_stride;
                }
                for (int x = x_begin; x < x_end; ++x) {
                    int sum = 0;
                    for (int k = 0; k < kernel_size; ++k) {
                        sum += rows[k][x] * kernel_traits<kernel_size>::taps[k];
                    }
                    convolved[y * source_width + x] = static_cast<unsigned short>(sum);
                }
            }
        }

        template <int kernel_size>
        void horizontal_scalar(const unsigned short* __restrict const convolved, const int source_width, const int source_stride, const int y, const int x_begin, const int x_end, unsigned char* __restrict const target_data) {
            constexpr static const int kernel_radius = kernel_size / 2;
            constexpr static const int shift = kernel_traits<kernel_size>::shift;
            const unsigned short* const row = convolved + y * source_width;
            unsigned char* const out = target_data + y * source_stride;
            for (int x = x_begin; x < x_end; ++x) {
                unsigned int sum = 0;
                if ((x >= kernel_radius) && (x + kernel_radius < source_width)) {
                    for (int k = 0; k < kernel_size; ++k) {
                        sum += static_cast<unsigned int>(row[x + k - kernel_radius]) * static_cast<unsigned int>(kernel_traits<kernel_size>::taps[k]);
                    }
                }
                else {
                    for (int k = 0; k < kernel_size; ++k) {
                        sum += static_cast<unsigned int>(row[reflect(x + k - kernel_radius, source_width)]) * static_cast<unsigned int>(kernel_traits<kernel_size>::taps[k]);
                    }
                }
                out[x] = static_cast<unsigned char>((sum + (1u << (shift - 1))) >> shift);
            }
        }

        template <int kernel_size>
        uint8x8_t horizontal_eight(const unsigned short* const window_start, const uint16x4_t* const weights) {
            constexpr static const int shift = kernel_traits<kernel_size>::shift;
            const uint32x4_t rounding = vdupq_n_u32(1u << (shift - 1));
            uint32x4_t sum_low = rounding;
            uint32x4_t sum_high = rounding;
            for (int k = 0; k < kernel_size; ++k) {
                const uint16x8_t window = vld1q_u16(window_start + k);
                sum_low = vmlal_u16(sum_low, vget_low_u16(window), weights[k]);
                sum_high = vmlal_u16(sum_high, vget_high_u16(window), weights[k]);
            }
            return vmovn_u16(vcombine_u16(vshrn_n_u32(sum_low, shift), vshrn_n_u32(sum_high, shift)));
        }

        template <int kernel_size>
        void gaussian_kernel(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data) {
            constexpr static const int kernel_radius = kernel_size / 2;
            if ((source_width <= 0) || (source_height <= 0)) {
                return;
            }
            unsigned short* convolved = new unsigned short[static_cast<unsigned long int>(source_width) * static_cast<unsigned long int>(source_height)];
            uint16x8_t weights_wide[static_cast<size_t>(kernel_size)];
            uint16x4_t weights[static_cast<size_t>(kernel_size)];
            for (int k = 0; k < kernel_size; ++k) {
                weights_wide[k] = vdupq_n_u16(static_cast<uint16_t>(kernel_traits<kernel_size>::taps[k]));
                weights[k] = vdup_n_u16(static_cast<uint16_t>(kernel_traits<kernel_size>::taps[k]));
            }

            // Vertical pass into the 16 bit intermediate, sixteen columns at a time, rows reflected at the edges.
            const int vector_width = source_width - (source_width % 16);
            for (int y = 0; y < source_height; ++y) {
                const unsigned char* rows[static_cast<size_t>(kernel_size)];
                for (int k = 0; k < kernel_size; ++k) {
                    rows[k] = source_data + reflect(y + k - kernel_radius, source_height) * source_stride;
                }
                unsigned short* const out = convolved + y * source_width;
                for (int x = 0; x < vector_width; x += 16) {
                    const uint8x16_t first = vld1q_u8(rows[0] + x);
                    uint16x8_t sum_low = vmulq_u16(vmovl_u8(vget_low_u8(first)), weights_wide[0]);
                    uint16x8_t sum_high = vmulq_u16(vmovl_u8(vget_high_u8(first)), weights_wide[0]);
                    for (int k = 1; k < kernel_size; ++k) {
                        const uint8x16_t pixels = vld1q_u8(rows[k] + x);
                        sum_low = vmlaq_u16(sum_low, vmovl_u8(vget_low_u8(pixels)), weights_wide[k]);
                        sum_high = vmlaq_u16(sum_high, vmovl_u8(vget_high_u8(pixels)), weights_wide[k]);
                    }
                    vst1q_u16(out + x, sum_low);
                    vst1q_u16(out + x + 8, sum_high);
                }
            }
            if (vector_width < source_width) {
                vertical_scalar<kernel_size>(source_data, source_width, source_height, source_stride, vector_width, source_width, convolved);
            }

            // Horizontal pass, sixteen outputs at a time accumulated in 32 bits, the reflected edges done scalar.
            for (int y = 0; y < source_height; ++y) {
                const unsigned short* const row = convolved + y * source_width;
                unsigned char* const out = target_data + y * source_stride;
                int x = kernel_radius;
                for (; x + 16 + kernel_radius <= source_width; x += 16) {
                    vst1q_u8(out + x, vcombine_u8(horizontal_eight<kernel_size>(row + x - kernel_radius, weights), horizontal_eight<kernel_size>(row + x + 8 - kernel_radius, weights)));
                }
                horizontal_scalar<kernel_size>(convolved, source_width, source_stride, y, 0, (kernel_radius < source_width) ? kernel_radius : source_width, target_data);
                horizontal_scalar<kernel_size>(convolved, source_width, source_stride, y, x, source_width, target_data);
            }
            delete[] convolved;
        }
    }

    void gaussian_5x5_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data) {
        gaussian_kernel<5>(source_data, source_width, source_height, source_stride, target_data);
    }

    void gaussian_7x7_neon(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data) {
        gaussian_kernel<7>(source_data, source_width, source_height, source_stride, target_data);
    }
}
