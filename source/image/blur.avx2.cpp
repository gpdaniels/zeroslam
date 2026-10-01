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

namespace {
    using size_t = decltype(sizeof(0));
}

namespace image {
    void gaussian_5x5_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
    void gaussian_5x5_decimate_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data, const int target_stride);
    void gaussian_7x7_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);

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
                __m256i sum = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(rows[0] + x))), _mm256_set1_epi16(static_cast<short>(kernel_traits<kernel_size>::taps[0])));
                for (int k = 1; k < kernel_size; ++k) {
                    sum = _mm256_add_epi16(sum, _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(rows[k] + x))), _mm256_set1_epi16(static_cast<short>(kernel_traits<kernel_size>::taps[k]))));
                }
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(strip + (x - first)), sum);
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

        __m256i load_sixteen(const unsigned short* const values) {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
        }

        // Narrows sixteen 16 bit results in order to bytes.
        __m128i narrow_sixteen(const __m256i values) {
            return _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(values, values), 0x08));
        }

        // Two taps as the 16 bit pair that one multiply add applies to two neighbouring values.
        template <int kernel_size>
        __m256i tap_pair(const int first) {
            const int second = (first + 1 < kernel_size) ? kernel_traits<kernel_size>::taps[first + 1] : 0;
            return _mm256_set1_epi32(kernel_traits<kernel_size>::taps[first] | (second << 16));
        }

        // Sixteen outputs from strip[x + k].
        template <int kernel_size>
        __m128i horizontal_sixteen(const unsigned short* const window);

        // The 5 tap sums are at most 16 * 4080, so they fit 16 bits.
        template <>
        __m128i horizontal_sixteen<5>(const unsigned short* const window) {
            constexpr static const int shift = kernel_traits<5>::shift;
            const __m256i centre = load_sixteen(window + 2);
            const __m256i outer = _mm256_add_epi16(load_sixteen(window), load_sixteen(window + 4));
            const __m256i inner = _mm256_slli_epi16(_mm256_add_epi16(_mm256_add_epi16(load_sixteen(window + 1), load_sixteen(window + 3)), centre), 2);
            const __m256i sum = _mm256_add_epi16(_mm256_add_epi16(outer, inner), _mm256_add_epi16(_mm256_slli_epi16(centre, 1), _mm256_set1_epi16(1 << (shift - 1))));
            return narrow_sixteen(_mm256_srli_epi16(sum, shift));
        }

        // The 7 tap sums need 32 bits, neighbouring columns are interleaved so each multiply adds a pair of taps.
        template <>
        __m128i horizontal_sixteen<7>(const unsigned short* const window) {
            constexpr static const int shift = kernel_traits<7>::shift;
            __m256i sum_low = _mm256_set1_epi32(1 << (shift - 1));
            __m256i sum_high = sum_low;
            for (int first = 0; first < 7; first += 2) {
                const __m256i values = load_sixteen(window + first);
                const __m256i next = (first + 1 < 7) ? load_sixteen(window + first + 1) : _mm256_setzero_si256();
                sum_low = _mm256_add_epi32(sum_low, _mm256_madd_epi16(_mm256_unpacklo_epi16(values, next), tap_pair<7>(first)));
                sum_high = _mm256_add_epi32(sum_high, _mm256_madd_epi16(_mm256_unpackhi_epi16(values, next), tap_pair<7>(first)));
            }
            return narrow_sixteen(_mm256_packus_epi32(_mm256_srli_epi32(sum_low, shift), _mm256_srli_epi32(sum_high, shift)));
        }

        // Eight outputs of the 5 tap kernel at every second column, strip[2 i + k], each multiply adding a pair of neighbouring taps into 32 bits.
        __m256i decimated_eight(const unsigned short* const window) {
            constexpr static const int shift = kernel_traits<5>::shift;
            const __m256i first = _mm256_madd_epi16(load_sixteen(window), tap_pair<5>(0));
            const __m256i second = _mm256_madd_epi16(load_sixteen(window + 2), tap_pair<5>(2));
            const __m256i third = _mm256_madd_epi16(load_sixteen(window + 4), tap_pair<5>(4));
            return _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(first, second), _mm256_add_epi32(third, _mm256_set1_epi32(1 << (shift - 1)))), shift);
        }

        template <int kernel_size>
        void gaussian_kernel(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data) {
            constexpr static const int kernel_radius = kernel_size / 2;
            if ((source_width <= 0) || (source_height <= 0)) {
                return;
            }
            alignas(32) unsigned short strip[strip_capacity] = {};
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
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + x), horizontal_sixteen<kernel_size>(strip + (x - x_begin)));
                    }
                    for (; x < x_end; ++x) {
                        out[x] = horizontal_one<kernel_size>(strip + (x - x_begin));
                    }
                }
            }
        }
    }

    void gaussian_5x5_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data) {
        gaussian_kernel<5>(source_data, source_width, source_height, source_stride, target_data);
    }

    void gaussian_5x5_decimate_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data, const int target_stride) {
        const int target_width = source_width / 2;
        const int target_height = source_height / 2;
        if ((target_width <= 0) || (target_height <= 0)) {
            return;
        }
        alignas(32) unsigned short strip[strip_capacity] = {};
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
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + x), narrow_sixteen(_mm256_permute4x64_epi64(_mm256_packus_epi32(decimated_eight(window), decimated_eight(window + 16)), 0xD8)));
                }
                for (; x < x_end; ++x) {
                    out[x] = horizontal_one<5>(strip + (2 * (x - x_begin)));
                }
            }
        }
    }

    void gaussian_7x7_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data) {
        gaussian_kernel<7>(source_data, source_width, source_height, source_stride, target_data);
    }
}
