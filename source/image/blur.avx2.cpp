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
    void gaussian_7x7_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);

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
        void gaussian_kernel(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data) {
            constexpr static const int kernel_radius = kernel_size / 2;
            constexpr static const int shift = kernel_traits<kernel_size>::shift;
            if ((source_width <= 0) || (source_height <= 0)) {
                return;
            }
            unsigned short* convolved = new unsigned short[static_cast<unsigned long int>(source_width) * static_cast<unsigned long int>(source_height)];
            __m256i weights[static_cast<size_t>(kernel_size)];
            for (int k = 0; k < kernel_size; ++k) {
                weights[k] = _mm256_set1_epi16(static_cast<short>(kernel_traits<kernel_size>::taps[k]));
            }
            const __m256i rounding = _mm256_set1_epi32(1 << (shift - 1));

            // Vertical pass into the 16 bit intermediate, sixteen columns at a time, rows reflected at the edges.
            const int vector_width = source_width - (source_width % 16);
            for (int y = 0; y < source_height; ++y) {
                const unsigned char* rows[static_cast<size_t>(kernel_size)];
                for (int k = 0; k < kernel_size; ++k) {
                    rows[k] = source_data + reflect(y + k - kernel_radius, source_height) * source_stride;
                }
                unsigned short* const out = convolved + y * source_width;
                for (int x = 0; x < vector_width; x += 16) {
                    __m256i sum = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(rows[0] + x))), weights[0]);
                    for (int k = 1; k < kernel_size; ++k) {
                        sum = _mm256_add_epi16(sum, _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(rows[k] + x))), weights[k]));
                    }
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + x), sum);
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
                    __m256i sum_low = rounding;
                    __m256i sum_high = rounding;
                    for (int k = 0; k < kernel_size; ++k) {
                        const __m256i window = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + x + k - kernel_radius));
                        const __m256i weight = _mm256_set1_epi32(kernel_traits<kernel_size>::taps[k]);
                        sum_low = _mm256_add_epi32(sum_low, _mm256_mullo_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(window)), weight));
                        sum_high = _mm256_add_epi32(sum_high, _mm256_mullo_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(window, 1)), weight));
                    }
                    const __m256i packed16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(_mm256_srli_epi32(sum_low, shift), _mm256_srli_epi32(sum_high, shift)), 0xD8);
                    const __m256i packed8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(packed16, _mm256_setzero_si256()), 0xD8);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + x), _mm256_castsi256_si128(packed8));
                }
                horizontal_scalar<kernel_size>(convolved, source_width, source_stride, y, 0, (kernel_radius < source_width) ? kernel_radius : source_width, target_data);
                horizontal_scalar<kernel_size>(convolved, source_width, source_stride, y, x, source_width, target_data);
            }
            delete[] convolved;
        }
    }

    void gaussian_5x5_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data) {
        gaussian_kernel<5>(source_data, source_width, source_height, source_stride, target_data);
    }

    void gaussian_7x7_avx2(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data) {
        gaussian_kernel<7>(source_data, source_width, source_height, source_stride, target_data);
    }
}
