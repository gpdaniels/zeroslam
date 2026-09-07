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

namespace {
    using size_t = decltype(sizeof(0));
}

namespace image {
    void gaussian_5x5_cpu(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);
    void gaussian_7x7_cpu(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data);

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
            if ((source_width <= 0) || (source_height <= 0)) {
                return;
            }
            unsigned short* convolved = new unsigned short[static_cast<unsigned long int>(source_width) * static_cast<unsigned long int>(source_height)];
            vertical_scalar<kernel_size>(source_data, source_width, source_height, source_stride, 0, source_width, convolved);
            for (int y = 0; y < source_height; ++y) {
                horizontal_scalar<kernel_size>(convolved, source_width, source_stride, y, 0, source_width, target_data);
            }
            delete[] convolved;
        }
    }

    void gaussian_5x5_cpu(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data) {
        gaussian_kernel<5>(source_data, source_width, source_height, source_stride, target_data);
    }

    void gaussian_7x7_cpu(const unsigned char* __restrict const source_data, const int source_width, const int source_height, const int source_stride, unsigned char* __restrict const target_data) {
        gaussian_kernel<7>(source_data, source_width, source_height, source_stride, target_data);
    }
}
