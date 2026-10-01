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

#include "feature/descriptor/teblid.hpp"

#include "math/math.hpp"

namespace feature::descriptor {
    constexpr const teblid::test teblid::tests[teblid::test_count] = {
        { 25, 14, 13, 15, 6, 21.65f },
        { 16, 15, 14, 11, 1, 5.65f },
        { 14, 14, 7, 8, 6, 4.95f },
        { 10, 9, 6, 20, 6, 2.45f },
        { 13, 26, 13, 19, 5, 2.25f },
        { 19, 14, 19, 5, 4, 0.85f },
        { 16, 19, 15, 13, 2, 3.35f },
        { 26, 26, 21, 12, 5, 1.75f },
        { 18, 23, 15, 20, 2, 4.55f },
        { 12, 15, 10, 20, 1, -1.55f },
        { 26, 4, 18, 8, 3, 4.55f },
        { 8, 21, 2, 29, 2, -5.05f },
        { 19, 16, 17, 19, 1, 3.15f },
        { 10, 3, 5, 13, 3, 4.85f },
        { 16, 10, 10, 14, 1, 9.95f },
        { 19, 12, 18, 17, 1, 1.35f },
        { 21, 26, 21, 19, 5, -2.05f },
        { 6, 7, 5, 5, 5, -0.15f },
        { 22, 12, 20, 14, 2, 1.55f },
        { 14, 12, 13, 17, 1, 3.35f },
        { 11, 16, 10, 13, 2, 0.25f },
        { 7, 23, 7, 17, 3, 0.35f },
        { 27, 13, 25, 8, 4, 2.45f },
        { 20, 19, 16, 14, 1, 2.75f },
        { 27, 10, 24, 16, 2, -1.65f },
        { 13, 12, 13, 6, 2, -0.05f },
        { 14, 18, 13, 23, 1, -0.75f },
        { 14, 8, 11, 1, 1, 0.85f },
        { 14, 23, 12, 9, 2, 2.95f },
        { 6, 19, 2, 13, 2, -1.65f },
        { 8, 19, 6, 19, 3, -0.05f },
        { 18, 28, 17, 25, 3, -0.25f },
        { 29, 28, 25, 22, 2, -3.85f },
        { 15, 19, 15, 17, 3, -0.05f },
        { 23, 21, 19, 19, 1, 3.35f },
        { 20, 20, 20, 16, 3, 0.05f },
        { 29, 4, 25, 8, 2, -3.55f },
        { 17, 6, 16, 25, 2, 2.65f },
        { 12, 21, 8, 29, 1, 1.95f },
        { 14, 15, 9, 17, 2, 6.35f },
        { 18, 5, 17, 3, 3, 0.85f },
        { 21, 12, 18, 10, 1, 2.65f },
        { 17, 14, 14, 14, 2, 12.45f },
        { 5, 26, 3, 6, 3, 0.05f },
        { 16, 13, 15, 14, 1, 3.35f },
        { 28, 21, 24, 22, 3, 1.75f },
        { 13, 12, 13, 10, 1, -1.05f },
        { 22, 3, 21, 11, 3, -1.05f },
        { 27, 27, 4, 16, 4, 28.25f },
        { 12, 13, 7, 10, 1, 0.35f },
        { 15, 25, 15, 22, 2, -0.15f },
        { 19, 10, 18, 12, 1, 2.05f },
        { 17, 16, 17, 9, 2, 2.55f },
        { 21, 17, 21, 14, 2, 0.85f },
        { 13, 19, 12, 16, 1, 1.35f },
        { 11, 11, 9, 15, 1, 1.15f },
        { 15, 26, 14, 28, 3, 1.25f },
        { 17, 22, 17, 20, 1, 1.35f },
        { 10, 26, 2, 27, 2, 1.85f },
        { 28, 12, 26, 23, 3, 3.95f },
        { 4, 5, 3, 14, 3, 0.75f },
        { 17, 7, 17, 4, 3, 1.65f },
        { 19, 15, 17, 15, 1, -3.15f },
        { 7, 8, 2, 5, 2, -6.35f },
        { 22, 15, 19, 14, 2, 2.05f },
        { 15, 16, 12, 20, 1, -5.15f },
        { 13, 19, 12, 20, 1, 1.75f },
        { 17, 10, 17, 8, 2, -0.65f },
        { 26, 16, 19, 15, 4, -0.65f },
        { 9, 14, 8, 20, 2, 1.05f },
        { 27, 14, 27, 4, 4, -0.85f },
        { 17, 14, 15, 9, 1, 0.85f },
        { 5, 4, 5, 3, 3, -0.35f },
        { 15, 30, 9, 5, 1, 9.05f },
        { 7, 25, 7, 23, 6, 0.75f },
        { 12, 24, 11, 16, 1, -1.75f },
        { 20, 29, 20, 20, 2, 0.75f },
        { 19, 18, 15, 19, 1, 16.05f },
        { 9, 11, 7, 11, 7, 0.35f },
        { 27, 26, 26, 15, 4, 0.75f },
        { 10, 28, 10, 27, 3, 0.05f },
        { 8, 12, 8, 6, 3, 0.05f },
        { 21, 23, 16, 22, 1, 3.75f },
        { 22, 7, 4, 25, 4, 14.15f },
        { 17, 19, 16, 15, 1, -8.95f },
        { 28, 21, 11, 15, 3, 67.25f },
        { 15, 3, 15, 2, 2, -0.45f },
        { 16, 16, 14, 17, 3, 1.65f },
        { 10, 17, 7, 18, 3, -1.95f },
        { 12, 18, 12, 15, 1, 1.15f },
        { 18, 16, 16, 13, 1, 1.85f },
        { 20, 16, 19, 15, 1, 3.95f },
        { 16, 15, 11, 11, 1, -1.75f },
        { 4, 14, 2, 13, 2, 0.45f },
        { 29, 18, 27, 17, 2, -1.55f },
        { 16, 18, 14, 16, 1, 1.05f },
        { 23, 29, 22, 27, 2, -0.25f },
        { 18, 13, 18, 11, 1, -1.05f },
        { 26, 23, 21, 27, 4, 3.05f },
        { 18, 22, 17, 18, 1, -1.05f },
        { 3, 11, 2, 21, 2, 1.95f },
        { 13, 18, 13, 9, 3, -0.05f },
        { 15, 14, 14, 5, 2, 0.85f },
        { 1, 14, 1, 1, 1, 3.05f },
        { 29, 2, 5, 9, 2, 34.85f },
        { 12, 17, 11, 17, 1, -0.15f },
        { 13, 10, 12, 25, 4, 4.35f },
        { 5, 13, 1, 25, 1, -10.65f },
        { 13, 16, 13, 12, 1, 2.35f },
        { 16, 23, 16, 12, 1, -1.35f },
        { 27, 14, 22, 14, 2, 0.05f },
        { 29, 29, 27, 27, 2, 1.05f },
        { 23, 6, 22, 4, 4, 1.05f },
        { 22, 16, 22, 8, 3, -0.15f },
        { 14, 1, 11, 9, 1, 0.45f },
        { 12, 11, 10, 8, 2, -0.55f },
        { 24, 19, 7, 16, 7, 10.45f },
        { 5, 29, 2, 20, 2, 1.35f },
        { 19, 15, 19, 13, 1, -0.95f },
        { 15, 18, 8, 24, 2, 0.45f },
        { 4, 24, 1, 30, 1, -0.85f },
        { 17, 30, 17, 26, 1, 1.45f },
        { 9, 8, 7, 5, 2, -1.85f },
        { 15, 20, 15, 18, 1, 1.65f },
        { 27, 5, 14, 26, 4, 2.75f },
        { 18, 19, 18, 15, 1, 1.05f },
        { 24, 14, 9, 12, 1, 81.45f },
        { 20, 6, 18, 10, 1, 3.35f },
        { 21, 23, 21, 21, 1, 0.85f },
        { 19, 17, 6, 6, 6, 2.65f },
        { 10, 13, 6, 12, 3, 9.35f },
        { 30, 10, 27, 14, 1, 1.15f },
        { 9, 5, 6, 3, 3, 1.35f },
        { 26, 21, 18, 19, 2, -1.55f },
        { 23, 5, 23, 4, 4, 0.85f },
        { 14, 11, 11, 12, 1, 20.65f },
        { 18, 13, 16, 13, 1, 2.05f },
        { 7, 8, 3, 16, 3, 12.85f },
        { 16, 15, 16, 12, 2, 7.95f },
        { 25, 20, 24, 25, 3, 2.25f },
        { 20, 14, 19, 14, 1, 0.05f },
        { 12, 29, 12, 5, 1, 0.85f },
        { 23, 17, 13, 13, 5, 8.75f },
        { 27, 27, 23, 22, 4, -8.25f },
        { 11, 4, 11, 3, 3, -0.35f },
        { 9, 18, 7, 15, 1, 1.65f },
        { 18, 17, 18, 14, 1, -3.95f },
        { 28, 2, 6, 17, 2, 92.55f },
        { 5, 20, 3, 22, 3, 0.55f },
        { 30, 30, 30, 2, 1, 0.35f },
        { 16, 8, 15, 13, 1, -0.75f },
        { 15, 16, 14, 13, 1, -12.25f },
        { 28, 5, 27, 5, 3, 0.55f },
        { 13, 13, 12, 12, 1, 1.05f },
        { 7, 8, 6, 7, 6, 0.95f },
        { 10, 21, 10, 17, 1, 1.15f },
        { 11, 17, 3, 30, 1, -43.25f },
        { 16, 17, 9, 14, 7, 3.05f },
        { 17, 16, 9, 14, 1, 4.35f },
        { 14, 29, 13, 27, 2, 7.15f },
        { 19, 5, 19, 3, 2, 0.15f },
        { 18, 16, 14, 14, 1, 57.95f },
        { 10, 23, 8, 25, 2, 4.35f },
        { 17, 17, 15, 18, 1, 0.75f },
        { 16, 22, 16, 16, 6, 0.05f },
        { 29, 11, 27, 11, 2, 0.05f },
        { 13, 9, 7, 11, 1, 5.45f },
        { 18, 23, 17, 19, 4, 0.55f },
        { 12, 14, 11, 17, 1, 0.95f },
        { 13, 23, 11, 18, 2, 20.55f },
        { 27, 8, 23, 20, 4, -4.45f },
        { 18, 18, 18, 11, 4, 0.75f },
        { 8, 21, 5, 8, 5, 4.55f },
        { 23, 5, 21, 10, 1, -0.15f },
        { 16, 16, 16, 12, 1, 8.65f },
        { 18, 17, 14, 19, 1, 42.65f },
        { 16, 27, 16, 24, 2, -0.45f },
        { 21, 17, 15, 15, 1, -1.25f },
        { 16, 5, 15, 9, 2, -1.75f },
        { 24, 16, 1, 30, 1, 11.25f },
        { 15, 14, 14, 19, 1, -8.15f },
        { 19, 12, 12, 14, 2, 2.85f },
        { 5, 5, 3, 4, 3, -2.85f },
        { 16, 11, 16, 9, 1, -5.05f },
        { 16, 9, 6, 18, 6, 44.65f },
        { 25, 24, 23, 14, 1, 1.45f },
        { 5, 26, 5, 17, 5, -0.75f },
        { 9, 16, 6, 18, 1, 11.85f },
        { 29, 25, 9, 24, 2, 2.05f },
        { 25, 22, 24, 30, 1, 1.25f },
        { 22, 2, 20, 5, 2, 4.45f },
        { 27, 1, 25, 11, 1, -1.35f },
        { 15, 12, 14, 10, 1, 5.95f },
        { 17, 6, 16, 8, 1, 1.35f },
        { 28, 8, 23, 7, 3, -2.55f },
        { 24, 24, 23, 22, 7, 5.05f },
        { 7, 18, 5, 20, 3, -2.85f },
        { 22, 15, 20, 20, 1, 7.35f },
        { 30, 21, 28, 20, 1, -1.35f },
        { 3, 18, 2, 18, 2, -0.45f },
        { 6, 14, 5, 15, 1, 0.45f },
        { 15, 18, 15, 16, 1, -11.85f },
        { 7, 11, 5, 2, 1, -39.65f },
        { 17, 17, 13, 15, 3, 1.65f },
        { 12, 15, 7, 15, 5, -0.05f },
        { 16, 12, 15, 18, 1, 3.65f },
        { 14, 26, 14, 25, 5, -0.35f },
        { 11, 17, 8, 18, 1, 0.05f },
        { 23, 13, 15, 21, 7, 1.85f },
        { 10, 9, 10, 2, 2, -0.45f },
        { 17, 13, 12, 19, 1, -1.75f },
        { 20, 25, 19, 22, 1, 3.95f },
        { 9, 26, 8, 21, 1, 5.25f },
        { 19, 22, 19, 18, 1, -1.05f },
        { 8, 15, 3, 12, 1, -11.95f },
        { 26, 13, 16, 19, 5, 37.05f },
        { 24, 12, 21, 13, 1, -1.15f },
        { 12, 14, 12, 9, 1, 1.25f },
        { 3, 7, 1, 1, 1, 0.75f },
        { 16, 9, 15, 3, 3, -6.05f },
        { 23, 20, 23, 8, 7, -1.55f },
        { 24, 16, 22, 15, 1, -1.65f },
        { 20, 19, 20, 14, 1, 0.85f },
        { 30, 27, 29, 22, 1, 0.35f },
        { 27, 17, 4, 16, 4, 101.55f },
        { 8, 13, 5, 13, 5, -5.05f },
        { 19, 8, 10, 16, 3, 3.65f },
        { 30, 11, 30, 4, 1, -2.35f },
        { 14, 21, 14, 20, 1, -0.35f },
        { 14, 11, 13, 13, 1, -1.65f },
        { 30, 2, 28, 5, 1, 0.65f },
        { 17, 29, 12, 24, 2, 6.35f },
        { 15, 25, 6, 30, 1, 2.85f },
        { 4, 1, 1, 1, 1, 5.25f },
        { 12, 16, 5, 20, 5, 24.05f },
        { 16, 20, 14, 15, 1, 38.15f },
        { 6, 17, 6, 9, 3, -1.05f },
        { 20, 17, 12, 20, 4, 3.05f },
        { 15, 15, 12, 4, 4, 0.35f },
        { 28, 20, 22, 21, 3, -16.05f },
        { 14, 18, 9, 18, 5, -1.25f },
        { 26, 1, 23, 5, 1, 0.25f },
        { 21, 24, 11, 10, 7, 1.95f },
        { 15, 19, 14, 12, 1, -0.85f },
        { 27, 29, 11, 16, 1, 107.65f },
        { 23, 19, 22, 29, 1, -1.55f },
        { 2, 30, 2, 29, 1, -0.25f },
        { 14, 16, 6, 5, 3, 26.95f },
        { 17, 13, 14, 16, 1, 35.95f },
        { 19, 14, 15, 16, 1, -4.85f },
        { 20, 25, 13, 15, 6, 1.55f },
        { 19, 18, 11, 12, 5, 10.85f },
        { 30, 30, 30, 13, 1, -7.15f },
        { 3, 14, 1, 9, 1, -4.25f },
        { 20, 17, 1, 18, 1, -25.15f },
        { 16, 20, 12, 19, 1, 2.75f }
    };

    namespace {
        // Every test compares the means of two boxes centred on rotated pattern points, box_sum takes a box [x0, x1) by [y0, y1) in the coordinates of the window around the centre pixel.
        template <typename box_sum_type>
        void describe_boxes(const float angle_radians, const box_sum_type& box_sum, binary<256>& descriptor) {
            constexpr static const int window_size = (2 * teblid::window_radius) + 1;
            const auto box_mean = [&](int x0, int y0, int x1, int y1) {
                x0 = math::max(0, math::min(x0, window_size - 1));
                y0 = math::max(0, math::min(y0, window_size - 1));
                x1 = math::max(x0 + 1, math::min(x1, window_size));
                y1 = math::max(y0 + 1, math::min(y1, window_size));
                return box_sum(x0, y0, x1, y1) / static_cast<float>((x1 - x0) * (y1 - y0));
            };
            // Note: Every rounded value lies between 0.5 and 2^22, where adding a half and truncating equals math::round without its library call.
            const auto round_positive = [](const float value) {
                return static_cast<int>(value + 0.5f);
            };
            const float scale = teblid::keypoint_size / static_cast<float>(teblid::patch_size);
            const float cosine = scale * math::cos(angle_radians);
            const float sine = scale * math::sin(angle_radians);
            const float half_patch = 0.5f * static_cast<float>(teblid::patch_size);
            const auto rectify = [&](const int patch_x, const int patch_y, int& window_x, int& window_y) {
                const float px = static_cast<float>(patch_x) - half_patch;
                const float py = static_cast<float>(patch_y) - half_patch;
                window_x = round_positive((cosine * px) - (sine * py) + static_cast<float>(teblid::window_radius));
                window_y = round_positive((sine * px) + (cosine * py) + static_cast<float>(teblid::window_radius));
            };
            for (int byte = 0; byte < 32; ++byte) {
                descriptor.data[byte] = 0;
            }
            for (int index = 0; index < teblid::test_count; ++index) {
                const teblid::test& t = teblid::tests[index];
                int x1 = 0;
                int y1 = 0;
                int x2 = 0;
                int y2 = 0;
                rectify(t.x1, t.y1, x1, y1);
                rectify(t.x2, t.y2, x2, y2);
                const int radius = round_positive(scale * static_cast<float>(t.radius));
                const float response = box_mean(x1 - radius, y1 - radius, x1 + radius + 1, y1 + radius + 1) - box_mean(x2 - radius, y2 - radius, x2 + radius + 1, y2 + radius + 1);
                if (response <= t.threshold) {
                    descriptor.data[index >> 3] = static_cast<unsigned char>(descriptor.data[index >> 3] | (1u << (index & 7)));
                }
            }
        }
    }

    void teblid::describe(const unsigned char* __restrict const data, const int stride, const float angle_radians, binary<256>& descriptor) {
        constexpr static const int window_size = (2 * teblid::window_radius) + 1;
        constexpr static const int integral_size = window_size + 1;
        int integral_window[integral_size * integral_size];
        for (int x = 0; x < integral_size; ++x) {
            integral_window[x] = 0;
        }
        for (int y = 1; y < integral_size; ++y) {
            const unsigned char* const row = data + static_cast<long>(y - 1 - teblid::window_radius) * stride - teblid::window_radius;
            int row_sum = 0;
            integral_window[y * integral_size] = 0;
            for (int x = 1; x < integral_size; ++x) {
                row_sum += row[x - 1];
                integral_window[(y * integral_size) + x] = integral_window[((y - 1) * integral_size) + x] + row_sum;
            }
        }
        const auto box_sum = [&](const int x0, const int y0, const int x1, const int y1) {
            const int sum = integral_window[(y1 * integral_size) + x1] - integral_window[(y0 * integral_size) + x1] - integral_window[(y1 * integral_size) + x0] + integral_window[(y0 * integral_size) + x0];
            return static_cast<float>(sum);
        };
        describe_boxes(angle_radians, box_sum, descriptor);
    }

    void teblid::integral(const unsigned char* __restrict const data, const int width, const int height, const int stride, unsigned int* __restrict const integral_data) {
        const size_t integral_stride = static_cast<size_t>(width) + 1;
        for (size_t x = 0; x < integral_stride; ++x) {
            integral_data[x] = 0;
        }
        for (int y = 0; y < height; ++y) {
            const unsigned char* __restrict const row = data + static_cast<size_t>(y) * static_cast<size_t>(stride);
            const unsigned int* __restrict const above = integral_data + static_cast<size_t>(y) * integral_stride;
            unsigned int* __restrict const current = integral_data + static_cast<size_t>(y + 1) * integral_stride;
            unsigned int row_sum = 0;
            current[0] = 0;
            for (int x = 0; x < width; ++x) {
                row_sum += row[x];
                current[x + 1] = above[x + 1] + row_sum;
            }
        }
    }

    void teblid::describe_integral(const unsigned int* __restrict const integral_data, const int integral_stride, const float x, const float y, const float angle_radians, binary<256>& descriptor) {
        const float pixel_x = math::floor(x);
        const float pixel_y = math::floor(y);
        const float fraction_x = x - pixel_x;
        const float fraction_y = y - pixel_y;
        const unsigned int* __restrict const window = integral_data + static_cast<long>(static_cast<int>(pixel_y) - teblid::window_radius) * integral_stride + (static_cast<int>(pixel_x) - teblid::window_radius);
        // Note: Large levels wrap the sums, so combine the four corners in unsigned arithmetic before anything else.
        const auto pixel_box_sum = [&](const int x0, const int y0, const int x1, const int y1) {
            const unsigned int sum = window[(y1 * integral_stride) + x1] - window[(y0 * integral_stride) + x1] - window[(y1 * integral_stride) + x0] + window[(y0 * integral_stride) + x0];
            return static_cast<float>(sum);
        };
        // Note: The corners of a moved box all share the centre's fraction, so its area sampled sum blends the box with its copies one pixel right and one pixel down.
        const auto row_box_sum = [&](const int x0, const int y0, const int x1, const int y1) {
            const float sum = pixel_box_sum(x0, y0, x1, y1);
            if (fraction_x == 0.0f) {
                return sum;
            }
            return sum + fraction_x * (pixel_box_sum(x0 + 1, y0, x1 + 1, y1) - sum);
        };
        const auto box_sum = [&](const int x0, const int y0, const int x1, const int y1) {
            const float sum = row_box_sum(x0, y0, x1, y1);
            if (fraction_y == 0.0f) {
                return sum;
            }
            return sum + fraction_y * (row_box_sum(x0, y0 + 1, x1, y1 + 1) - sum);
        };
        describe_boxes(angle_radians, box_sum, descriptor);
    }
}
