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

#include "feature/refiner/subpixel.hpp"

#include "math/math.hpp"

namespace feature::refiner {
    bool subpixel::refine(
        const unsigned char* __restrict const data,
        const int stride,
        float& offset_x,
        float& offset_y
    ) {
        constexpr static const int max_iterations = 10;
        constexpr static const int window_size = 5;
        constexpr static const float max_offset = 4.0f;
        constexpr static const float eps = 1e-6f;

        constexpr static const int samples_stride = 2 * window_size + 3;

        // Initialize offset.
        offset_x = 0.0f;
        offset_y = 0.0f;

        float single_x = 0.0f;
        float single_y = 0.0f;
        bool single_accepted = false;

        // Iterative refinement.
        for (int iteration = 0; iteration < max_iterations; iteration++) {
            const int integer_x = static_cast<int>(math::floor(static_cast<double>(offset_x)));
            const int integer_y = static_cast<int>(math::floor(static_cast<double>(offset_y)));
            const float fraction_x = offset_x - static_cast<float>(integer_x);
            const float fraction_y = offset_y - static_cast<float>(integer_y);
            const float w00 = (1.0f - fraction_x) * (1.0f - fraction_y);
            const float w01 = fraction_x * (1.0f - fraction_y);
            const float w10 = (1.0f - fraction_x) * fraction_y;
            const float w11 = fraction_x * fraction_y;

            float samples[samples_stride * samples_stride];
            const unsigned char* __restrict const centre = data + integer_y * stride + integer_x;
            for (int y = 0; y < samples_stride; ++y) {
                for (int x = 0; x < samples_stride; ++x) {
                    const unsigned char* __restrict const base = centre + (y - window_size - 1) * stride + (x - window_size - 1);
                    samples[y * samples_stride + x] = static_cast<float>(base[0]) * w00 + static_cast<float>(base[1]) * w01 + static_cast<float>(base[stride]) * w10 + static_cast<float>(base[stride + 1]) * w11;
                }
            }

            float A11 = 0.0f;
            float A12 = 0.0f;
            float A22 = 0.0f;
            float b1 = 0.0f;
            float b2 = 0.0f;

            for (int dy = -window_size; dy <= window_size; dy++) {
                for (int dx = -window_size; dx <= window_size; dx++) {
                    const float* __restrict const ptr = samples + (dy + window_size + 1) * samples_stride + (dx + window_size + 1);

                    // Calculate gradients
                    const float gx = (ptr[1] - ptr[-1]) * 0.5f;
                    const float gy = (ptr[samples_stride] - ptr[-samples_stride]) * 0.5f;
                    const float norm = gx * gx + gy * gy;
                    if (norm < eps) {
                        continue;
                    }

                    // At the corner, edge normals should intersect
                    // The edge normal at point (dx,dy) is (gx,gy)
                    // We want to find where most lines intersect.

                    A11 += gx * gx;
                    A12 += gx * gy;
                    A22 += gy * gy;

                    b1 += gx * (gx * static_cast<float>(dx) + gy * static_cast<float>(dy));
                    b2 += gy * (gx * static_cast<float>(dx) + gy * static_cast<float>(dy));
                }
            }

            // Solve the 2x2 system
            const float det = A11 * A22 - A12 * A12;
            if (math::abs(det) < eps) {
                break;
            }

            const float inv_det = 1.0f / det;
            const float delta_x = (A22 * b1 - A12 * b2) * inv_det;
            const float delta_y = (A11 * b2 - A12 * b1) * inv_det;

            offset_x += delta_x;
            offset_y += delta_y;

            if (iteration == 0) {
                single_x = offset_x;
                single_y = offset_y;
                single_accepted = (math::abs(offset_x) <= max_offset) && (math::abs(offset_y) <= max_offset);
            }

            // Note: The refined location must remain strictly interior to the measurement window, as gradient estimates at the window edge already sample pixels outside it.
            if ((math::abs(offset_x) > max_offset) || (math::abs(offset_y) > max_offset)) {
                break;
            }

            // Check for convergence
            if (delta_x * delta_x + delta_y * delta_y < eps) {
                return true;
            }
        }

        offset_x = single_x;
        offset_y = single_y;
        return single_accepted;
    }

    void subpixel::patch_41x41_bilinear(
        const unsigned char* __restrict const data,
        const int stride,
        const float offset_x,
        const float offset_y,
        unsigned char* __restrict patch
    ) {
        constexpr static const auto clamp = [](float value) -> unsigned char {
            return static_cast<unsigned char>(math::min(math::max(int(math::round(value)), 0), 255));
        };

        constexpr int patch_size = 41;
        constexpr int half_size = patch_size / 2;

        // Split the offset into an integer part and a fractional part in [0, 1).
        // Note: The floor ensures negative offsets are handled consistently, e.g. an offset of -0.3 becomes an integer part of -1 and a fraction of +0.7.
        const int int_x = static_cast<int>(math::floor(static_cast<double>(offset_x)));
        const int int_y = static_cast<int>(math::floor(static_cast<double>(offset_y)));
        const float fx = offset_x - static_cast<float>(int_x);
        const float fy = offset_y - static_cast<float>(int_y);

        // The bilinear interpolation weights are constant across the patch.
        const float w00 = (1 - fx) * (1 - fy);
        const float w01 = fx * (1 - fy);
        const float w10 = (1 - fx) * fy;
        const float w11 = fx * fy;

        // Pointer to the pixel at the integer part of the refined location, so the patch is centered on the refined subpixel location.
        const unsigned char* __restrict const center = data + int_y * stride + int_x;

        for (int dy = -half_size; dy <= half_size; ++dy) {
            for (int dx = -half_size; dx <= half_size; ++dx) {
                // Pointer to top-left pixel of the interpolation square
                const unsigned char* base = center + dy * stride + dx;

                const float i00 = static_cast<float>(base[0]);
                const float i01 = static_cast<float>(base[1]);
                const float i10 = static_cast<float>(base[stride]);
                const float i11 = static_cast<float>(base[stride + 1]);

                const float val = i00 * w00 + i01 * w01 + i10 * w10 + i11 * w11;

                patch[(dy + half_size) * patch_size + (dx + half_size)] = clamp(val);
            }
        }
    }
}
