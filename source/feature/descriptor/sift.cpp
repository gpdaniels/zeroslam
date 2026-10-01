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

#include "feature/descriptor/sift.hpp"

#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::descriptor {
    namespace {
        // The keypoint independent part of every sample: its Gaussian weight, and along each axis its lower cell and the share of the upper cell.
        struct sample_table final {
            float gaussian[sift::samples][sift::samples];
            int cell[sift::samples];
            float cell_share[sift::samples];
        };

        const sample_table& cached_samples() {
            static const sample_table table = []() {
                sample_table built;
                const float sigma = 0.5f * static_cast<float>(sift::samples);
                const float half = 0.5f * static_cast<float>(sift::samples);
                for (int grid = 0; grid < sift::samples; ++grid) {
                    const float u = (static_cast<float>(grid) + 0.5f) - half;
                    const float c = ((u + half) / static_cast<float>(sift::samples)) * static_cast<float>(sift::cells) - 0.5f;
                    built.cell[grid] = static_cast<int>(math::floor(c));
                    built.cell_share[grid] = c - static_cast<float>(built.cell[grid]);
                }
                for (int gy = 0; gy < sift::samples; ++gy) {
                    for (int gx = 0; gx < sift::samples; ++gx) {
                        const float ux = (static_cast<float>(gx) + 0.5f) - half;
                        const float uy = (static_cast<float>(gy) + 0.5f) - half;
                        built.gaussian[gy][gx] = math::exp(-((ux * ux) + (uy * uy)) / (2.0f * sigma * sigma));
                    }
                }
                return built;
            }();
            return table;
        }
    }

    void sift::describe_float(const unsigned char* __restrict const data, const int stride, const float angle_radians, const float* const affine, float (&vector)[sift::dimensions]) {
        sift::describe_float(data, stride, 0.0f, 0.0f, angle_radians, affine, vector);
    }

    void sift::describe_float(const unsigned char* __restrict const data, const int stride, const float offset_x, const float offset_y, const float angle_radians, const float* const affine, float (&vector)[sift::dimensions]) {
        const sample_table& table = cached_samples();
        for (int d = 0; d < sift::dimensions; ++d) {
            vector[d] = 0.0f;
        }
        const float cosine = math::cos(angle_radians);
        const float sine = math::sin(angle_radians);
        const float half = 0.5f * static_cast<float>(sift::samples);
        const auto sample = [&](const float x, const float y) {
            const int x0 = static_cast<int>(math::floor(x));
            const int y0 = static_cast<int>(math::floor(y));
            const float fx = x - static_cast<float>(x0);
            const float fy = y - static_cast<float>(y0);
            const unsigned char* const row0 = data + (static_cast<long>(y0) * stride) + x0;
            const unsigned char* const row1 = row0 + stride;
            const float top = (static_cast<float>(row0[0]) * (1.0f - fx)) + (static_cast<float>(row0[1]) * fx);
            const float bottom = (static_cast<float>(row1[0]) * (1.0f - fx)) + (static_cast<float>(row1[1]) * fx);
            return (top * (1.0f - fy)) + (bottom * fy);
        };
        for (int gy = 0; gy < sift::samples; ++gy) {
            for (int gx = 0; gx < sift::samples; ++gx) {
                const float ux = (static_cast<float>(gx) + 0.5f) - half;
                const float uy = (static_cast<float>(gy) + 0.5f) - half;
                float ax = ux;
                float ay = uy;
                if (affine != nullptr) {
                    ax = (affine[0] * ux) + (affine[1] * uy);
                    ay = (affine[2] * ux) + (affine[3] * uy);
                }
                const float px = offset_x + sift::sample_spacing * ((cosine * ax) - (sine * ay));
                const float py = offset_y + sift::sample_spacing * ((sine * ax) + (cosine * ay));
                const float dx = 0.5f * (sample(px + 1.0f, py) - sample(px - 1.0f, py));
                const float dy = 0.5f * (sample(px, py + 1.0f) - sample(px, py - 1.0f));
                const float rotated_x = (cosine * dx) + (sine * dy);
                const float rotated_y = (-sine * dx) + (cosine * dy);
                // Note: The samples sit at R A u, so the gradient of the sampled patch is the image gradient times (R A) transposed.
                const float rx = (affine != nullptr) ? ((affine[0] * rotated_x) + (affine[2] * rotated_y)) : rotated_x;
                const float ry = (affine != nullptr) ? ((affine[1] * rotated_x) + (affine[3] * rotated_y)) : rotated_y;
                const float magnitude = math::sqrt((rx * rx) + (ry * ry));
                if (magnitude <= 0.0f) {
                    continue;
                }
                const float weight = magnitude * table.gaussian[gy][gx];
                float orientation = math::atan2(ry, rx);
                if (orientation < 0.0f) {
                    orientation += 2.0f * 3.14159265358979323846f;
                }
                const float ob = orientation * (static_cast<float>(sift::bins) / (2.0f * 3.14159265358979323846f));
                const int cx0 = table.cell[gx];
                const int cy0 = table.cell[gy];
                const int ob0 = static_cast<int>(math::floor(ob));
                const float wx = table.cell_share[gx];
                const float wy = table.cell_share[gy];
                const float wo = ob - static_cast<float>(ob0);
                for (int iy = 0; iy < 2; ++iy) {
                    const int cell_y = cy0 + iy;
                    if ((cell_y < 0) || (cell_y >= sift::cells)) {
                        continue;
                    }
                    for (int ix = 0; ix < 2; ++ix) {
                        const int cell_x = cx0 + ix;
                        if ((cell_x < 0) || (cell_x >= sift::cells)) {
                            continue;
                        }
                        for (int io = 0; io < 2; ++io) {
                            const int bin = (ob0 + io) % sift::bins;
                            const float share = weight * (ix ? wx : (1.0f - wx)) * (iy ? wy : (1.0f - wy)) * (io ? wo : (1.0f - wo));
                            vector[(((cell_y * sift::cells) + cell_x) * sift::bins) + bin] += share;
                        }
                    }
                }
            }
        }
        const auto normalise_l2 = [&]() {
            float sum = 0.0f;
            for (int d = 0; d < sift::dimensions; ++d) {
                sum += vector[d] * vector[d];
            }
            const float inverse = (sum > 0.0f) ? (1.0f / math::sqrt(sum)) : 0.0f;
            for (int d = 0; d < sift::dimensions; ++d) {
                vector[d] *= inverse;
            }
        };
        normalise_l2();
        for (int d = 0; d < sift::dimensions; ++d) {
            vector[d] = math::min(vector[d], sift::clip);
        }
        normalise_l2();
        float l1 = 0.0f;
        for (int d = 0; d < sift::dimensions; ++d) {
            l1 += vector[d];
        }
        const float inverse_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
        for (int d = 0; d < sift::dimensions; ++d) {
            vector[d] = math::sqrt(vector[d] * inverse_l1);
        }
    }

    void sift::binarise(const float (&vector)[sift::dimensions], binary<256>& descriptor) {
        for (int byte = 0; byte < 32; ++byte) {
            descriptor.data[byte] = 0;
        }
        float sorted[sift::dimensions];
        for (int d = 0; d < sift::dimensions; ++d) {
            sorted[d] = vector[d];
        }
        std::nth_element(&sorted[0], &sorted[sift::dimensions / 2], &sorted[0] + sift::dimensions);
        const float median = sorted[sift::dimensions / 2];
        for (int d = 0; d < sift::dimensions; ++d) {
            if (vector[d] > median) {
                descriptor.data[d >> 3] = static_cast<unsigned char>(descriptor.data[d >> 3] | (1u << (d & 7)));
            }
            // Note: Pairing every dimension with the one half the descriptor later compares each pair twice, so the second half pairs with the same bin of the next cell.
            const int other = (d < (sift::dimensions / 2)) ? (d + (sift::dimensions / 2)) : ((d + sift::bins) % sift::dimensions);
            if (vector[d] > vector[other]) {
                const int bit = sift::dimensions + d;
                descriptor.data[bit >> 3] = static_cast<unsigned char>(descriptor.data[bit >> 3] | (1u << (bit & 7)));
            }
        }
    }

    void sift::describe(const unsigned char* __restrict const data, const int stride, const float angle_radians, binary<256>& descriptor) {
        sift::describe(data, stride, 0.0f, 0.0f, angle_radians, descriptor);
    }

    void sift::describe(const unsigned char* __restrict const data, const int stride, const float offset_x, const float offset_y, const float angle_radians, binary<256>& descriptor) {
        float vector[sift::dimensions];
        sift::describe_float(data, stride, offset_x, offset_y, angle_radians, nullptr, vector);
        sift::binarise(vector, descriptor);
    }
}
