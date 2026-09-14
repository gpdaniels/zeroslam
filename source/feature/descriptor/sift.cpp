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

namespace feature::descriptor {
    void sift::describe_float(const unsigned char* __restrict const data, const int stride, const float angle_radians, const float* const affine, float (&vector)[sift::dimensions]) {
        for (int d = 0; d < sift::dimensions; ++d) {
            vector[d] = 0.0f;
        }
        const float cosine = math::cos(angle_radians);
        const float sine = math::sin(angle_radians);
        const float sigma = 0.5f * static_cast<float>(sift::samples);
        const float half = 0.5f * static_cast<float>(sift::samples);
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
                const float px = sift::sample_spacing * ((cosine * ax) - (sine * ay));
                const float py = sift::sample_spacing * ((sine * ax) + (cosine * ay));
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
                const float dx = 0.5f * (sample(px + 1.0f, py) - sample(px - 1.0f, py));
                const float dy = 0.5f * (sample(px, py + 1.0f) - sample(px, py - 1.0f));
                const float rx = (cosine * dx) + (sine * dy);
                const float ry = (-sine * dx) + (cosine * dy);
                const float magnitude = math::sqrt((rx * rx) + (ry * ry));
                if (magnitude <= 0.0f) {
                    continue;
                }
                const float weight = magnitude * math::exp(-((ux * ux) + (uy * uy)) / (2.0f * sigma * sigma));
                float orientation = math::atan2(ry, rx);
                if (orientation < 0.0f) {
                    orientation += 2.0f * 3.14159265358979323846f;
                }
                const float cx = ((ux + half) / static_cast<float>(sift::samples)) * static_cast<float>(sift::cells) - 0.5f;
                const float cy = ((uy + half) / static_cast<float>(sift::samples)) * static_cast<float>(sift::cells) - 0.5f;
                const float ob = orientation * (static_cast<float>(sift::bins) / (2.0f * 3.14159265358979323846f));
                const int cx0 = static_cast<int>(math::floor(cx));
                const int cy0 = static_cast<int>(math::floor(cy));
                const int ob0 = static_cast<int>(math::floor(ob));
                const float wx = cx - static_cast<float>(cx0);
                const float wy = cy - static_cast<float>(cy0);
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
        for (int i = 0; i <= sift::dimensions / 2; ++i) {
            int smallest = i;
            for (int j = i + 1; j < sift::dimensions; ++j) {
                if (sorted[j] < sorted[smallest]) {
                    smallest = j;
                }
            }
            const float swap = sorted[i];
            sorted[i] = sorted[smallest];
            sorted[smallest] = swap;
        }
        const float median = sorted[sift::dimensions / 2];
        for (int d = 0; d < sift::dimensions; ++d) {
            if (vector[d] > median) {
                descriptor.data[d >> 3] = static_cast<unsigned char>(descriptor.data[d >> 3] | (1u << (d & 7)));
            }
            const int other = (d + (2 * sift::cells * sift::bins)) % sift::dimensions;
            if (vector[d] > vector[other]) {
                const int bit = sift::dimensions + d;
                descriptor.data[bit >> 3] = static_cast<unsigned char>(descriptor.data[bit >> 3] | (1u << (bit & 7)));
            }
        }
    }

    void sift::describe(const unsigned char* __restrict const data, const int stride, const float angle_radians, binary<256>& descriptor) {
        float vector[sift::dimensions];
        sift::describe_float(data, stride, angle_radians, nullptr, vector);
        sift::binarise(vector, descriptor);
    }
}
