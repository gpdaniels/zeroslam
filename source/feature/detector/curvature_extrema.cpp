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

#include "feature/detector/curvature_extrema.hpp"

#include "image/blur.hpp"
#include "image/derivative.hpp"
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::detector {
    bool curvature_extrema::refine_subpixel(
        const std::int64_t* __restrict const kappa,
        const int stride,
        const int cell_x,
        const int cell_y,
        float& offset_x,
        float& offset_y
    ) {
        double m00 = 0.0;
        double m10 = 0.0;
        double m01 = 0.0;
        double m20 = 0.0;
        double m02 = 0.0;
        double m11 = 0.0;
        for (int v = -1; v <= 1; ++v) {
            for (int u = -1; u <= 1; ++u) {
                const double value = static_cast<double>(kappa[(cell_y + v) * stride + (cell_x + u)]);
                m00 += value;
                m10 += static_cast<double>(u) * value;
                m01 += static_cast<double>(v) * value;
                m20 += static_cast<double>(u * u) * value;
                m02 += static_cast<double>(v * v) * value;
                m11 += static_cast<double>(u * v) * value;
            }
        }
        const double b = m10 / 6.0;
        const double c = m01 / 6.0;
        const double e = m11 / 4.0;
        const double d_minus_f = (m20 - m02) / 2.0;
        const double d_plus_f = (m20 + m02) / 2.0 - (2.0 / 3.0) * m00;
        const double d = (d_plus_f + d_minus_f) / 2.0;
        const double f = (d_plus_f - d_minus_f) / 2.0;
        const double determinant = 4.0 * d * f - e * e;
        if (math::abs(determinant) < 1.0e-12) {
            return false;
        }
        const double du = (-2.0 * f * b + e * c) / determinant;
        const double dv = (e * b - 2.0 * d * c) / determinant;
        if ((math::abs(du) >= 1.0) || (math::abs(dv) >= 1.0)) {
            return false;
        }
        offset_x = static_cast<float>(du);
        offset_y = static_cast<float>(dv);
        return true;
    }

    size_t curvature_extrema::detect(
        const std::int64_t* __restrict const kappa,
        const int width,
        const int height,
        const int stride,
        const options& settings,
        std::vector<extremum>& extrema,
        std::int64_t* const threshold_out
    ) {
        extrema.clear();
        const int border = math::max(2, settings.border);
        if ((width <= 2 * border) || (height <= 2 * border)) {
            if (threshold_out != nullptr) {
                *threshold_out = 0;
            }
            return 0;
        }

        for (int y = border; y < height - border; ++y) {
            const std::int64_t* __restrict const row_above = kappa + (y - 1) * stride;
            const std::int64_t* __restrict const row_centre = kappa + y * stride;
            const std::int64_t* __restrict const row_below = kappa + (y + 1) * stride;
            for (int x = border; x < width - border; ++x) {
                const std::int64_t centre = row_centre[x];
                const bool is_maximum =
                    (centre > row_above[x - 1]) && (centre > row_above[x]) && (centre > row_above[x + 1]) &&
                    (centre > row_centre[x - 1]) && (centre > row_centre[x + 1]) &&
                    (centre > row_below[x - 1]) && (centre > row_below[x]) && (centre > row_below[x + 1]);
                const bool is_minimum =
                    (centre < row_above[x - 1]) && (centre < row_above[x]) && (centre < row_above[x + 1]) &&
                    (centre < row_centre[x - 1]) && (centre < row_centre[x + 1]) &&
                    (centre < row_below[x - 1]) && (centre < row_below[x]) && (centre < row_below[x + 1]);
                if (!is_maximum && !is_minimum) {
                    continue;
                }
                extremum found;
                found.x = static_cast<float>(x);
                found.y = static_cast<float>(y);
                found.cell_x = x;
                found.cell_y = y;
                found.sign = is_maximum ? 1 : -1;
                found.response = (centre < 0) ? -centre : centre;
                extrema.push_back(found);
            }
        }

        std::int64_t threshold = 0;
        if ((settings.quantile > 0.0f) && (!extrema.empty())) {
            std::vector<std::int64_t> responses;
            responses.reserve(extrema.size());
            for (const extremum& candidate : extrema) {
                responses.push_back(candidate.response);
            }
            std::sort(responses.begin(), responses.end());
            const float quantile = math::min(settings.quantile, 1.0f);
            const size_t index = static_cast<size_t>(static_cast<double>(quantile) * static_cast<double>(responses.size() - 1));
            threshold = responses[index];
            size_t write = 0;
            for (size_t read = 0; read < extrema.size(); ++read) {
                if (extrema[read].response >= threshold) {
                    extrema[write++] = extrema[read];
                }
            }
            extrema.resize(write);
        }
        if (threshold_out != nullptr) {
            *threshold_out = threshold;
        }

        if (extrema.size() > settings.maximum) {
            std::vector<size_t> order(extrema.size());
            for (size_t i = 0; i < order.size(); ++i) {
                order[i] = i;
            }
            std::sort(order.begin(), order.end(), [&extrema](const size_t lhs, const size_t rhs) {
                if (extrema[lhs].response != extrema[rhs].response) {
                    return extrema[lhs].response > extrema[rhs].response;
                }
                return lhs < rhs;
            });
            order.resize(settings.maximum);
            std::sort(order.begin(), order.end());
            std::vector<extremum> kept;
            kept.reserve(order.size());
            for (const size_t index : order) {
                kept.push_back(extrema[index]);
            }
            extrema = static_cast<std::vector<extremum>&&>(kept);
        }

        if (settings.subpixel) {
            for (extremum& candidate : extrema) {
                float offset_x = 0.0f;
                float offset_y = 0.0f;
                if (curvature_extrema::refine_subpixel(kappa, stride, candidate.cell_x, candidate.cell_y, offset_x, offset_y)) {
                    candidate.x = static_cast<float>(candidate.cell_x) + offset_x;
                    candidate.y = static_cast<float>(candidate.cell_y) + offset_y;
                }
            }
        }
        return extrema.size();
    }

    size_t curvature_extrema::detect(
        const unsigned char* __restrict const data,
        const int width,
        const int height,
        const int stride,
        const options& settings,
        std::vector<std::int64_t>& kappa,
        std::vector<extremum>& extrema,
        std::int64_t* const threshold_out
    ) {
        const size_t plane_size = static_cast<size_t>(height) * static_cast<size_t>(stride);
        std::vector<unsigned char> blurred(plane_size);
        image::blur::gaussian_7x7(data, width, height, stride, blurred.data());
        std::vector<std::int16_t> dx(plane_size);
        std::vector<std::int16_t> dy(plane_size);
        std::vector<std::int16_t> dxx(plane_size);
        std::vector<std::int16_t> dyy(plane_size);
        std::vector<std::int16_t> dxy(plane_size);
        kappa.resize(plane_size);
        image::derivative::curvature(blurred.data(), width, height, stride, dx.data(), dy.data(), dxx.data(), dyy.data(), dxy.data(), kappa.data());
        options adjusted = settings;
        adjusted.border = math::max(settings.border, 5);
        return curvature_extrema::detect(kappa.data(), width, height, stride, adjusted, extrema, threshold_out);
    }
}
