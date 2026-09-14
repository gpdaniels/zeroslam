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

#pragma once
#ifndef ZEROSLAM_FEATURE_DETECTOR_CURVATURE_EXTREMA_HPP
#define ZEROSLAM_FEATURE_DETECTOR_CURVATURE_EXTREMA_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdint>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace feature::detector {
    class curvature_extrema final {
    public:
        struct extremum final {
            float x;
            float y;
            int cell_x;
            int cell_y;
            int sign;
            std::int64_t response;
        };

        struct options final {
            float quantile = 0.95f;
            bool subpixel = true;
            int border = 2;
            size_t maximum = 100000;
        };

    public:
        static bool refine_subpixel(
            const std::int64_t* __restrict const kappa,
            const int stride,
            const int cell_x,
            const int cell_y,
            float& offset_x,
            float& offset_y
        );

        static size_t detect(
            const std::int64_t* __restrict const kappa,
            const int width,
            const int height,
            const int stride,
            const options& settings,
            std::vector<extremum>& extrema,
            std::int64_t* const threshold_out = nullptr
        );

        static size_t detect(
            const unsigned char* __restrict const data,
            const int width,
            const int height,
            const int stride,
            const options& settings,
            std::vector<std::int64_t>& kappa,
            std::vector<extremum>& extrema,
            std::int64_t* const threshold_out = nullptr
        );
    };
}

#endif // ZEROSLAM_FEATURE_DETECTOR_CURVATURE_EXTREMA_HPP
