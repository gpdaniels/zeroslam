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
#ifndef ZEROSLAM_FEATURE_SCORE_STRUCTURE_TENSOR_HPP
#define ZEROSLAM_FEATURE_SCORE_STRUCTURE_TENSOR_HPP

namespace feature::score {
    class structure_tensor final {
    public:
        enum class measure {
            klt,
            forstner,
            harris,
            rohr,
            kenney
        };

        constexpr static const float harris_k = 0.04f;

        constexpr static const float sigma_maximum = 3.5f;
        constexpr static const int smoothing_radius_maximum = 25;

    public:
        static float score(const measure kind, const float ixx, const float ixy, const float iyy);

        static float strength(const measure kind, const float response);

        static int smoothing_radius(const float sigma);

        static void smoothing_weights(const float sigma, float* __restrict const weights);

        static int margin(const float sigma);

        static void respond(
            const unsigned char* __restrict const data,
            const int width,
            const int height,
            const int stride,
            const measure kind,
            const float sigma,
            float* __restrict const response
        );
    };
}

#endif // ZEROSLAM_FEATURE_SCORE_STRUCTURE_TENSOR_HPP
