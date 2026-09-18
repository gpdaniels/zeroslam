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
#ifndef ZEROSLAM_FEATURE_TRACKER_DOMINANT_FLOW_HPP
#define ZEROSLAM_FEATURE_TRACKER_DOMINANT_FLOW_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace feature::tracker {
    class dominant_flow final {
    public:
        struct affine final {
            float a00 = 1.0f;
            float a01 = 0.0f;
            float a10 = 0.0f;
            float a11 = 1.0f;
            float bx = 0.0f;
            float by = 0.0f;

            void apply(const float x, const float y, float& out_x, float& out_y) const;

            bool invert(affine& inverse) const;
        };

        struct options final {
            float sigma = 2.0f;
            int iterations = 10;
            float convergence = 1.0e-5f;
        };

    public:
        static bool fit(
            const float* __restrict const from_x,
            const float* __restrict const from_y,
            const float* __restrict const to_x,
            const float* __restrict const to_y,
            const size_t count,
            const options& settings,
            affine& flow
        );
    };
}

#endif // ZEROSLAM_FEATURE_TRACKER_DOMINANT_FLOW_HPP
