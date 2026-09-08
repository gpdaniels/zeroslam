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
#ifndef ZEROSLAM_ESTIMATION_POSE_HOMOGRAPHY_HPP
#define ZEROSLAM_ESTIMATION_POSE_HOMOGRAPHY_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::pose {
    // Relative pose [R | t] of the rhs camera from a homography mapping rhs points to lhs points (Faugeras and Lustman) and its normalised correspondences by a two view cheirality vote; same outputs and acceptance rule as pose::essential.
    template <typename type>
    class homography final {
    public:
        static bool recover(
            const type* const __restrict homography,
            const type* const __restrict lhs_points,
            const type* const __restrict rhs_points,
            const size_t point_count,
            type* const __restrict rotation,
            type* const __restrict translation,
            type* const __restrict triangulated_points,
            size_t* const __restrict support_count = nullptr
        );
    };

    extern template class homography<float>;
    extern template class homography<double>;
}

#endif // ZEROSLAM_ESTIMATION_POSE_HOMOGRAPHY_HPP
