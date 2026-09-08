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
#ifndef ZEROSLAM_ESTIMATION_MINIMAL_HOMOGRAPHY_4_POINT_HPP
#define ZEROSLAM_ESTIMATION_MINIMAL_HOMOGRAPHY_4_POINT_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::minimal {
    // Homography mapping rhs points to lhs points from four normalised correspondences, lhs ~ H rhs, by the normalised direct linear transform; points are packed x, y pairs, H is 3 by 3 row major with unit Frobenius norm.
    template <typename type>
    class homography_4_point final {
    public:
        static bool solve(
            const type* const __restrict lhs_points,
            const type* const __restrict rhs_points,
            type* const __restrict homography
        );
    };

    extern template class homography_4_point<float>;
    extern template class homography_4_point<double>;
}

#endif // ZEROSLAM_ESTIMATION_MINIMAL_HOMOGRAPHY_4_POINT_HPP
