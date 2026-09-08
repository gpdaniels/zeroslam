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
#ifndef ZEROSLAM_ESTIMATION_MINIMAL_ESSENTIAL_5_POINT_HPP
#define ZEROSLAM_ESTIMATION_MINIMAL_ESSENTIAL_5_POINT_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::minimal {
    // Essential matrices from five normalised correspondences such that rhs^T E lhs = 0; points are packed x, y pairs and up to ten 3 by 3 row major matrices are written; returns how many.
    template <typename type>
    class essential_5_point final {
    public:
        static int solve(
            const type* const __restrict lhs_points,
            const type* const __restrict rhs_points,
            type* const __restrict essentials
        );
    };

    extern template class essential_5_point<float>;
    extern template class essential_5_point<double>;
}

#endif // ZEROSLAM_ESTIMATION_MINIMAL_ESSENTIAL_5_POINT_HPP
