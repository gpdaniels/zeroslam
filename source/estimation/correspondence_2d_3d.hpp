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
#ifndef ZEROSLAM_ESTIMATION_CORRESPONDENCE_2D_3D_HPP
#define ZEROSLAM_ESTIMATION_CORRESPONDENCE_2D_3D_HPP

#include "math/matrix.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation {
    template <typename type>
    class correspondence_2d_3d final {
    public:
        math::matrix<type, 2, 1> lhs;
        math::matrix<type, 3, 1> rhs;
    };
}

#endif // ZEROSLAM_ESTIMATION_CORRESPONDENCE_2D_3D_HPP
