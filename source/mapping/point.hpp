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
#ifndef ZEROSLAM_MAPPING_POINT_HPP
#define ZEROSLAM_MAPPING_POINT_HPP

#include "math/matrix.hpp"

namespace mapping {
    class point final {
    public:
        int id;
        math::matrix<double, 3, 1> location;
        math::matrix<double, 3, 1> colour;
        unsigned char descriptor[32] = {};

    public:
        point();
        point(int input_id, const math::matrix<double, 3, 1>& input_location, const math::matrix<double, 3, 1>& input_colour);
    };
}

#endif // ZEROSLAM_MAPPING_POINT_HPP
