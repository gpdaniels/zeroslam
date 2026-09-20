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
#ifndef ZEROSLAM_MAPPING_LINE_HPP
#define ZEROSLAM_MAPPING_LINE_HPP

#include "geometry/plucker.hpp"
#include "math/matrix.hpp"

namespace mapping {
    class line final {
    public:
        int id;
        geometry::plucker plucker_line;
        math::matrix<double, 3, 1> locations[2];

    public:
        line();
        line(int input_id, const geometry::plucker& input_line, const math::matrix<double, 3, 1>& input_location_1, const math::matrix<double, 3, 1>& input_location_2);

        void resynchronise_endpoints();
    };
}

#endif // ZEROSLAM_MAPPING_LINE_HPP
