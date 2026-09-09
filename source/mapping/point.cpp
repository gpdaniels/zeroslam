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

#include "mapping/point.hpp"

namespace mapping {
    point::point() {
        this->id = -1;
        this->location = math::matrix<double, 3, 1>::zero();
        this->colour = math::matrix<double, 3, 1>::zero();
    }

    point::point(int input_id, const math::matrix<double, 3, 1>& input_location, const math::matrix<double, 3, 1>& input_colour) {
        this->id = input_id;
        this->location = input_location;
        this->colour = input_colour;
    }
}
