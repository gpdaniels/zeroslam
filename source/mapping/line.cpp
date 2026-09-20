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

#include "mapping/line.hpp"

namespace mapping {
    line::line()
        : id(-1)
        , plucker_line()
        , locations{ math::matrix<double, 3, 1>::zero(), math::matrix<double, 3, 1>::zero() } {
    }

    line::line(int input_id, const geometry::plucker& input_line, const math::matrix<double, 3, 1>& input_location_1, const math::matrix<double, 3, 1>& input_location_2)
        : id(input_id)
        , plucker_line(input_line)
        , locations{ input_location_1, input_location_2 } {
    }

    void line::resynchronise_endpoints() {
        this->locations[0] = this->plucker_line.project_point(this->locations[0]);
        this->locations[1] = this->plucker_line.project_point(this->locations[1]);
    }
}
