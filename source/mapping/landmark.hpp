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
#ifndef ZEROSLAM_MAPPING_LANDMARK_HPP
#define ZEROSLAM_MAPPING_LANDMARK_HPP

#include "math/matrix.hpp"

namespace mapping {
    class point {
    public:
        static inline int id_generator = 0;
        int id;
        math::matrix<double, 3, 1> location;
        math::matrix<double, 3, 1> colour;

    public:
        point();
        point(const math::matrix<double, 3, 1>& input_location, const math::matrix<double, 3, 1>& input_colour);
    };

    class line {
    public:
        static inline int id_generator = 0;
        int id;
        math::matrix<double, 3, 1> locations[2];

    public:
        line();
        line(const math::matrix<double, 3, 1>& input_location_1, const math::matrix<double, 3, 1>& input_location_2);
    };
}

namespace mapping {
    inline point::point() {
        this->id = -1;
        this->location = math::matrix<double, 3, 1>::zero();
        this->colour = math::matrix<double, 3, 1>::zero();
    }

    inline point::point(const math::matrix<double, 3, 1>& input_location, const math::matrix<double, 3, 1>& input_colour) {
        this->id = point::id_generator++;
        this->location = input_location;
        this->colour = input_colour;
    }

    inline line::line() {
        this->id = -1;
        this->locations[0] = math::matrix<double, 3, 1>::zero();
        this->locations[1] = math::matrix<double, 3, 1>::zero();
    }

    inline line::line(const math::matrix<double, 3, 1>& input_location_1, const math::matrix<double, 3, 1>& input_location_2) {
        this->id = line::id_generator++;
        this->locations[0] = input_location_1;
        this->locations[1] = input_location_2;
    }
}

#endif // ZEROSLAM_MAPPING_LANDMARK_HPP
