/*
Copyright (C) 2025 Geoffrey Daniels. https://gpdaniels.com/

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
#ifndef LANDMARK_HPP
#define LANDMARK_HPP

#include "matrix.hpp"

namespace landmark {
    class point {
    public:
        static inline int id_generator = 0;
        int id;
        matrix::matrix<double, 3, 1> location;
        matrix::matrix<double, 3, 1> colour;

    public:
        point();
        point(const matrix::matrix<double, 3, 1>& input_location, const matrix::matrix<double, 3, 1>& input_colour);
    };

    class line {
    public:
        static inline int id_generator = 0;
        int id;
        matrix::matrix<double, 3, 1> locations[2];

    public:
        line();
        line(const matrix::matrix<double, 3, 1>& input_location_1, const matrix::matrix<double, 3, 1>& input_location_2);
    };
}

namespace landmark {
    point::point() {
        this->id = -1;
        this->location = matrix::matrix<double, 3, 1>::zero();
        this->colour = matrix::matrix<double, 3, 1>::zero();
    }

    point::point(const matrix::matrix<double, 3, 1>& input_location, const matrix::matrix<double, 3, 1>& input_colour) {
        this->id = point::id_generator++;
        this->location = input_location;
        this->colour = input_colour;
    }

    line::line() {
        this->id = -1;
        this->locations[0] = matrix::matrix<double, 3, 1>::zero();
        this->locations[1] = matrix::matrix<double, 3, 1>::zero();
    }

    line::line(const matrix::matrix<double, 3, 1>& input_location_1, const matrix::matrix<double, 3, 1>& input_location_2) {
        this->id = line::id_generator++;
        this->locations[0] = input_location_1;
        this->locations[1] = input_location_2;
    }
}

#endif // LANDMARK_HPP