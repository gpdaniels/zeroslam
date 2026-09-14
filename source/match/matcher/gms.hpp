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
#ifndef ZEROSLAM_MATCH_MATCHER_GMS_HPP
#define ZEROSLAM_MATCH_MATCHER_GMS_HPP

#include "feature/point.hpp"
#include "match/pair.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace match::matcher {
    class gms final {
    public:
        struct options final {
            int grid_size = 0;
            float grid_matches_per_cell = 6.0f;
            int grid_size_maximum = 20;
            int grid_size_minimum = 4;
            float alpha = 6.0f;
            float margin = 0.1f;
        };

    public:
        static size_t filter(
            const feature::point* const lhs_points,
            const float lhs_width,
            const float lhs_height,
            const feature::point* const rhs_points,
            const float rhs_width,
            const float rhs_height,
            match::pair* const matches,
            const size_t matches_size,
            const options& settings
        );

        static size_t filter(
            const feature::point* const lhs_points,
            const float lhs_width,
            const float lhs_height,
            const feature::point* const rhs_points,
            const float rhs_width,
            const float rhs_height,
            match::pair* const matches,
            const size_t matches_size
        );
    };
}

#endif // ZEROSLAM_MATCH_MATCHER_GMS_HPP
