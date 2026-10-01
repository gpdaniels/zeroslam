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
    // Grid-based motion statistics (Bian et al. 2017): keeps the matches whose grid cell pair is supported by the matches of the
    // neighbouring cell pairs. Each rotation and scale hypothesis enabled is tried and the one keeping the most matches is used.
    class gms final {
    public:
        struct options final {
            int grid_size = 0;
            float grid_matches_per_cell = 6.0f;
            int grid_size_maximum = 20;
            int grid_size_minimum = 4;
            float alpha = 6.0f;
            float margin = 0.1f;
            // Pair the neighbourhoods turned by each multiple of 45 degrees too, for an in-plane rotation between the images.
            bool rotation = true;
            // Try the right grid at 1/2, 1/sqrt(2), sqrt(2) and 2 times the left resolution too, for a change of scale between the images.
            // Off by default: with a few tens of matches the extra hypotheses let some false matches of an unrelated pair through.
            bool scale = false;
        };

    public:
        // Moves the kept matches to the front, in their original order, and returns how many were kept.
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
