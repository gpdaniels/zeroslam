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
#ifndef ZEROSLAM_FEATURE_DISTRIBUTOR_SQUARE_COVERING_HPP
#define ZEROSLAM_FEATURE_DISTRIBUTOR_SQUARE_COVERING_HPP

#include "feature/point.hpp"

namespace feature::distributor {
    class square_covering final {
    public:
        static int distribute(
            const point* __restrict const features_detected_sorted,
            const int features_detected_sorted_size,
            const int max_width,
            const int max_height,
            const int min_features,
            const int max_features,
            point* __restrict features_distributed
        );
    };
}

#endif // ZEROSLAM_FEATURE_DISTRIBUTOR_SQUARE_COVERING_HPP
