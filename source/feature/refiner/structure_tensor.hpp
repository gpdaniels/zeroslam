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
#ifndef ZEROSLAM_FEATURE_REFINER_STRUCTURE_TENSOR_HPP
#define ZEROSLAM_FEATURE_REFINER_STRUCTURE_TENSOR_HPP

#include "feature/score/structure_tensor.hpp"

namespace feature::refiner {
    class structure_tensor final {
    public:
        constexpr static const int search_radius = 2;

        constexpr static const float sigma_maximum = 3.2f;

    public:
        static int footprint(const float sigma);

        static bool refine(
            const unsigned char* __restrict const data,
            const int stride,
            const score::structure_tensor::measure kind,
            const float sigma,
            float& offset_x,
            float& offset_y
        );
    };
}

#endif // ZEROSLAM_FEATURE_REFINER_STRUCTURE_TENSOR_HPP
