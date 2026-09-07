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
#ifndef ZEROSLAM_FEATURE_SUPPRESSOR_FAST_HPP
#define ZEROSLAM_FEATURE_SUPPRESSOR_FAST_HPP

#include "feature/point.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace feature::suppressor {
    class fast final {
    public:
        static size_t suppress(
            const point* __restrict const features,
            const size_t features_count,
            const size_t max_row,
            point* __restrict const features_suppressed
        );
    };
}

#endif // ZEROSLAM_FEATURE_SUPPRESSOR_FAST_HPP
