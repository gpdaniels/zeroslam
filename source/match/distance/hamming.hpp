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
#ifndef ZEROSLAM_MATCH_DISTANCE_HAMMING_HPP
#define ZEROSLAM_MATCH_DISTANCE_HAMMING_HPP

#include "feature/descriptor/binary.hpp"

namespace match::distance {
    class hamming final {
    public:
        static unsigned int distance(
            const feature::descriptor::binary<256>& lhs,
            const feature::descriptor::binary<256>& rhs
        );
    };
}

#endif // ZEROSLAM_MATCH_DISTANCE_HAMMING_HPP
