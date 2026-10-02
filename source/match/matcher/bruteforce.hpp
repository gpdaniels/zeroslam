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
#ifndef ZEROSLAM_MATCH_MATCHER_BRUTEFORCE_HPP
#define ZEROSLAM_MATCH_MATCHER_BRUTEFORCE_HPP

#include "feature/descriptor/binary.hpp"
#include "match/pair.hpp"

namespace match::matcher {
    class bruteforce final {
    public:
        // The matches_count nearest right descriptors of each left descriptor, ascending, a tie going to the lower right index.
        // A left descriptor is reported only when its best score is under the threshold, and then with every runner up it has (fewer only
        // when there are fewer right descriptors) whatever their scores, so a caller can ratio test the best against the second.
        // Each left descriptor needs matches_count free slots, matching stops at the first without room, and the count written is returned.
        static size_t find_matches(
            const feature::descriptor::stored* lhs_descriptors,
            const size_t lhs_descriptors_size,
            const feature::descriptor::stored* rhs_descriptors,
            const size_t rhs_descriptors_size,
            const float threshold,
            const size_t matches_count,
            match::pair* matches,
            const size_t matches_size
        );
    };
}

#endif // ZEROSLAM_MATCH_MATCHER_BRUTEFORCE_HPP
