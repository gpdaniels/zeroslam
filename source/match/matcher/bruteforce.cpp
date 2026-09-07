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

#include "match/matcher/bruteforce.hpp"

#include "match/distance/hamming.hpp"

namespace match::matcher {
    size_t bruteforce::find_matches(
        const feature::descriptor::binary<256>* lhs_descriptors,
        const size_t lhs_descriptors_size,
        const feature::descriptor::binary<256>* rhs_descriptors,
        const size_t rhs_descriptors_size,
        const float threshold,
        const size_t matches_count,
        match::pair* matches,
        const size_t matches_size
    ) {
        if ((matches_count == 0) || (matches_size == 0)) {
            return 0;
        }
        size_t count = 0;
        for (size_t lhs_index = 0; lhs_index < lhs_descriptors_size; ++lhs_index) {
            if (count + matches_count > matches_size) {
                break;
            }
            for (size_t matches_index = 0; matches_index < matches_count; ++matches_index) {
                matches[count + matches_index].lhs_index = lhs_index;
                matches[count + matches_index].score = threshold;
            }
            for (size_t rhs_index = 0; rhs_index < rhs_descriptors_size; ++rhs_index) {
                const float score = static_cast<float>(distance::hamming::distance(lhs_descriptors[lhs_index], rhs_descriptors[rhs_index]));
                for (size_t matches_index = 0; matches_index < matches_count; ++matches_index) {
                    if (score < matches[count + matches_index].score) {
                        for (size_t shift_index = matches_count - 1; shift_index > matches_index; --shift_index) {
                            matches[count + shift_index].score = matches[count + shift_index - 1].score;
                            matches[count + shift_index].rhs_index = matches[count + shift_index - 1].rhs_index;
                        }
                        matches[count + matches_index].score = score;
                        matches[count + matches_index].rhs_index = rhs_index;
                        break;
                    }
                }
            }
            const size_t save_index = count;
            for (size_t matches_index = 0; matches_index < matches_count; ++matches_index) {
                if (matches[save_index + matches_index].score < threshold) {
                    ++count;
                }
            }
        }
        return count;
    }
}
