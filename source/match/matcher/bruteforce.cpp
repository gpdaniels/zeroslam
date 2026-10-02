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
#include "math/math.hpp"

namespace match::matcher {
    namespace {
        // Insert into the ascending best list, after any equal score so the earlier candidate wins a tie.
        void insert(match::pair* const best, const size_t best_size, const size_t rhs_index, const float score) {
            size_t slot = best_size - 1;
            while ((slot > 0) && (score < best[slot - 1].score)) {
                best[slot] = best[slot - 1];
                --slot;
            }
            best[slot].rhs_index = rhs_index;
            best[slot].score = score;
        }

        // Nothing unless the best is under the threshold, otherwise every candidate found.
        size_t reported(const match::pair* const best, const size_t best_size, const float threshold) {
            if (!(best[0].score < threshold)) {
                return 0;
            }
            size_t count = 1;
            while ((count < best_size) && math::isfinite(best[count].score)) {
                ++count;
            }
            return count;
        }
    }

    size_t bruteforce::find_matches(
        const feature::descriptor::stored* lhs_descriptors,
        const size_t lhs_descriptors_size,
        const feature::descriptor::stored* rhs_descriptors,
        const size_t rhs_descriptors_size,
        const float threshold,
        const size_t matches_count,
        match::pair* matches,
        const size_t matches_size
    ) {
        if ((matches_count == 0) || (matches_size == 0)) {
            return 0;
        }
        constexpr static const size_t block_size = 256;
        unsigned int distances[block_size];
        size_t count = 0;
        for (size_t lhs_index = 0; lhs_index < lhs_descriptors_size; ++lhs_index) {
            if (count + matches_count > matches_size) {
                break;
            }
            match::pair* const best = &matches[count];
            for (size_t matches_index = 0; matches_index < matches_count; ++matches_index) {
                best[matches_index] = match::pair{ lhs_index, 0, math::inf<float>() };
            }
            float worst = math::inf<float>();
            for (size_t block_begin = 0; block_begin < rhs_descriptors_size; block_begin += block_size) {
                const size_t block_count = math::min(block_size, rhs_descriptors_size - block_begin);
                distance::hamming::distances(lhs_descriptors[lhs_index], &rhs_descriptors[block_begin], block_count, &distances[0]);
                for (size_t block_index = 0; block_index < block_count; ++block_index) {
                    const float score = static_cast<float>(distances[block_index]);
                    if (score < worst) {
                        insert(best, matches_count, block_begin + block_index, score);
                        worst = best[matches_count - 1].score;
                    }
                }
            }
            count += reported(best, matches_count, threshold);
        }
        return count;
    }
}
