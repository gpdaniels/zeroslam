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
    // The simd tier is chosen once, on first use, and every tier gives the same distances.
    class hamming final {
    public:
        static unsigned int distance(
            const feature::descriptor::binary<256>& lhs,
            const feature::descriptor::binary<256>& rhs
        );

        // The distance from the query to each of the descriptors, results[i] for descriptors[i].
        static void distances(
            const feature::descriptor::binary<256>& query,
            const feature::descriptor::binary<256>* __restrict const descriptors,
            const size_t descriptors_size,
            unsigned int* __restrict const results
        );

        // The distance from the query to each indexed descriptor, results[i] for descriptors[indices[i]].
        static void distances(
            const feature::descriptor::binary<256>& query,
            const feature::descriptor::binary<256>* __restrict const descriptors,
            const size_t* __restrict const indices,
            const size_t indices_size,
            unsigned int* __restrict const results
        );

        static unsigned int distance(
            const feature::descriptor::binary<512>& lhs,
            const feature::descriptor::binary<512>& rhs
        );

        static void distances(
            const feature::descriptor::binary<512>& query,
            const feature::descriptor::binary<512>* __restrict const descriptors,
            const size_t descriptors_size,
            unsigned int* __restrict const results
        );

        static void distances(
            const feature::descriptor::binary<512>& query,
            const feature::descriptor::binary<512>* __restrict const descriptors,
            const size_t* __restrict const indices,
            const size_t indices_size,
            unsigned int* __restrict const results
        );
    };
}

#endif // ZEROSLAM_MATCH_DISTANCE_HAMMING_HPP
