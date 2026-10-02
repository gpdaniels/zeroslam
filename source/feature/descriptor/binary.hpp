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
#ifndef ZEROSLAM_FEATURE_DESCRIPTOR_BINARY_HPP
#define ZEROSLAM_FEATURE_DESCRIPTOR_BINARY_HPP

#include "core/assert.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace feature::descriptor {
    template <size_t size_bits>
    class binary final {
    private:
        static_assert(size_bits % 8 == 0, "Binary descriptor size must be a round number of bytes");

    public:
        constexpr static const size_t size_bytes = size_bits / 8;

    public:
        alignas(size_bytes >= 8 ? 8 : size_bytes >= 4 ? 4
                                  : size_bytes >= 2   ? 2
                                                      : 1) unsigned char data[size_bytes];

    public:
        const unsigned char& operator[](size_t index) const {
            ASSERT(index < (size_bytes), "Index out of bounds.");
            return this->data[index];
        }

        unsigned char& operator[](size_t index) {
            ASSERT(index < (size_bytes), "Index out of bounds.");
            return this->data[index];
        }

        // A copy of a narrower descriptor in the first bytes with the rest zero, so the Hamming distance between two copies equals the distance between the originals.
        template <size_t narrow_bits>
        static binary widened(const binary<narrow_bits>& narrow) {
            static_assert(narrow_bits <= size_bits, "A descriptor only widens.");
            binary wide;
            for (size_t byte = 0; byte < size_bytes; ++byte) {
                wide.data[byte] = (byte < binary<narrow_bits>::size_bytes) ? narrow.data[byte] : static_cast<unsigned char>(0);
            }
            return wide;
        }
    };

    // Every descriptor the pipeline stores, indexes and matches has this width. ORB, TEBLID-256 and bSIFT-256 fill the first
    // 256 bits and leave the rest zero, which adds nothing to a Hamming distance between two of them.
    constexpr static const size_t stored_bits = 512;
    using stored = binary<stored_bits>;
}

#endif // ZEROSLAM_FEATURE_DESCRIPTOR_BINARY_HPP
