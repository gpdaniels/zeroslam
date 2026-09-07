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

namespace match::distance {
    unsigned int distance_cpu(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);

    namespace {
        unsigned int distance_64(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs) {
            constexpr static const unsigned long long int Ox01 = 0x0101010101010101ull;
            constexpr static const unsigned long long int Ox0F = 0x0F0F0F0F0F0F0F0Full;
            constexpr static const unsigned long long int Ox33 = 0x3333333333333333ull;
            constexpr static const unsigned long long int Ox55 = 0x5555555555555555ull;
            unsigned long long int internal_lhs;
            unsigned long long int internal_rhs;
            unsigned char* __restrict const data_internal_lhs = reinterpret_cast<unsigned char*>(&internal_lhs);
            unsigned char* __restrict const data_internal_rhs = reinterpret_cast<unsigned char*>(&internal_rhs);
            for (decltype(sizeof(unsigned long long int)) i = 0; i < sizeof(unsigned long long int); ++i) {
                data_internal_lhs[i] = data_lhs[i];
                data_internal_rhs[i] = data_rhs[i];
            }
            unsigned long long int difference = internal_lhs ^ internal_rhs;
            difference = difference - ((difference >> 1u) & Ox55);
            difference = (difference & Ox33) + ((difference >> 2u) & Ox33);
            return static_cast<unsigned int>(static_cast<unsigned long long int>(((difference + (difference >> 4u)) & Ox0F) * Ox01) >> 56u);
        }
    }

    unsigned int distance_cpu(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs) {
        return (distance_64(&data_lhs[0], &data_rhs[0]) + distance_64(&data_lhs[8], &data_rhs[8])) + (distance_64(&data_lhs[16], &data_rhs[16]) + distance_64(&data_lhs[24], &data_rhs[24]));
    }
}
