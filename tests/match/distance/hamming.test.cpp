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

#include "match/distance/hamming.hpp"

#include "core/cpu.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

namespace match::distance {
    unsigned int distance_cpu(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
#if defined(ZEROSLAM_SIMD_AVX)
    unsigned int distance_avx(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
#endif
#if defined(ZEROSLAM_SIMD_AVX2)
    unsigned int distance_avx2(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
#endif
#if defined(ZEROSLAM_SIMD_NEON)
    unsigned int distance_neon(const unsigned char* __restrict const data_lhs, const unsigned char* __restrict const data_rhs);
#endif
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        feature::descriptor::binary<256> zero{};
        feature::descriptor::binary<256> ones;
        for (size_t i = 0; i < 32; ++i) {
            ones.data[i] = 0xFF;
        }
        REQUIRE(match::distance::hamming::distance(zero, zero) == 0);
        REQUIRE(match::distance::hamming::distance(ones, ones) == 0);
        REQUIRE(match::distance::hamming::distance(zero, ones) == 256);
        REQUIRE(match::distance::hamming::distance(ones, zero) == 256);
        feature::descriptor::binary<256> one{};
        one.data[5] = 0x11;
        REQUIRE(match::distance::hamming::distance(zero, one) == 2);
        REQUIRE(match::distance::hamming::distance(one, ones) == 254);
    }

    {
        unsigned long long int lhs[4] = { 0, 0, 0, 1 };
        unsigned long long int rhs[4] = { 0, 0, 0, 0 };
        while ((lhs[0] != 0) || (lhs[1] != 0) || (lhs[2] != 0) || (lhs[3] != 0)) {
            REQUIRE(match::distance::distance_cpu(reinterpret_cast<unsigned char*>(&lhs[0]), reinterpret_cast<unsigned char*>(&rhs[0])) == 1);
            lhs[3] <<= 1;
            for (int i = 2; i >= 0; --i) {
                if ((lhs[0] == 0) && (lhs[1] == 0) && (lhs[2] == 0) && (lhs[3] == 0)) {
                    lhs[i] = 0b0000000000000000000000000000000000000000000000000000000000000001;
                }
                else {
                    lhs[i] <<= 1;
                }
            }
        }
    }

#if defined(ZEROSLAM_SIMD_AVX)
    {
        if (core::cpu::has_avx() && core::cpu::has_popcnt()) {
            std::vector<feature::descriptor::binary<256>> lhs(256);
            std::vector<feature::descriptor::binary<256>> rhs(256);
            for (size_t i = 0; i < 256; ++i) {
                for (size_t j = 0; j < 32; ++j) {
                    lhs[i].data[j] = static_cast<unsigned char>((i * 31 + j * 17) % 256);
                    rhs[i].data[j] = static_cast<unsigned char>((i * 13 + j * 29 + 7) % 256);
                }
            }
            for (size_t i = 0; i < 256; ++i) {
                REQUIRE(match::distance::distance_cpu(lhs[i].data, rhs[i].data) == match::distance::distance_avx(lhs[i].data, rhs[i].data));
            }
            REQUIRE(match::distance::hamming::distance(lhs[0], rhs[0]) == match::distance::distance_cpu(lhs[0].data, rhs[0].data));
        }
    }
#endif

#if defined(ZEROSLAM_SIMD_AVX2)
    {
        if (core::cpu::has_avx2() && core::cpu::has_popcnt()) {
            std::vector<feature::descriptor::binary<256>> lhs(256);
            std::vector<feature::descriptor::binary<256>> rhs(256);
            for (size_t i = 0; i < 256; ++i) {
                for (size_t j = 0; j < 32; ++j) {
                    lhs[i].data[j] = static_cast<unsigned char>((i * 31 + j * 17) % 256);
                    rhs[i].data[j] = static_cast<unsigned char>((i * 13 + j * 29 + 7) % 256);
                }
            }
            for (size_t i = 0; i < 256; ++i) {
                REQUIRE(match::distance::distance_cpu(lhs[i].data, rhs[i].data) == match::distance::distance_avx2(lhs[i].data, rhs[i].data));
            }
            REQUIRE(match::distance::hamming::distance(lhs[0], rhs[0]) == match::distance::distance_cpu(lhs[0].data, rhs[0].data));
        }
    }
#endif

#if defined(ZEROSLAM_SIMD_NEON)
    {
        if (core::cpu::has_neon()) {
            std::vector<feature::descriptor::binary<256>> lhs(256);
            std::vector<feature::descriptor::binary<256>> rhs(256);
            for (size_t i = 0; i < 256; ++i) {
                for (size_t j = 0; j < 32; ++j) {
                    lhs[i].data[j] = static_cast<unsigned char>((i * 31 + j * 17) % 256);
                    rhs[i].data[j] = static_cast<unsigned char>((i * 13 + j * 29 + 7) % 256);
                }
            }
            for (size_t i = 0; i < 256; ++i) {
                REQUIRE(match::distance::distance_cpu(lhs[i].data, rhs[i].data) == match::distance::distance_neon(lhs[i].data, rhs[i].data));
            }
            REQUIRE(match::distance::hamming::distance(lhs[0], rhs[0]) == match::distance::distance_cpu(lhs[0].data, rhs[0].data));
        }
    }
#endif

    return EXIT_SUCCESS;
}
