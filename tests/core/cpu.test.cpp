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

#include "core/cpu.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        REQUIRE(core::cpu::has_avx() == core::cpu::has_avx());
        REQUIRE(core::cpu::has_avx2() == core::cpu::has_avx2());
        REQUIRE(core::cpu::has_neon() == core::cpu::has_neon());
    }

    {
        REQUIRE(!core::cpu::has_avx512_foundation() || core::cpu::has_avx2());
        REQUIRE(!core::cpu::has_avx2() || core::cpu::has_avx());
        REQUIRE(!core::cpu::has_fma() || core::cpu::has_avx());
        REQUIRE(!core::cpu::has_avx() || core::cpu::has_sse4_2());
        REQUIRE(!core::cpu::has_sse4_2() || core::cpu::has_sse4_1());
        REQUIRE(!core::cpu::has_sse4_1() || core::cpu::has_ssse3());
        REQUIRE(!core::cpu::has_ssse3() || core::cpu::has_sse3());
        REQUIRE(!core::cpu::has_sse3() || core::cpu::has_sse2());
        REQUIRE(!core::cpu::has_sse2() || core::cpu::has_sse());
        REQUIRE(!core::cpu::has_sse() || core::cpu::has_mmx());
    }

    {
        size_t length = 0;
        REQUIRE(!core::cpu::get_manufacturer_id(nullptr, length));
        REQUIRE(length >= 1);
        char short_buffer[2] = {};
        size_t short_length = sizeof(short_buffer);
        REQUIRE(core::cpu::get_manufacturer_id(short_buffer, short_length) == (length <= sizeof(short_buffer)));
        REQUIRE(short_length == (length <= sizeof(short_buffer) ? length - 1 : length));
        char manufacturer_id[64] = {};
        size_t manufacturer_id_length = sizeof(manufacturer_id);
        REQUIRE(core::cpu::get_manufacturer_id(manufacturer_id, manufacturer_id_length));
        REQUIRE(manufacturer_id_length == length - 1);
        REQUIRE(std::strlen(manufacturer_id) == manufacturer_id_length);
        char exact[64] = {};
        size_t exact_length = length;
        REQUIRE(core::cpu::get_manufacturer_id(exact, exact_length));
        REQUIRE(std::strcmp(exact, manufacturer_id) == 0);
    }

    {
        size_t length = 0;
        REQUIRE(!core::cpu::get_brand_string(nullptr, length));
        REQUIRE(length >= 1);
        REQUIRE(length <= 49);
        char brand_string[64] = {};
        size_t brand_string_length = sizeof(brand_string);
        REQUIRE(core::cpu::get_brand_string(brand_string, brand_string_length));
        REQUIRE(brand_string_length == length - 1);
        REQUIRE(std::strlen(brand_string) == brand_string_length);
        REQUIRE((brand_string_length == 0) || ((brand_string[0] != ' ') && (brand_string[brand_string_length - 1] != ' ')));
        size_t too_short = length - 1;
        REQUIRE(!core::cpu::get_brand_string(brand_string, too_short));
        REQUIRE(too_short == length);
    }

    {
        char manufacturer_id[13] = {};
        size_t manufacturer_id_length = sizeof(manufacturer_id);
        REQUIRE(core::cpu::get_manufacturer_id(manufacturer_id, manufacturer_id_length));
        char brand_string[49] = {};
        size_t brand_string_length = sizeof(brand_string);
        REQUIRE(core::cpu::get_brand_string(brand_string, brand_string_length));
        std::printf("CPU manufacturer id: %s\n", manufacturer_id);
        std::printf("CPU brand string: %s\n", brand_string);
        std::printf("CPU mmx: %s\n", core::cpu::has_mmx() ? "yes" : "no");
        std::printf("CPU fma: %s\n", core::cpu::has_fma() ? "yes" : "no");
        std::printf("CPU sse: %s\n", core::cpu::has_sse() ? "yes" : "no");
        std::printf("CPU sse2: %s\n", core::cpu::has_sse2() ? "yes" : "no");
        std::printf("CPU sse3: %s\n", core::cpu::has_sse3() ? "yes" : "no");
        std::printf("CPU ssse3: %s\n", core::cpu::has_ssse3() ? "yes" : "no");
        std::printf("CPU sse4.1: %s\n", core::cpu::has_sse4_1() ? "yes" : "no");
        std::printf("CPU sse4.2: %s\n", core::cpu::has_sse4_2() ? "yes" : "no");
        std::printf("CPU popcnt: %s\n", core::cpu::has_popcnt() ? "yes" : "no");
        std::printf("CPU avx: %s\n", core::cpu::has_avx() ? "yes" : "no");
        std::printf("CPU avx2: %s\n", core::cpu::has_avx2() ? "yes" : "no");
        std::printf("CPU avx512 foundation: %s\n", core::cpu::has_avx512_foundation() ? "yes" : "no");
        std::printf("CPU bmi: %s\n", core::cpu::has_bmi() ? "yes" : "no");
        std::printf("CPU bmi2: %s\n", core::cpu::has_bmi2() ? "yes" : "no");
        std::printf("CPU neon: %s\n", core::cpu::has_neon() ? "yes" : "no");
    }

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    {
        REQUIRE(core::cpu::has_sse2());
        REQUIRE(!core::cpu::has_neon());
        char manufacturer_id[13] = {};
        size_t manufacturer_id_length = sizeof(manufacturer_id);
        REQUIRE(core::cpu::get_manufacturer_id(manufacturer_id, manufacturer_id_length));
        REQUIRE(manufacturer_id_length == 12);
#if !defined(_MSC_VER)
        REQUIRE(core::cpu::has_sse2() == (__builtin_cpu_supports("sse2") != 0));
        REQUIRE(core::cpu::has_popcnt() == (__builtin_cpu_supports("popcnt") != 0));
        REQUIRE(core::cpu::has_avx() == (__builtin_cpu_supports("avx") != 0));
        REQUIRE(core::cpu::has_avx2() == (__builtin_cpu_supports("avx2") != 0));
        REQUIRE(core::cpu::has_fma() == (__builtin_cpu_supports("fma") != 0));
        REQUIRE(core::cpu::has_bmi2() == (__builtin_cpu_supports("bmi2") != 0));
        REQUIRE(core::cpu::has_avx512_foundation() == (__builtin_cpu_supports("avx512f") != 0));
#endif
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    {
        REQUIRE(core::cpu::has_neon());
        REQUIRE(!core::cpu::has_avx());
        REQUIRE(!core::cpu::has_avx2());
        size_t length = 0;
        REQUIRE(!core::cpu::get_manufacturer_id(nullptr, length));
        REQUIRE(length == 1);
    }
#endif

    return EXIT_SUCCESS;
}
