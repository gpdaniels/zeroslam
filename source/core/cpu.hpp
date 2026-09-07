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
#ifndef ZEROSLAM_CORE_CPU_HPP
#define ZEROSLAM_CORE_CPU_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstddef>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace core {
    class cpu final {
    private:
        bool mmx = false;
        bool fma = false;
        bool sse = false;
        bool sse2 = false;
        bool sse3 = false;
        bool ssse3 = false;
        bool sse4_1 = false;
        bool sse4_2 = false;
        bool popcnt = false;
        bool avx = false;
        bool avx2 = false;
        bool avx512_foundation = false;
        bool bmi = false;
        bool bmi2 = false;
        bool neon = false;

    private:
        cpu();

        ~cpu() = default;

        cpu(const cpu&) = delete;
        cpu(cpu&&) = delete;
        cpu& operator=(const cpu&) = delete;
        cpu& operator=(cpu&&) = delete;

        static const cpu& instance();

    public:
        static bool has_mmx();
        static bool has_fma();
        static bool has_sse();
        static bool has_sse2();
        static bool has_sse3();
        static bool has_ssse3();
        static bool has_sse4_1();
        static bool has_sse4_2();
        static bool has_popcnt();
        static bool has_avx();
        static bool has_avx2();
        static bool has_avx512_foundation();
        static bool has_bmi();
        static bool has_bmi2();
        static bool has_neon();

    public:
        static bool get_manufacturer_id(char* buffer, size_t& length);
        static bool get_brand_string(char* buffer, size_t& length);
    };
}

#endif // ZEROSLAM_CORE_CPU_HPP
