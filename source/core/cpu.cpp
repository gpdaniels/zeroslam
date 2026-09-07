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

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define ZEROSLAM_CORE_CPU_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#else
#define ZEROSLAM_CORE_CPU_X86 0
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace core {
    namespace {
        struct cpuid_registers final {
            unsigned int eax = 0;
            unsigned int ebx = 0;
            unsigned int ecx = 0;
            unsigned int edx = 0;
        };

#if ZEROSLAM_CORE_CPU_X86
        cpuid_registers query_cpuid(const unsigned int leaf_id) {
            cpuid_registers registers;
#if ZEROSLAM_CORE_CPU_X86
#if defined(_MSC_VER)
            int values[4] = { 0, 0, 0, 0 };
            __cpuidex(values, static_cast<int>(leaf_id), 0);
            registers.eax = static_cast<unsigned int>(values[0]);
            registers.ebx = static_cast<unsigned int>(values[1]);
            registers.ecx = static_cast<unsigned int>(values[2]);
            registers.edx = static_cast<unsigned int>(values[3]);
#else
            __cpuid_count(leaf_id, 0u, registers.eax, registers.ebx, registers.ecx, registers.edx);
#endif
#else
            static_cast<void>(leaf_id);
#endif
            return registers;
        }

        unsigned long long int query_extended_control_register() {
#if ZEROSLAM_CORE_CPU_X86
#if defined(_MSC_VER)
            return _xgetbv(0);
#else
            unsigned int eax = 0;
            unsigned int edx = 0;
            __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
            return (static_cast<unsigned long long int>(edx) << 32u) | eax;
#endif
#else
            return 0;
#endif
        }
#endif

        void copy_bytes(void* const destination, const void* const source, const size_t count) {
            for (size_t index = 0; index < count; ++index) {
                static_cast<char*>(destination)[index] = static_cast<const char*>(source)[index];
            }
        }

#if ZEROSLAM_CORE_CPU_X86
        bool bit(const unsigned int value, const unsigned int index) {
            return (value & (1u << index)) != 0u;
        }
#endif

        bool copy_string(const char* const text, const size_t text_length, char* const buffer, size_t& length) {
            if ((buffer == nullptr) || (length < text_length + 1)) {
                length = text_length + 1;
                return false;
            }
            copy_bytes(buffer, text, text_length);
            buffer[text_length] = 0;
            length = text_length;
            return true;
        }
    }

    cpu::cpu() {
#if ZEROSLAM_CORE_CPU_X86
        const unsigned int max_leaf_id = query_cpuid(0x00000000u).eax;
        if (max_leaf_id >= 0x00000001u) {
            const cpuid_registers features = query_cpuid(0x00000001u);
            this->mmx = bit(features.edx, 23);
            this->sse = bit(features.edx, 25);
            this->sse2 = bit(features.edx, 26);
            this->sse3 = bit(features.ecx, 0);
            this->ssse3 = bit(features.ecx, 9);
            this->sse4_1 = bit(features.ecx, 19);
            this->sse4_2 = bit(features.ecx, 20);
            this->popcnt = bit(features.ecx, 23);
            const bool operating_system_saves_state = bit(features.ecx, 27);
            const unsigned long long int extended_control = operating_system_saves_state ? query_extended_control_register() : 0;
            const bool ymm_enabled = ((extended_control & 0x06u) == 0x06u);
            const bool zmm_enabled = ((extended_control & 0xE6u) == 0xE6u);
            this->avx = bit(features.ecx, 28) && ymm_enabled;
            this->fma = bit(features.ecx, 12) && ymm_enabled;
            if (max_leaf_id >= 0x00000007u) {
                const cpuid_registers extended_features = query_cpuid(0x00000007u);
                this->bmi = bit(extended_features.ebx, 3);
                this->avx2 = bit(extended_features.ebx, 5) && this->avx;
                this->bmi2 = bit(extended_features.ebx, 8);
                this->avx512_foundation = bit(extended_features.ebx, 16) && zmm_enabled;
            }
        }
#elif defined(__aarch64__) || defined(_M_ARM64)
        this->neon = true;
#endif
    }

    const cpu& cpu::instance() {
        static const cpu singleton;
        return singleton;
    }

    bool cpu::has_mmx() {
        return cpu::instance().mmx;
    }

    bool cpu::has_fma() {
        return cpu::instance().fma;
    }

    bool cpu::has_sse() {
        return cpu::instance().sse;
    }

    bool cpu::has_sse2() {
        return cpu::instance().sse2;
    }

    bool cpu::has_sse3() {
        return cpu::instance().sse3;
    }

    bool cpu::has_ssse3() {
        return cpu::instance().ssse3;
    }

    bool cpu::has_sse4_1() {
        return cpu::instance().sse4_1;
    }

    bool cpu::has_sse4_2() {
        return cpu::instance().sse4_2;
    }

    bool cpu::has_popcnt() {
        return cpu::instance().popcnt;
    }

    bool cpu::has_avx() {
        return cpu::instance().avx;
    }

    bool cpu::has_avx2() {
        return cpu::instance().avx2;
    }

    bool cpu::has_avx512_foundation() {
        return cpu::instance().avx512_foundation;
    }

    bool cpu::has_bmi() {
        return cpu::instance().bmi;
    }

    bool cpu::has_bmi2() {
        return cpu::instance().bmi2;
    }

    bool cpu::has_neon() {
        return cpu::instance().neon;
    }

    bool cpu::get_manufacturer_id(char* buffer, size_t& length) {
        char text[12] = {};
        size_t text_length = 0;
#if ZEROSLAM_CORE_CPU_X86
        const cpuid_registers registers = query_cpuid(0x00000000u);
        copy_bytes(&text[0], &registers.ebx, 4);
        copy_bytes(&text[4], &registers.edx, 4);
        copy_bytes(&text[8], &registers.ecx, 4);
        text_length = 12;
#endif
        return copy_string(text, text_length, buffer, length);
    }

    bool cpu::get_brand_string(char* buffer, size_t& length) {
        char text[48] = {};
        size_t text_length = 0;
#if ZEROSLAM_CORE_CPU_X86
        if ((query_cpuid(0x80000000u).eax & 0x7FFFFFFFu) >= 0x00000004u) {
            for (unsigned int leaf = 0; leaf < 3; ++leaf) {
                const cpuid_registers registers = query_cpuid(0x80000002u + leaf);
                copy_bytes(&text[(leaf * 16) + 0], &registers.eax, 4);
                copy_bytes(&text[(leaf * 16) + 4], &registers.ebx, 4);
                copy_bytes(&text[(leaf * 16) + 8], &registers.ecx, 4);
                copy_bytes(&text[(leaf * 16) + 12], &registers.edx, 4);
            }
            text_length = 48;
            while ((text_length > 0) && ((text[text_length - 1] == 0) || (text[text_length - 1] == ' '))) {
                --text_length;
            }
        }
#endif
        size_t text_start = 0;
        while ((text_start < text_length) && (text[text_start] == ' ')) {
            ++text_start;
        }
        return copy_string(&text[text_start], text_length - text_start, buffer, length);
    }
}

#undef ZEROSLAM_CORE_CPU_X86
