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
#ifndef ZEROSLAM_CORE_TIMESTAMP_HPP
#define ZEROSLAM_CORE_TIMESTAMP_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#if defined(_WIN32) || defined(_WIN64)
#if defined(_M_X64) || defined(_M_AMD64) || defined(__x86_64__) || defined(__amd64__)
#define _AMD64_
#elif defined(_M_IX86) || defined(__i386__)
#define _X86_
#elif defined(_M_ARM64) || defined(__aarch64__)
#define _ARM64_
#elif defined(_M_ARM) || defined(__arm__)
#define _ARM_
#endif
#include <realtimeapiset.h>
#pragma comment(lib, "mincore.lib")
#else
#include <ctime>
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace core {
    // Get a timestamp representing now, in nanoseconds of a monotonically increasing clock. A failure to read the clock returns zero.
    inline long long int timestamp() {
#if defined(_WIN32) || defined(_WIN64)
        // The interrupt time counts hundreds of nanoseconds since boot, including time spent asleep.
        unsigned long long int hundreds_of_nanoseconds = 0;
        QueryInterruptTimePrecise(&hundreds_of_nanoseconds);
        return static_cast<long long int>(hundreds_of_nanoseconds) * 100ll;
#elif defined(__APPLE__)
        // The raw monotonic clock is not slewed by time adjustments.
        return static_cast<long long int>(clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW));
#elif defined(__linux__)
        // The boot time clock is the monotonic clock that also counts time spent asleep.
        struct timespec time_specification = {};
        if (clock_gettime(CLOCK_BOOTTIME, &time_specification) != 0) {
            return 0;
        }
        return (1000000000ll * static_cast<long long int>(time_specification.tv_sec)) + static_cast<long long int>(time_specification.tv_nsec);
#else
#error "Failed to define the timestamp function for this platform."
#endif
    }
}

#endif // ZEROSLAM_CORE_TIMESTAMP_HPP
