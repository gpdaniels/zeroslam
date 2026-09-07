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
#ifndef ZEROSLAM_CORE_ASSERT_HPP
#define ZEROSLAM_CORE_ASSERT_HPP

#ifndef NDEBUG
#include "core/logger.hpp"
#endif

namespace core {
#ifndef NDEBUG
#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define ASSERT(ASSERTION, MESSAGE) static_cast<void>((ASSERTION) || (core::logger::log(core::logger::level::error, "ASSERT[%s:%d]: '%s' failed: %s", __FILE__, __LINE__, #ASSERTION, MESSAGE), __builtin_trap(), 0))
#else
#define ASSERT(ASSERTION, MESSAGE) static_cast<void>(0)
#endif
}

#endif // ZEROSLAM_CORE_ASSERT_HPP
