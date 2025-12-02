/*
Copyright (C) 2025 Geoffrey Daniels. https://gpdaniels.com/

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
#ifndef ASSERT_HPP
#define ASSERT_HPP

#ifndef NDEBUG
#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define ASSERT(ASSERTION, MESSAGE) static_cast<void>((ASSERTION) || (__builtin_trap(), 0))
#else
#define ASSERT(ASSERTION, MESSAGE) static_cast<void>(0)
#endif

#endif // ASSERT_HPP