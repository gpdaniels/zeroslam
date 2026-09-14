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
#ifndef ZEROSLAM_CORE_ARENA_ALLOCATOR_HPP
#define ZEROSLAM_CORE_ARENA_ALLOCATOR_HPP

#include "core/arena.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace core {
    template <typename type>
    class arena_allocator {
    public:
        using value_type = type;

    public:
        arena_allocator() = default;

        template <typename other_type>
        arena_allocator(const arena_allocator<other_type>&) {
        }

        type* allocate(const size_t count) {
            return static_cast<type*>(arena::instance().allocate(count * sizeof(type), alignof(type)));
        }

        void deallocate(type* const pointer, const size_t count) {
            arena::instance().deallocate(pointer, count * sizeof(type));
        }

        template <typename other_type>
        bool operator==(const arena_allocator<other_type>&) const {
            return true;
        }

        template <typename other_type>
        bool operator!=(const arena_allocator<other_type>&) const {
            return false;
        }
    };
}

#endif // ZEROSLAM_CORE_ARENA_ALLOCATOR_HPP
