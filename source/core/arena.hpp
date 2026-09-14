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
#ifndef ZEROSLAM_CORE_ARENA_HPP
#define ZEROSLAM_CORE_ARENA_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <mutex>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace core {
    class arena final {
    public:
        constexpr static const size_t chunk_size = 1u << 20;

        class scope final {
        private:
            size_t chunk_index;
            size_t offset;

        public:
            scope();
            ~scope();
            scope(const scope&) = delete;
            scope(scope&&) = delete;
            scope& operator=(const scope&) = delete;
            scope& operator=(scope&&) = delete;
        };

    private:
        struct chunk final {
            unsigned char* data;
            size_t capacity;
        };

        struct thread_state final {
            std::vector<chunk> chunks;
            size_t offset = 0;
            int depth = 0;
        };

        std::mutex mutex;
        std::vector<chunk> free_chunks;
        size_t allocated_chunks = 0;
        size_t allocated_bytes = 0;

    private:
        arena();
        ~arena();
        arena(const arena&) = delete;
        arena(arena&&) = delete;
        arena& operator=(const arena&) = delete;
        arena& operator=(arena&&) = delete;

        static thread_state& state();
        chunk acquire(size_t minimum_capacity);
        void release(std::vector<chunk>& chunks, size_t keep);

    public:
        static arena& instance();

        void* allocate(size_t bytes, size_t alignment);

        void deallocate(void* pointer, size_t bytes);

        size_t chunk_count();
        size_t chunk_bytes();
        static int depth();
    };
}

#endif // ZEROSLAM_CORE_ARENA_HPP
