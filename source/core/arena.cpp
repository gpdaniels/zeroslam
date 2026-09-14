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

#include "core/arena.hpp"

#include "core/assert.hpp"

namespace core {
    arena::scope::scope()
        : chunk_index(arena::state().chunks.size())
        , offset(arena::state().offset) {
        ++arena::state().depth;
    }

    arena::scope::~scope() {
        thread_state& current = arena::state();
        --current.depth;
        arena::instance().release(current.chunks, this->chunk_index);
        current.offset = this->offset;
    }

    arena::arena()
        : mutex()
        , free_chunks()
        , allocated_chunks(0)
        , allocated_bytes(0) {
    }

    arena::~arena() {
        for (const chunk& block : this->free_chunks) {
            delete[] block.data;
        }
    }

    arena::thread_state& arena::state() {
        static thread_local thread_state current;
        return current;
    }

    arena::chunk arena::acquire(const size_t minimum_capacity) {
        std::lock_guard<std::mutex> lock(this->mutex);
        for (size_t i = 0; i < this->free_chunks.size(); ++i) {
            if (this->free_chunks[i].capacity >= minimum_capacity) {
                const chunk found = this->free_chunks[i];
                this->free_chunks[i] = this->free_chunks.back();
                this->free_chunks.pop_back();
                return found;
            }
        }
        const size_t capacity = (minimum_capacity > arena::chunk_size) ? minimum_capacity : arena::chunk_size;
        chunk created;
        created.data = new unsigned char[capacity];
        created.capacity = capacity;
        ++this->allocated_chunks;
        this->allocated_bytes += capacity;
        return created;
    }

    void arena::release(std::vector<chunk>& chunks, const size_t keep) {
        if (chunks.size() <= keep) {
            return;
        }
        std::lock_guard<std::mutex> lock(this->mutex);
        for (size_t i = keep; i < chunks.size(); ++i) {
            this->free_chunks.push_back(chunks[i]);
        }
        chunks.resize(keep);
    }

    arena& arena::instance() {
        static arena singleton;
        return singleton;
    }

    void* arena::allocate(const size_t bytes, const size_t alignment) {
        ASSERT((alignment != 0) && ((alignment & (alignment - 1)) == 0), "The alignment must be a power of two.");
        thread_state& current = arena::state();
        ASSERT(current.depth > 0, "Arena allocation outside of an arena::scope is never released.");
        const size_t request = (bytes == 0) ? 1 : bytes;
        const auto aligned_offset = [alignment](const chunk& block, const size_t offset) {
            const size_t address = reinterpret_cast<size_t>(block.data) + offset;
            return (((address + alignment - 1) & ~(alignment - 1)) - reinterpret_cast<size_t>(block.data));
        };
        if (!current.chunks.empty()) {
            const chunk& last = current.chunks.back();
            const size_t aligned = aligned_offset(last, current.offset);
            if ((aligned <= last.capacity) && (request <= last.capacity - aligned)) {
                current.offset = aligned + request;
                return last.data + aligned;
            }
        }
        current.chunks.push_back(this->acquire(request + alignment));
        const chunk& taken = current.chunks.back();
        const size_t aligned = aligned_offset(taken, 0);
        current.offset = aligned + request;
        return taken.data + aligned;
    }

    void arena::deallocate(void* const pointer, const size_t bytes) {
        static_cast<void>(pointer);
        static_cast<void>(bytes);
    }

    size_t arena::chunk_count() {
        std::lock_guard<std::mutex> lock(this->mutex);
        return this->allocated_chunks;
    }

    size_t arena::chunk_bytes() {
        std::lock_guard<std::mutex> lock(this->mutex);
        return this->allocated_bytes;
    }

    int arena::depth() {
        return arena::state().depth;
    }
}
