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
#ifndef ZEROSLAM_TOOLS_COMMON_TRIPLE_BUFFER_HPP
#define ZEROSLAM_TOOLS_COMMON_TRIPLE_BUFFER_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <atomic>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace gtl {
    template <typename buffer_type>
    class triple_buffer final {
    public:
        using type = buffer_type;

    private:
        struct index_type final {
            unsigned char read    : 2;
            unsigned char swap    : 2;
            unsigned char write   : 2;
            unsigned char updated : 2;
        };

        static_assert(sizeof(index_type) == 1, "The size of the index_type is assumed to be one byte.");

    private:
        std::atomic<index_type> indexes;

        buffer_type buffers[3];

    public:
        ~triple_buffer() = default;

        triple_buffer() {
            this->indexes.store(index_type{ 0, 1, 2, 0 });
        }

        triple_buffer(const triple_buffer&) = delete;

        triple_buffer(triple_buffer&&) = delete;

        triple_buffer& operator=(const triple_buffer&) = delete;

        triple_buffer& operator=(triple_buffer&&) = delete;

    public:
        bool update_read() {
            index_type current_indexes = this->indexes.load();
            if (current_indexes.updated == 0) {
                return false;
            }
            while (!this->indexes.compare_exchange_weak(current_indexes, index_type{ current_indexes.swap, current_indexes.read, current_indexes.write, 0 })) {
            }
            return true;
        }

        type& get_read() {
            return this->buffers[this->indexes.load().read];
        }

        const type& get_read() const {
            return this->buffers[this->indexes.load().read];
        }

        type& get_write() {
            return this->buffers[this->indexes.load().write];
        }

        void update_write() {
            index_type current_indexes = this->indexes.load();
            while (!this->indexes.compare_exchange_weak(current_indexes, index_type{ current_indexes.read, current_indexes.write, current_indexes.swap, 1 })) {
            }
        }
    };
}

#endif // ZEROSLAM_TOOLS_COMMON_TRIPLE_BUFFER_HPP
