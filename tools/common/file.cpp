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

#include "file.hpp"

#if defined(linux) || defined(__linux) || defined(__linux__)
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#if defined(_WIN32)

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <io.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif

#if defined(__APPLE__)
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace gtl {

    bool file::is_open() const {
        return (this->handle >= 0);
    }

    bool file::open(
        const char* const __restrict path,
        access_type access_mode,
        creation_type creation_mode,
        cursor_type cursor_mode
    ) {
        if (this->is_open()) {
            return false;
        }

#if (defined(linux) || defined(__linux) || defined(__linux__))
        constexpr static const int flag_access_read_only = 0;      // O_RDONLY;
        constexpr static const int flag_access_write_only = 1;     // O_WRONLY;
        constexpr static const int flag_access_read_and_write = 2; // O_RDWR;

        constexpr static const int flag_creation_open_only = 0;          // 0;
        constexpr static const int flag_creation_create_only = 64 | 128; // O_CREAT | O_EXCL;
        constexpr static const int flag_creation_create_or_open = 64;    // O_CREAT;

        constexpr static const int flag_cursor_start = 0;                 // 0;
        constexpr static const int flag_cursor_start_truncate = 512;      // O_TRUNC;
        constexpr static const int flag_cursor_end = 1024;                // O_APPEND;
        constexpr static const int flag_cursor_end_truncate = 512 | 1024; // O_TRUNC | O_APPEND;
#endif

#if defined(_WIN32)
        constexpr static const int flag_access_read_only = 0;      // O_RDONLY;
        constexpr static const int flag_access_write_only = 1;     // O_WRONLY;
        constexpr static const int flag_access_read_and_write = 2; // O_RDWR;

        constexpr static const int flag_creation_open_only = 0;            // 0;
        constexpr static const int flag_creation_create_only = 256 | 1024; // O_CREAT | O_EXCL;
        constexpr static const int flag_creation_create_or_open = 256;     // O_CREAT;

        constexpr static const int flag_cursor_start = 0;              // 0;
        constexpr static const int flag_cursor_start_truncate = 512;   // O_TRUNC;
        constexpr static const int flag_cursor_end = 8;                // O_APPEND;
        constexpr static const int flag_cursor_end_truncate = 512 | 8; // O_TRUNC | O_APPEND;
#endif

#if defined(__APPLE__)
        constexpr static const int flag_access_read_only = 0;      // O_RDONLY;
        constexpr static const int flag_access_write_only = 1;     // O_WRONLY;
        constexpr static const int flag_access_read_and_write = 2; // O_RDWR;

        constexpr static const int flag_creation_open_only = 0;            // 0;
        constexpr static const int flag_creation_create_only = 512 | 2048; // O_CREAT | O_EXCL;
        constexpr static const int flag_creation_create_or_open = 512;     // O_CREAT;

        constexpr static const int flag_cursor_start = 0;               // 0;
        constexpr static const int flag_cursor_start_truncate = 1024;   // O_TRUNC;
        constexpr static const int flag_cursor_end = 8;                 // O_APPEND;
        constexpr static const int flag_cursor_end_truncate = 1024 | 8; // O_TRUNC | O_APPEND;
#endif

        int mode_flags = 0;
        switch (access_mode) {
            case access_type::read_only:
                mode_flags |= flag_access_read_only;
                break;
            case access_type::write_only:
                mode_flags |= flag_access_write_only;
                break;
            case access_type::read_and_write:
                mode_flags |= flag_access_read_and_write;
                break;
        }
        switch (creation_mode) {
            case creation_type::open_only:
                mode_flags |= flag_creation_open_only;
                break;
            case creation_type::create_only:
                mode_flags |= flag_creation_create_only;
                break;
            case creation_type::create_or_open:
                mode_flags |= flag_creation_create_or_open;
                break;
        }
        switch (cursor_mode) {
            case cursor_type::start_of_file:
                mode_flags |= flag_cursor_start;
                break;
            case cursor_type::start_of_truncated:
                mode_flags |= flag_cursor_start_truncate;
                break;
            case cursor_type::end_of_file:
                mode_flags |= flag_cursor_end;
                break;
            case cursor_type::end_of_truncated:
                mode_flags |= flag_cursor_end_truncate;
                break;
        }

        this->handle = ::open(path, mode_flags, 0666);

        // Validate that the opened handle refers to a real file, and not a directory, pipe, or other special file.
        // Note: Write modes cannot open directories or pipes, so only read mode needs validation.
        // Note: On windows the open call above already fails for directories, so this check passes trivially.
        if (this->is_open() && (access_mode == access_type::read_only)) {
            char probe = 0;
            const offset_type probe_length = ::read(this->handle, &probe, 1);
            const offset_type restored_position = (probe_length > 0) ? static_cast<offset_type>(::lseek(this->handle, 0, 0)) : 0;
            if ((probe_length < 0) || (restored_position != 0)) {
                ::close(this->handle);
                this->handle = -1;
            }
        }

        return this->is_open();
    }

    bool file::close() {
        if (!this->is_open()) {
            return false;
        }

        if (::close(this->handle) != 0) {
            this->handle = -1;
            return false;
        }

        this->handle = -1;
        return true;
    }

    int file::get_handle() const {
        return this->handle;
    }

    bool file::is_eof(bool& eof) const {
        eof = true;

        if (!this->is_open()) {
            return false;
        }
        const offset_type position_current = static_cast<offset_type>(::lseek(this->handle, 0, 1));
        if (position_current < 0) {
            return false;
        }
        const offset_type position_end = static_cast<offset_type>(::lseek(this->handle, 0, 2));
        if (position_end < 0) {
            return false;
        }
        const offset_type position_restored = static_cast<offset_type>(::lseek(this->handle, static_cast<long>(position_current), 0));
        if (position_restored < 0) {
            return false;
        }

        eof = (position_current == position_end);
        return true;
    }

    bool file::get_size(size_type& size) const {
        size = 0;

        if (!this->is_open()) {
            return false;
        }

        const offset_type position_current = static_cast<offset_type>(::lseek(this->handle, 0, 1));
        if (position_current < 0) {
            return false;
        }
        const offset_type position_end = static_cast<offset_type>(::lseek(this->handle, 0, 2));
        if (position_end < 0) {
            return false;
        }
        const offset_type position_restored = static_cast<offset_type>(::lseek(this->handle, static_cast<long>(position_current), 0));
        if (position_restored < 0) {
            return false;
        }

        size = static_cast<size_type>(position_end);
        return true;
    }

    bool file::get_cursor_position(size_type& position) const {
        position = 0;

        if (!this->is_open()) {
            return false;
        }

        offset_type position_current = static_cast<offset_type>(::lseek(this->handle, 0, 1));
        if (position_current < 0) {
            return false;
        }

        position = static_cast<size_type>(position_current);
        return true;
    }

    bool file::set_cursor_position(offset_type position, position_type relative_to) const {
        if (!this->is_open()) {
            return false;
        }

        int direction_flags = 0;
        switch (relative_to) {
            case position_type::start:
                direction_flags = 0;
                break;
            case position_type::current:
                direction_flags = 1;
                break;
            case position_type::end:
                direction_flags = 2;
                break;
        }

        const offset_type position_current = static_cast<offset_type>(::lseek(this->handle, static_cast<long>(position), direction_flags));
        if (position_current < 0) {
            return false;
        }

        return true;
    }

    bool file::read(char* const __restrict buffer, size_type& length) const {
        if (!this->is_open()) {
            length = 0;
            return false;
        }

        if (length == 0) {
            return true;
        }

#if defined(_WIN32)
        const offset_type read_length = ::read(this->handle, buffer, static_cast<unsigned int>(length));
#else
        const offset_type read_length = ::read(this->handle, buffer, length);
#endif
        if (read_length < 0) {
            return false;
        }

        length = static_cast<size_type>(read_length);
        return true;
    }

    bool file::write(const char* const __restrict buffer, size_type& length) const {
        if (!this->is_open()) {
            length = 0;
            return false;
        }

        if (length == 0) {
            return true;
        }

#if defined(_WIN32)
        const offset_type write_length = ::write(this->handle, buffer, static_cast<unsigned int>(length));
#else
        const offset_type write_length = ::write(this->handle, buffer, length);
#endif
        if (write_length < 0) {
            return false;
        }

        length = static_cast<size_type>(write_length);
        return true;
    }
}
