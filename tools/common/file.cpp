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

#if defined(__linux__) && !defined(_FILE_OFFSET_BITS)
#define _FILE_OFFSET_BITS 64
#endif

#include "file.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include <climits>
#include <cstdio>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    long long seek(const int handle, const long long offset, const int whence) {
#if defined(_WIN32)
        return ::_lseeki64(handle, offset, whence);
#else
        return static_cast<long long>(::lseek(handle, static_cast<off_t>(offset), whence));
#endif
    }
}

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

#if defined(_WIN32)
        constexpr static const int flag_access_read_only = _O_RDONLY | _O_BINARY;
        constexpr static const int flag_access_write_only = _O_WRONLY | _O_BINARY;
        constexpr static const int flag_access_read_and_write = _O_RDWR | _O_BINARY;
        constexpr static const int flag_creation_open_only = 0;
        constexpr static const int flag_creation_create_only = _O_CREAT | _O_EXCL;
        constexpr static const int flag_creation_create_or_open = _O_CREAT;
        constexpr static const int flag_cursor_start = 0;
        constexpr static const int flag_cursor_start_truncate = _O_TRUNC;
        constexpr static const int flag_cursor_end = _O_APPEND;
        constexpr static const int flag_cursor_end_truncate = _O_TRUNC | _O_APPEND;
        constexpr static const int permissions = _S_IREAD | _S_IWRITE;
#else
        constexpr static const int flag_access_read_only = O_RDONLY;
        constexpr static const int flag_access_write_only = O_WRONLY;
        constexpr static const int flag_access_read_and_write = O_RDWR;
        constexpr static const int flag_creation_open_only = 0;
        constexpr static const int flag_creation_create_only = O_CREAT | O_EXCL;
        constexpr static const int flag_creation_create_or_open = O_CREAT;
        constexpr static const int flag_cursor_start = 0;
        constexpr static const int flag_cursor_start_truncate = O_TRUNC;
        constexpr static const int flag_cursor_end = O_APPEND;
        constexpr static const int flag_cursor_end_truncate = O_TRUNC | O_APPEND;
        constexpr static const int permissions = 0666;
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

#if defined(_WIN32)
        this->handle = ::_open(path, mode_flags, permissions);
#else
        this->handle = ::open(path, mode_flags, permissions);
#endif

        // Validate that the opened handle refers to a real file, and not a directory, pipe, or other special file.
        // Note: Write modes cannot open directories or pipes, so only read mode needs validation.
        // Note: On windows the open call above already fails for directories, so this check passes trivially.
        if (this->is_open() && (access_mode == access_type::read_only)) {
            char probe = 0;
#if defined(_WIN32)
            const long long probe_length = ::_read(this->handle, &probe, 1);
#else
            const long long probe_length = static_cast<long long>(::read(this->handle, &probe, 1));
#endif
            const long long restored_position = (probe_length > 0) ? seek(this->handle, 0, SEEK_SET) : 0;
            if ((probe_length < 0) || (restored_position != 0)) {
                this->close();
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
        const long long position_current = seek(this->handle, 0, SEEK_CUR);
        if (position_current < 0) {
            return false;
        }
        const long long position_end = seek(this->handle, 0, SEEK_END);
        if (position_end < 0) {
            return false;
        }
        const long long position_restored = seek(this->handle, position_current, SEEK_SET);
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

        const long long position_current = seek(this->handle, 0, SEEK_CUR);
        if (position_current < 0) {
            return false;
        }
        const long long position_end = seek(this->handle, 0, SEEK_END);
        if (position_end < 0) {
            return false;
        }
        const long long position_restored = seek(this->handle, position_current, SEEK_SET);
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

        const long long position_current = seek(this->handle, 0, SEEK_CUR);
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

        int direction_flags = SEEK_SET;
        switch (relative_to) {
            case position_type::start:
                direction_flags = SEEK_SET;
                break;
            case position_type::current:
                direction_flags = SEEK_CUR;
                break;
            case position_type::end:
                direction_flags = SEEK_END;
                break;
        }

        const long long position_current = seek(this->handle, static_cast<long long>(position), direction_flags);
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

        const size_type request = (length > static_cast<size_type>(INT_MAX)) ? static_cast<size_type>(INT_MAX) : length;
#if defined(_WIN32)
        const long long read_length = ::_read(this->handle, buffer, static_cast<unsigned int>(request));
#else
        const long long read_length = static_cast<long long>(::read(this->handle, buffer, request));
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

        const size_type request = (length > static_cast<size_type>(INT_MAX)) ? static_cast<size_type>(INT_MAX) : length;
#if defined(_WIN32)
        const long long write_length = ::_write(this->handle, buffer, static_cast<unsigned int>(request));
#else
        const long long write_length = static_cast<long long>(::write(this->handle, buffer, request));
#endif
        if (write_length < 0) {
            return false;
        }

        length = static_cast<size_type>(write_length);
        return true;
    }
}
