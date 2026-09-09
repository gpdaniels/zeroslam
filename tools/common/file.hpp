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
#ifndef ZEROSLAM_TOOLS_COMMON_FILE_HPP
#define ZEROSLAM_TOOLS_COMMON_FILE_HPP

// Summary: An RAII file handle that wraps file operation functions. [wip]

namespace {
    using size_t = decltype(sizeof(0));
    using ssize_t = decltype(static_cast<char*>(nullptr) - static_cast<char*>(nullptr));
}

namespace gtl {

    /// @brief A class to hold an RAII file handle and provide member functions to operate on it.
    class file final {
    public:
        /// @brief The size and offset types used by the member functions and the underlying file IO functions.
        using size_type = size_t;
        using offset_type = ssize_t;

    public:
        /// @brief Types of file access: reading, writing, or both.
        enum class access_type {
            /// @brief Open in read only mode with the cursor positioned at the start of the file.
            read_only,

            /// @brief Open in write only mode with the cursor positioned at the start of the file.
            write_only,

            /// @brief Open in both read and write mode with the cursor positioned at the start of the file.
            read_and_write
        };

        /// @brief Types of file creation: don't create, only create, or either.
        enum class creation_type {
            /// @brief Only open the file if it already exists.
            open_only,

            /// @brief Only open the file by creating it.
            create_only,

            /// @brief Open the file either if it exists or by creating it.
            create_or_open
        };

        /// @brief Types of file cursor: start or end, and untouched or truncated.
        enum class cursor_type {
            /// @brief The cursor is positioned at the start of the current file contents.
            start_of_file,

            /// @brief The cursor is positioned at the start of the file after its contents have been truncated.
            start_of_truncated,

            /// @brief The cursor is positioned at the end of the current file contents.
            /// @note The cursor is reposition at thend end for every write.
            end_of_file,

            /// @brief The cursor is positioned at the end of the file after its contents have been truncated.
            /// @note The cursor is reposition at thend end for every write.
            end_of_truncated
        };

        /// @brief References for position offsets: from start, from current location, or from end.
        enum class position_type {
            /// @brief Offset position from the start of the file.
            /// @note Using this position the offset should be zero or greater.
            start,

            /// @brief Offset position from the current location in the file.
            current,

            /// @brief Offset position from the end of the file.
            /// @note Using this position the offset should be zero or less.
            end
        };

    private:
        /// @brief handle is the file descriptor which represents the opened file to the operating system.
        int handle = -1;

    public:
        /// @brief Destructor ensures the file handle is closed when this class is desctructed.
        ~file() {
            this->close();
        }

        /// @brief Empty constructor is defaulted.
        file() = default;

        /// @brief Copy constructor is deleted.
        file(const file& other) = delete;

        /// @brief Move constructor is defaulted.
        file(file&& other) = default;

        /// @brief Copy assignment operator is deleted.
        file& operator=(const file& other) = delete;

        /// @brief Move assignment operator is defaulted.
        file& operator=(file&& other) = default;

        /// @brief Parameterised constructor passes arguments onto the open member function.
        /// @param path The path to the file to open.
        /// @param access_mode The access mode used to read or write the file.
        /// @param creation_mode The creation mode used to create or open the file.
        /// @param cursor_mode The cursor mode used to position reads and writes, whether to truncate the file, and if all writes should be appended.
        file(
            const char* const __restrict path,
            access_type access_mode = access_type::read_only,
            creation_type creation_mode = creation_type::open_only,
            cursor_type cursor_mode = cursor_type::start_of_file
        ) {
            this->open(path, access_mode, creation_mode, cursor_mode);
        }

    public:
        /// @brief A function which returns the open status of the file handle within this class.
        /// @return true if a file is open, false otherwise.
        bool is_open() const;

        /// @brief A function to open a file.
        /// @param path The path to the file to open.
        /// @param access_mode The access mode used to read or write the file.
        /// @param creation_mode The creation mode used to create or open the file.
        /// @param cursor_mode The writing mode used when reading, writing, truncate, or append to the file.
        /// @return true if the file was successfully opened, false otherwise.
        bool open(
            const char* const __restrict path,
            access_type access_mode = access_type::read_only,
            creation_type creation_mode = creation_type::open_only,
            cursor_type cursor_mode = cursor_type::start_of_file
        );

        /// @brief A function to close an open file.
        /// @return true if the file was successfully closed, false otherwise.
        bool close();

    public:
        /// @brief A function get the internal file handle used to control access to an opened file.
        /// @return The raw file handle.
        int get_handle() const;

    public:
        /// @brief A function which returns the eof status of the file handle within this class.
        /// @param[out] eof true if the file position is at the eof, false otherwise.
        /// @return true if the cursor is at the end of the file, false otherwise.
        bool is_eof(bool& eof) const;

    public:
        /// @brief A function to get the current size of an opened file.
        /// @param[out] size The current size of the file from the start to the end.
        /// @return true if the size was successfully calculated, false otherwise.
        bool get_size(size_type& size) const;

    public:
        /// @brief A function to get the current location within an opened file.
        /// @param[out] position The current position within the file from the start.
        /// @return true if the position was successfully calculated, false otherwise.
        bool get_cursor_position(size_type& position) const;

        /// @brief A function to set the current location within an opened file.
        /// @param position The desired offset within the file relative to either the start, end, or the current position.
        /// @param relative_to The desired position used as a base upon which to apply the offset.
        /// @return true if the position was successfully set, false otherwise.
        bool set_cursor_position(offset_type position, position_type relative_to = gtl::file::position_type::start) const;

    public:
        /// @brief A function to read an array of characters from an opened file.
        /// @param[out] buffer The array to be filled by characters.
        /// @param[in,out] length The number of characters to attempt to read, set to the number of characters read.
        /// @return true if no errors were encountered, false otherwise.
        bool read(char* const __restrict buffer, size_type& length) const;

        /// @brief A function to write an array of characters to an opened file.
        /// @param buffer The array of characters to output.
        /// @param[in,out] length The number of characters to attempt to write, set to the number of characters written.
        /// @return true if no errors were encountered, false otherwise.
        bool write(const char* const __restrict buffer, size_type& length) const;
    };
}

#endif // ZEROSLAM_TOOLS_COMMON_FILE_HPP
