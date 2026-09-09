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
#ifndef ZEROSLAM_TOOLS_COMMON_DIRECTORY_HPP
#define ZEROSLAM_TOOLS_COMMON_DIRECTORY_HPP

// Summary: Collection of cross platform functions to create and read directories.

#include <string>
#include <vector>

namespace gtl {

    /// @brief A class which provides cross platform functions to create and read directories.
    class directory final {
    private:
        /// @brief  Deleted destructor.
        ~directory() = delete;

        /// @brief  Deleted constructor.
        directory() = delete;

        /// @brief  Deleted copy constructor.
        directory(directory&) = delete;

        /// @brief  Deleted move constructor.
        directory(directory&&) = delete;

        /// @brief  Deleted copy assignment operator.
        directory& operator=(directory& other) = delete;

        /// @brief  Deleted move assignment operator.
        directory& operator=(directory&& other) = delete;

    public:
        /// @brief  Create one directory, treating an existing directory as success.
        /// @param path The path of the directory to create.
        /// @return true if the directory exists after the call, false otherwise.
        static bool make_directory(const std::string& path);

        /// @brief  Create a directory and any missing parents above it.
        /// @param path The path of the directory to create.
        /// @return true if the directory exists after the call, false otherwise.
        static bool make_directories(const std::string& path);

        /// @brief  The names of the entries of a directory, excluding "." and "..", sorted so that a directory reads the same on every filesystem.
        /// @param path The path of the directory to list.
        /// @param[out] names The names of the entries of the directory.
        /// @return true if the directory could be read, false otherwise.
        static bool list_directory(const std::string& path, std::vector<std::string>& names);
    };
}

#endif // ZEROSLAM_TOOLS_COMMON_DIRECTORY_HPP
