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
#ifndef ZEROSLAM_TOOLS_COMMON_PATHS_HPP
#define ZEROSLAM_TOOLS_COMMON_PATHS_HPP

// Summary: Collection of cross platform functions to provide useful paths.

#include <string>
#include <vector>

namespace gtl {

    /// @brief A class which provides cross platform functions to provide useful paths.
    class paths final {
    private:
        /// @brief  Deleted destructor.
        ~paths() = delete;

        /// @brief  Deleted constructor.
        paths() = delete;

        /// @brief  Deleted copy constructor.
        paths(paths&) = delete;

        /// @brief  Deleted move constructor.
        paths(paths&&) = delete;

        /// @brief  Deleted copy assignment operator.
        paths& operator=(paths& other) = delete;

        /// @brief  Deleted move assignment operator.
        paths& operator=(paths&& other) = delete;

    public:
        /// @brief  Get the full path to the current executable.
        /// @param[out] executable_path the full path to the current executable.
        /// @return true if the executable path was successfully returned, false otherwise.
        static bool get_executable_path(std::string& executable_path);

        /// @brief  Get the name of the current executable.
        /// @param[out] executable_name the name of the current executable.
        /// @return true if the executable name was successfully returned, false otherwise.
        static bool get_executable_name(std::string& executable_name);

        /// @brief  Get the directory of the current executable.
        /// @param[out] executable_directory the directory of the current executable.
        /// @return true if the executable directory was successfully returned, false otherwise.
        static bool get_executable_directory(std::string& executable_directory);

        /// @brief  The extension of a file name, including the dot, or empty when there is none.
        /// @param name The name of the file.
        /// @return The extension of the file name, including the dot, or empty when there is none.
        static std::string path_extension(const std::string& name);

        /// @brief  The stem of a file name: everything before its extension.
        /// @param name The name of the file.
        /// @return The stem of the file name: everything before its extension.
        static std::string path_stem(const std::string& name);

        /// @brief  Everything before the final path separator, or empty when there is none.
        /// @param path The path to the file.
        /// @return Everything before the final path separator, or empty when there is none.
        static std::string path_parent_directory(const std::string& path);

        /// @brief  Check if there is something at a path.
        /// @param path The path to inspect.
        /// @return true if there is something at the path, false otherwise.
        static bool exists(const std::string& path);

        /// @brief  Check if the thing at a path is a directory.
        /// @param path The path to inspect.
        /// @return true if the thing at the path is a directory, false otherwise.
        static bool is_directory(const std::string& path);

        /// @brief  Check if the thing at a path is a regular file.
        /// @param path The path to inspect.
        /// @return true if the thing at the path is a regular file, false otherwise.
        static bool is_regular_file(const std::string& path);

        /// @brief  The size of a file, or false when it cannot be determined.
        /// @param path The path of the file.
        /// @param[out] size The size of the file.
        /// @return true if the size of the file was successfully returned, false otherwise.
        static bool get_file_size(const std::string& path, unsigned long long& size);

        /// @brief  Get the current working directory.
        /// @param[out] current_working_directory The current working directory.
        /// @return true if the current working directory was successfully returned, false otherwise.
        static bool get_current_working_directory(std::string& current_working_directory);

        /// @brief  Get the users home directory.
        /// @param[out] home_directory The users home directory.
        /// @return true if the users home directory was successfully returned, false otherwise.
        static bool get_home_directory(std::string& home_directory);

        /// @brief  Get the users application data directory.
        /// @param[out] application_data_directory The users application data directory.
        /// @return true if the users application data directory was successfully returned, false otherwise.
        static bool get_application_data_directory(std::string& application_data_directory);
    };
}

#endif // ZEROSLAM_TOOLS_COMMON_PATHS_HPP
