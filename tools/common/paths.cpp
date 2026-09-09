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

#include "paths.hpp"

#include "directory.hpp"

#if defined(linux) || defined(__linux) || defined(__linux__)
#include <cstring>
#include <dirent.h>
#include <linux/limits.h>
#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(_WIN32)

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define STRICT

#include <sdkddkver.h>

#if defined(_AFXDLL)
#include <afxwin.h>
#else
#include <Windows.h>
#endif

#include <Shlobj.h>
#include <direct.h>
#include <sys/stat.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif

#if defined(__APPLE__)
#include <cstring>
#include <dirent.h>
#include <mach-o/dyld.h>
#include <pwd.h>
#include <sys/stat.h>
#include <sys/syslimits.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace gtl {

    bool paths::get_executable_path(std::string& executable_path) {
        executable_path.clear();

#if defined(linux) || defined(__linux) || defined(__linux__)

        executable_path.resize(PATH_MAX + 1, '\0');
        ssize_t executable_path_size = ::readlink("/proc/self/exe", &executable_path[0], PATH_MAX);
        if (executable_path_size <= 0) {
            return false;
        }
        executable_path.resize(static_cast<std::size_t>(executable_path_size));
        return true;

#elif defined(_WIN32)

        executable_path.resize(MAX_PATH + 1, '\0');
        GetModuleFileNameA(nullptr, &executable_path[0], MAX_PATH);
        std::size_t executable_path_size = executable_path.find_last_not_of('\0');
        if (executable_path_size == std::string::npos) {
            return false;
        }
        executable_path.resize(executable_path_size + 1);
        return true;

#elif defined(__APPLE__)

        uint32_t executable_path_size = 0;
        if (_NSGetExecutablePath(nullptr, &executable_path_size) == 0) {
            return false;
        }
        if (executable_path_size == 0) {
            return false;
        }
        executable_path.resize(executable_path_size, '\0');
        if (_NSGetExecutablePath(&executable_path[0], &executable_path_size) != 0) {
            return false;
        }
        executable_path.resize(executable_path_size - 1);
        return true;

#else

        return false;

#endif
    }

    bool paths::get_executable_name(std::string& executable_name) {
        executable_name.clear();
        std::string executable_path;
        if (!get_executable_path(executable_path)) {
            return false;
        }
        std::string::size_type executable_name_start_index = executable_path.find_last_of("\\/");
        if (executable_name_start_index == std::string::npos) {
            return false;
        }
        executable_name = executable_path.substr(executable_name_start_index + 1);
        executable_name = executable_name.substr(0, executable_name.find_last_of("."));
        return true;
    }

    bool paths::get_executable_directory(std::string& executable_directory) {
        executable_directory.clear();
        std::string executable_path;
        if (!get_executable_path(executable_path)) {
            return false;
        }
        std::string::size_type executable_directory_end_index = executable_path.find_last_of("\\/");
        if (executable_directory_end_index == std::string::npos) {
            return false;
        }
        executable_directory = executable_path.substr(0, executable_directory_end_index);
        return true;
    }

    std::string paths::path_extension(const std::string& name) {
        const std::size_t dot = name.find_last_of('.');
        if ((dot == std::string::npos) || (dot == 0)) {
            return std::string();
        }
        return name.substr(dot);
    }

    std::string paths::path_stem(const std::string& name) {
        return name.substr(0, name.size() - path_extension(name).size());
    }

    std::string paths::path_parent_directory(const std::string& path) {
        const std::size_t separator = path.find_last_of("/\\");
        if (separator == std::string::npos) {
            return std::string();
        }
        return path.substr(0, separator);
    }

    bool paths::exists(const std::string& path) {
#if defined(_WIN32)
        struct __stat64 status = {};
        return ::_stat64(path.c_str(), &status) == 0;
#else
        struct stat status = {};
        return ::stat(path.c_str(), &status) == 0;
#endif
    }

    bool paths::is_directory(const std::string& path) {
#if defined(_WIN32)
        struct __stat64 status = {};
        if (::_stat64(path.c_str(), &status) != 0) {
            return false;
        }
        return (status.st_mode & _S_IFMT) == _S_IFDIR;
#else
        struct stat status = {};
        if (::stat(path.c_str(), &status) != 0) {
            return false;
        }
        return S_ISDIR(status.st_mode);
#endif
    }

    bool paths::is_regular_file(const std::string& path) {
#if defined(_WIN32)
        struct __stat64 status = {};
        if (::_stat64(path.c_str(), &status) != 0) {
            return false;
        }
        return (status.st_mode & _S_IFMT) == _S_IFREG;
#else
        struct stat status = {};
        if (::stat(path.c_str(), &status) != 0) {
            return false;
        }
        return S_ISREG(status.st_mode);
#endif
    }

    bool paths::get_file_size(const std::string& path, unsigned long long& size) {
        size = 0;
#if defined(_WIN32)
        struct __stat64 status = {};
        if (::_stat64(path.c_str(), &status) != 0) {
            return false;
        }
        if ((status.st_mode & _S_IFMT) != _S_IFREG) {
            return false;
        }
#else
        struct stat status = {};
        if (::stat(path.c_str(), &status) != 0) {
            return false;
        }
        if (!S_ISREG(status.st_mode)) {
            return false;
        }
#endif
        size = (status.st_size > 0) ? static_cast<unsigned long long>(status.st_size) : 0ull;
        return true;
    }

    bool paths::get_current_working_directory(std::string& current_working_directory) {
        current_working_directory.clear();

#if (defined(linux) || defined(__linux) || defined(__linux__)) || defined(__APPLE__)

        // Get the current working directory status.
        struct stat status_current_directory = {};
        if (stat(".", &status_current_directory) != 0) {
            return false;
        }
        // Get the root directory status.
        struct stat status_root_directory = {};
        if (stat("/", &status_root_directory) != 0) {
            return false;
        }
        // Attempt to keep stepping up the directory tree until we find the root.
        std::string parent_directory = "./..";
        std::vector<std::string> path_components;
        while ((status_current_directory.st_dev != status_root_directory.st_dev) || (status_current_directory.st_ino != status_root_directory.st_ino)) {
            // Get the status of the parent directory.
            struct stat status_parent_directory = {};
            if (stat(parent_directory.c_str(), &status_parent_directory) != 0) {
                return false;
            }
            bool pushed = false;
            // Check each directory in the parent until we find the current directory.
            std::vector<std::string> directory_names;
            directory::list_directory(parent_directory, directory_names);
            std::sort(directory_names.begin(), directory_names.end());
            for (const std::string& directory_name : directory_names) {
                // Get the potentially current directory status.
                const std::string directory = parent_directory + "/" + directory_name;
                struct stat status_potential_directory = {};
                if (!lstat(directory.c_str(), &status_potential_directory)) {
                    // Compare the current directory and the potentially current directory.
                    if ((status_current_directory.st_dev == status_potential_directory.st_dev) && (status_current_directory.st_ino == status_potential_directory.st_ino)) {
                        // If we have successfully found the current directory update the current to the parent and continue the search.
                        status_current_directory = status_parent_directory;
                        parent_directory += "/..";
                        path_components.emplace_back(directory_name);
                        pushed = true;
                        break;
                    }
                }
            }
            // If we failed to find the current directory inside the parent directory exit the loop.
            if (!pushed) {
                break;
            }
        }
        // Double check we found managed to navigate all the way to the root directory.
        if ((status_current_directory.st_dev != status_root_directory.st_dev) || (status_current_directory.st_ino != status_root_directory.st_ino)) {
            return false;
        }
        // Convert the path components back into a path.
        for (std::vector<std::string>::reverse_iterator path_component = path_components.rbegin(); path_component != path_components.rend(); ++path_component) {
            current_working_directory += "/" + *path_component;
        }
        return true;

#elif defined(_WIN32)

        current_working_directory.resize(MAX_PATH + 1, '\0');
        if (!_getcwd(&current_working_directory[0], MAX_PATH)) {
            return false;
        }
        std::size_t current_working_directory_size = current_working_directory.find_last_not_of('\0');
        if (current_working_directory_size == std::string::npos) {
            return false;
        }
        current_working_directory.resize(current_working_directory_size + 1);
        return true;

#else

        return false;

#endif
    }

    bool paths::get_home_directory(std::string& home_directory) {
        home_directory.clear();

#if defined(linux) || defined(__linux) || defined(__linux__)

        passwd* user_information = getpwuid(getuid());
        if (!user_information) {
            return false;
        }
        home_directory = std::string(user_information->pw_dir);
        return true;

#elif defined(_WIN32)

        home_directory.resize(MAX_PATH + 1, '\0');
        if (SHGetFolderPathA(nullptr, CSIDL_MYDOCUMENTS, nullptr, 0, &home_directory[0]) < 0) {
            return false;
        }
        std::size_t home_path_size = home_directory.find_last_not_of('\0');
        if (home_path_size == std::string::npos) {
            return false;
        }
        home_directory.resize(home_path_size + 1);
        return true;

#elif defined(__APPLE__)

        passwd* user_information = getpwuid(getuid());
        if (!user_information) {
            return false;
        }
        home_directory = std::string(user_information->pw_dir);
        return true;

#else

        return false;

#endif
    }

    bool paths::get_application_data_directory(std::string& application_data_directory) {
        application_data_directory.clear();

#if defined(linux) || defined(__linux) || defined(__linux__)

        std::string home_directory;
        if (!get_home_directory(home_directory)) {
            return false;
        }
        application_data_directory = home_directory + "/.config";
        return true;

#elif defined(_WIN32)

        application_data_directory.resize(MAX_PATH + 1, '\0');
        if (SHGetFolderPathA(nullptr, CSIDL_COMMON_APPDATA, nullptr, 0, &application_data_directory[0]) < 0) {
            return false;
        }
        std::size_t pathLength = application_data_directory.find_last_not_of('\0');
        if (pathLength == std::string::npos) {
            return false;
        }
        application_data_directory.resize(pathLength + 1);
        return true;

#elif defined(__APPLE__)

        std::string home_directory;
        if (!get_home_directory(home_directory)) {
            return false;
        }
        application_data_directory = home_directory + "/Library/Application Support";
        return true;

#else

        return false;

#endif
    }
}
