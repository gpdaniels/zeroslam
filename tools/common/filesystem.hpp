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
#ifndef ZEROSLAM_TOOLS_COMMON_FILESYSTEM_HPP
#define ZEROSLAM_TOOLS_COMMON_FILESYSTEM_HPP

// Summary: Minimal platform file system queries standing in for <filesystem>.

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <string>
#include <sys/stat.h>
#include <vector>

#if !defined(_WIN32)
#include <dirent.h>
#include <unistd.h>
#else
#include <direct.h>
#include <io.h>
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// A small set of the directory queries the tools need, built directly on the platform
// apis so nothing here depends on <filesystem> or the posix source feature macros.
namespace platform {
    // The extension of a file name, including the dot, or empty when there is none.
    inline std::string path_extension(const std::string& name) {
        const std::size_t dot = name.find_last_of('.');
        if ((dot == std::string::npos) || (dot == 0)) {
            return std::string();
        }
        return name.substr(dot);
    }

    // The stem of a file name: everything before its extension.
    inline std::string path_stem(const std::string& name) {
        return name.substr(0, name.size() - path_extension(name).size());
    }

    // Everything before the final path separator, or empty when there is none.
    inline std::string path_parent_directory(const std::string& path) {
        const std::size_t separator = path.find_last_of("/\\");
        if (separator == std::string::npos) {
            return std::string();
        }
        return path.substr(0, separator);
    }

    // The kind and size of the thing at a path, or false when there is nothing.
    inline bool inspect_path(const std::string& path, bool& directory, bool& regular_file, unsigned long long& size) {
        directory = false;
        regular_file = false;
        size = 0;
#if defined(_WIN32)
        struct __stat64 status = {};
        if (::_stat64(path.c_str(), &status) != 0) {
            return false;
        }
        directory = ((status.st_mode & _S_IFMT) == _S_IFDIR);
        regular_file = ((status.st_mode & _S_IFMT) == _S_IFREG);
#else
        struct stat status = {};
        if (::stat(path.c_str(), &status) != 0) {
            return false;
        }
        directory = S_ISDIR(status.st_mode);
        regular_file = S_ISREG(status.st_mode);
#endif
        size = (status.st_size > 0) ? static_cast<unsigned long long>(status.st_size) : 0ull;
        return true;
    }

    inline bool is_directory(const std::string& path) {
        bool directory = false;
        bool regular_file = false;
        unsigned long long size = 0;
        return inspect_path(path, directory, regular_file, size) && directory;
    }

    inline bool is_regular_file(const std::string& path) {
        bool directory = false;
        bool regular_file = false;
        unsigned long long size = 0;
        return inspect_path(path, directory, regular_file, size) && regular_file;
    }

    // The size of a file, or false when it cannot be determined.
    inline bool get_file_size(const std::string& path, unsigned long long& size) {
        bool directory = false;
        bool regular_file = false;
        return inspect_path(path, directory, regular_file, size) && regular_file;
    }

    // Create one directory, treating an existing directory as success.
    inline bool make_directory(const std::string& path) {
#if defined(_WIN32)
        if (::_mkdir(path.c_str()) == 0) {
            return true;
        }
#else
        if (::mkdir(path.c_str(), 0755) == 0) {
            return true;
        }
#endif
        return is_directory(path);
    }

    // Create a directory and any missing parents above it.
    inline bool make_directories(const std::string& path) {
        if (path.empty()) {
            return false;
        }
        bool directory = false;
        bool regular_file = false;
        unsigned long long size = 0;
        if (inspect_path(path, directory, regular_file, size)) {
            return directory;
        }
        const std::size_t separator = path.find_last_of("/\\");
        if ((separator != std::string::npos) && (separator > 0)) {
            if (!make_directories(path.substr(0, separator))) {
                return false;
            }
        }
        return make_directory(path);
    }

    // The names of the entries of a directory, excluding '.' and '..', in no particular
    // order; false when the directory cannot be read.
    inline bool list_directory(const std::string& path, std::vector<std::string>& names) {
        names.clear();
#if defined(_WIN32)
        struct _finddata_t entry = {};
        const intptr_t handle = ::_findfirst((path + "\\*").c_str(), &entry);
        if (handle == -1) {
            return false;
        }
        do {
            const std::string name = entry.name;
            if ((name != ".") && (name != "..")) {
                names.push_back(name);
            }
        } while (::_findnext(handle, &entry) == 0);
        ::_findclose(handle);
#else
        ::DIR* directory = ::opendir(path.c_str());
        if (directory == nullptr) {
            return false;
        }
        while (const struct dirent* entry = ::readdir(directory)) {
            const std::string name = entry->d_name;
            if ((name != ".") && (name != "..")) {
                names.push_back(name);
            }
        }
        ::closedir(directory);
#endif
        return true;
    }
}

#endif // ZEROSLAM_TOOLS_COMMON_FILESYSTEM_HPP
