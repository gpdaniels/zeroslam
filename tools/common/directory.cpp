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

#include "directory.hpp"

#include "paths.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <sys/stat.h>

#if defined(_WIN32)
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace gtl {

    bool directory::make_directory(const std::string& path) {
#if defined(_WIN32)
        if (::_mkdir(path.c_str()) == 0) {
            return true;
        }
#else
        if (::mkdir(path.c_str(), 0755) == 0) {
            return true;
        }
#endif
        return paths::is_directory(path);
    }

    bool directory::make_directories(const std::string& path) {
        if (path.empty()) {
            return false;
        }
        if (paths::exists(path)) {
            return paths::is_directory(path);
        }
        const std::size_t separator = path.find_last_of("/\\");
        if ((separator != std::string::npos) && (separator > 0)) {
            if (!make_directories(path.substr(0, separator))) {
                return false;
            }
        }
        return make_directory(path);
    }

    bool directory::list_directory(const std::string& path, std::vector<std::string>& names) {
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
        ::DIR* directory_handle = ::opendir(path.c_str());
        if (directory_handle == nullptr) {
            return false;
        }
        while (const struct dirent* entry = ::readdir(directory_handle)) {
            const std::string name = entry->d_name;
            if ((name != ".") && (name != "..")) {
                names.push_back(name);
            }
        }
        ::closedir(directory_handle);
#endif
        std::sort(names.begin(), names.end());
        return true;
    }
}
