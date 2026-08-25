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
#include "process.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    // Tool names map directly onto binary names, restrict them to safe characters.
    bool is_valid_tool_name(const char* name) {
        if ((name == nullptr) || (name[0] == 0) || (name[0] == '-')) {
            return false;
        }
        for (const char* character = name; *character != 0; ++character) {
            const bool valid = ((*character >= 'a') && (*character <= 'z')) || ((*character >= '0') && (*character <= '9')) || (*character == '-') || (*character == '_');
            if (!valid) {
                return false;
            }
        }
        return true;
    }

    void print_banner() {
        std::printf(".-----------------------------------------------.\n");
        std::printf("|   _____             _____ __    _____ _____   |\n");
        std::printf("|  |__   |___ ___ ___|   __|  |  |  _  |     |  |\n");
        std::printf("|  |   __| -_|  _| . |__   |  |__|     | | | |  |\n");
        std::printf("|  |_____|___|_| |___|_____|_____|__|__|_|_|_|  |\n");
        std::printf("|                                               |\n");
        std::printf("| This software is a:                           |\n");
        std::printf("|  |- simple                                    |\n");
        std::printf("|  |- minimal                                   |\n");
        std::printf("|  |- indirect                                  |\n");
        std::printf("|  |- monocular                                 |\n");
        std::printf("|  |- factor-graph                              |\n");
        std::printf("|  |- deterministic                             |\n");
        std::printf("|  |- dependency-free                           |\n");
        std::printf("|  '- visual SLAM system written in pure C++.   |\n");
        std::printf("|                                               |\n");
        std::printf("| No external libraries. No frills. Just SLAM.  |\n");
        std::printf("|                                               |\n");
        std::printf("| >   https://github.com/gpdaniels/zeroslam   < |\n");
        std::printf("|                                               |\n");
        std::printf("| Licensed under GPLv3                          |\n");
        std::printf("| Get in touch for commercial licensing.        |\n");
        std::printf("'-----------------------------------------------'\n");
        std::printf("\n");
        // Flushed so the banner survives the exec into the tool and orders before its output.
        std::fflush(stdout);
    }

    void print_usage(const char* argv0, const std::string& launcher_directory) {
        std::printf("Usage: %s [--quiet] [tool] [arguments...]\n", argv0);
        std::printf("    tool      - The name of a zeroslam tool, run as the standalone 'zeroslam-[tool]' binary.\n");
        std::printf("    arguments - Forwarded unchanged to the tool.\n");
        std::printf("    --quiet   - Suppress the banner, only consumed before the tool name.\n");
        std::printf("Tools are searched for next to this launcher first, then on the PATH.\n");
        std::vector<std::string> tools;
        if (!launcher_directory.empty()) {
            constexpr static const char prefix[] = "zeroslam-";
            constexpr static const std::size_t prefix_length = sizeof(prefix) - 1;
            std::error_code error;
            std::filesystem::directory_iterator iterator(launcher_directory, error);
            if (!error) {
                for (const std::filesystem::directory_iterator end; iterator != end; iterator.increment(error)) {
                    if (error) {
                        break;
                    }
                    std::error_code entry_error;
                    if (!iterator->is_regular_file(entry_error) || entry_error) {
                        continue;
                    }
                    std::string name = iterator->path().filename().string();
                    if (name.compare(0, prefix_length, prefix) != 0) {
                        continue;
                    }
                    constexpr static const char extension[] = ".exe";
                    constexpr static const std::size_t extension_length = sizeof(extension) - 1;
                    if ((name.size() > extension_length) && (name.compare(name.size() - extension_length, extension_length, extension) == 0)) {
                        name.resize(name.size() - extension_length);
                    }
                    tools.push_back(name.substr(prefix_length));
                }
            }
        }
        std::sort(tools.begin(), tools.end());
        tools.erase(std::unique(tools.begin(), tools.end()), tools.end());
        if (!tools.empty()) {
            std::printf("Available tools:\n");
            for (const std::string& tool : tools) {
                std::printf("    %s\n", tool.c_str());
            }
        }
    }
}

int main(int argc, char* argv[]) {
    const char* argv0 = (argc > 0) ? argv[0] : "zeroslam";
    std::string launcher_directory;
    gtl::paths::get_executable_directory(launcher_directory);

    // A leading --quiet suppresses the banner; arguments after the tool name are
    // forwarded untouched.
    bool quiet = false;
    int tool_argument = 1;
    while ((tool_argument < argc) && ((std::strcmp(argv[tool_argument], "--quiet") == 0) || (std::strcmp(argv[tool_argument], "-q") == 0))) {
        quiet = true;
        ++tool_argument;
    }
    if (!quiet) {
        print_banner();
    }

    if ((tool_argument >= argc) || (std::strcmp(argv[tool_argument], "help") == 0) || (std::strcmp(argv[tool_argument], "--help") == 0) || (std::strcmp(argv[tool_argument], "-h") == 0)) {
        print_usage(argv0, launcher_directory);
        return EXIT_SUCCESS;
    }

    if (!is_valid_tool_name(argv[tool_argument])) {
        std::fprintf(stderr, "Invalid tool name: '%s'\n", argv[tool_argument]);
        return EXIT_FAILURE;
    }
    const std::string tool_name = std::string("zeroslam-") + argv[tool_argument];

    // Tools are launched by name like any installed command, with the launcher's own
    // directory searched first.
    if (!launcher_directory.empty()) {
        platform::prepend_to_search_path(launcher_directory);
    }

    std::vector<char*> tool_arguments;
    tool_arguments.push_back(const_cast<char*>(tool_name.c_str()));
    for (int i = tool_argument + 1; i < argc; ++i) {
        tool_arguments.push_back(argv[i]);
    }
    tool_arguments.push_back(nullptr);

    platform::replace_process_searching_path(tool_name, tool_arguments);

    std::fprintf(stderr, "Failed to run tool '%s': no '%s' found next to the launcher or on the PATH.\n", argv[tool_argument], tool_name.c_str());
    return 127;
}
