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
#ifndef ZEROSLAM_TOOLS_COMMON_PROCESS_HPP
#define ZEROSLAM_TOOLS_COMMON_PROCESS_HPP

// Summary: Launch other programs by name through the PATH search, and run commands
// through the shell, capturing their output.

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <sys/wait.h>
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if !defined(_WIN32)
// The popen and pclose declarations are hidden by strict ansi modes such as -std=c++17,
// so declare them here rather than raising the posix source feature macros.
#if defined(__APPLE__)
#define ZEROSLAM_PLATFORM_ALIAS(name) __asm__("_" #name)
#else
#define ZEROSLAM_PLATFORM_ALIAS(name)
#endif

extern "C" std::FILE* popen(const char* command, const char* type) ZEROSLAM_PLATFORM_ALIAS(popen);
extern "C" int pclose(std::FILE* stream) ZEROSLAM_PLATFORM_ALIAS(pclose);

#undef ZEROSLAM_PLATFORM_ALIAS
#endif

namespace platform {
#if !defined(_WIN32)
    namespace details {
        // The setenv and execvp declarations are hidden by strict ansi modes such as
        // -std=c++17, so declare them here rather than raising the posix source feature macros.
        extern "C" int setenv(const char* name, const char* value, int overwrite);
        extern "C" int execvp(const char* file, char* const arguments[]);
    }

    // Replace the current process with 'file' found by searching the PATH environment
    // variable, passing 'arguments' in the argv convention: arguments.front() is the program
    // name and arguments.back() is nullptr. Only returns when the program could not be started.
    inline void replace_process_searching_path(const std::string& file, std::vector<char*> arguments) {
        static_cast<void>(details::execvp(file.c_str(), arguments.data()));
    }
#else
    namespace details {
        // Windows cannot replace a running process, so the program is started as a child of
        // this one and waited for; ending this process with the child's exit code afterwards
        // mirrors the posix exec behaviour of not returning.
        inline void finish_with_exit_code(const int exit_code) {
            std::fflush(stdout);
            std::fflush(stderr);
            std::_Exit(exit_code);
        }

        // The _spawnv functions pass the arguments through one command line without quoting
        // them, so any argument containing whitespace must be re-quoted to survive the round
        // trip.
        inline std::vector<const char*> windows_command_line(const std::vector<char*>& arguments) {
            std::vector<std::string> quoted_arguments;
            quoted_arguments.reserve(arguments.size());
            for (const char* const argument : arguments) {
                if (argument == nullptr) {
                    break;
                }
                std::string quoted = argument;
                if (quoted.find_first_of(" \t") != std::string::npos) {
                    quoted = "\"" + quoted + "\"";
                }
                quoted_arguments.push_back(std::move(quoted));
            }
            std::vector<const char*> command_line;
            command_line.reserve(quoted_arguments.size() + 1);
            for (const std::string& argument : quoted_arguments) {
                command_line.push_back(argument.c_str());
            }
            command_line.push_back(nullptr);
            return command_line;
        }
    }

    // Replace the current process with 'file' found by searching the PATH environment
    // variable, passing 'arguments' in the argv convention: arguments.front() is the program
    // name and arguments.back() is nullptr. Only returns when the program could not be started.
    inline void replace_process_searching_path(const std::string& file, std::vector<char*> arguments) {
        const std::vector<const char*> command_line = details::windows_command_line(arguments);
        const intptr_t status = ::_spawnvp(_P_WAIT, file.c_str(), command_line.data());
        if (status >= 0) {
            details::finish_with_exit_code(static_cast<int>(status));
        }
    }
#endif

    // Prepend 'directory' to the PATH environment variable, so programs started by name are
    // found there first.
    inline void prepend_to_search_path(const std::string& directory) {
#if defined(_WIN32)
        const char* const current_path = std::getenv("PATH");
        const std::string updated_path = directory + (((current_path != nullptr) && (*current_path != '\0')) ? (";" + std::string(current_path)) : std::string());
        static_cast<void>(::_putenv(("PATH=" + updated_path).c_str()));
#else
        const char* const current_path = std::getenv("PATH");
        const std::string updated_path = directory + (((current_path != nullptr) && (*current_path != '\0')) ? (":" + std::string(current_path)) : std::string());
        static_cast<void>(details::setenv("PATH", updated_path.c_str(), 1));
#endif
    }

    // Quote a path or argument so the shell treats it as a single word.
    inline std::string quote_for_shell(const std::string& text) {
#if defined(_WIN32)
        return "\"" + text + "\"";
#else
        std::string quoted = "'";
        for (const char character : text) {
            if (character == '\'') {
                quoted += "'\\''";
            }
            else {
                quoted += character;
            }
        }
        quoted += "'";
        return quoted;
#endif
    }

    // The last non-empty line of a command's output, for concise error reporting.
    inline std::string last_line(const std::string& output) {
        std::size_t end = output.size();
        while (end > 0) {
            std::size_t start = output.find_last_of('\n', end - 1);
            start = (start == std::string::npos) ? 0 : (start + 1);
            std::string line = output.substr(start, end - start);
            while (!line.empty() && ((line.back() == '\r') || (line.back() == ' '))) {
                line.pop_back();
            }
            if (!line.empty()) {
                return line;
            }
            if (start == 0) {
                break;
            }
            end = start - 1;
        }
        return std::string();
    }

    // Run a command through the shell, capturing (and optionally echoing) its combined output.
    inline bool run_command(const std::string& command, const bool echo_output, std::string& output, int& exit_code) {
        output.clear();
        exit_code = -1;
#if defined(_WIN32)
        std::FILE* pipe = ::_popen(command.c_str(), "r");
#else
        std::FILE* pipe = ::popen(command.c_str(), "r");
#endif
        if (pipe == nullptr) {
            return false;
        }
        char buffer[4096];
        for (std::size_t bytes_read = 0; (bytes_read = std::fread(&buffer[0], 1, sizeof(buffer), pipe)) > 0;) {
            output.append(&buffer[0], bytes_read);
            if (echo_output) {
                std::fwrite(&buffer[0], 1, bytes_read, stdout);
                std::fflush(stdout);
            }
        }
#if defined(_WIN32)
        const int status = ::_pclose(pipe);
        if (status == -1) {
            return false;
        }
        exit_code = status;
#else
        const int status = ::pclose(pipe);
        if (status == -1) {
            return false;
        }
        if (WIFEXITED(status)) {
            exit_code = WEXITSTATUS(status);
        }
        else if (WIFSIGNALED(status)) {
            exit_code = 128 + WTERMSIG(status);
        }
#endif
        return true;
    }
}

#endif // ZEROSLAM_TOOLS_COMMON_PROCESS_HPP
