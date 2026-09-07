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

#include "core/logger.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

namespace {
    class capture final {
    private:
        FILE* file;

    public:
        capture()
            : file(std::tmpfile()) {
            REQUIRE(this->file != nullptr);
            core::logger::set_stream(this->file);
        }

        ~capture() {
            core::logger::set_stream(nullptr);
            std::fclose(this->file);
        }

        size_t read(char* const buffer, const size_t capacity) {
            std::rewind(this->file);
            const size_t length = std::fread(buffer, 1, capacity - 1, this->file);
            buffer[length] = '\0';
            std::fclose(this->file);
            this->file = std::tmpfile();
            REQUIRE(this->file != nullptr);
            core::logger::set_stream(this->file);
            return length;
        }
    };

    // A line is "[<15 digits>][<5 chars>] <message>\n": check the fixed-width prefix and the message.
    bool matches_line(const char* const line, const char* const level, const char* const message) {
        if (std::strlen(line) < 24) {
            return false;
        }
        if (line[0] != '[') {
            return false;
        }
        for (int i = 1; i <= 15; ++i) {
            if ((line[i] < '0') || (line[i] > '9')) {
                return false;
            }
        }
        if ((line[16] != ']') || (line[17] != '[')) {
            return false;
        }
        if (std::strncmp(line + 18, level, 5) != 0) {
            return false;
        }
        if ((line[23] != ']') || (line[24] != ' ')) {
            return false;
        }
        const size_t message_length = std::strlen(message);
        if (std::strncmp(line + 25, message, message_length) != 0) {
            return false;
        }
        return std::strcmp(line + 25 + message_length, "\n") == 0;
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        REQUIRE(static_cast<int>(core::logger::level::error) == 1);
        REQUIRE(static_cast<int>(core::logger::level::warn) == 2);
        REQUIRE(static_cast<int>(core::logger::level::note) == 3);
        REQUIRE(static_cast<int>(core::logger::level::info) == 4);
        REQUIRE(static_cast<int>(core::logger::level::debug) == 5);
    }

    {
        REQUIRE(std::strcmp(core::logger::level_to_string(core::logger::level::error), "ERROR") == 0);
        REQUIRE(std::strcmp(core::logger::level_to_string(core::logger::level::warn), "WARN-") == 0);
        REQUIRE(std::strcmp(core::logger::level_to_string(core::logger::level::note), "NOTE-") == 0);
        REQUIRE(std::strcmp(core::logger::level_to_string(core::logger::level::info), "INFO-") == 0);
        REQUIRE(std::strcmp(core::logger::level_to_string(core::logger::level::debug), "DEBUG") == 0);
    }

    {
        REQUIRE(core::logger::get_verbosity() == static_cast<int>(core::logger::level::error));
        REQUIRE(core::logger::get_stream() == stderr);
        REQUIRE(core::logger::enabled(core::logger::level::error));
        REQUIRE(!core::logger::enabled(core::logger::level::warn));
    }

    {
        capture captured;
        char buffer[1024];
        core::logger::set_verbosity(0);
        REQUIRE(core::logger::log(core::logger::level::error, "silent") == 0);
        REQUIRE(captured.read(buffer, sizeof(buffer)) == 0);

        core::logger::set_verbosity(2);
        REQUIRE(core::logger::enabled(core::logger::level::error));
        REQUIRE(core::logger::enabled(core::logger::level::warn));
        REQUIRE(!core::logger::enabled(core::logger::level::note));
        REQUIRE(core::logger::log(core::logger::level::note, "dropped") == 0);
        REQUIRE(core::logger::log(core::logger::level::info, "dropped") == 0);
        REQUIRE(core::logger::log(core::logger::level::debug, "dropped") == 0);
        REQUIRE(captured.read(buffer, sizeof(buffer)) == 0);
        REQUIRE(core::logger::log(core::logger::level::warn, "kept") > 0);
        REQUIRE(captured.read(buffer, sizeof(buffer)) > 0);
        REQUIRE(matches_line(buffer, "WARN-", "kept"));

        core::logger::set_verbosity(5);
        REQUIRE(core::logger::log(core::logger::level::debug, "kept") > 0);
        REQUIRE(captured.read(buffer, sizeof(buffer)) > 0);
        REQUIRE(matches_line(buffer, "DEBUG", "kept"));
        core::logger::set_verbosity(0);
    }

    {
        capture captured;
        char buffer[1024];
        core::logger::set_verbosity(5);
        REQUIRE(core::logger::log(core::logger::level::error, "value %d and %s (%.2f%%)", 42, "text", 12.5) > 0);
        REQUIRE(captured.read(buffer, sizeof(buffer)) > 0);
        REQUIRE(matches_line(buffer, "ERROR", "value 42 and text (12.50%)"));
        REQUIRE(core::logger::log(core::logger::level::note, "no arguments") > 0);
        REQUIRE(captured.read(buffer, sizeof(buffer)) > 0);
        REQUIRE(matches_line(buffer, "NOTE-", "no arguments"));
        const int written = core::logger::log(core::logger::level::info, "abc");
        const size_t length = captured.read(buffer, sizeof(buffer));
        REQUIRE(written == static_cast<int>(length));
        REQUIRE(length == 25 + 3 + 1);
        core::logger::set_verbosity(0);
    }

    {
        capture captured;
        char buffer[1024];
        core::logger::set_verbosity(1);
        REQUIRE(core::logger::log(core::logger::level::error, "first") > 0);
        REQUIRE(core::logger::log(core::logger::level::error, "second") > 0);
        REQUIRE(captured.read(buffer, sizeof(buffer)) > 0);
        const char* const second = std::strchr(buffer, '\n') + 1;
        REQUIRE(matches_line(second, "ERROR", "second"));
        REQUIRE(std::strncmp(buffer + 1, second + 1, 15) <= 0);
        core::logger::set_verbosity(0);
    }

    {
        capture captured;
        char format[1500];
        std::memset(format, 'x', sizeof(format) - 1);
        format[sizeof(format) - 1] = '\0';
        std::memcpy(format, "%d ", 3);
        char buffer[2048];
        core::logger::set_verbosity(1);
        REQUIRE(core::logger::log(core::logger::level::error, format, 7) > 0);
        const size_t length = captured.read(buffer, sizeof(buffer));
        REQUIRE(length == 25 + (sizeof(format) - 1 - 1) + 1);
        REQUIRE(std::strncmp(buffer + 25, "7 xxxx", 6) == 0);
        REQUIRE(buffer[length - 1] == '\n');
        core::logger::set_verbosity(0);
    }

    return EXIT_SUCCESS;
}
