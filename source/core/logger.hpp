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
#ifndef ZEROSLAM_CORE_LOGGER_HPP
#define ZEROSLAM_CORE_LOGGER_HPP

#include "core/timestamp.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace core {
    class logger final {
    public:
        enum class level : int {
            error = 1,
            warn = 2,
            note = 3,
            info = 4,
            debug = 5
        };

    private:
        int verbosity;
        FILE* stream;
        long long int start;

    private:
        logger()
            : verbosity(static_cast<int>(level::error))
            , stream(nullptr)
            , start(timestamp()) {
        }

        ~logger() = default;

        logger(const logger&) = delete;
        logger(logger&&) = delete;
        logger& operator=(const logger&) = delete;
        logger& operator=(logger&&) = delete;

        // The one instance.
        static logger& instance() {
            static logger singleton;
            return singleton;
        }

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-security"
#pragma clang diagnostic ignored "-Wformat-nonliteral"
#endif

#if (defined(__GNUC__) || defined(__GNUG__)) && (!defined(__clang__) && (!defined(__INTEL_COMPILER)))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif

        template <typename... argument_types>
        static int write(const level message_level, const char* const format, argument_types... arguments) {
            FILE* const stream = logger::get_stream();
            const long long int nanoseconds = timestamp() - logger::instance().start;
            const char* const level_string = logger::level_to_string(message_level);
            char prefixed[1024];
            const int prefixed_length = std::snprintf(prefixed, sizeof(prefixed), "[%015lld][%s] %s\n", nanoseconds, level_string, format);
            int written = 0;
            if ((prefixed_length < 0) || (prefixed_length >= static_cast<int>(sizeof(prefixed)))) {
                written += std::fprintf(stream, "[%015lld][%s] ", nanoseconds, level_string);
                written += std::fprintf(stream, format, arguments...);
                written += (std::fputc('\n', stream) == EOF) ? 0 : 1;
            }
            else {
                written = std::fprintf(stream, prefixed, arguments...);
            }
            std::fflush(stream);
            return written;
        }

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if (defined(__GNUC__) || defined(__GNUG__)) && (!defined(__clang__) && (!defined(__INTEL_COMPILER)))
#pragma GCC diagnostic pop
#endif

    public:
        static const char* level_to_string(const level message_level) {
            switch (message_level) {
                case level::error:
                    return "ERROR";
                case level::warn:
                    return "WARN-";
                case level::note:
                    return "NOTE-";
                case level::info:
                    return "INFO-";
                case level::debug:
                    return "DEBUG";
            }
            return "?????";
        }

        static int get_verbosity() {
            return logger::instance().verbosity;
        }

        static void set_verbosity(const int verbosity) {
            logger::instance().verbosity = verbosity;
        }

        static bool enabled(const level message_level) {
            return static_cast<int>(message_level) <= logger::get_verbosity();
        }

        static FILE* get_stream() {
            FILE* const stream = logger::instance().stream;
            return (stream != nullptr) ? stream : stderr;
        }

        static void set_stream(FILE* const stream) {
            logger::instance().stream = stream;
        }

        template <typename... argument_types>
        static int log(const level message_level, const char* const format, argument_types... arguments) {
            if (!logger::enabled(message_level)) {
                return 0;
            }
            return logger::write(message_level, format, arguments...);
        }
    };
}

#endif // ZEROSLAM_CORE_LOGGER_HPP
