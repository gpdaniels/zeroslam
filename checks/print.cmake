#[[
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
]]

# Define the minimum version of CMake that is required.
CMAKE_MINIMUM_REQUIRED(VERSION 3.22)
CMAKE_POLICY(VERSION 3.22)

MESSAGE(STATUS "Checking for direct prints...")

# Get parameters passed from the main CMakeLists.txt.
SET(CMAKE_SOURCE_DIR ${SOURCE_DIR})

# Shared file enumeration, which leaves out every path the .gitignore files exclude.
INCLUDE("${CMAKE_SOURCE_DIR}/checks/project_files.cmake")

# Find all public header and library source files, the tests and tools print as they please.
GET_PROJECT_FILES(SOURCE_FILES
    "include/*"
    "source/*"
)

# The logger header is the one place that writes to a stream.
SET(LOGGER_HEADER "source/core/logger.hpp")

# The output functions of the standard library, a call to any of these belongs in core::logger::log instead.
SET(PRINT_FUNCTIONS "printf|fprintf|vprintf|vfprintf|puts|fputs|putc|fputc|putchar|perror")
SET(PRINT_FUNCTION_REGEX "(^|[^A-Za-z0-9_:])(std::)?(${PRINT_FUNCTIONS})[ \t]*[(]")
SET(PRINT_STREAM_REGEX "std::(cout|cerr|clog|wcout|wcerr|wclog)")

# Check each file for direct prints.
FOREACH(SOURCE_FILE ${SOURCE_FILES})

    # Print progress.
    #MESSAGE(STATUS "Processing '${SOURCE_FILE}'...")

    IF(SOURCE_FILE STREQUAL LOGGER_HEADER)
        CONTINUE()
    ENDIF()

    # Get content.
    FILE(READ "${CMAKE_SOURCE_DIR}/${SOURCE_FILE}" SOURCE_FILE_CONTENT)

    # If a print function call is detected raise an error.
    IF(SOURCE_FILE_CONTENT MATCHES "${PRINT_FUNCTION_REGEX}")
        SET(PRINT_FUNCTION "${CMAKE_MATCH_3}")
        MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${SOURCE_FILE}:0 (MESSAGE):")
        MESSAGE(FATAL_ERROR "Detected a call to '${PRINT_FUNCTION}' in '${SOURCE_FILE}', use the logger instead.")
    ENDIF()

    # If a standard stream is detected raise an error.
    IF(SOURCE_FILE_CONTENT MATCHES "${PRINT_STREAM_REGEX}")
        MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${SOURCE_FILE}:0 (MESSAGE):")
        MESSAGE(FATAL_ERROR "Detected a use of '${CMAKE_MATCH_0}' in '${SOURCE_FILE}', use the logger instead.")
    ENDIF()

ENDFOREACH()

MESSAGE(STATUS "Finished direct prints check.")
