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

MESSAGE(STATUS "Checking assert macros...")

# Get parameters passed from the main CMakeLists.txt.
SET(CMAKE_SOURCE_DIR ${SOURCE_DIR})

# Shared file enumeration, which leaves out every path the .gitignore files exclude.
INCLUDE("${CMAKE_SOURCE_DIR}/checks/project_files.cmake")

# Find all public header, source, test, and tool files.
GET_PROJECT_FILES(SOURCE_FILES
    "include/*"
    "source/*"
    "tests/*"
    "tools/*"
)

# The project has one assert macro, ASSERT in the assert header.
SET(ASSERT_HEADER "source/core/assert.hpp")
SET(ASSERT_INCLUDE "#include \"core/assert.hpp\"")

# The build files are not code.
LIST(FILTER SOURCE_FILES EXCLUDE REGEX "(^|/)CMakeLists[.]txt$")

# Check each file for assert macro usage.
FOREACH(SOURCE_FILE ${SOURCE_FILES})

    # Print progress.
    #MESSAGE(STATUS "Processing '${SOURCE_FILE}'...")

    # Get content.
    FILE(READ "${CMAKE_SOURCE_DIR}/${SOURCE_FILE}" SOURCE_FILE_CONTENT)

    # Add a trailing newline so a marker on an unterminated last line is checked too.
    SET(SOURCE_FILE_CONTENT "${SOURCE_FILE_CONTENT}\n")

    # A use of the shared assert macro requires its header to be included directly.
    IF(NOT SOURCE_FILE STREQUAL ASSERT_HEADER)
        IF(SOURCE_FILE_CONTENT MATCHES "(^|[^A-Za-z0-9_])ASSERT[ \t]*[(]")
            STRING(FIND "${SOURCE_FILE_CONTENT}" "${ASSERT_INCLUDE}" FOUND_INCLUDE_LOCATION)
            IF("${FOUND_INCLUDE_LOCATION}" MATCHES "-1")
                MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${SOURCE_FILE}:0 (MESSAGE):")
                MESSAGE(FATAL_ERROR "File '${SOURCE_FILE}' uses ASSERT without the include '${ASSERT_INCLUDE}'.")
            ENDIF()
        ENDIF()
    ENDIF()

    # Any assert-like macro a file defines for itself must be undefined before the file ends.
    STRING(REGEX MATCHALL "#[ \t]*define[ \t]+[A-Za-z_][A-Za-z0-9_]*ASSERT[ \t]*[(]" DEFINED_ASSERTS "${SOURCE_FILE_CONTENT}")
    FOREACH(DEFINED_ASSERT ${DEFINED_ASSERTS})
        STRING(REGEX REPLACE "#[ \t]*define[ \t]+([A-Za-z_][A-Za-z0-9_]*ASSERT)[ \t]*[(]" "\\1" DEFINED_ASSERT "${DEFINED_ASSERT}")

        # The shared macro is meant to stay defined.
        IF(SOURCE_FILE STREQUAL ASSERT_HEADER AND DEFINED_ASSERT STREQUAL "ASSERT")
            CONTINUE()
        ENDIF()

        IF(NOT SOURCE_FILE_CONTENT MATCHES "#[ \t]*undef[ \t]+${DEFINED_ASSERT}[^A-Za-z0-9_]")
            MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${SOURCE_FILE}:0 (MESSAGE):")
            MESSAGE(FATAL_ERROR "Assert macro '${DEFINED_ASSERT}' in file '${SOURCE_FILE}' is missing its undef.")
        ENDIF()
    ENDFOREACH()

ENDFOREACH()

MESSAGE(STATUS "Finished assert macros check.")
