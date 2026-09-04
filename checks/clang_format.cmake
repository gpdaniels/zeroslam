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

MESSAGE(STATUS "Checking for clang format...")

# Get parameters passed from the main CMakeLists.txt.
SET(CMAKE_SOURCE_DIR ${SOURCE_DIR})
SET(CMAKE_BINARY_DIR ${BINARY_DIR})

# Shared file enumeration, which leaves out every path the .gitignore files exclude.
INCLUDE("${CMAKE_SOURCE_DIR}/checks/project_files.cmake")

# Find all public header, source, test, and tool files.
GET_PROJECT_FILES(CLANG_FORMAT_SOURCES
    "include/*.h"
    "include/*.hpp"
    "source/*.cpp"
    "source/*.hpp"
    "tests/*.cpp"
    "tests/*.hpp"
    "tools/*.c"
    "tools/*.cpp"
    "tools/*.hpp"
)

# The style file is required, without it clang-format would apply its own default style.
IF(NOT EXISTS "${CMAKE_SOURCE_DIR}/.clang-format")
    MESSAGE(STATUS "No .clang-format file in the project, skipping check.")
    RETURN()
ENDIF()

# Get the executable.
FIND_PROGRAM(CLANG_FORMAT_EXECUTABLE NAMES clang-format DOC "clang-format command line tool")
IF(DEFINED ENV{ZEROSLAM_REQUIRE_CLANG_FORMAT})
    SET(CLANG_FORMAT_MISSING_MESSAGE FATAL_ERROR)
ELSE()
    SET(CLANG_FORMAT_MISSING_MESSAGE STATUS)
ENDIF()
IF(NOT CLANG_FORMAT_EXECUTABLE)
    MESSAGE(${CLANG_FORMAT_MISSING_MESSAGE} "Failed to find clang-format executable, skipping check.")
    RETURN()
ENDIF()

# Get the version.
EXECUTE_PROCESS(
    COMMAND ${CLANG_FORMAT_EXECUTABLE} --version
    OUTPUT_VARIABLE CLANG_FORMAT_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Verify the version.
STRING(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" CLANG_FORMAT_VERSION "${CLANG_FORMAT_VERSION_OUTPUT}")
IF(NOT CLANG_FORMAT_VERSION VERSION_EQUAL "21.1.2")
    MESSAGE(${CLANG_FORMAT_MISSING_MESSAGE} "Failed to find clang-format version 21.1.2, found: ${CLANG_FORMAT_VERSION}.")
    RETURN()
ENDIF()

SET(FORMAT_ERRORS_FOUND 0)

# Check each file is formatted, reporting every file that is not before failing.
FOREACH(SOURCE_FILE ${CLANG_FORMAT_SOURCES})

    # Print progress.
    #MESSAGE(STATUS "Processing '${SOURCE_FILE}'...")

    EXECUTE_PROCESS(
        COMMAND             "${CLANG_FORMAT_EXECUTABLE}" "--dry-run" "-Werror" "-style=file" "${CMAKE_SOURCE_DIR}/${SOURCE_FILE}"
        WORKING_DIRECTORY   "${CMAKE_BINARY_DIR}"
        RESULT_VARIABLE     RETURN_CODE
        ERROR_VARIABLE      FORMAT_RESULT
        OUTPUT_QUIET
    )

    # Check for formatting differences.
    IF(RETURN_CODE)
        MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${SOURCE_FILE}:0 (MESSAGE): clang-format reported:\n${FORMAT_RESULT}")
        # Count up the error messages printed.
        MATH(EXPR FORMAT_ERRORS_FOUND "${FORMAT_ERRORS_FOUND}+1")
    ENDIF()

ENDFOREACH()

IF(${FORMAT_ERRORS_FOUND} GREATER 0)
    MESSAGE(FATAL_ERROR "Found ${FORMAT_ERRORS_FOUND} formatting errors.")
ENDIF()

MESSAGE(STATUS "Finished clang format check.")
