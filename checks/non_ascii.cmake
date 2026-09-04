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

MESSAGE(STATUS "Checking for non-ascii characters...")

# Get parameters passed from the main CMakeLists.txt.
SET(CMAKE_SOURCE_DIR ${SOURCE_DIR})

# Shared file enumeration, which leaves out every path the .gitignore files exclude.
INCLUDE("${CMAKE_SOURCE_DIR}/checks/project_files.cmake")

# Find all check, public header, source, test, and tool files.
# The documentation (README.md) is left out as it holds unicode (box drawing characters, units).
GET_PROJECT_FILES(SOURCE_FILES
    "checks/*"
    "include/*"
    "source/*"
    "tests/*"
    "tools/*"
)

# Check each file for bytes outside the ascii range.
FOREACH(SOURCE_FILE ${SOURCE_FILES})

    # Print progress.
    #MESSAGE(STATUS "Processing '${SOURCE_FILE}'...")

    # Get content as hex.
    FILE(READ "${CMAKE_SOURCE_DIR}/${SOURCE_FILE}" SOURCE_FILE_CONTENT HEX)

    # Search for hex bytes greater than or equal to 0x80 by splitting and matching.
    STRING(REGEX MATCHALL "[0-9a-f][0-9a-f]" SOURCE_FILE_BYTES "${SOURCE_FILE_CONTENT}")
    LIST(FILTER SOURCE_FILE_BYTES INCLUDE REGEX "^[89a-f]")

    # Fail on the first file with a non-ascii byte.
    IF(SOURCE_FILE_BYTES)
        LIST(GET SOURCE_FILE_BYTES 0 SOURCE_FILE_BYTE)
        MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${SOURCE_FILE}:0 (MESSAGE):")
        MESSAGE(FATAL_ERROR "Found a non-ascii byte (0x${SOURCE_FILE_BYTE}) in file '${SOURCE_FILE}'.")
    ENDIF()

ENDFOREACH()

MESSAGE(STATUS "Finished non-ascii characters check.")
