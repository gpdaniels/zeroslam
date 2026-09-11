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

MESSAGE(STATUS "Checking raw allocations...")

# Get parameters passed from the main CMakeLists.txt.
SET(CMAKE_SOURCE_DIR ${SOURCE_DIR})

# Shared file enumeration, which leaves out every path the .gitignore files exclude.
INCLUDE("${CMAKE_SOURCE_DIR}/checks/project_files.cmake")

# Find all public header and library source files.
GET_PROJECT_FILES(SOURCE_FILES
    "include/*.h"
    "include/*.hpp"
    "source/*.cpp"
    "source/*.hpp"
)

SET(ALLOCATION_REGEX "[^A-Za-z0-9_:](new[ ]*[(]|new[ ]+[A-Za-z_][A-Za-z_0-9:<>, ]*[ ]*[[({]|delete[ ]*[[]|delete[ ]+[A-Za-z_(*]|malloc[(]|calloc[(]|realloc[(]|free[(])")
SET(PLACEMENT_REGEX "new[ ]*[(].*,[ ]*static_cast<[^>]*>[(]nullptr[)][)]")
SET(OPERATOR_REGEX "operator[ ]+(new|delete)")

# Check each file for raw allocations, these are reported but do not fail the check.
FOREACH(SOURCE_FILE ${SOURCE_FILES})

    # Print progress.
    #MESSAGE(STATUS "Processing '${SOURCE_FILE}'...")

    # Get content, with the statement separators removed so a line is never split into list items.
    FILE(READ "${CMAKE_SOURCE_DIR}/${SOURCE_FILE}" SOURCE_FILE_CONTENT)
    STRING(REPLACE ";" "" SOURCE_FILE_CONTENT "${SOURCE_FILE_CONTENT}")

    # Keep the lines with an allocation expression, then drop comments and placement construction.
    STRING(REGEX MATCHALL "[^\n]*${ALLOCATION_REGEX}[^\n]*" SOURCE_FILE_LINES "${SOURCE_FILE_CONTENT}")
    LIST(FILTER SOURCE_FILE_LINES EXCLUDE REGEX "^[ ]*//")
    LIST(FILTER SOURCE_FILE_LINES EXCLUDE REGEX "${PLACEMENT_REGEX}")
    LIST(FILTER SOURCE_FILE_LINES EXCLUDE REGEX "${OPERATOR_REGEX}")

    # Print a message for each found allocation.
    FOREACH(SOURCE_FILE_LINE ${SOURCE_FILE_LINES})
        STRING(STRIP "${SOURCE_FILE_LINE}" SOURCE_FILE_LINE)
        # TODO: Reported without failing the check for now.
        MESSAGE("CMake Warning at ${CMAKE_SOURCE_DIR}/${SOURCE_FILE}:0 (MESSAGE):")
        MESSAGE(WARNING "Found a raw allocation/deallocation: '${SOURCE_FILE_LINE}'.")
    ENDFOREACH()

ENDFOREACH()

MESSAGE(STATUS "Finished raw allocations check.")
