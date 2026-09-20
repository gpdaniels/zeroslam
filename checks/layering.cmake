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

MESSAGE(STATUS "Checking the layering of the tools...")

# Get parameters passed from the main CMakeLists.txt.
SET(CMAKE_SOURCE_DIR ${SOURCE_DIR})

# Shared file enumeration, which leaves out every path the .gitignore files exclude.
INCLUDE("${CMAKE_SOURCE_DIR}/checks/project_files.cmake")

# Find all tool files, which may only use the public api in include/.
GET_PROJECT_FILES(TOOL_FILES
    "tools/*.c"
    "tools/*.cpp"
    "tools/*.h"
    "tools/*.hpp"
)

# Find all library headers as include paths.
GET_PROJECT_FILES(LIBRARY_HEADERS
    "source/*.h"
    "source/*.hpp"
)
SET(LIBRARY_INCLUDE_PATHS "")
FOREACH(LIBRARY_HEADER ${LIBRARY_HEADERS})
    STRING(REGEX REPLACE "^source/" "" LIBRARY_INCLUDE_PATH "${LIBRARY_HEADER}")
    LIST(APPEND LIBRARY_INCLUDE_PATHS "${LIBRARY_INCLUDE_PATH}")
ENDFOREACH()

# Check each tool file for an include of a library header.
FOREACH(TOOL_FILE ${TOOL_FILES})

    # Print progress.
    #MESSAGE(STATUS "Processing '${TOOL_FILE}'...")

    # Get content.
    FILE(READ "${CMAKE_SOURCE_DIR}/${TOOL_FILE}" TOOL_FILE_CONTENT)

    # Find the include directives.
    STRING(REGEX MATCHALL "#[ \t]*include[ \t]*[<\"][^>\"]+[>\"]" INCLUDE_DIRECTIVES "${TOOL_FILE_CONTENT}")
    FOREACH(INCLUDE_DIRECTIVE ${INCLUDE_DIRECTIVES})
        STRING(REGEX REPLACE "#[ \t]*include[ \t]*[<\"]([^>\"]+)[>\"]" "\\1" INCLUDE_PATH "${INCLUDE_DIRECTIVE}")
        IF(INCLUDE_PATH IN_LIST LIBRARY_INCLUDE_PATHS)
            MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${TOOL_FILE}:0 (MESSAGE):")
            MESSAGE(FATAL_ERROR "Detected an include of the library header '${INCLUDE_PATH}' in '${TOOL_FILE}', the tools only use the public api in include/zeroslam/.")
        ENDIF()
    ENDFOREACH()

ENDFOREACH()

MESSAGE(STATUS "Finished layering check.")
