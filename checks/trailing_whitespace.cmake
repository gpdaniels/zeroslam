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

MESSAGE(STATUS "Checking for trailing whitespace...")

# Get parameters passed from the main CMakeLists.txt.
SET(CMAKE_SOURCE_DIR ${SOURCE_DIR})

# Shared file enumeration, which leaves out every path the .gitignore files exclude.
INCLUDE("${CMAKE_SOURCE_DIR}/checks/project_files.cmake")

# Find all project files.
GET_PROJECT_FILES(PROJECT_FILES)

# Check each file for trailing whitespace.
FOREACH(PROJECT_FILE ${PROJECT_FILES})

    # Print progress.
    #MESSAGE(STATUS "Processing '${PROJECT_FILE}'...")

    # Get content.
    FILE(READ "${CMAKE_SOURCE_DIR}/${PROJECT_FILE}" PROJECT_FILE_CONTENT)

    # Add a trailing newline so a marker on an unterminated last line is checked too.
    SET(PROJECT_FILE_CONTENT "${PROJECT_FILE_CONTENT}\n")

    # Search for a space or tab directly before a line end.
    IF(PROJECT_FILE_CONTENT MATCHES "[ \t]\n")
        MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${PROJECT_FILE}:0 (MESSAGE):")
        MESSAGE(FATAL_ERROR "Found trailing whitespace in file '${PROJECT_FILE}'.")
    ENDIF()

ENDFOREACH()

MESSAGE(STATUS "Finished trailing whitespace check.")
