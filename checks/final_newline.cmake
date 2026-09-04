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

MESSAGE(STATUS "Checking for final newlines...")

# Get parameters passed from the main CMakeLists.txt.
SET(CMAKE_SOURCE_DIR ${SOURCE_DIR})

# Shared file enumeration, which leaves out every path the .gitignore files exclude.
INCLUDE("${CMAKE_SOURCE_DIR}/checks/project_files.cmake")

# Find all project files.
GET_PROJECT_FILES(PROJECT_FILES)

# Check each file ends with exactly one newline.
FOREACH(PROJECT_FILE ${PROJECT_FILES})

    # Print progress.
    #MESSAGE(STATUS "Processing '${PROJECT_FILE}'...")

    # Get content as hex.
    FILE(READ "${CMAKE_SOURCE_DIR}/${PROJECT_FILE}" PROJECT_FILE_CONTENT HEX)

    # An empty file is fine.
    IF(PROJECT_FILE_CONTENT STREQUAL "")
        CONTINUE()
    ENDIF()

    # The last byte must be a newline (0x0a), and the one before it must not be.
    IF(NOT PROJECT_FILE_CONTENT MATCHES "0a$")
        MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${PROJECT_FILE}:0 (MESSAGE):")
        MESSAGE(FATAL_ERROR "Missing final newline in file '${PROJECT_FILE}'.")
    ENDIF()
    STRING(LENGTH "${PROJECT_FILE_CONTENT}" PROJECT_FILE_LENGTH)
    IF(PROJECT_FILE_LENGTH GREATER 2 AND PROJECT_FILE_CONTENT MATCHES "0a0a$")
        MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${PROJECT_FILE}:0 (MESSAGE):")
        MESSAGE(FATAL_ERROR "Found more than one final newline in file '${PROJECT_FILE}'.")
    ENDIF()

ENDFOREACH()

MESSAGE(STATUS "Finished final newlines check.")
