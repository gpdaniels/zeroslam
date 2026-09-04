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

MESSAGE(STATUS "Checking for tabs...")

# Get parameters passed from the main CMakeLists.txt.
SET(CMAKE_SOURCE_DIR ${SOURCE_DIR})

# Shared file enumeration, which leaves out every path the .gitignore files exclude.
INCLUDE("${CMAKE_SOURCE_DIR}/checks/project_files.cmake")

# Find all project files.
GET_PROJECT_FILES(PROJECT_FILES)

# Check each file for tabs.
FOREACH(PROJECT_FILE ${PROJECT_FILES})

    # Print progress.
    #MESSAGE(STATUS "Processing '${PROJECT_FILE}'...")

    # Get content.
    FILE(READ "${CMAKE_SOURCE_DIR}/${PROJECT_FILE}" PROJECT_FILE_CONTENT)

    # Search for tab characters.
    STRING(FIND "${PROJECT_FILE_CONTENT}" "\t" FOUND_TAB_LOCATION)

    # If any are found the match location will be something other than -1.
    IF(NOT "${FOUND_TAB_LOCATION}" MATCHES "-1")
        MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${PROJECT_FILE}:0 (MESSAGE):")
        MESSAGE(FATAL_ERROR "Found a tab in file '${PROJECT_FILE}'.")
    ENDIF()

ENDFOREACH()

MESSAGE(STATUS "Finished tab check.")
