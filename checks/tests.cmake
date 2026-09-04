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

MESSAGE(STATUS "Checking for tests...")

# Get parameters passed from the main CMakeLists.txt.
SET(CMAKE_SOURCE_DIR ${SOURCE_DIR})

# Shared file enumeration, which leaves out every path the .gitignore files exclude.
INCLUDE("${CMAKE_SOURCE_DIR}/checks/project_files.cmake")

# Find all source files and test files.
GET_PROJECT_FILES(SOURCE_FILES
    "source/*.hpp"
)
GET_PROJECT_FILES(TEST_FILES
    "tests/*.test.cpp"
)

# Check if each source file has a test file.
FOREACH(SOURCE_FILE ${SOURCE_FILES})

    # Print progress.
    #MESSAGE(STATUS "Processing '${SOURCE_FILE}'...")

    # The source file 'source/.../name.hpp' is tested by 'tests/.../name.test.cpp'.
    STRING(REGEX REPLACE "^source/(.*)[.]hpp$" "tests/\\1.test.cpp" TEST_FILE "${SOURCE_FILE}")

    IF(NOT "${TEST_FILE}" IN_LIST TEST_FILES)
        MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${SOURCE_FILE}:0 (MESSAGE):")
        MESSAGE(FATAL_ERROR "Failed to find test file for '${SOURCE_FILE}'.")
    ENDIF()

    LIST(REMOVE_ITEM TEST_FILES "${TEST_FILE}")

ENDFOREACH()

IF(TEST_FILES)
    MESSAGE(FATAL_ERROR "Found more tests than source files: '${TEST_FILES}'")
ENDIF()

MESSAGE(STATUS "Finished test check.")
