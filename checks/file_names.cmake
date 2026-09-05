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

MESSAGE(STATUS "Checking file names...")

# Get parameters passed from the main CMakeLists.txt.
SET(CMAKE_SOURCE_DIR ${SOURCE_DIR})

# Shared file enumeration, which leaves out every path the .gitignore files exclude.
INCLUDE("${CMAKE_SOURCE_DIR}/checks/project_files.cmake")

# Find all project files.
GET_PROJECT_FILES(PROJECT_FILES)

# Check each file name against the pattern for its directory.
FOREACH(PROJECT_FILE ${PROJECT_FILES})

    # Print progress.
    #MESSAGE(STATUS "Processing '${PROJECT_FILE}'...")

    # Special files.
    IF("${PROJECT_FILE}" MATCHES "^(CMakeLists[.]txt|LICENSE|README[.]md|[.]clang-format|[.]github/.+)$")
        CONTINUE()
    ENDIF()

    # Check specific patterns for specific directories.
    IF("${PROJECT_FILE}" MATCHES "(^|/)[.]gitignore$")
        # Ignore files, any directory may hold one.
        CONTINUE()
    ELSEIF(NOT "${PROJECT_FILE}" MATCHES "/")
        # Any other root file is not allowed.
        MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${PROJECT_FILE}:0 (MESSAGE):")
        MESSAGE(FATAL_ERROR "Found a file in the root directory that is not allowed: '${PROJECT_FILE}'.")
    ELSEIF("${PROJECT_FILE}" MATCHES "^checks/[a-z]+[a-z_0-9]*[.]cmake$")
        # Check files.
        CONTINUE()
    ELSEIF("${PROJECT_FILE}" MATCHES "^include/([a-z]+[a-z_0-9]*/)+[a-z]+[a-z_0-9]*[.](h|hpp|py)$")
        # Public header files.
        CONTINUE()
    ELSEIF("${PROJECT_FILE}" MATCHES "^source/([a-z]+[a-z_0-9]*/)*[a-z]+[a-z_0-9]*[.][ch]pp$")
        # Source files, inside the directory that names their namespace or at the root for the system level.
        CONTINUE()
    ELSEIF("${PROJECT_FILE}" MATCHES "^source/([a-z]+[a-z_0-9]*/)+[a-z]+[a-z_0-9]*[.](avx|avx2|neon)[.]cpp$")
        # Source files holding the simd implementations declared in the header of the same name.
        CONTINUE()
    ELSEIF("${PROJECT_FILE}" MATCHES "^tests/([a-z]+[a-z_0-9]*/)*[a-z]+[a-z_0-9]*[.]test[.]cpp$")
        # Test files.
        CONTINUE()
    ELSEIF("${PROJECT_FILE}" MATCHES "^tools/[a-z]+[-a-z_0-9]*/(CMakeLists[.]txt|[a-z]+[a-z_0-9]*[.]([ch]pp|c|py))$")
        # Tool files, in a directory named after the command so hyphens are allowed, in c++, c, or python.
        CONTINUE()
    ENDIF()

    # Otherwise the name is invalid.
    MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${PROJECT_FILE}:0 (MESSAGE):")
    MESSAGE(FATAL_ERROR "Found invalid file name: '${PROJECT_FILE}'.")

ENDFOREACH()

MESSAGE(STATUS "Finished file name check.")
