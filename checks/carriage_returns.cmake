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

MESSAGE(STATUS "Checking for carriage returns...")

# Get parameters passed from the main CMakeLists.txt.
SET(CMAKE_SOURCE_DIR ${SOURCE_DIR})

# Shared file enumeration, which leaves out every path the .gitignore files exclude.
INCLUDE("${CMAKE_SOURCE_DIR}/checks/project_files.cmake")

# Find all project files.
GET_PROJECT_FILES(PROJECT_FILES)

IF(CMAKE_HOST_WIN32)
    # Git converts line endings on checkout on windows, so every file would fail there.
    MESSAGE(STATUS "Skipping carriage returns check on windows host.")
    RETURN()
ENDIF()

# Check each file for carriage returns.
FOREACH(PROJECT_FILE ${PROJECT_FILES})

    # Print progress.
    #MESSAGE(STATUS "Processing '${PROJECT_FILE}'...")

    # Get content as hex.
    FILE(READ "${CMAKE_SOURCE_DIR}/${PROJECT_FILE}" PROJECT_FILE_CONTENT HEX)

    # Search for the carriage return hex bytes. Must match at even offset (each byte is two chars don't match overlaps).
    SET(FOUND_CARRIAGE_RETURN FALSE)
    STRING(FIND "${PROJECT_FILE_CONTENT}" "0d" FOUND_CARRIAGE_RETURN_LOCATION)
    WHILE(NOT FOUND_CARRIAGE_RETURN_LOCATION EQUAL -1)
        MATH(EXPR FOUND_CARRIAGE_RETURN_PARITY "${FOUND_CARRIAGE_RETURN_LOCATION} % 2")
        IF(FOUND_CARRIAGE_RETURN_PARITY EQUAL 0)
            SET(FOUND_CARRIAGE_RETURN TRUE)
            BREAK()
        ENDIF()
        MATH(EXPR FOUND_CARRIAGE_RETURN_LOCATION "${FOUND_CARRIAGE_RETURN_LOCATION} + 1")
        STRING(SUBSTRING "${PROJECT_FILE_CONTENT}" ${FOUND_CARRIAGE_RETURN_LOCATION} -1 PROJECT_FILE_CONTENT)
        STRING(FIND "${PROJECT_FILE_CONTENT}" "0d" FOUND_CARRIAGE_RETURN_LOCATION)
    ENDWHILE()

    # Fail on the first file with a carriage return.
    IF(FOUND_CARRIAGE_RETURN)
        MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${PROJECT_FILE}:0 (MESSAGE):")
        MESSAGE(FATAL_ERROR "Found a carriage return in file '${PROJECT_FILE}'.")
    ENDIF()

ENDFOREACH()

MESSAGE(STATUS "Finished carriage return check.")
