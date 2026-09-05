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

MESSAGE(STATUS "Checking namespaces...")

# Get parameters passed from the main CMakeLists.txt.
SET(CMAKE_SOURCE_DIR ${SOURCE_DIR})

# Shared file enumeration, which leaves out every path the .gitignore files exclude.
INCLUDE("${CMAKE_SOURCE_DIR}/checks/project_files.cmake")

# Find all source files.
GET_PROJECT_FILES(SOURCE_FILES
    "source/*.hpp"
    "source/*.cpp"
)

# Check each file opens the namespace named after its directory and no other.
FOREACH(SOURCE_FILE ${SOURCE_FILES})

    # Print progress.
    #MESSAGE(STATUS "Processing '${SOURCE_FILE}'...")

    # Determine the required namespace from the directory below the top level directory.
    STRING(REGEX REPLACE "^source/([a-z_0-9]+)/.*$" "\\1" SOURCE_DIRECTORY "${SOURCE_FILE}")
    IF(SOURCE_DIRECTORY STREQUAL SOURCE_FILE)
        # Files at the root of the source directory tie the others together and belong to no namespace.
        CONTINUE()
    ENDIF()

    # Get content.
    FILE(READ "${CMAKE_SOURCE_DIR}/${SOURCE_FILE}" SOURCE_FILE_CONTENT)

    # Replace special list chars.
    STRING(REGEX REPLACE "([[]|[]])" "\\\\\\1" SOURCE_FILE_CONTENT "${SOURCE_FILE_CONTENT}")

    # Replace newlines.
    STRING(REGEX REPLACE "[\r]?[\n]" ";" SOURCE_FILE_LINES "${SOURCE_FILE_CONTENT}")

    # Anonymous namespaces belong in translation units.
    IF(SOURCE_FILE MATCHES "[.]hpp$")
        SET(ANONYMOUS_NAMESPACE_LINES ${SOURCE_FILE_LINES})
        LIST(FILTER ANONYMOUS_NAMESPACE_LINES INCLUDE REGEX "^[ \t]*(inline[ \t]+)?namespace[ \t]*{")
        IF(ANONYMOUS_NAMESPACE_LINES)
            # TODO: Reported without failing the check for now.
            MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${SOURCE_FILE}:0 (MESSAGE):")
            MESSAGE(STATUS "Header '${SOURCE_FILE}' opens an anonymous namespace.")
        ENDIF()
    ENDIF()

    # Keep the top level namespace lines: unindented, so nested namespaces (indented) are skipped.
    LIST(FILTER SOURCE_FILE_LINES INCLUDE REGEX "^namespace[ ]")

    # Check each opened namespace, and that the directory namespace is opened at least once.
    SET(DIRECTORY_NAMESPACE_OPENED FALSE)
    FOREACH(SOURCE_FILE_LINE ${SOURCE_FILE_LINES})
        STRING(REGEX REPLACE "^namespace[ ]+([A-Za-z_0-9:]*).*$" "\\1" SOURCE_NAMESPACE "${SOURCE_FILE_LINE}")
        # Anonymous namespaces checked above.
        IF(SOURCE_NAMESPACE STREQUAL "")
            CONTINUE()
        ENDIF()
        # Split nested definitions e.g. 'namespace a::b'.
        STRING(REGEX REPLACE "::.*$" "" SOURCE_NAMESPACE_TOP "${SOURCE_NAMESPACE}")
        # Check for directory namespace.
        IF(SOURCE_NAMESPACE_TOP STREQUAL SOURCE_DIRECTORY)
            SET(DIRECTORY_NAMESPACE_OPENED TRUE)
        ELSE()
            MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${SOURCE_FILE}:0 (MESSAGE):")
            MESSAGE(FATAL_ERROR "Namespace '${SOURCE_NAMESPACE}' in file '${SOURCE_FILE}' does not match its directory, expected 'namespace ${SOURCE_DIRECTORY}'.")
        ENDIF()
    ENDFOREACH()
    IF(NOT DIRECTORY_NAMESPACE_OPENED)
        MESSAGE("CMake Error at ${CMAKE_SOURCE_DIR}/${SOURCE_FILE}:0 (MESSAGE):")
        MESSAGE(FATAL_ERROR "File '${SOURCE_FILE}' does not open its namespace, expected 'namespace ${SOURCE_DIRECTORY}'.")
    ENDIF()

ENDFOREACH()

MESSAGE(STATUS "Finished namespaces check.")
