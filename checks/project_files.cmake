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

# This script has two jobs.
# - When included by another check it defines the shared file enumeration functions.
# - When run on its own as a check it is a self test of the file enumeration functions.

# Convert a .gitignore pattern into a regular expression that matches a path relative to the directory of the .gitignore file.
# The pattern has already had its leading '/' removed and been prefixed with '**/' when it is not anchored.
FUNCTION(CONVERT_IGNORE_PATTERN_TO_REGEX PATTERN OUTPUT_VARIABLE)

    SET(REGEX "^")
    STRING(LENGTH "${PATTERN}" PATTERN_LENGTH)
    SET(INDEX 0)
    WHILE(INDEX LESS PATTERN_LENGTH)

        # Get the current character and the one after it.
        STRING(SUBSTRING "${PATTERN}" ${INDEX} 1 CHARACTER)
        MATH(EXPR NEXT_INDEX "${INDEX} + 1")
        STRING(SUBSTRING "${PATTERN}" ${NEXT_INDEX} 1 NEXT_CHARACTER)

        IF(CHARACTER STREQUAL "\\")
            # A backslash escapes the character after it, which is then matched literally.
            IF(NEXT_CHARACTER STREQUAL "")
                STRING(APPEND REGEX "\\\\")
            ELSE()
                IF(NEXT_CHARACTER MATCHES "^[.^$*+?()|]$" OR NEXT_CHARACTER STREQUAL "[" OR NEXT_CHARACTER STREQUAL "]" OR NEXT_CHARACTER STREQUAL "\\")
                    STRING(APPEND REGEX "\\${NEXT_CHARACTER}")
                ELSE()
                    STRING(APPEND REGEX "${NEXT_CHARACTER}")
                ENDIF()
                MATH(EXPR INDEX "${INDEX} + 1")
            ENDIF()
        ELSEIF(CHARACTER STREQUAL "*")
            IF(NEXT_CHARACTER STREQUAL "*")
                MATH(EXPR AFTER_INDEX "${INDEX} + 2")
                STRING(SUBSTRING "${PATTERN}" ${AFTER_INDEX} 1 AFTER_CHARACTER)
                IF(AFTER_CHARACTER STREQUAL "/")
                    # A '**/' matches zero or more directories.
                    STRING(APPEND REGEX "(.*/)?")
                    MATH(EXPR INDEX "${INDEX} + 2")
                ELSEIF(AFTER_CHARACTER STREQUAL "")
                    # A trailing '**' matches everything inside.
                    STRING(APPEND REGEX ".*")
                    MATH(EXPR INDEX "${INDEX} + 1")
                ELSE()
                    # Any other '**' is an ordinary '*'.
                    STRING(APPEND REGEX "[^/]*")
                    MATH(EXPR INDEX "${INDEX} + 1")
                ENDIF()
            ELSE()
                # A '*' matches anything except a directory separator.
                STRING(APPEND REGEX "[^/]*")
            ENDIF()
        ELSEIF(CHARACTER STREQUAL "?")
            # A '?' matches one character except a directory separator.
            STRING(APPEND REGEX "[^/]")
        ELSEIF(CHARACTER STREQUAL "[")
            # A '[...]' is a character class, with '!' rather than '^' for negation. Without a
            # closing bracket it is a literal bracket.
            STRING(SUBSTRING "${PATTERN}" ${NEXT_INDEX} -1 REMAINDER)
            STRING(FIND "${REMAINDER}" "]" CLOSE_OFFSET)
            IF(CLOSE_OFFSET EQUAL -1)
                STRING(APPEND REGEX "\\[")
            ELSE()
                STRING(SUBSTRING "${REMAINDER}" 0 ${CLOSE_OFFSET} CLASS)
                STRING(REGEX REPLACE "^!" "^" CLASS "${CLASS}")
                STRING(APPEND REGEX "[${CLASS}]")
                MATH(EXPR INDEX "${INDEX} + ${CLOSE_OFFSET} + 1")
            ENDIF()
        ELSEIF(CHARACTER MATCHES "^[.^$+()|]$" OR CHARACTER STREQUAL "]")
            # Anything special to a regular expression is matched literally.
            STRING(APPEND REGEX "\\${CHARACTER}")
        ELSE()
            STRING(APPEND REGEX "${CHARACTER}")
        ENDIF()

        MATH(EXPR INDEX "${INDEX} + 1")
    ENDWHILE()
    STRING(APPEND REGEX "$")

    SET(${OUTPUT_VARIABLE} "${REGEX}" PARENT_SCOPE)

ENDFUNCTION()

# Read the patterns of one .gitignore file and append them to the rules in the calling scope.
# Each rule is held in the IGNORE_RULE_<index>_ variables (DIRECTORY, REGEX, NEGATED, DIRECTORY_ONLY).
# IGNORE_RULE_COUNT holds how many rules there are.
# Requires CMAKE_SOURCE_DIR to be set to the source directory.
FUNCTION(READ_IGNORE_RULES IGNORE_FILE)

    # The directory the patterns are scoped to, relative to the source directory, empty for the root.
    GET_FILENAME_COMPONENT(IGNORE_DIRECTORY "${IGNORE_FILE}" DIRECTORY)

    # Read the content and take one line at a time, without a list so that the characters special
    # to a list are kept in the patterns.
    FILE(READ "${CMAKE_SOURCE_DIR}/${IGNORE_FILE}" IGNORE_FILE_CONTENT)
    STRING(REPLACE "\r" "" IGNORE_FILE_CONTENT "${IGNORE_FILE_CONTENT}")
    SET(IGNORE_FILE_CONTENT "${IGNORE_FILE_CONTENT}\n")
    WHILE(NOT IGNORE_FILE_CONTENT STREQUAL "")

        # Split off the next line.
        STRING(FIND "${IGNORE_FILE_CONTENT}" "\n" NEWLINE_OFFSET)
        STRING(SUBSTRING "${IGNORE_FILE_CONTENT}" 0 ${NEWLINE_OFFSET} PATTERN)
        MATH(EXPR NEXT_OFFSET "${NEWLINE_OFFSET} + 1")
        STRING(SUBSTRING "${IGNORE_FILE_CONTENT}" ${NEXT_OFFSET} -1 IGNORE_FILE_CONTENT)

        # Trailing spaces are ignored unless they are escaped with a backslash.
        STRING(REGEX REPLACE "([^\\\\])[ \t]+$" "\\1" PATTERN "${PATTERN}")

        # Blank lines and comments hold no pattern.
        IF(PATTERN STREQUAL "" OR PATTERN MATCHES "^#")
            CONTINUE()
        ENDIF()

        # A leading '!' negates the pattern, re-including what an earlier pattern excluded.
        SET(NEGATED FALSE)
        IF(PATTERN MATCHES "^!")
            SET(NEGATED TRUE)
            STRING(SUBSTRING "${PATTERN}" 1 -1 PATTERN)
        ENDIF()

        # A trailing '/' matches directories only.
        SET(DIRECTORY_ONLY FALSE)
        IF(PATTERN MATCHES "/$")
            SET(DIRECTORY_ONLY TRUE)
            STRING(REGEX REPLACE "/+$" "" PATTERN "${PATTERN}")
        ENDIF()

        # A '/' at the start or in the middle anchors the pattern to the directory of the .gitignore
        # file, otherwise the pattern matches at any depth below it.
        IF(PATTERN MATCHES "^/")
            STRING(SUBSTRING "${PATTERN}" 1 -1 PATTERN)
        ELSEIF(NOT PATTERN MATCHES "/")
            SET(PATTERN "**/${PATTERN}")
        ENDIF()

        # Nothing left to match, for example the pattern was a lone '/'.
        IF(PATTERN STREQUAL "")
            CONTINUE()
        ENDIF()

        # Store the rule.
        CONVERT_IGNORE_PATTERN_TO_REGEX("${PATTERN}" REGEX)
        SET(IGNORE_RULE_${IGNORE_RULE_COUNT}_DIRECTORY "${IGNORE_DIRECTORY}" PARENT_SCOPE)
        SET(IGNORE_RULE_${IGNORE_RULE_COUNT}_REGEX "${REGEX}" PARENT_SCOPE)
        SET(IGNORE_RULE_${IGNORE_RULE_COUNT}_NEGATED ${NEGATED} PARENT_SCOPE)
        SET(IGNORE_RULE_${IGNORE_RULE_COUNT}_DIRECTORY_ONLY ${DIRECTORY_ONLY} PARENT_SCOPE)
        MATH(EXPR IGNORE_RULE_COUNT "${IGNORE_RULE_COUNT} + 1")
    ENDWHILE()

    SET(IGNORE_RULE_COUNT ${IGNORE_RULE_COUNT} PARENT_SCOPE)

ENDFUNCTION()

# Decide whether a path, relative to the source directory, is excluded by the rules in the calling scope.
# The rules are ordered from the shallowest .gitignore file to the deepest and the last matching rule decides.
# This means a deeper file and a later pattern override the ones before.
# The parent directories of the path have already been decided by the walk, so only the path itself is matched here.
FUNCTION(IS_PATH_IGNORED PATH IS_PATH_DIRECTORY OUTPUT_VARIABLE)

    SET(IGNORED FALSE)
    SET(INDEX 0)
    WHILE(INDEX LESS IGNORE_RULE_COUNT)

        # The pattern only applies to paths below the directory of its .gitignore file, and matches
        # against the path relative to that directory.
        SET(IGNORE_DIRECTORY "${IGNORE_RULE_${INDEX}_DIRECTORY}")
        IF(IGNORE_DIRECTORY STREQUAL "")
            SET(RELATIVE_PATH "${PATH}")
        ELSE()
            STRING(LENGTH "${IGNORE_DIRECTORY}/" IGNORE_DIRECTORY_LENGTH)
            STRING(SUBSTRING "${PATH}" 0 ${IGNORE_DIRECTORY_LENGTH} PATH_PREFIX)
            IF(NOT PATH_PREFIX STREQUAL "${IGNORE_DIRECTORY}/")
                MATH(EXPR INDEX "${INDEX} + 1")
                CONTINUE()
            ENDIF()
            STRING(SUBSTRING "${PATH}" ${IGNORE_DIRECTORY_LENGTH} -1 RELATIVE_PATH)
        ENDIF()

        # A directory only pattern does not apply to a file.
        IF(IGNORE_RULE_${INDEX}_DIRECTORY_ONLY AND NOT IS_PATH_DIRECTORY)
            MATH(EXPR INDEX "${INDEX} + 1")
            CONTINUE()
        ENDIF()

        # The last matching rule decides.
        IF(RELATIVE_PATH MATCHES "${IGNORE_RULE_${INDEX}_REGEX}")
            IF(IGNORE_RULE_${INDEX}_NEGATED)
                SET(IGNORED FALSE)
            ELSE()
                SET(IGNORED TRUE)
            ENDIF()
        ENDIF()

        MATH(EXPR INDEX "${INDEX} + 1")
    ENDWHILE()

    SET(${OUTPUT_VARIABLE} ${IGNORED} PARENT_SCOPE)

ENDFUNCTION()

# Find the project files, optionally restricted to the glob patterns given after the output variable.
# Patterns are relative to the project directory and without any the whole tree is returned.
# The returned paths are relative to the project directory and sorted.
# Requires CMAKE_SOURCE_DIR to be set to the source directory, and uses BINARY_DIR when set to leave out a build directory inside the source tree.
FUNCTION(GET_PROJECT_FILES OUTPUT_VARIABLE)

    # The patterns to restrict the result to, if any.
    SET(PATTERNS ${ARGN})

    # The build directory when it is inside the source tree, a build directory that is not named to
    # match the .gitignore patterns is still not part of the project.
    SET(BINARY_DIR_RELATIVE "")
    IF(BINARY_DIR)
        FILE(RELATIVE_PATH BINARY_DIR_RELATIVE "${CMAKE_SOURCE_DIR}" "${BINARY_DIR}")
        IF(BINARY_DIR_RELATIVE MATCHES "^[.][.]")
            SET(BINARY_DIR_RELATIVE "")
        ENDIF()
    ENDIF()

    # Find every .gitignore file in the tree, other than in the git and build directories, and read
    # them from the shallowest to the deepest so that the rules of a deeper file come later and win.
    FILE(GLOB_RECURSE IGNORE_FILES RELATIVE "${CMAKE_SOURCE_DIR}/" "${CMAKE_SOURCE_DIR}/.gitignore")
    LIST(FILTER IGNORE_FILES EXCLUDE REGEX "(^|/)[.]git/")
    IF(BINARY_DIR_RELATIVE)
        LIST(FILTER IGNORE_FILES EXCLUDE REGEX "^${BINARY_DIR_RELATIVE}/")
    ENDIF()
    LIST(SORT IGNORE_FILES)
    SET(IGNORE_RULE_COUNT 0)
    SET(IGNORE_FILES_MAX_DEPTH 0)
    FOREACH(IGNORE_FILE ${IGNORE_FILES})
        STRING(REGEX MATCHALL "/" IGNORE_FILE_SLASHES "${IGNORE_FILE}")
        LIST(LENGTH IGNORE_FILE_SLASHES IGNORE_FILE_DEPTH)
        IF(IGNORE_FILE_DEPTH GREATER IGNORE_FILES_MAX_DEPTH)
            SET(IGNORE_FILES_MAX_DEPTH ${IGNORE_FILE_DEPTH})
        ENDIF()
    ENDFOREACH()
    FOREACH(DEPTH RANGE ${IGNORE_FILES_MAX_DEPTH})
        FOREACH(IGNORE_FILE ${IGNORE_FILES})
            STRING(REGEX MATCHALL "/" IGNORE_FILE_SLASHES "${IGNORE_FILE}")
            LIST(LENGTH IGNORE_FILE_SLASHES IGNORE_FILE_DEPTH)
            IF(IGNORE_FILE_DEPTH EQUAL DEPTH)
                READ_IGNORE_RULES("${IGNORE_FILE}")
            ENDIF()
        ENDFOREACH()
    ENDFOREACH()

    # Walk the tree from the source directory, globbing the entries of each directory and leaving
    # out the ones the rules exclude. An excluded directory is not walked, nothing inside it can be
    # re-included. The git directory is never part of the tree, and a symbolic link is a file.
    SET(PROJECT_FILES "")
    SET(PENDING_DIRECTORIES ".")
    WHILE(PENDING_DIRECTORIES)
        LIST(POP_FRONT PENDING_DIRECTORIES DIRECTORY)
        IF(DIRECTORY STREQUAL ".")
            SET(DIRECTORY_PREFIX "")
        ELSE()
            SET(DIRECTORY_PREFIX "${DIRECTORY}/")
        ENDIF()
        FILE(GLOB ENTRIES RELATIVE "${CMAKE_SOURCE_DIR}/${DIRECTORY}" LIST_DIRECTORIES true "${CMAKE_SOURCE_DIR}/${DIRECTORY}/*")
        FOREACH(ENTRY ${ENTRIES})
            SET(ENTRY_PATH "${DIRECTORY_PREFIX}${ENTRY}")
            IF(ENTRY STREQUAL ".git")
                CONTINUE()
            ENDIF()
            IF(BINARY_DIR_RELATIVE AND ENTRY_PATH STREQUAL BINARY_DIR_RELATIVE)
                CONTINUE()
            ENDIF()
            IF(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/${ENTRY_PATH}" AND NOT IS_SYMLINK "${CMAKE_SOURCE_DIR}/${ENTRY_PATH}")
                SET(ENTRY_IS_DIRECTORY TRUE)
            ELSE()
                SET(ENTRY_IS_DIRECTORY FALSE)
            ENDIF()
            IS_PATH_IGNORED("${ENTRY_PATH}" ${ENTRY_IS_DIRECTORY} ENTRY_IGNORED)
            IF(ENTRY_IGNORED)
                CONTINUE()
            ENDIF()
            IF(ENTRY_IS_DIRECTORY)
                LIST(APPEND PENDING_DIRECTORIES "${ENTRY_PATH}")
            ELSE()
                LIST(APPEND PROJECT_FILES "${ENTRY_PATH}")
            ENDIF()
        ENDFOREACH()
    ENDWHILE()

    # Restrict the result to the requested patterns.
    IF(PATTERNS)
        SET(ABSOLUTE_PATTERNS "")
        FOREACH(PATTERN ${PATTERNS})
            LIST(APPEND ABSOLUTE_PATTERNS "${CMAKE_SOURCE_DIR}/${PATTERN}")
        ENDFOREACH()
        FILE(GLOB_RECURSE MATCHED_FILES RELATIVE "${CMAKE_SOURCE_DIR}/" ${ABSOLUTE_PATTERNS})
        SET(SELECTED_FILES "")
        FOREACH(MATCHED_FILE ${MATCHED_FILES})
            IF("${MATCHED_FILE}" IN_LIST PROJECT_FILES)
                LIST(APPEND SELECTED_FILES "${MATCHED_FILE}")
            ENDIF()
        ENDFOREACH()
        SET(PROJECT_FILES ${SELECTED_FILES})
    ENDIF()

    # Sort list of files.
    LIST(SORT PROJECT_FILES)

    SET(${OUTPUT_VARIABLE} ${PROJECT_FILES} PARENT_SCOPE)

ENDFUNCTION()

# The rest of the script is the check, so it runs when this file is the script passed to cmake and
# not when another check includes it for the function above.
GET_FILENAME_COMPONENT(THIS_FILE "${CMAKE_CURRENT_LIST_FILE}" REALPATH)
GET_FILENAME_COMPONENT(SCRIPT_FILE "${CMAKE_SCRIPT_MODE_FILE}" REALPATH)
IF(NOT THIS_FILE STREQUAL SCRIPT_FILE)
    RETURN()
ENDIF()

MESSAGE(STATUS "Checking project files...")

# Get parameters passed from the main CMakeLists.txt.
SET(CMAKE_SOURCE_DIR ${SOURCE_DIR})

# The pattern conversion is the heart of the enumeration, so check it on the forms git documents.
SET(PATTERN_CASES
    "**/foo|foo|TRUE"
    "**/foo|a/b/foo|TRUE"
    "**/foo|a/b/foo/bar|FALSE"
    "foo|a/foo|FALSE"
    "foo/bar|foo/bar|TRUE"
    "foo/**|foo/a/b|TRUE"
    "a/**/b|a/b|TRUE"
    "a/**/b|a/x/y/b|TRUE"
    "**/*.txt|a/b.txt|TRUE"
    "**/*.txt|a/b/txt|FALSE"
    "**/?.txt|a/b.txt|TRUE"
    "**/?.txt|a/bb.txt|FALSE"
    "**/[Bb]uild*|Build-x|TRUE"
    "**/[!a]x|bx|TRUE"
    "**/[!a]x|ax|FALSE"
    "**/\\#*|#tmp|TRUE"
    "**/*~|a/b~|TRUE"
    "**/\$RECYCLE.BIN|\$RECYCLE.BIN|TRUE"
    "**/\$RECYCLE.BIN|\$RECYCLExBIN|FALSE"
    "**/a+b(c)|a+b(c)|TRUE"
)
FOREACH(PATTERN_CASE ${PATTERN_CASES})

    STRING(REPLACE "|" ";" PATTERN_CASE "${PATTERN_CASE}")
    LIST(GET PATTERN_CASE 0 PATTERN)
    LIST(GET PATTERN_CASE 1 PATH)
    LIST(GET PATTERN_CASE 2 EXPECTED)
    CONVERT_IGNORE_PATTERN_TO_REGEX("${PATTERN}" REGEX)
    IF(PATH MATCHES "${REGEX}")
        SET(MATCHED TRUE)
    ELSE()
        SET(MATCHED FALSE)
    ENDIF()
    IF(NOT MATCHED STREQUAL EXPECTED)
        MESSAGE(FATAL_ERROR "Pattern '${PATTERN}' converted to '${REGEX}' matched '${PATH}' as ${MATCHED}, expected ${EXPECTED}.")
    ENDIF()

ENDFOREACH()

# Find all project files.
GET_PROJECT_FILES(PROJECT_FILES)
LIST(LENGTH PROJECT_FILES PROJECT_FILES_LENGTH)
IF(PROJECT_FILES_LENGTH EQUAL 0)
    MESSAGE(FATAL_ERROR "Failed to find any project files in '${CMAKE_SOURCE_DIR}'.")
ENDIF()
MESSAGE(STATUS "Found ${PROJECT_FILES_LENGTH} project files.")

# The build directory holds generated files that no check should look at.
IF(BINARY_DIR)
    FILE(RELATIVE_PATH BINARY_DIR_RELATIVE "${CMAKE_SOURCE_DIR}" "${BINARY_DIR}")
ENDIF()
FOREACH(PROJECT_FILE ${PROJECT_FILES})

    # Print progress.
    #MESSAGE(STATUS "Processing '${PROJECT_FILE}'...")

    # Every path is relative to the source directory and readable, the other checks read them all.
    IF(NOT EXISTS "${CMAKE_SOURCE_DIR}/${PROJECT_FILE}")
        MESSAGE(FATAL_ERROR "Found a project file that does not exist: '${PROJECT_FILE}'.")
    ENDIF()
    IF(IS_ABSOLUTE "${PROJECT_FILE}" OR "${PROJECT_FILE}" MATCHES "(^|/)[.][.](/|$)")
        MESSAGE(FATAL_ERROR "Found a project file that is not a path inside the source directory: '${PROJECT_FILE}'.")
    ENDIF()

    # There should never be a project file in the .git directory.
    IF("${PROJECT_FILE}" MATCHES "^[.]git/")
        MESSAGE(FATAL_ERROR "Found a project file inside the git directory: '${PROJECT_FILE}'.")
    ENDIF()
    IF(BINARY_DIR_RELATIVE AND NOT BINARY_DIR_RELATIVE MATCHES "^[.][.]" AND "${PROJECT_FILE}" MATCHES "^${BINARY_DIR_RELATIVE}/")
        MESSAGE(FATAL_ERROR "Found a project file inside the build directory: '${PROJECT_FILE}'.")
    ENDIF()

ENDFOREACH()

# Validate a pattern restricts the result to the files it matches, and to nothing the whole tree left out.
GET_PROJECT_FILES(SOURCE_FILES
    "source/*.hpp"
)
IF(NOT SOURCE_FILES)
    MESSAGE(FATAL_ERROR "Failed to find any source headers with the pattern 'source/*.hpp'.")
ENDIF()
FOREACH(SOURCE_FILE ${SOURCE_FILES})

    IF(NOT "${SOURCE_FILE}" MATCHES "^source/.*[.]hpp$")
        MESSAGE(FATAL_ERROR "Found a file that does not match the requested pattern: '${SOURCE_FILE}'.")
    ENDIF()
    IF(NOT "${SOURCE_FILE}" IN_LIST PROJECT_FILES)
        MESSAGE(FATAL_ERROR "Found a file that is not one of the project files: '${SOURCE_FILE}'.")
    ENDIF()

ENDFOREACH()

MESSAGE(STATUS "Finished project files check.")
