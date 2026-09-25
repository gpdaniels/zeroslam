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
GET_FILENAME_COMPONENT(TOOL_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
STRING(PREPEND TOOL_NAME "${PROJECT_NAME}-")
STRING(TOUPPER "${TOOL_NAME}" TOOL_NAME_UPPER)
STRING(REPLACE "-" "_" TOOL_NAME_UPPER "${TOOL_NAME_UPPER}")
OPTION(BUILD_${TOOL_NAME_UPPER} "Build the '${TOOL_NAME}' tool." OFF)
IF(NOT BUILD_ALL_TOOLS AND NOT BUILD_${TOOL_NAME_UPPER})
    MESSAGE(STATUS "Skipping '${TOOL_NAME}', define BUILD_${TOOL_NAME_UPPER} or BUILD_ALL_TOOLS as TRUE to enable it.")
    RETURN()
ENDIF()
FILE(GLOB_RECURSE TOOL_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
ADD_CUSTOM_TARGET(${TOOL_NAME} SOURCES ${TOOL_FILES})
SET_TARGET_PROPERTIES(${TOOL_NAME} PROPERTIES FOLDER tools)

FIND_PACKAGE(Python3 QUIET COMPONENTS Interpreter)
IF(NOT Python3_Interpreter_FOUND)
    MESSAGE(STATUS "Skipping the '${TOOL_NAME}' test: no python3 interpreter found.")
    RETURN()
ENDIF()

IF(ZEROSLAM_SANITIZE_ADDRESS OR ZEROSLAM_SANITIZE_UNDEFINED OR ZEROSLAM_SANITIZE_THREAD OR ZEROSLAM_SANITIZE_MEMORY)
    MESSAGE(STATUS "Skipping the '${TOOL_NAME}' test: the sanitizers cannot be loaded into python.")
    RETURN()
ENDIF()

ADD_TEST(NAME build_tool_api_python COMMAND "${CMAKE_COMMAND}" "--build" "${CMAKE_BINARY_DIR}" "--config" "$<CONFIG>" "--target" "${PROJECT_NAME}")
SET_TESTS_PROPERTIES(build_tool_api_python PROPERTIES RESOURCE_LOCK build_tree)
ADD_TEST(NAME tool_api_python COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/main.py")
SET_TESTS_PROPERTIES(tool_api_python PROPERTIES DEPENDS build_tool_api_python)
SET_TESTS_PROPERTIES(tool_api_python PROPERTIES ENVIRONMENT "ZEROSLAM_API_LIBRARY=$<TARGET_FILE:${PROJECT_NAME}>;PYTHONPATH=${CMAKE_SOURCE_DIR}/include/zeroslam;PYTHONDONTWRITEBYTECODE=1")
