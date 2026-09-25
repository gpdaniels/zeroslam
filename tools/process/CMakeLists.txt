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
FILE(GLOB_RECURSE TOOL_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
ADD_EXECUTABLE(${TOOL_NAME} ${TOOL_FILES})
TARGET_INCLUDE_DIRECTORIES(${TOOL_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/")
TARGET_LINK_LIBRARIES(${TOOL_NAME} PRIVATE zeroslam)
TARGET_LINK_LIBRARIES(${TOOL_NAME} PRIVATE zeroslam-common)
TARGET_LINK_LIBRARIES(${TOOL_NAME} PRIVATE Threads::Threads)
SET_TARGET_PROPERTIES(${TOOL_NAME} PROPERTIES FOLDER tools)
