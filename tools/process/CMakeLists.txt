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
FILE(GLOB_RECURSE TOOL_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
ADD_EXECUTABLE(${TOOL_NAME} ${TOOL_FILES})
TARGET_INCLUDE_DIRECTORIES(${TOOL_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/")
TARGET_LINK_LIBRARIES(${TOOL_NAME} PRIVATE zeroslam)
TARGET_LINK_LIBRARIES(${TOOL_NAME} PRIVATE zeroslam-common)
TARGET_LINK_LIBRARIES(${TOOL_NAME} PRIVATE Threads::Threads)
SET_TARGET_PROPERTIES(${TOOL_NAME} PROPERTIES FOLDER tools)
