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
###############################################################################
# DESKTOP
###############################################################################
# The parts of the app that need no phone, the camera model and the session that runs the system, are built on a desktop
# into a test that feeds them a rendered corridor.
SET(SHARED_FILES "${CMAKE_CURRENT_SOURCE_DIR}/camera_model.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/session.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/session.cpp")
IF(NOT ANDROID)
    ADD_EXECUTABLE(${TOOL_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp" ${SHARED_FILES})
    TARGET_INCLUDE_DIRECTORIES(${TOOL_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/")
    TARGET_LINK_LIBRARIES(${TOOL_NAME} PRIVATE zeroslam zeroslam-common Threads::Threads)
    SET_TARGET_PROPERTIES(${TOOL_NAME} PROPERTIES FOLDER tools)

    SET(TOOL_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/output")
    ADD_TEST(NAME build_tool_android COMMAND "${CMAKE_COMMAND}" "--build" "${CMAKE_BINARY_DIR}" "--config" "$<CONFIG>" "--target" "${TOOL_NAME}")
    SET_TESTS_PROPERTIES(build_tool_android PROPERTIES RESOURCE_LOCK build_tree)
    ADD_TEST(NAME clean_tool_android COMMAND "${CMAKE_COMMAND}" "-E" "rm" "-rf" "${TOOL_OUTPUT_DIRECTORY}")
    SET_TESTS_PROPERTIES(clean_tool_android PROPERTIES FIXTURES_SETUP tool_android_output)
    ADD_TEST(NAME tool_android COMMAND $<TARGET_FILE:${TOOL_NAME}> "${TOOL_OUTPUT_DIRECTORY}")
    SET_TESTS_PROPERTIES(tool_android PROPERTIES DEPENDS build_tool_android FIXTURES_REQUIRED tool_android_output)
    RETURN()
ENDIF()
###############################################################################
# APP
###############################################################################
# The app is a native activity, with no java: it runs the camera through the ndk's camera2 api, draws with opengl es, and is
# packaged by the sdk's build tools rather than gradle.
# The api level the app is built for, which the ndk's legacy toolchain file gives in ANDROID_PLATFORM_LEVEL and its newer one
# in CMAKE_SYSTEM_VERSION.
SET(MINIMUM_SDK_VERSION "${ANDROID_PLATFORM_LEVEL}")
IF(NOT MINIMUM_SDK_VERSION)
    SET(MINIMUM_SDK_VERSION "${CMAKE_SYSTEM_VERSION}")
ENDIF()
IF(NOT MINIMUM_SDK_VERSION OR (MINIMUM_SDK_VERSION LESS 24))
    MESSAGE(FATAL_ERROR "The app needs the ndk camera of android 7 (api level 24) or later, the platform is '${MINIMUM_SDK_VERSION}': configure with -DANDROID_PLATFORM=android-24 or later.")
ENDIF()
SET(ZEROSLAM_ANDROID_NDK "${CMAKE_ANDROID_NDK}")
IF(NOT ZEROSLAM_ANDROID_NDK)
    SET(ZEROSLAM_ANDROID_NDK "${ANDROID_NDK}")
ENDIF()
SET(GLUE_SOURCE "${ZEROSLAM_ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c")
IF(NOT EXISTS "${GLUE_SOURCE}")
    MESSAGE(FATAL_ERROR "Failed to find the ndk's native app glue at '${GLUE_SOURCE}'.")
ENDIF()
# The ndk's own code, compiled as it is.
ADD_LIBRARY(${TOOL_NAME}-glue STATIC "${GLUE_SOURCE}")
TARGET_INCLUDE_DIRECTORIES(${TOOL_NAME}-glue PUBLIC "${ZEROSLAM_ANDROID_NDK}/sources/android/native_app_glue/")
SET_SOURCE_FILES_PROPERTIES("${GLUE_SOURCE}" PROPERTIES COMPILE_OPTIONS "-w")
SET_TARGET_PROPERTIES(${TOOL_NAME}-glue PROPERTIES FOLDER tools)

FILE(GLOB APP_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
LIST(REMOVE_ITEM APP_FILES "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp")
ADD_LIBRARY(${TOOL_NAME} SHARED ${APP_FILES})
TARGET_INCLUDE_DIRECTORIES(${TOOL_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/")
TARGET_LINK_LIBRARIES(${TOOL_NAME} PRIVATE zeroslam zeroslam-common ${TOOL_NAME}-glue Threads::Threads android log camera2ndk mediandk EGL GLESv2)
# The activity finds the glue's entry point by name, so it is kept although nothing in the library calls it.
TARGET_LINK_OPTIONS(${TOOL_NAME} PRIVATE "-u" "ANativeActivity_onCreate")
# The shared tool code stays inside the library.
TARGET_LINK_OPTIONS(${TOOL_NAME} PRIVATE "-Wl,--exclude-libs,libzeroslam-common.a")
SET_TARGET_PROPERTIES(${TOOL_NAME} PROPERTIES FOLDER tools)
###############################################################################
# PACKAGE
###############################################################################
SET(ZEROSLAM_ANDROID_PACKAGE "com.gpdaniels.zeroslam" CACHE STRING "The app's package name.")
SET(ZEROSLAM_ANDROID_SDK "" CACHE PATH "The android sdk whose build tools and platform package the app, by default ANDROID_HOME, ANDROID_SDK_ROOT, or the sdk the ndk is in.")
# From api level 35 an app is drawn under the system bars whatever it asks, and from 36 a large screen turns it whatever it
# asks, neither of which the app needs, so it targets 34 unless asked otherwise.
SET(ZEROSLAM_ANDROID_TARGET_SDK "34" CACHE STRING "The api level the app targets, at most the newest platform of the sdk.")
SET(ZEROSLAM_ANDROID_KEYSTORE "" CACHE FILEPATH "The keystore to sign the app with, by default ~/.android/debug.keystore when it exists and otherwise one made in the build directory.")
SET(ANDROID_SDK_DIRECTORY "${ZEROSLAM_ANDROID_SDK}")
FOREACH(CANDIDATE "$ENV{ANDROID_HOME}" "$ENV{ANDROID_SDK_ROOT}" "${ZEROSLAM_ANDROID_NDK}/../..")
    IF(NOT ANDROID_SDK_DIRECTORY AND CANDIDATE AND IS_DIRECTORY "${CANDIDATE}/build-tools")
        GET_FILENAME_COMPONENT(ANDROID_SDK_DIRECTORY "${CANDIDATE}" ABSOLUTE)
    ENDIF()
ENDFOREACH()
# The newest build tools and platform the sdk has.
FILE(GLOB BUILD_TOOLS_DIRECTORIES "${ANDROID_SDK_DIRECTORY}/build-tools/*")
LIST(SORT BUILD_TOOLS_DIRECTORIES COMPARE NATURAL ORDER DESCENDING)
LIST(GET BUILD_TOOLS_DIRECTORIES 0 BUILD_TOOLS_DIRECTORY)
FILE(GLOB PLATFORM_JARS "${ANDROID_SDK_DIRECTORY}/platforms/android-*/android.jar")
LIST(SORT PLATFORM_JARS COMPARE NATURAL ORDER DESCENDING)
LIST(GET PLATFORM_JARS 0 PLATFORM_JAR)
FIND_PROGRAM(AAPT2_EXECUTABLE NAMES aapt2 PATHS "${BUILD_TOOLS_DIRECTORY}" NO_DEFAULT_PATH)
FIND_PROGRAM(ZIPALIGN_EXECUTABLE NAMES zipalign PATHS "${BUILD_TOOLS_DIRECTORY}" NO_DEFAULT_PATH)
FIND_PROGRAM(APKSIGNER_EXECUTABLE NAMES apksigner apksigner.bat PATHS "${BUILD_TOOLS_DIRECTORY}" NO_DEFAULT_PATH)
FIND_PROGRAM(ADB_EXECUTABLE NAMES adb PATHS "${ANDROID_SDK_DIRECTORY}/platform-tools" NO_DEFAULT_PATH)
FIND_PACKAGE(Java COMPONENTS Development)
GET_FILENAME_COMPONENT(JAVA_BINARY_DIRECTORY "${Java_JAVA_EXECUTABLE}" DIRECTORY)
FIND_PROGRAM(KEYTOOL_EXECUTABLE NAMES keytool PATHS "${JAVA_BINARY_DIRECTORY}")
IF(NOT AAPT2_EXECUTABLE OR NOT ZIPALIGN_EXECUTABLE OR NOT APKSIGNER_EXECUTABLE OR NOT PLATFORM_JAR OR NOT Java_JAR_EXECUTABLE OR NOT KEYTOOL_EXECUTABLE)
    MESSAGE(STATUS "Building '${TOOL_NAME}' without its package: the android sdk's build tools and a platform (from '${ANDROID_SDK_DIRECTORY}', set ZEROSLAM_ANDROID_SDK), and a java development kit, package the app.")
    RETURN()
ENDIF()
STRING(REGEX REPLACE "^.*/android-([0-9]+)[^/]*/android.jar$" "\\1" PLATFORM_SDK_VERSION "${PLATFORM_JAR}")
SET(TARGET_SDK_VERSION "${ZEROSLAM_ANDROID_TARGET_SDK}")
IF(TARGET_SDK_VERSION GREATER PLATFORM_SDK_VERSION)
    SET(TARGET_SDK_VERSION "${PLATFORM_SDK_VERSION}")
ENDIF()
GET_FILENAME_COMPONENT(BUILD_TOOLS_VERSION "${BUILD_TOOLS_DIRECTORY}" NAME)
MESSAGE(STATUS "Packaging '${TOOL_NAME}' with the build tools ${BUILD_TOOLS_VERSION} for api levels ${MINIMUM_SDK_VERSION} to ${TARGET_SDK_VERSION} against the platform of ${PLATFORM_SDK_VERSION}.")

SET(APK_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/apk")
SET(APK_FILE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TOOL_NAME}.apk")
# The manifest of a native activity, which loads the library named here and holds no code of its own.
FILE(CONFIGURE OUTPUT "${APK_DIRECTORY}/AndroidManifest.xml" @ONLY CONTENT [=[<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="@ZEROSLAM_ANDROID_PACKAGE@" android:versionCode="1" android:versionName="1.0">
    <uses-feature android:name="android.hardware.camera.any" android:required="true"/>
    <uses-feature android:glEsVersion="0x00020000" android:required="true"/>
    <uses-permission android:name="android.permission.CAMERA"/>
    <application android:label="ZeroSLAM" android:icon="@mipmap/icon" android:hasCode="false" android:extractNativeLibs="false" android:theme="@android:style/Theme.NoTitleBar.Fullscreen">
        <activity android:name="android.app.NativeActivity" android:exported="true" android:launchMode="singleTask" android:screenOrientation="sensorLandscape" android:configChanges="orientation|screenSize|screenLayout|smallestScreenSize|keyboard|keyboardHidden|navigation|density|uiMode">
            <meta-data android:name="android.app.lib_name" android:value="@TOOL_NAME@"/>
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
    </application>
</manifest>
]=])
CONFIGURE_FILE("${CMAKE_SOURCE_DIR}/icons/logo.png" "${APK_DIRECTORY}/res/mipmap-xxxhdpi/icon.png" COPYONLY)

SET(KEYSTORE "${ZEROSLAM_ANDROID_KEYSTORE}")
IF(NOT KEYSTORE AND EXISTS "$ENV{HOME}/.android/debug.keystore")
    SET(KEYSTORE "$ENV{HOME}/.android/debug.keystore")
ENDIF()
IF(NOT KEYSTORE)
    SET(KEYSTORE "${APK_DIRECTORY}/debug.keystore")
    ADD_CUSTOM_COMMAND(
        OUTPUT "${KEYSTORE}"
        COMMAND "${KEYTOOL_EXECUTABLE}" "-genkeypair" "-keystore" "${KEYSTORE}" "-storepass" "android" "-alias" "androiddebugkey" "-keypass" "android" "-keyalg" "RSA" "-keysize" "2048" "-validity" "10000" "-dname" "CN=Android Debug,O=Android,C=US"
        COMMENT "Making a debug keystore to sign ${TOOL_NAME} with..."
        VERBATIM
    )
ENDIF()

# The libraries go in uncompressed and page aligned, so that they are loaded straight from the package, and stripped of
# what loading them does not need. The shared c++ library goes in with them when the build uses it.
SET(LIBRARY_DIRECTORY "lib/${ANDROID_ABI}")
SET(STAGE_COMMANDS
    COMMAND "${CMAKE_COMMAND}" "-E" "rm" "-rf" "${APK_DIRECTORY}/stage"
    COMMAND "${CMAKE_COMMAND}" "-E" "make_directory" "${APK_DIRECTORY}/stage/${LIBRARY_DIRECTORY}"
    COMMAND "${CMAKE_STRIP}" "--strip-unneeded" "-o" "${APK_DIRECTORY}/stage/${LIBRARY_DIRECTORY}/$<TARGET_FILE_NAME:${TOOL_NAME}>" "$<TARGET_FILE:${TOOL_NAME}>"
    COMMAND "${CMAKE_STRIP}" "--strip-unneeded" "-o" "${APK_DIRECTORY}/stage/${LIBRARY_DIRECTORY}/$<TARGET_FILE_NAME:zeroslam>" "$<TARGET_FILE:zeroslam>"
)
IF(ANDROID_STL STREQUAL "c++_shared")
    SET(STL_TRIPLE_arm64-v8a "aarch64-linux-android")
    SET(STL_TRIPLE_armeabi-v7a "arm-linux-androideabi")
    SET(STL_TRIPLE_x86_64 "x86_64-linux-android")
    SET(STL_TRIPLE_x86 "i686-linux-android")
    FILE(GLOB SHARED_STL "${ZEROSLAM_ANDROID_NDK}/toolchains/llvm/prebuilt/*/sysroot/usr/lib/${STL_TRIPLE_${ANDROID_ABI}}/libc++_shared.so")
    IF(NOT SHARED_STL)
        MESSAGE(FATAL_ERROR "Failed to find libc++_shared.so for '${ANDROID_ABI}' in the ndk.")
    ENDIF()
    LIST(GET SHARED_STL 0 SHARED_STL)
    LIST(APPEND STAGE_COMMANDS COMMAND "${CMAKE_COMMAND}" "-E" "copy" "${SHARED_STL}" "${APK_DIRECTORY}/stage/${LIBRARY_DIRECTORY}/")
ENDIF()
# Build tools 35 and later align the libraries to 16 KiB pages, which devices with 16 KiB pages need, earlier ones to 4 KiB.
SET(ZIPALIGN_PAGE_FLAGS "-p")
IF(BUILD_TOOLS_VERSION VERSION_GREATER_EQUAL 35)
    SET(ZIPALIGN_PAGE_FLAGS "-P" "16")
ENDIF()
ADD_CUSTOM_COMMAND(
    OUTPUT "${APK_FILE}"
    COMMAND "${CMAKE_COMMAND}" "-E" "rm" "-rf" "${APK_DIRECTORY}/compiled" "${APK_DIRECTORY}/unaligned.apk" "${APK_DIRECTORY}/unsigned.apk"
    COMMAND "${CMAKE_COMMAND}" "-E" "make_directory" "${APK_DIRECTORY}/compiled"
    COMMAND "${AAPT2_EXECUTABLE}" "compile" "-o" "${APK_DIRECTORY}/compiled" "${APK_DIRECTORY}/res/mipmap-xxxhdpi/icon.png"
    COMMAND "${AAPT2_EXECUTABLE}" "link" "-o" "${APK_DIRECTORY}/unaligned.apk" "-I" "${PLATFORM_JAR}" "--manifest" "${APK_DIRECTORY}/AndroidManifest.xml" "--min-sdk-version" "${MINIMUM_SDK_VERSION}" "--target-sdk-version" "${TARGET_SDK_VERSION}" "${APK_DIRECTORY}/compiled/mipmap-xxxhdpi_icon.png.flat"
    ${STAGE_COMMANDS}
    COMMAND "${Java_JAR_EXECUTABLE}" "uf0M" "${APK_DIRECTORY}/unaligned.apk" "-C" "${APK_DIRECTORY}/stage" "lib"
    COMMAND "${ZIPALIGN_EXECUTABLE}" "-f" ${ZIPALIGN_PAGE_FLAGS} "4" "${APK_DIRECTORY}/unaligned.apk" "${APK_DIRECTORY}/unsigned.apk"
    COMMAND "${APKSIGNER_EXECUTABLE}" "sign" "--ks" "${KEYSTORE}" "--ks-pass" "pass:android" "--ks-key-alias" "androiddebugkey" "--key-pass" "pass:android" "--out" "${APK_FILE}" "${APK_DIRECTORY}/unsigned.apk"
    DEPENDS ${TOOL_NAME} zeroslam "${APK_DIRECTORY}/AndroidManifest.xml" "${APK_DIRECTORY}/res/mipmap-xxxhdpi/icon.png" "${KEYSTORE}"
    COMMENT "Packaging ${APK_FILE}..."
    VERBATIM
)
ADD_CUSTOM_TARGET(${TOOL_NAME}-package ALL DEPENDS "${APK_FILE}")
SET_TARGET_PROPERTIES(${TOOL_NAME}-package PROPERTIES FOLDER tools)
IF(ADB_EXECUTABLE)
    # Install the package on the connected device, and start the app.
    ADD_CUSTOM_TARGET(${TOOL_NAME}-install
        COMMAND "${ADB_EXECUTABLE}" "install" "-r" "${APK_FILE}"
        DEPENDS "${APK_FILE}"
        VERBATIM
    )
    ADD_CUSTOM_TARGET(${TOOL_NAME}-run
        COMMAND "${ADB_EXECUTABLE}" "shell" "am" "start" "-n" "${ZEROSLAM_ANDROID_PACKAGE}/android.app.NativeActivity"
        DEPENDS ${TOOL_NAME}-install
        VERBATIM
    )
ENDIF()
