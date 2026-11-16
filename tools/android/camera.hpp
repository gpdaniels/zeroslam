/*
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
*/

#pragma once
#ifndef ZEROSLAM_TOOLS_ANDROID_CAMERA_HPP
#define ZEROSLAM_TOOLS_ANDROID_CAMERA_HPP

#include "camera_model.hpp"
#include "session.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <atomic>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <functional>
#include <media/NdkImageReader.h>
#include <mutex>
#include <string>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mobile {
    // How the camera is run, from the app's configuration.
    struct camera_settings final {
        // The stream closest to this width with the sensor's aspect ratio. A quarter of vga keeps a phone near the camera's
        // rate, and on TUM RGB-D fr1_desk, halved, the intensity flow tracked better than at the full size (2.9 cm of
        // absolute error against 4.0 cm) in half the time.
        int preferred_width = 320;
        // A fixed focus in dioptres (1 / metres), or continuous autofocus when negative. A fixed focus keeps the intrinsics
        // fixed, which the system assumes, and one dioptre keeps from about half a metre to a few metres sharp.
        double focus_dioptres = 1.0;
    };

    // The first back facing camera (or the first camera, when none faces back), streaming its luminance through the ndk's
    // camera2 api. The geometry is held still for the system: distortion correction, optical and video stabilisation are
    // off, and the crop is the whole array.
    class camera final {
    public:
        using frame_callback = std::function<void(const frame_view&)>;

        camera();
        ~camera();

        camera(const camera&) = delete;
        camera& operator=(const camera&) = delete;

        bool open(const camera_settings& settings, std::string& error);
        void close();
        bool is_open() const;

        // Streams into the callback, called on the camera's thread, until stopped.
        bool start(frame_callback callback, std::string& error);
        void stop();

        // The library's camera for the stream, once known: from the characteristics, or from the first capture result for
        // a device that reports its calibration only there, or estimated from the lens when it reports none.
        bool get_parameters(zeroslam_sensor_parameters_camera_struct& parameters) const;
        // Where the camera's intrinsics came from, for the display.
        std::string get_description() const;
        // The clockwise turn, in degrees, that shows the sensor's image upright on the device in its natural orientation.
        int get_sensor_orientation() const;
        // Set when the device has been lost to the app, as when another app takes the camera.
        bool has_failed() const;

    private:
        static void on_disconnected(void* context, ACameraDevice* device);
        static void on_error(void* context, ACameraDevice* device, int error);
        static void on_image_available(void* context, AImageReader* reader);
        static void on_capture_completed(void* context, ACameraCaptureSession* session, ACaptureRequest* request, const ACameraMetadata* result);
        static void on_session_closed(void* context, ACameraCaptureSession* session);
        static void on_session_ready(void* context, ACameraCaptureSession* session);
        static void on_session_active(void* context, ACameraCaptureSession* session);

        void resolve_parameters(const ACameraMetadata* result);

        ACameraManager* manager = nullptr;
        ACameraDevice* device = nullptr;
        AImageReader* reader = nullptr;
        ANativeWindow* window = nullptr;
        ACaptureSessionOutputContainer* outputs = nullptr;
        ACaptureSessionOutput* output = nullptr;
        ACameraOutputTarget* target = nullptr;
        ACaptureRequest* request = nullptr;
        ACameraCaptureSession* capture_session = nullptr;
        ACameraDevice_StateCallbacks device_callbacks{};
        ACameraCaptureSession_stateCallbacks session_callbacks{};
        ACameraCaptureSession_captureCallbacks capture_callbacks{};

        std::string identifier;
        camera_settings settings;
        camera_characteristics characteristics;
        stream_size size;
        int sensor_orientation = 0;
        // Whether the device's distortion correction could be turned off, without which its output is corrected and the
        // reported distortion is not the image's.
        bool correction_off = false;

        mutable std::mutex mutex;
        frame_callback callback;
        bool resolved = false;
        zeroslam_sensor_parameters_camera_struct parameters{};
        std::string description;
        std::atomic<bool> failed{ false };
    };
}

#endif // ZEROSLAM_TOOLS_ANDROID_CAMERA_HPP
