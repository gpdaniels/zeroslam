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

#include "camera.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <camera/NdkCameraMetadata.h>
#include <cstdio>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mobile {
    namespace {
        bool find_entry(const ACameraMetadata* metadata, const unsigned int tag, ACameraMetadata_const_entry& entry) {
            return (ACameraMetadata_getConstEntry(metadata, tag, &entry) == ACAMERA_OK) && (entry.count > 0);
        }

        bool has_byte(const ACameraMetadata* metadata, const unsigned int tag, const int value) {
            ACameraMetadata_const_entry entry{};
            if (!find_entry(metadata, tag, entry)) {
                return false;
            }
            for (unsigned int i = 0; i < entry.count; ++i) {
                if (static_cast<int>(entry.data.u8[i]) == value) {
                    return true;
                }
            }
            return false;
        }

        void set_byte(ACaptureRequest* request, const unsigned int tag, const int value) {
            const unsigned char byte = static_cast<unsigned char>(value);
            ACaptureRequest_setEntry_u8(request, tag, 1, &byte);
        }

        // The intrinsics and distortion of a characteristics or capture result, when it has them.
        void read_calibration(const ACameraMetadata* metadata, camera_characteristics& characteristics) {
            ACameraMetadata_const_entry entry{};
            if (find_entry(metadata, ACAMERA_LENS_INTRINSIC_CALIBRATION, entry) && (entry.count >= 5) && (entry.data.f[0] > 0.0f) && (entry.data.f[1] > 0.0f)) {
                characteristics.calibrated = true;
                for (int i = 0; i < 5; ++i) {
                    characteristics.intrinsics[i] = static_cast<double>(entry.data.f[i]);
                }
            }
            if (find_entry(metadata, ACAMERA_LENS_DISTORTION, entry) && (entry.count >= 5)) {
                characteristics.distorted = true;
                for (int i = 0; i < 5; ++i) {
                    characteristics.distortion[i] = static_cast<double>(entry.data.f[i]);
                }
            }
        }
    }

    camera::camera() {
        this->manager = ACameraManager_create();
    }

    camera::~camera() {
        this->close();
        if (this->manager != nullptr) {
            ACameraManager_delete(this->manager);
        }
    }

    bool camera::open(const camera_settings& settings_opened, std::string& error) {
        this->close();
        this->settings = settings_opened;
        this->failed.store(false);
        if (this->manager == nullptr) {
            error = "Failed to reach the camera service";
            return false;
        }
        ACameraIdList* list = nullptr;
        if ((ACameraManager_getCameraIdList(this->manager, &list) != ACAMERA_OK) || (list == nullptr) || (list->numCameras <= 0)) {
            if (list != nullptr) {
                ACameraManager_deleteCameraIdList(list);
            }
            error = "Found no camera";
            return false;
        }
        // The first camera facing back, which is the main camera on the phones that have several.
        std::string chosen = list->cameraIds[0];
        for (int i = 0; i < list->numCameras; ++i) {
            ACameraMetadata* metadata = nullptr;
            if (ACameraManager_getCameraCharacteristics(this->manager, list->cameraIds[i], &metadata) != ACAMERA_OK) {
                continue;
            }
            const bool back = has_byte(metadata, ACAMERA_LENS_FACING, ACAMERA_LENS_FACING_BACK);
            ACameraMetadata_free(metadata);
            if (back) {
                chosen = list->cameraIds[i];
                break;
            }
        }
        ACameraManager_deleteCameraIdList(list);
        this->identifier = chosen;

        ACameraMetadata* metadata = nullptr;
        if (ACameraManager_getCameraCharacteristics(this->manager, this->identifier.c_str(), &metadata) != ACAMERA_OK) {
            error = "Failed to read the characteristics of camera " + this->identifier;
            return false;
        }
        this->characteristics = camera_characteristics{};
        ACameraMetadata_const_entry entry{};
        // With distortion correction off the stream is cut from the pre-correction array, in whose coordinates the device
        // calibrates, and a device without the control has its active array the same as that array.
        this->correction_off = has_byte(metadata, ACAMERA_DISTORTION_CORRECTION_AVAILABLE_MODES, ACAMERA_DISTORTION_CORRECTION_MODE_OFF);
        const bool correction_controlled = find_entry(metadata, ACAMERA_DISTORTION_CORRECTION_AVAILABLE_MODES, entry);
        if (find_entry(metadata, ACAMERA_SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE, entry) && (entry.count >= 4) && (!correction_controlled || this->correction_off)) {
            this->characteristics.region[2] = static_cast<double>(entry.data.i32[2]);
            this->characteristics.region[3] = static_cast<double>(entry.data.i32[3]);
        }
        else if (find_entry(metadata, ACAMERA_SENSOR_INFO_ACTIVE_ARRAY_SIZE, entry) && (entry.count >= 4)) {
            this->characteristics.region[2] = static_cast<double>(entry.data.i32[2]);
            this->characteristics.region[3] = static_cast<double>(entry.data.i32[3]);
        }
        read_calibration(metadata, this->characteristics);
        if (correction_controlled && !this->correction_off) {
            // The device corrects its output whatever is asked, so its distortion is not the image's.
            this->characteristics.distorted = false;
        }
        if (find_entry(metadata, ACAMERA_LENS_INFO_AVAILABLE_FOCAL_LENGTHS, entry)) {
            this->characteristics.focal_length_millimetres = static_cast<double>(entry.data.f[0]);
        }
        if (find_entry(metadata, ACAMERA_SENSOR_INFO_PHYSICAL_SIZE, entry) && (entry.count >= 2)) {
            this->characteristics.physical_size_millimetres[0] = static_cast<double>(entry.data.f[0]);
            this->characteristics.physical_size_millimetres[1] = static_cast<double>(entry.data.f[1]);
        }
        if (find_entry(metadata, ACAMERA_SENSOR_INFO_PIXEL_ARRAY_SIZE, entry) && (entry.count >= 2)) {
            this->characteristics.pixel_array_size[0] = entry.data.i32[0];
            this->characteristics.pixel_array_size[1] = entry.data.i32[1];
        }
        if (find_entry(metadata, ACAMERA_SENSOR_ORIENTATION, entry)) {
            this->sensor_orientation = entry.data.i32[0];
        }
        std::vector<stream_size> sizes;
        if (find_entry(metadata, ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, entry)) {
            for (unsigned int i = 0; (i + 3) < entry.count; i += 4) {
                if ((entry.data.i32[i] == AIMAGE_FORMAT_YUV_420_888) && (entry.data.i32[i + 3] == ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS_OUTPUT)) {
                    sizes.push_back(stream_size{ entry.data.i32[i + 1], entry.data.i32[i + 2] });
                }
            }
        }
        // The focus is fixed where the lens can be focused and asked to, otherwise left to the device.
        const bool focus_fixed = (this->settings.focus_dioptres >= 0.0) && has_byte(metadata, ACAMERA_CONTROL_AF_AVAILABLE_MODES, ACAMERA_CONTROL_AF_MODE_OFF);
        float minimum_focus_dioptres = 0.0f;
        if (find_entry(metadata, ACAMERA_LENS_INFO_MINIMUM_FOCUS_DISTANCE, entry)) {
            minimum_focus_dioptres = entry.data.f[0];
        }
        const bool optical_stabilisation_off = has_byte(metadata, ACAMERA_LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION, ACAMERA_LENS_OPTICAL_STABILIZATION_MODE_OFF);
        // The frame rate range that reaches 30 frames a second with the highest floor, which keeps the exposures short.
        int frame_rate_range[2] = { 0, 0 };
        if (find_entry(metadata, ACAMERA_CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES, entry)) {
            for (unsigned int i = 0; (i + 1) < entry.count; i += 2) {
                const int low = entry.data.i32[i];
                const int high = entry.data.i32[i + 1];
                if ((high <= 30) && ((high > frame_rate_range[1]) || ((high == frame_rate_range[1]) && (low > frame_rate_range[0])))) {
                    frame_rate_range[0] = low;
                    frame_rate_range[1] = high;
                }
            }
        }
        ACameraMetadata_free(metadata);

        this->size = stream_size{};
        if (!choose_stream_size(sizes, this->characteristics, this->settings.preferred_width, this->size)) {
            error = "Camera " + this->identifier + " has no luminance stream";
            return false;
        }
        if (this->correction_off || !correction_controlled) {
            // The stream is cut from the whole region, in the pre-correction array's coordinates when correction is off.
            this->characteristics.region[0] = 0.0;
            this->characteristics.region[1] = 0.0;
        }

        this->device_callbacks.context = this;
        this->device_callbacks.onDisconnected = &camera::on_disconnected;
        this->device_callbacks.onError = &camera::on_error;
        if (ACameraManager_openCamera(this->manager, this->identifier.c_str(), &this->device_callbacks, &this->device) != ACAMERA_OK) {
            this->device = nullptr;
            error = "Failed to open camera " + this->identifier;
            return false;
        }
        if ((AImageReader_new(this->size.width, this->size.height, AIMAGE_FORMAT_YUV_420_888, 4, &this->reader) != AMEDIA_OK) || (AImageReader_getWindow(this->reader, &this->window) != AMEDIA_OK)) {
            error = "Failed to make an image reader of " + std::to_string(this->size.width) + "x" + std::to_string(this->size.height);
            this->close();
            return false;
        }
        AImageReader_ImageListener listener{};
        listener.context = this;
        listener.onImageAvailable = &camera::on_image_available;
        AImageReader_setImageListener(this->reader, &listener);
        ACaptureSessionOutputContainer_create(&this->outputs);
        ACaptureSessionOutput_create(this->window, &this->output);
        ACaptureSessionOutputContainer_add(this->outputs, this->output);
        ACameraOutputTarget_create(this->window, &this->target);
        if (ACameraDevice_createCaptureRequest(this->device, TEMPLATE_RECORD, &this->request) != ACAMERA_OK) {
            error = "Failed to make a capture request";
            this->close();
            return false;
        }
        ACaptureRequest_addTarget(this->request, this->target);
        set_byte(this->request, ACAMERA_CONTROL_MODE, ACAMERA_CONTROL_MODE_AUTO);
        set_byte(this->request, ACAMERA_CONTROL_AE_MODE, ACAMERA_CONTROL_AE_MODE_ON);
        if (frame_rate_range[1] > 0) {
            ACaptureRequest_setEntry_i32(this->request, ACAMERA_CONTROL_AE_TARGET_FPS_RANGE, 2, &frame_rate_range[0]);
        }
        set_byte(this->request, ACAMERA_CONTROL_VIDEO_STABILIZATION_MODE, ACAMERA_CONTROL_VIDEO_STABILIZATION_MODE_OFF);
        if (optical_stabilisation_off) {
            set_byte(this->request, ACAMERA_LENS_OPTICAL_STABILIZATION_MODE, ACAMERA_LENS_OPTICAL_STABILIZATION_MODE_OFF);
        }
        if (this->correction_off) {
            set_byte(this->request, ACAMERA_DISTORTION_CORRECTION_MODE, ACAMERA_DISTORTION_CORRECTION_MODE_OFF);
        }
        if ((this->characteristics.region[2] > 0.0) && (this->characteristics.region[3] > 0.0)) {
            const int crop[4] = { 0, 0, static_cast<int>(this->characteristics.region[2]), static_cast<int>(this->characteristics.region[3]) };
            ACaptureRequest_setEntry_i32(this->request, ACAMERA_SCALER_CROP_REGION, 4, &crop[0]);
        }
        if (focus_fixed && (minimum_focus_dioptres > 0.0f)) {
            set_byte(this->request, ACAMERA_CONTROL_AF_MODE, ACAMERA_CONTROL_AF_MODE_OFF);
            const float focus = (static_cast<float>(this->settings.focus_dioptres) < minimum_focus_dioptres) ? static_cast<float>(this->settings.focus_dioptres) : minimum_focus_dioptres;
            ACaptureRequest_setEntry_float(this->request, ACAMERA_LENS_FOCUS_DISTANCE, 1, &focus);
        }
        else if (minimum_focus_dioptres > 0.0f) {
            set_byte(this->request, ACAMERA_CONTROL_AF_MODE, ACAMERA_CONTROL_AF_MODE_CONTINUOUS_VIDEO);
        }
        this->session_callbacks.context = this;
        this->session_callbacks.onClosed = &camera::on_session_closed;
        this->session_callbacks.onReady = &camera::on_session_ready;
        this->session_callbacks.onActive = &camera::on_session_active;
        this->capture_callbacks = ACameraCaptureSession_captureCallbacks{};
        this->capture_callbacks.context = this;
        this->capture_callbacks.onCaptureCompleted = &camera::on_capture_completed;
        if (ACameraDevice_createCaptureSession(this->device, this->outputs, &this->session_callbacks, &this->capture_session) != ACAMERA_OK) {
            this->capture_session = nullptr;
            error = "Failed to start a capture session";
            this->close();
            return false;
        }

        std::lock_guard<std::mutex> lock(this->mutex);
        this->resolved = false;
        if (this->characteristics.calibrated) {
            this->resolve_parameters(nullptr);
        }
        return true;
    }

    void camera::close() {
        this->stop();
        if (this->capture_session != nullptr) {
            ACameraCaptureSession_close(this->capture_session);
            this->capture_session = nullptr;
        }
        if (this->request != nullptr) {
            ACaptureRequest_free(this->request);
            this->request = nullptr;
        }
        if (this->target != nullptr) {
            ACameraOutputTarget_free(this->target);
            this->target = nullptr;
        }
        if (this->outputs != nullptr) {
            if (this->output != nullptr) {
                ACaptureSessionOutputContainer_remove(this->outputs, this->output);
            }
            ACaptureSessionOutputContainer_free(this->outputs);
            this->outputs = nullptr;
        }
        if (this->output != nullptr) {
            ACaptureSessionOutput_free(this->output);
            this->output = nullptr;
        }
        if (this->device != nullptr) {
            ACameraDevice_close(this->device);
            this->device = nullptr;
        }
        if (this->reader != nullptr) {
            // The reader owns its window, and once deleted calls its listener no more.
            AImageReader_delete(this->reader);
            this->reader = nullptr;
            this->window = nullptr;
        }
    }

    bool camera::is_open() const {
        return this->device != nullptr;
    }

    bool camera::start(frame_callback frame_handler, std::string& error) {
        if (this->capture_session == nullptr) {
            error = "The camera is not open";
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(this->mutex);
            this->callback = std::move(frame_handler);
        }
        if (ACameraCaptureSession_setRepeatingRequest(this->capture_session, &this->capture_callbacks, 1, &this->request, nullptr) != ACAMERA_OK) {
            error = "Failed to start streaming from camera " + this->identifier;
            return false;
        }
        return true;
    }

    void camera::stop() {
        if (this->capture_session != nullptr) {
            ACameraCaptureSession_stopRepeating(this->capture_session);
        }
        std::lock_guard<std::mutex> lock(this->mutex);
        this->callback = nullptr;
    }

    bool camera::get_parameters(zeroslam_sensor_parameters_camera_struct& stream_parameters) const {
        std::lock_guard<std::mutex> lock(this->mutex);
        if (!this->resolved) {
            return false;
        }
        stream_parameters = this->parameters;
        return true;
    }

    std::string camera::get_description() const {
        std::lock_guard<std::mutex> lock(this->mutex);
        return this->description;
    }

    int camera::get_sensor_orientation() const {
        return this->sensor_orientation;
    }

    bool camera::has_failed() const {
        return this->failed.load();
    }

    void camera::resolve_parameters(const ACameraMetadata* result) {
        camera_characteristics resolved_characteristics = this->characteristics;
        const char* source = "calibrated";
        if (!resolved_characteristics.calibrated && (result != nullptr)) {
            read_calibration(result, resolved_characteristics);
            if (!this->correction_off && resolved_characteristics.distorted) {
                ACameraMetadata_const_entry entry{};
                resolved_characteristics.distorted = !find_entry(result, ACAMERA_DISTORTION_CORRECTION_MODE, entry);
            }
            source = "calibrated per frame";
        }
        if (!resolved_characteristics.calibrated) {
            source = "estimated from the lens";
        }
        if (!stream_camera(resolved_characteristics, this->size.width, this->size.height, this->parameters)) {
            this->failed.store(true);
            this->description = "Camera " + this->identifier + " reports neither a calibration nor its lens";
            return;
        }
        char text[256];
        std::snprintf(&text[0], sizeof(text), "Camera %s %dx%d %s: f %.1f %.1f c %.1f %.1f k %.3f %.3f %.3f", this->identifier.c_str(), this->size.width, this->size.height, source, this->parameters.focal_x, this->parameters.focal_y, this->parameters.centre_x, this->parameters.centre_y, this->parameters.distortion[0], this->parameters.distortion[1], this->parameters.distortion[4]);
        this->description = &text[0];
        this->resolved = true;
    }

    void camera::on_disconnected(void* context, ACameraDevice* device) {
        static_cast<void>(device);
        static_cast<camera*>(context)->failed.store(true);
    }

    void camera::on_error(void* context, ACameraDevice* device, int error) {
        static_cast<void>(device);
        static_cast<void>(error);
        static_cast<camera*>(context)->failed.store(true);
    }

    void camera::on_image_available(void* context, AImageReader* reader) {
        camera* const self = static_cast<camera*>(context);
        AImage* image = nullptr;
        if ((AImageReader_acquireLatestImage(reader, &image) != AMEDIA_OK) || (image == nullptr)) {
            return;
        }
        // The luminance is the first plane, one byte a pixel.
        unsigned char* data = nullptr;
        int length = 0;
        int row_stride = 0;
        long long timestamp = 0;
        int64_t timestamp_nanoseconds = 0;
        if ((AImage_getPlaneData(image, 0, &data, &length) == AMEDIA_OK) && (AImage_getPlaneRowStride(image, 0, &row_stride) == AMEDIA_OK) && (AImage_getTimestamp(image, &timestamp_nanoseconds) == AMEDIA_OK)) {
            timestamp = static_cast<long long>(timestamp_nanoseconds);
            frame_view frame;
            frame.pixels = data;
            frame.width = self->size.width;
            frame.height = self->size.height;
            frame.row_stride = row_stride;
            frame.timestamp_nanoseconds = timestamp;
            const bool complete = (length >= ((row_stride * (frame.height - 1)) + frame.width));
            std::lock_guard<std::mutex> lock(self->mutex);
            if (complete && self->resolved && self->callback) {
                self->callback(frame);
            }
        }
        AImage_delete(image);
    }

    void camera::on_capture_completed(void* context, ACameraCaptureSession* session, ACaptureRequest* request, const ACameraMetadata* result) {
        static_cast<void>(session);
        static_cast<void>(request);
        camera* const self = static_cast<camera*>(context);
        std::lock_guard<std::mutex> lock(self->mutex);
        if (!self->resolved && !self->failed.load()) {
            self->resolve_parameters(result);
        }
    }

    void camera::on_session_closed(void* context, ACameraCaptureSession* session) {
        static_cast<void>(context);
        static_cast<void>(session);
    }

    void camera::on_session_ready(void* context, ACameraCaptureSession* session) {
        static_cast<void>(context);
        static_cast<void>(session);
    }

    void camera::on_session_active(void* context, ACameraCaptureSession* session) {
        static_cast<void>(context);
        static_cast<void>(session);
    }
}
