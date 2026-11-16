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

// The app: a native activity that runs the system on the main camera and shows what it makes of it. Its settings are read
// from configuration.txt in the app's files directory (/sdcard/Android/data/[package]/files/), one key=value a line: the
// app's own keys start with "app_" (see app_settings) and every other line is a setting of the library. Saved runs and
// recordings are written to the same directory.

#include "camera.hpp"
#include "dataset.hpp"
#include "renderer.hpp"
#include "session.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <EGL/egl.h>
#include <algorithm>
#include <android/log.h>
#include <android/window.h>
#include <android_native_app_glue.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <jni.h>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    constexpr static const char log_tag[] = "zeroslam";

    // The library's settings the app starts from, cheaper than its defaults so that a phone keeps up: the intensity flow
    // in place of the wavelet flow, and no subpixel refinement of the corners.
    constexpr static const char default_configuration[] = "flow=intensity\nrefiner=none\n";

    void log_line(const int priority, const std::string& text) {
        __android_log_write(priority, &log_tag[0], text.c_str());
    }

    // The library logs to stderr, which an app's process sends nowhere, so both standard streams go to the log instead.
    void forward_output_to_log() {
        static int pipe_ends[2] = { -1, -1 };
        if ((pipe_ends[0] >= 0) || (pipe(&pipe_ends[0]) != 0)) {
            return;
        }
        std::setvbuf(stdout, nullptr, _IOLBF, 0);
        std::setvbuf(stderr, nullptr, _IOLBF, 0);
        dup2(pipe_ends[1], STDOUT_FILENO);
        dup2(pipe_ends[1], STDERR_FILENO);
        std::thread([]() {
            std::string line;
            char buffer[512];
            for (;;) {
                const ssize_t length = read(pipe_ends[0], &buffer[0], sizeof(buffer));
                if (length <= 0) {
                    return;
                }
                for (ssize_t i = 0; i < length; ++i) {
                    if (buffer[i] == '\n') {
                        log_line(ANDROID_LOG_INFO, line);
                        line.clear();
                    }
                    else {
                        line += buffer[i];
                    }
                }
            }
        }).detach();
    }

    // The big cores: those within 70% of the fastest core's top frequency. The library splits its parallel loops evenly
    // over its threads, so a loop spread onto the little cores waits on them.
    // The kernel's files report a size they do not have, so they are read as a stream.
    unsigned int count_big_cores() {
        std::vector<long> frequencies;
        for (unsigned int core = 0; core < 64; ++core) {
            const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(core) + "/cpufreq/cpuinfo_max_freq";
            std::FILE* const file = std::fopen(path.c_str(), "r");
            if (file == nullptr) {
                continue;
            }
            long frequency = 0;
            if (std::fscanf(file, "%ld", &frequency) == 1) {
                frequencies.push_back(frequency);
            }
            std::fclose(file);
        }
        if (frequencies.empty()) {
            return 0;
        }
        const long fastest = *std::max_element(frequencies.begin(), frequencies.end());
        return static_cast<unsigned int>(std::count_if(frequencies.begin(), frequencies.end(), [fastest](const long frequency) {
            return (10 * frequency) >= (7 * fastest);
        }));
    }

    // The settings of configuration.txt, split into the app's and the library's.
    struct app_settings final {
        mobile::camera_settings camera;
        // How many threads the library runs on, 0 for the big cores.
        unsigned int threads = 0;
        std::string library;
        std::string error;
    };

    app_settings read_settings(const std::string& path) {
        app_settings settings;
        std::map<std::string, std::string> library;
        std::vector<std::string> order;
        const auto add = [&](const std::string& text, const bool from_file) {
            std::size_t start = 0;
            while (start < text.size()) {
                std::size_t end = text.find('\n', start);
                end = (end == std::string::npos) ? text.size() : end;
                std::string line = text.substr(start, end - start);
                start = end + 1;
                line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
                const std::size_t equals = line.find('=');
                if (line.empty() || (line[0] == '#') || (equals == std::string::npos)) {
                    continue;
                }
                const std::string key = line.substr(0, equals);
                const std::string value = line.substr(equals + 1);
                if (from_file && (key == "app_camera_width")) {
                    settings.camera.preferred_width = std::max(64, std::atoi(value.c_str()));
                }
                else if (from_file && (key == "app_camera_focus")) {
                    settings.camera.focus_dioptres = (value == "auto") ? -1.0 : std::max(0.0, std::atof(value.c_str()));
                }
                else if (from_file && (key == "app_threads")) {
                    settings.threads = static_cast<unsigned int>(std::max(0, std::atoi(value.c_str())));
                }
                else if (from_file && (key.compare(0, 4, "app_") == 0)) {
                    settings.error = "Unknown setting '" + key + "' in " + path;
                }
                else {
                    if (library.find(key) == library.end()) {
                        order.push_back(key);
                    }
                    library[key] = value;
                }
            }
        };
        add(&default_configuration[0], false);
        std::string text;
        if (dataset::read_text_file(path, text)) {
            add(text, true);
        }
        for (const std::string& key : order) {
            settings.library += key + "=" + library[key] + "\n";
        }
        return settings;
    }

    // The java side of the activity, reached through jni from the app's thread.
    class activity_bridge final {
    public:
        explicit activity_bridge(ANativeActivity* native_activity)
            : activity(native_activity) {
            if (this->activity->vm->AttachCurrentThread(&this->environment, nullptr) != JNI_OK) {
                this->environment = nullptr;
            }
        }

        ~activity_bridge() {
            if (this->environment != nullptr) {
                this->activity->vm->DetachCurrentThread();
            }
        }

        activity_bridge(const activity_bridge&) = delete;
        activity_bridge& operator=(const activity_bridge&) = delete;

        bool has_camera_permission() {
            if (this->environment == nullptr) {
                return false;
            }
            JNIEnv* const env = this->environment;
            jclass activity_class = env->GetObjectClass(this->activity->clazz);
            jmethodID check = env->GetMethodID(activity_class, "checkSelfPermission", "(Ljava/lang/String;)I");
            jstring permission = env->NewStringUTF("android.permission.CAMERA");
            const jint granted = env->CallIntMethod(this->activity->clazz, check, permission);
            env->DeleteLocalRef(permission);
            env->DeleteLocalRef(activity_class);
            return !this->clear_exception() && (granted == 0);
        }

        void request_camera_permission() {
            if (this->environment == nullptr) {
                return;
            }
            JNIEnv* const env = this->environment;
            jclass activity_class = env->GetObjectClass(this->activity->clazz);
            jmethodID request = env->GetMethodID(activity_class, "requestPermissions", "([Ljava/lang/String;I)V");
            jclass string_class = env->FindClass("java/lang/String");
            jstring permission = env->NewStringUTF("android.permission.CAMERA");
            jobjectArray permissions = env->NewObjectArray(1, string_class, permission);
            env->CallVoidMethod(this->activity->clazz, request, permissions, 1);
            env->DeleteLocalRef(permissions);
            env->DeleteLocalRef(permission);
            env->DeleteLocalRef(string_class);
            env->DeleteLocalRef(activity_class);
            this->clear_exception();
        }

        // The display's turn from its natural orientation, counter clockwise in degrees.
        int display_rotation() {
            if (this->environment == nullptr) {
                return 0;
            }
            JNIEnv* const env = this->environment;
            jclass activity_class = env->GetObjectClass(this->activity->clazz);
            jmethodID get_window_manager = env->GetMethodID(activity_class, "getWindowManager", "()Landroid/view/WindowManager;");
            jobject window_manager = env->CallObjectMethod(this->activity->clazz, get_window_manager);
            int rotation = 0;
            if (!this->clear_exception() && (window_manager != nullptr)) {
                jclass window_manager_class = env->GetObjectClass(window_manager);
                jmethodID get_display = env->GetMethodID(window_manager_class, "getDefaultDisplay", "()Landroid/view/Display;");
                jobject display = env->CallObjectMethod(window_manager, get_display);
                if (!this->clear_exception() && (display != nullptr)) {
                    jclass display_class = env->GetObjectClass(display);
                    jmethodID get_rotation = env->GetMethodID(display_class, "getRotation", "()I");
                    rotation = 90 * static_cast<int>(env->CallIntMethod(display, get_rotation));
                    rotation = this->clear_exception() ? 0 : rotation;
                    env->DeleteLocalRef(display_class);
                    env->DeleteLocalRef(display);
                }
                env->DeleteLocalRef(window_manager_class);
                env->DeleteLocalRef(window_manager);
            }
            env->DeleteLocalRef(activity_class);
            return rotation;
        }

    private:
        bool clear_exception() {
            if (this->environment->ExceptionCheck() == JNI_TRUE) {
                this->environment->ExceptionClear();
                return true;
            }
            return false;
        }

        ANativeActivity* activity;
        JNIEnv* environment = nullptr;
    };

    struct application final {
        android_app* app = nullptr;
        std::unique_ptr<activity_bridge> bridge;
        app_settings settings;
        std::string files_directory;

        EGLDisplay display = EGL_NO_DISPLAY;
        EGLConfig config = nullptr;
        EGLContext context = EGL_NO_CONTEXT;
        EGLSurface surface = EGL_NO_SURFACE;
        mobile::renderer renderer;
        int window_width = 0;
        int window_height = 0;
        int image_rotation = -1;
        int frames_since_layout = 0;
        // Whether the window needs drawing again: a new snapshot, a touch, a new layout or a new surface.
        bool dirty = true;

        mobile::camera camera;
        std::unique_ptr<mobile::session> session;
        bool resumed = false;
        bool permission_requested = false;
        bool streaming = false;
        // Set when the system refused the camera, until the camera is opened again.
        bool refused = false;
        std::string status;
        std::chrono::steady_clock::time_point last_attempt{};

        // The touch being followed: where it went down, where it was last, and the spread of a pinch.
        bool touching = false;
        bool dragged = false;
        bool touch_in_map = false;
        float down_x = 0.0f;
        float down_y = 0.0f;
        float last_x = 0.0f;
        float last_y = 0.0f;
        float last_spread = 0.0f;

        float density() const {
            const int dpi = AConfiguration_getDensity(this->app->config);
            return ((dpi > 0) && (dpi != ACONFIGURATION_DENSITY_ANY) && (dpi != ACONFIGURATION_DENSITY_NONE)) ? (static_cast<float>(dpi) / 160.0f) : 1.0f;
        }

        bool create_surface() {
            if (this->display == EGL_NO_DISPLAY) {
                this->display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
                if ((this->display == EGL_NO_DISPLAY) || (eglInitialize(this->display, nullptr, nullptr) != EGL_TRUE)) {
                    this->status = "Failed to reach the display";
                    return false;
                }
                const EGLint attributes[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE };
                EGLint count = 0;
                if ((eglChooseConfig(this->display, &attributes[0], &this->config, 1, &count) != EGL_TRUE) || (count < 1)) {
                    this->status = "Failed to find a display configuration";
                    return false;
                }
            }
            this->surface = eglCreateWindowSurface(this->display, this->config, this->app->window, nullptr);
            if (this->surface == EGL_NO_SURFACE) {
                this->status = "Failed to make a surface of the window";
                return false;
            }
            bool fresh_context = false;
            if (this->context == EGL_NO_CONTEXT) {
                const EGLint attributes[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
                this->context = eglCreateContext(this->display, this->config, EGL_NO_CONTEXT, &attributes[0]);
                fresh_context = true;
            }
            if ((this->context == EGL_NO_CONTEXT) || (eglMakeCurrent(this->display, this->surface, this->surface, this->context) != EGL_TRUE)) {
                this->status = "Failed to make an opengl es 2 context";
                return false;
            }
            eglSwapInterval(this->display, 1);
            if (fresh_context) {
                std::string error;
                if (!this->renderer.create(error)) {
                    this->status = error;
                    return false;
                }
            }
            this->image_rotation = -1;
            this->dirty = true;
            return true;
        }

        void destroy_surface() {
            if (this->display == EGL_NO_DISPLAY) {
                return;
            }
            eglMakeCurrent(this->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (this->surface != EGL_NO_SURFACE) {
                eglDestroySurface(this->display, this->surface);
                this->surface = EGL_NO_SURFACE;
            }
        }

        // The context takes the renderer's objects with it.
        void destroy_display() {
            this->destroy_surface();
            if (this->context != EGL_NO_CONTEXT) {
                eglDestroyContext(this->display, this->context);
                this->context = EGL_NO_CONTEXT;
            }
            if (this->display != EGL_NO_DISPLAY) {
                eglTerminate(this->display);
                this->display = EGL_NO_DISPLAY;
            }
        }

        // Follows the window's size and the turn that shows the camera's image upright on it.
        void update_layout() {
            // A turn of half a circle changes nothing but the rotation, so the rotation is looked at now and again.
            if ((this->surface == EGL_NO_SURFACE) || ((this->image_rotation >= 0) && (++this->frames_since_layout < 30))) {
                return;
            }
            this->frames_since_layout = 0;
            EGLint width = 0;
            EGLint height = 0;
            eglQuerySurface(this->display, this->surface, EGL_WIDTH, &width);
            eglQuerySurface(this->display, this->surface, EGL_HEIGHT, &height);
            const int rotation = ((this->camera.get_sensor_orientation() - this->bridge->display_rotation()) % 360 + 360) % 360;
            if ((width != this->window_width) || (height != this->window_height) || (rotation != this->image_rotation)) {
                this->window_width = width;
                this->window_height = height;
                this->image_rotation = rotation;
                this->renderer.resize(width, height, this->density(), rotation);
                this->dirty = true;
            }
        }

        // Opens the camera once the app is in front and may, and starts the system once the camera's geometry is known.
        void update_camera() {
            if (!this->resumed) {
                return;
            }
            if (this->streaming && this->camera.has_failed()) {
                this->camera.close();
                this->streaming = false;
                this->status = "Lost the camera, retrying";
            }
            const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            if (!this->streaming && ((now - this->last_attempt) > std::chrono::seconds(1))) {
                this->last_attempt = now;
                if (!this->bridge->has_camera_permission()) {
                    if (!this->permission_requested) {
                        this->bridge->request_camera_permission();
                        this->permission_requested = true;
                    }
                    this->status = "Waiting for permission to use the camera";
                    return;
                }
                std::string error;
                if (!this->camera.open(this->settings.camera, error) || !this->camera.start([this](const mobile::frame_view& frame) {
                        this->session->submit(frame);
                    },
                                                                                            error)) {
                    this->camera.close();
                    this->status = error;
                    log_line(ANDROID_LOG_ERROR, error);
                    return;
                }
                this->streaming = true;
                this->refused = false;
                this->image_rotation = -1;
                this->status = "Waiting for the camera's calibration";
            }
            zeroslam_sensor_parameters_camera_struct parameters{};
            if (this->streaming && !this->refused && this->camera.get_parameters(parameters)) {
                const zeroslam_sensor_parameters_camera_struct& running = this->session->current().camera;
                const bool same = this->session->is_running() && (running.width == parameters.width) && (running.height == parameters.height) && (running.focal_x == parameters.focal_x) && (running.focal_y == parameters.focal_y) && (running.centre_x == parameters.centre_x) && (running.centre_y == parameters.centre_y);
                if (!same) {
                    std::string error;
                    if (!this->session->start(parameters, this->settings.library, error)) {
                        // The configuration file is the likely cause, so the app falls back to its own settings.
                        this->status = error + ", running without configuration.txt";
                        log_line(ANDROID_LOG_ERROR, this->status);
                        if (!this->session->start(parameters, &default_configuration[0], error)) {
                            this->status = error;
                            this->refused = true;
                        }
                    }
                    else {
                        this->status = this->settings.error.empty() ? this->camera.get_description() : this->settings.error;
                        log_line(ANDROID_LOG_INFO, this->camera.get_description());
                    }
                }
            }
        }

        void stop_camera() {
            this->camera.close();
            this->streaming = false;
        }

        bool handle_motion(const AInputEvent* event) {
            this->dirty = true;
            const int32_t action = AMotionEvent_getAction(event) & AMOTION_EVENT_ACTION_MASK;
            const std::size_t pointers = AMotionEvent_getPointerCount(event);
            const float x = AMotionEvent_getX(event, 0);
            const float y = AMotionEvent_getY(event, 0);
            const float spread = (pointers >= 2) ? std::hypot(AMotionEvent_getX(event, 1) - x, AMotionEvent_getY(event, 1) - y) : 0.0f;
            const float slop = 10.0f * this->density();
            switch (action) {
                case AMOTION_EVENT_ACTION_DOWN:
                    this->touching = true;
                    this->dragged = false;
                    this->touch_in_map = this->renderer.in_map(x, y);
                    this->down_x = x;
                    this->down_y = y;
                    this->last_x = x;
                    this->last_y = y;
                    return true;
                case AMOTION_EVENT_ACTION_POINTER_DOWN:
                case AMOTION_EVENT_ACTION_POINTER_UP:
                    this->dragged = true;
                    this->last_spread = spread;
                    this->last_x = x;
                    this->last_y = y;
                    return true;
                case AMOTION_EVENT_ACTION_MOVE:
                    if (!this->touching) {
                        return true;
                    }
                    if (std::hypot(x - this->down_x, y - this->down_y) > slop) {
                        this->dragged = true;
                    }
                    if (this->touch_in_map && (pointers >= 2) && (this->last_spread > 0.0f) && (spread > 0.0f)) {
                        this->renderer.zoom(spread / this->last_spread);
                    }
                    else if (this->touch_in_map && (pointers == 1) && this->dragged) {
                        this->renderer.orbit(x - this->last_x, y - this->last_y);
                    }
                    this->last_spread = spread;
                    this->last_x = x;
                    this->last_y = y;
                    return true;
                case AMOTION_EVENT_ACTION_UP:
                    if (this->touching && !this->dragged) {
                        switch (this->renderer.hit(x, y)) {
                            case mobile::button::reset:
                                this->session->reset();
                                break;
                            case mobile::button::save:
                                this->session->save();
                                break;
                            case mobile::button::record:
                                this->session->toggle_recording();
                                break;
                            case mobile::button::none:
                                break;
                        }
                    }
                    this->touching = false;
                    return true;
                case AMOTION_EVENT_ACTION_CANCEL:
                    this->touching = false;
                    return true;
                default:
                    return false;
            }
        }
    };

    void handle_command(android_app* app, const int32_t command) {
        application* const state = static_cast<application*>(app->userData);
        switch (command) {
            case APP_CMD_INIT_WINDOW:
                if ((app->window != nullptr) && !state->create_surface()) {
                    log_line(ANDROID_LOG_ERROR, state->status);
                }
                break;
            case APP_CMD_TERM_WINDOW:
                state->destroy_surface();
                break;
            case APP_CMD_WINDOW_RESIZED:
            case APP_CMD_CONFIG_CHANGED:
                state->image_rotation = -1;
                break;
            case APP_CMD_RESUME:
                // Coming back to the app asks again for a permission refused before.
                state->resumed = true;
                state->permission_requested = false;
                break;
            case APP_CMD_PAUSE:
                state->resumed = false;
                state->stop_camera();
                break;
            default:
                break;
        }
    }

    int32_t handle_input(android_app* app, AInputEvent* event) {
        application* const state = static_cast<application*>(app->userData);
        if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
            return state->handle_motion(event) ? 1 : 0;
        }
        return 0;
    }
}

void android_main(android_app* app) {
    forward_output_to_log();
    application state;
    state.app = app;
    state.bridge = std::make_unique<activity_bridge>(app->activity);
    state.files_directory = (app->activity->externalDataPath != nullptr) ? app->activity->externalDataPath : app->activity->internalDataPath;
    state.settings = read_settings(state.files_directory + "/configuration.txt");
    // The library's thread pool is sized when the first system is made, from this.
    const unsigned int threads = (state.settings.threads > 0) ? state.settings.threads : count_big_cores();
    if (threads > 0) {
        setenv("ZEROSLAM_THREADS", std::to_string(threads).c_str(), 1);
    }
    log_line(ANDROID_LOG_INFO, "Writing to " + state.files_directory + ", running the library on " + std::to_string(threads) + " threads with:\n" + state.settings.library);
    state.session = std::make_unique<mobile::session>(state.files_directory);
    app->userData = &state;
    app->onAppCmd = &handle_command;
    app->onInputEvent = &handle_input;
    ANativeActivity_setWindowFlags(app->activity, AWINDOW_FLAG_KEEP_SCREEN_ON | AWINDOW_FLAG_FULLSCREEN, 0);

    std::string shown_status;
    while (app->destroyRequested == 0) {
        // Take the events waiting, then draw when something has changed, looking again for a new snapshot as often as the
        // display refreshes, so that drawing takes no more from the system than it must.
        const bool showing = (state.surface != EGL_NO_SURFACE);
        int events = 0;
        android_poll_source* source = nullptr;
        const int result = ALooper_pollOnce(showing ? (state.dirty ? 0 : 16) : 250, nullptr, &events, reinterpret_cast<void**>(&source));
        if ((result >= 0) && (source != nullptr)) {
            source->process(app, source);
            continue;
        }
        state.update_camera();
        if (state.surface == EGL_NO_SURFACE) {
            continue;
        }
        state.update_layout();
        if (state.session->update() || (state.status != shown_status)) {
            state.dirty = true;
        }
        if (!state.dirty) {
            continue;
        }
        state.dirty = false;
        shown_status = state.status;
        state.renderer.draw(state.session->current(), state.status);
        if (eglSwapBuffers(state.display, state.surface) != EGL_TRUE) {
            // A lost context goes with everything made in it, which is made again for the window.
            const bool lost = (eglGetError() == EGL_CONTEXT_LOST);
            if (lost) {
                state.destroy_display();
            }
            else {
                state.destroy_surface();
            }
            if (lost && (app->window != nullptr)) {
                state.create_surface();
            }
        }
    }
    state.stop_camera();
    state.session->stop();
    state.destroy_display();
}
