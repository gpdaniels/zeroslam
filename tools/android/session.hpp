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
#ifndef ZEROSLAM_TOOLS_ANDROID_SESSION_HPP
#define ZEROSLAM_TOOLS_ANDROID_SESSION_HPP

#include "triple_buffer.hpp"
#include "zeroslam/zeroslam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mobile {
    // One camera frame's grey pixels as the camera hands them over: height rows of width bytes, each row_stride bytes apart.
    struct frame_view final {
        const unsigned char* pixels = nullptr;
        int width = 0;
        int height = 0;
        int row_stride = 0;
        long long timestamp_nanoseconds = 0;
    };

    enum class tracking_state {
        initialising,
        tracking,
        lost
    };

    const char* tracking_state_name(tracking_state state);

    // What the display needs of the system after a frame, copied out so that it reads it without a lock.
    struct snapshot final {
        // Changes with every processed frame, and with every refresh of the landmarks and keyframes.
        unsigned long long frame_sequence = 0;
        unsigned long long map_sequence = 0;
        zeroslam_sensor_parameters_camera_struct camera{};
        // The frame the system processed last, camera.width * camera.height bytes, and its pose when it has one.
        std::vector<unsigned char> image;
        long long timestamp_nanoseconds = 0;
        tracking_state state = tracking_state::initialising;
        bool posed = false;
        // The camera-to-world pose { x, y, z, qx, qy, qz, qw }.
        double pose[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
        // The landmarks, x y z each, and the camera centres of the keyframes, x y z each in time order.
        std::vector<float> landmarks;
        std::vector<float> keyframes;
        long long frames_received = 0;
        long long frames_processed = 0;
        long long frames_dropped = 0;
        double process_milliseconds = 0.0;
        double average_process_milliseconds = 0.0;
        // Whether every frame is being recorded, and how many have been written and how many the disk was too slow for.
        bool recording = false;
        long long frames_recorded = 0;
        long long frames_unrecorded = 0;
        // The outcome of the last save, reset or recording, for the display to show.
        std::string message;
    };

    // Runs the system on its own thread over the newest frame the camera has given it: a frame that arrives while the
    // system is still busy replaces the one waiting, so the system falls behind by at most one frame however long a frame
    // takes, and the frames it skips are counted.
    class session final {
    public:
        // Saved runs and recordings go into the output directory, each in a directory named by the time it was made.
        explicit session(std::string output_directory);
        ~session();

        session(const session&) = delete;
        session& operator=(const session&) = delete;

        // Starts the system for the camera, with any configuration (key=value lines) on top of the defaults.
        bool start(const zeroslam_sensor_parameters_camera_struct& camera, const std::string& configuration, std::string& error);
        void stop();
        bool is_running() const;

        // Hands over a frame, which is copied, from any thread.
        void submit(const frame_view& frame);

        // Requests handled on the worker between frames: a fresh system, and writing the trajectory and the map out.
        void reset();
        void save();

        // Starts or stops recording every frame submitted as a scene, written on a thread of its own.
        void toggle_recording();

        // Takes the newest snapshot, returning whether there was a new one, and reads it.
        bool update();
        const snapshot& current();

        // Blocks until the worker has nothing pending, for tests.
        void wait_until_idle();

    private:
        class recorder;

        void run();
        bool create_system(std::string& error);
        void process(long long timestamp_nanoseconds);
        void refresh_map();
        void publish();
        void save_run();
        std::string make_run_directory() const;
        void set_message(const std::string& text);

        std::string output_directory;
        zeroslam_sensor_parameters_camera_struct camera{};
        std::string configuration;
        zeroslam::system system;
        std::thread worker;

        // Shared with the threads that submit and request, under the mutex.
        mutable std::mutex mutex;
        std::condition_variable wake;
        std::condition_variable idle;
        bool running = false;
        bool stopping = false;
        bool busy = false;
        bool frame_waiting = false;
        bool reset_requested = false;
        bool save_requested = false;
        bool publish_requested = false;
        std::vector<unsigned char> waiting_image;
        long long waiting_timestamp_nanoseconds = 0;
        long long frames_received = 0;
        long long frames_dropped = 0;
        std::string message;
        std::unique_ptr<recorder> recording;

        // The worker's own.
        std::vector<unsigned char> working_image;
        std::vector<long long> fed_timestamps;
        long long last_timestamp_nanoseconds = -1;
        long long frames_processed = 0;
        bool ever_posed = false;
        int frames_since_refresh = 0;
        std::vector<zeroslam_point_struct> points;
        std::vector<long long> keyframe_timestamps;
        snapshot latest;
        gtl::triple_buffer<snapshot> snapshots;
    };
}

#endif // ZEROSLAM_TOOLS_ANDROID_SESSION_HPP
