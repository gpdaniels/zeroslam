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

#include "session.hpp"

#include "dataset.hpp"
#include "directory.hpp"
#include "file.hpp"
#include "import.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <chrono>
#include <cstdio>
#include <ctime>
#include <deque>
#include <utility>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mobile {
    namespace {
        constexpr static const int sensor_id = 1;
        // The landmarks and keyframes are copied out with each new keyframe, and at least this often between them, as the
        // adjustments move them.
        constexpr static const int refresh_frames = 15;
        // How many frames a recording holds while the disk catches up, before it skips them.
        constexpr static const std::size_t recording_queue_frames = 30;

        std::string local_time_name() {
            const std::time_t now = std::time(nullptr);
            std::tm parts{};
#if defined(_WIN32)
            localtime_s(&parts, &now);
#else
            localtime_r(&now, &parts);
#endif
            char name[32];
            if (std::strftime(&name[0], sizeof(name), "%Y%m%d-%H%M%S", &parts) == 0) {
                return "run";
            }
            return std::string(&name[0]);
        }

        gtl::file create_file(const std::string& path) {
            return gtl::file(path.c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
        }

        unsigned char to_byte(const float value) {
            const float clamped = (value < 0.0f) ? 0.0f : ((value > 1.0f) ? 1.0f : value);
            return static_cast<unsigned char>((clamped * 255.0f) + 0.5f);
        }
    }

    const char* tracking_state_name(const tracking_state state) {
        switch (state) {
            case tracking_state::initialising:
                return "initialising";
            case tracking_state::tracking:
                return "tracking";
            case tracking_state::lost:
                return "lost";
        }
        return "unknown";
    }

    // Writes every frame it is given as a scene in the directory form (see README.md "Scene format"), which
    // `zeroslam dataset collapse` packs into a scene file to process and view on a desktop.
    class session::recorder final {
    public:
        recorder(std::string scene_directory, const zeroslam_sensor_parameters_camera_struct& camera_recorded)
            : directory(std::move(scene_directory))
            , camera(camera_recorded) {
            if (!gtl::directory::make_directories(this->directory + "/sensor/image_01")) {
                return;
            }
            this->calibration = create_file(this->directory + "/sensor/image_01.txt");
            if (!this->calibration.is_open()) {
                return;
            }
            this->writer = std::thread(&recorder::run, this);
        }

        ~recorder() {
            this->finish();
        }

        // Writes what the recorder holds and stops it.
        void finish() {
            {
                std::lock_guard<std::mutex> lock(this->mutex);
                this->stopping = true;
            }
            this->wake.notify_all();
            if (this->writer.joinable()) {
                this->writer.join();
            }
        }

        recorder(const recorder&) = delete;
        recorder& operator=(const recorder&) = delete;

        bool is_open() const {
            return this->writer.joinable();
        }

        const std::string& get_directory() const {
            return this->directory;
        }

        void push(const frame_view& frame) {
            std::lock_guard<std::mutex> lock(this->mutex);
            if (this->failed || (this->queue.size() >= recording_queue_frames)) {
                ++this->skipped;
                return;
            }
            std::vector<unsigned char> image;
            if (!this->spare.empty()) {
                image = std::move(this->spare.back());
                this->spare.pop_back();
            }
            image.resize(static_cast<std::size_t>(frame.width) * static_cast<std::size_t>(frame.height));
            for (int row = 0; row < frame.height; ++row) {
                const unsigned char* const source = frame.pixels + (static_cast<std::ptrdiff_t>(row) * frame.row_stride);
                std::copy(source, source + frame.width, image.begin() + (static_cast<std::ptrdiff_t>(row) * frame.width));
            }
            this->queue.emplace_back(frame.timestamp_nanoseconds, std::move(image));
            this->wake.notify_one();
        }

        void counts(long long& written_frames, long long& skipped_frames, bool& has_failed) const {
            std::lock_guard<std::mutex> lock(this->mutex);
            written_frames = this->written;
            skipped_frames = this->skipped;
            has_failed = this->failed;
        }

    private:
        void run() {
            const double translation[3] = { 0.0, 0.0, 0.0 };
            const double rotation[4] = { 0.0, 0.0, 0.0, 1.0 };
            const double intrinsics[4] = { this->camera.focal_x, this->camera.focal_y, this->camera.centre_x, this->camera.centre_y };
            for (;;) {
                std::pair<long long, std::vector<unsigned char>> frame;
                {
                    std::unique_lock<std::mutex> lock(this->mutex);
                    this->wake.wait(lock, [this]() {
                        return this->stopping || !this->queue.empty();
                    });
                    if (this->queue.empty()) {
                        return;
                    }
                    frame = std::move(this->queue.front());
                    this->queue.pop_front();
                }
                char name[32];
                std::snprintf(&name[0], sizeof(name), "%020lld.pnm", frame.first);
                const std::string path = this->directory + "/sensor/image_01/" + std::string(&name[0]);
                const bool stored = dataset::write_pnm(path, dataset::pnm_format::pgm8, static_cast<unsigned int>(this->camera.width), static_cast<unsigned int>(this->camera.height), frame.second.data()) && import::write_calibration_line(this->calibration, frame.first, &translation[0], &rotation[0], "plumb_bob", &intrinsics[0], &this->camera.distortion[0], 5);
                std::lock_guard<std::mutex> lock(this->mutex);
                if (stored) {
                    ++this->written;
                }
                else {
                    this->failed = true;
                }
                this->spare.push_back(std::move(frame.second));
            }
        }

        const std::string directory;
        const zeroslam_sensor_parameters_camera_struct camera;
        gtl::file calibration;
        std::thread writer;
        mutable std::mutex mutex;
        std::condition_variable wake;
        bool stopping = false;
        bool failed = false;
        long long written = 0;
        long long skipped = 0;
        std::deque<std::pair<long long, std::vector<unsigned char>>> queue;
        std::vector<std::vector<unsigned char>> spare;
    };

    session::session(std::string directory)
        : output_directory(std::move(directory)) {
    }

    session::~session() {
        this->stop();
    }

    bool session::start(const zeroslam_sensor_parameters_camera_struct& camera_started, const std::string& configuration_started, std::string& error) {
        this->stop();
        {
            std::lock_guard<std::mutex> lock(this->mutex);
            this->camera = camera_started;
        }
        this->configuration = configuration_started;
        if (!this->create_system(error)) {
            return false;
        }
        this->fed_timestamps.clear();
        this->last_timestamp_nanoseconds = -1;
        this->frames_processed = 0;
        this->ever_posed = false;
        this->frames_since_refresh = 0;
        this->keyframe_timestamps.clear();
        this->latest = snapshot{};
        this->latest.camera = this->camera;
        {
            std::lock_guard<std::mutex> lock(this->mutex);
            this->running = true;
            this->stopping = false;
            this->busy = false;
            this->frame_waiting = false;
            this->reset_requested = false;
            this->save_requested = false;
            this->publish_requested = false;
            this->frames_received = 0;
            this->frames_dropped = 0;
            this->message.clear();
        }
        this->publish();
        this->worker = std::thread(&session::run, this);
        return true;
    }

    void session::stop() {
        std::unique_ptr<recorder> finished;
        {
            std::lock_guard<std::mutex> lock(this->mutex);
            this->stopping = true;
            finished = std::move(this->recording);
        }
        this->wake.notify_all();
        if (this->worker.joinable()) {
            this->worker.join();
        }
        finished.reset();
        std::lock_guard<std::mutex> lock(this->mutex);
        this->running = false;
    }

    bool session::is_running() const {
        std::lock_guard<std::mutex> lock(this->mutex);
        return this->running;
    }

    void session::submit(const frame_view& frame) {
        std::lock_guard<std::mutex> lock(this->mutex);
        if (!this->running || this->stopping || (frame.pixels == nullptr) || (frame.width != this->camera.width) || (frame.height != this->camera.height) || (frame.row_stride < frame.width)) {
            return;
        }
        ++this->frames_received;
        if (this->recording) {
            this->recording->push(frame);
        }
        if (this->frame_waiting) {
            ++this->frames_dropped;
        }
        this->waiting_image.resize(static_cast<std::size_t>(frame.width) * static_cast<std::size_t>(frame.height));
        for (int row = 0; row < frame.height; ++row) {
            const unsigned char* const source = frame.pixels + (static_cast<std::ptrdiff_t>(row) * frame.row_stride);
            std::copy(source, source + frame.width, this->waiting_image.begin() + (static_cast<std::ptrdiff_t>(row) * frame.width));
        }
        this->waiting_timestamp_nanoseconds = frame.timestamp_nanoseconds;
        this->frame_waiting = true;
        this->wake.notify_one();
    }

    void session::reset() {
        {
            std::lock_guard<std::mutex> lock(this->mutex);
            this->reset_requested = true;
        }
        this->wake.notify_one();
    }

    void session::save() {
        {
            std::lock_guard<std::mutex> lock(this->mutex);
            this->save_requested = true;
        }
        this->wake.notify_one();
    }

    void session::toggle_recording() {
        std::unique_ptr<recorder> finished;
        {
            std::lock_guard<std::mutex> lock(this->mutex);
            if (!this->running) {
                return;
            }
            if (this->recording) {
                finished = std::move(this->recording);
            }
            else {
                const std::string directory = this->make_run_directory();
                std::unique_ptr<recorder> started = directory.empty() ? nullptr : std::make_unique<recorder>(directory + "/scene", this->camera);
                if (!started || !started->is_open()) {
                    this->message = "Failed to start a recording in " + this->output_directory;
                }
                else {
                    this->message = "Recording to " + started->get_directory();
                    this->recording = std::move(started);
                }
                this->publish_requested = true;
            }
        }
        if (finished) {
            // The writer drains what it holds outside the lock, so that the camera is not held up.
            finished->finish();
            long long written = 0;
            long long skipped = 0;
            bool failed = false;
            finished->counts(written, skipped, failed);
            std::lock_guard<std::mutex> lock(this->mutex);
            this->message = (failed ? "Failed to write the recording to " : ("Recorded " + std::to_string(written) + " frames (" + std::to_string(skipped) + " skipped) to ")) + finished->get_directory();
            this->publish_requested = true;
        }
        this->wake.notify_one();
    }

    bool session::update() {
        return this->snapshots.update_read();
    }

    const snapshot& session::current() {
        return this->snapshots.get_read();
    }

    void session::wait_until_idle() {
        std::unique_lock<std::mutex> lock(this->mutex);
        this->idle.wait(lock, [this]() {
            return !this->running || (!this->busy && !this->frame_waiting && !this->reset_requested && !this->save_requested && !this->publish_requested);
        });
    }

    void session::run() {
        for (;;) {
            bool reset_now = false;
            bool save_now = false;
            bool frame_now = false;
            bool publish_now = false;
            long long timestamp_nanoseconds = 0;
            {
                std::unique_lock<std::mutex> lock(this->mutex);
                this->busy = false;
                this->idle.notify_all();
                this->wake.wait(lock, [this]() {
                    return this->stopping || this->frame_waiting || this->reset_requested || this->save_requested || this->publish_requested;
                });
                if (this->stopping) {
                    return;
                }
                this->busy = true;
                reset_now = std::exchange(this->reset_requested, false);
                save_now = std::exchange(this->save_requested, false);
                frame_now = std::exchange(this->frame_waiting, false);
                publish_now = std::exchange(this->publish_requested, false);
                if (frame_now) {
                    this->working_image.swap(this->waiting_image);
                    timestamp_nanoseconds = this->waiting_timestamp_nanoseconds;
                }
            }
            if (reset_now) {
                std::string error;
                const bool created = this->create_system(error);
                this->fed_timestamps.clear();
                this->frames_since_refresh = 0;
                this->ever_posed = false;
                this->keyframe_timestamps.clear();
                this->latest.posed = false;
                this->latest.state = tracking_state::initialising;
                this->latest.landmarks.clear();
                this->latest.keyframes.clear();
                ++this->latest.map_sequence;
                this->set_message(created ? std::string("Reset") : error);
                this->publish();
            }
            if (frame_now) {
                this->process(timestamp_nanoseconds);
            }
            if (save_now) {
                this->save_run();
                this->publish();
            }
            else if (publish_now && !frame_now && !reset_now) {
                this->publish();
            }
        }
    }

    bool session::create_system(std::string& error) {
        this->system = zeroslam::system();
        if (!this->system.is_valid()) {
            error = "Failed to create the system";
            return false;
        }
        if (!this->configuration.empty() && (this->system.set_configuration(this->configuration.data(), static_cast<int>(this->configuration.size())) != zeroslam_return_success)) {
            error = "The library rejected the configuration";
            return false;
        }
        zeroslam_sensor_parameters_camera_struct parameters = this->camera;
        zeroslam_sensor_rig_struct rig{};
        rig.type = zeroslam_sensor_camera;
        rig.sensor_id = sensor_id;
        rig.parameters_length = static_cast<int>(sizeof(parameters));
        rig.parameters_data = &parameters;
        const zeroslam_return_enum result = this->system.set_sensor_rig(&rig, 1);
        if (result != zeroslam_return_success) {
            error = std::string("The library rejected the camera: ") + zeroslam_return_enum_to_string(result);
            return false;
        }
        return true;
    }

    void session::process(const long long timestamp_nanoseconds) {
        // The library takes strictly increasing timestamps, which a camera restarted after a pause still gives.
        if (timestamp_nanoseconds <= this->last_timestamp_nanoseconds) {
            return;
        }
        zeroslam_sensor_data_struct measurement{};
        measurement.timestamp = timestamp_nanoseconds;
        measurement.sensor_id = sensor_id;
        measurement.measurement_length = static_cast<int>(this->working_image.size());
        measurement.measurement_data = this->working_image.data();
        const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
        const zeroslam_return_enum result = this->system.set_sensor_data(&measurement, 1);
        const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        if (result != zeroslam_return_success) {
            this->set_message(std::string("The library rejected a frame: ") + zeroslam_return_enum_to_string(result));
            return;
        }
        this->last_timestamp_nanoseconds = timestamp_nanoseconds;
        this->fed_timestamps.push_back(timestamp_nanoseconds);
        ++this->frames_processed;
        this->latest.process_milliseconds = milliseconds;
        this->latest.average_process_milliseconds = (this->frames_processed == 1) ? milliseconds : ((0.95 * this->latest.average_process_milliseconds) + (0.05 * milliseconds));

        zeroslam_pose_struct pose{};
        this->latest.posed = this->system.get_pose_at_timestamp(&pose, timestamp_nanoseconds) == zeroslam_return_success;
        if (this->latest.posed) {
            this->ever_posed = true;
            for (int i = 0; i < 7; ++i) {
                this->latest.pose[i] = pose.pose[i];
            }
        }
        this->latest.state = this->latest.posed ? tracking_state::tracking : (this->ever_posed ? tracking_state::lost : tracking_state::initialising);
        this->latest.image = this->working_image;
        this->latest.timestamp_nanoseconds = timestamp_nanoseconds;
        ++this->latest.frame_sequence;

        zeroslam_map_keyframes_struct keyframe_count{};
        const zeroslam_return_enum counted = this->system.get_map_keyframes(&keyframe_count);
        const std::size_t keyframes_now = (counted == zeroslam_return_failure_insufficient_data_length) ? static_cast<std::size_t>(keyframe_count.keyframes_length) : 0;
        if ((++this->frames_since_refresh >= refresh_frames) || (keyframes_now != this->keyframe_timestamps.size())) {
            this->refresh_map();
        }
        this->publish();
    }

    void session::refresh_map() {
        this->frames_since_refresh = 0;
        this->latest.landmarks.clear();
        zeroslam_map_chunk_struct chunk{};
        zeroslam_return_enum result = this->system.get_map_chunk(0.0f, 0.0f, 0.0f, &chunk);
        if (result == zeroslam_return_failure_insufficient_data_length) {
            this->points.resize(static_cast<std::size_t>(chunk.points_length));
            chunk.points = this->points.data();
            result = this->system.get_map_chunk(0.0f, 0.0f, 0.0f, &chunk);
        }
        if (result == zeroslam_return_success) {
            this->latest.landmarks.reserve(3 * static_cast<std::size_t>(chunk.points_length));
            for (int i = 0; i < chunk.points_length; ++i) {
                const zeroslam_point_struct& point = this->points[static_cast<std::size_t>(i)];
                this->latest.landmarks.push_back(point.x);
                this->latest.landmarks.push_back(point.y);
                this->latest.landmarks.push_back(point.z);
            }
        }
        else {
            this->points.clear();
        }
        this->latest.keyframes.clear();
        zeroslam_map_keyframes_struct keyframes{};
        result = this->system.get_map_keyframes(&keyframes);
        if (result == zeroslam_return_failure_insufficient_data_length) {
            this->keyframe_timestamps.resize(static_cast<std::size_t>(keyframes.keyframes_length));
            keyframes.keyframes = this->keyframe_timestamps.data();
            result = this->system.get_map_keyframes(&keyframes);
        }
        if (result != zeroslam_return_success) {
            this->keyframe_timestamps.clear();
        }
        this->keyframe_timestamps.resize(static_cast<std::size_t>((result == zeroslam_return_success) ? keyframes.keyframes_length : 0));
        for (const long long keyframe_timestamp : this->keyframe_timestamps) {
            zeroslam_pose_struct pose{};
            if (this->system.get_pose_at_timestamp(&pose, keyframe_timestamp) == zeroslam_return_success) {
                this->latest.keyframes.push_back(static_cast<float>(pose.pose[0]));
                this->latest.keyframes.push_back(static_cast<float>(pose.pose[1]));
                this->latest.keyframes.push_back(static_cast<float>(pose.pose[2]));
            }
        }
        ++this->latest.map_sequence;
    }

    void session::publish() {
        {
            std::lock_guard<std::mutex> lock(this->mutex);
            this->latest.frames_received = this->frames_received;
            this->latest.frames_dropped = this->frames_dropped;
            this->latest.frames_processed = this->frames_processed;
            this->latest.recording = static_cast<bool>(this->recording);
            if (this->recording) {
                bool failed = false;
                this->recording->counts(this->latest.frames_recorded, this->latest.frames_unrecorded, failed);
                if (failed) {
                    this->message = "Failed to write the recording to " + this->recording->get_directory();
                }
            }
            this->latest.message = this->message;
        }
        this->snapshots.get_write() = this->latest;
        this->snapshots.update_write();
    }

    void session::save_run() {
        const std::string directory = this->make_run_directory();
        if (directory.empty()) {
            this->set_message("Failed to create a directory in " + this->output_directory);
            return;
        }
        gtl::file trajectory = create_file(directory + "/trajectory.txt");
        std::size_t poses = 0;
        bool written = trajectory.is_open();
        for (std::size_t i = 0; written && (i < this->fed_timestamps.size()); ++i) {
            zeroslam_pose_struct pose{};
            if (this->system.get_pose_at_timestamp(&pose, this->fed_timestamps[i]) != zeroslam_return_success) {
                continue;
            }
            written = dataset::write_pose_line(trajectory, pose.timestamp, &pose.pose[0], &pose.pose[3]);
            ++poses;
        }
        written = trajectory.close() && written;
        // The landmarks as the last refresh saw them, coloured as the library colours them.
        gtl::file map = create_file(directory + "/map.ply");
        written = written && map.is_open();
        written = written && dataset::write_formatted_line(map, "ply\nformat ascii 1.0\nelement vertex %zu\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n", this->points.size());
        for (std::size_t i = 0; written && (i < this->points.size()); ++i) {
            const zeroslam_point_struct& point = this->points[i];
            written = dataset::write_formatted_line(map, "%.6f %.6f %.6f %u %u %u\n", static_cast<double>(point.x), static_cast<double>(point.y), static_cast<double>(point.z), static_cast<unsigned int>(to_byte(point.r)), static_cast<unsigned int>(to_byte(point.g)), static_cast<unsigned int>(to_byte(point.b)));
        }
        written = map.close() && written;
        this->set_message(written ? ("Saved " + std::to_string(poses) + " poses and " + std::to_string(this->points.size()) + " landmarks to " + directory) : ("Failed to write the run to " + directory));
    }

    std::string session::make_run_directory() const {
        const std::string base = this->output_directory + "/" + local_time_name();
        for (int attempt = 0; attempt < 100; ++attempt) {
            const std::string directory = (attempt == 0) ? base : (base + "-" + std::to_string(attempt));
            std::vector<std::string> names;
            if (gtl::directory::list_directory(directory, names)) {
                continue;
            }
            if (gtl::directory::make_directories(directory)) {
                return directory;
            }
            return "";
        }
        return "";
    }

    void session::set_message(const std::string& text) {
        std::lock_guard<std::mutex> lock(this->mutex);
        this->message = text;
    }
}
