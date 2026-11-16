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

// The app's camera model and session on a desktop, where the camera is a rendered corridor, so that the parts of the app that
// do not need a phone are tested wherever the library is.

#include "camera_model.hpp"
#include "dataset.hpp"
#include "directory.hpp"
#include "session.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#define REQUIRE(EXPRESSION) require((EXPRESSION), #EXPRESSION, __LINE__)

namespace {
    void require(const bool passed, const char* const expression, const int line) {
        if (passed) {
            return;
        }
        std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", line, expression);
        std::exit(EXIT_FAILURE);
    }

    bool close_to(const double value, const double expected) {
        return std::abs(value - expected) <= 1e-9 * (1.0 + std::abs(expected));
    }

    void test_choose_stream_size() {
        mobile::camera_characteristics characteristics;
        characteristics.region[2] = 4032.0;
        characteristics.region[3] = 3024.0;
        const std::vector<mobile::stream_size> sizes = { { 1920, 1080 }, { 1280, 960 }, { 720, 480 }, { 640, 480 }, { 320, 240 }, { 640, 360 } };
        mobile::stream_size chosen;
        REQUIRE(mobile::choose_stream_size(sizes, characteristics, 640, chosen));
        REQUIRE((chosen.width == 640) && (chosen.height == 480));
        REQUIRE(mobile::choose_stream_size(sizes, characteristics, 1000, chosen));
        REQUIRE((chosen.width == 1280) && (chosen.height == 960));
        // Without a size of the region's aspect ratio the closest in width is taken.
        const std::vector<mobile::stream_size> wide = { { 1920, 1080 }, { 640, 360 } };
        REQUIRE(mobile::choose_stream_size(wide, characteristics, 640, chosen));
        REQUIRE((chosen.width == 640) && (chosen.height == 360));
        REQUIRE(!mobile::choose_stream_size({}, characteristics, 640, chosen));
        std::printf("choose_stream_size: passed\n");
    }

    void test_stream_camera() {
        mobile::camera_characteristics characteristics;
        characteristics.region[0] = 0.0;
        characteristics.region[1] = 0.0;
        characteristics.region[2] = 4000.0;
        characteristics.region[3] = 3000.0;
        characteristics.calibrated = true;
        characteristics.intrinsics[0] = 3200.0;
        characteristics.intrinsics[1] = 3210.0;
        characteristics.intrinsics[2] = 2010.0;
        characteristics.intrinsics[3] = 1490.0;
        characteristics.distorted = true;
        characteristics.distortion[0] = 0.1;
        characteristics.distortion[1] = -0.2;
        characteristics.distortion[2] = 0.3;
        characteristics.distortion[3] = 0.004;
        characteristics.distortion[4] = -0.005;

        // A stream of the region's aspect ratio is the whole region scaled.
        zeroslam_sensor_parameters_camera_struct camera{};
        REQUIRE(mobile::stream_camera(characteristics, 640, 480, camera));
        REQUIRE((camera.width == 640) && (camera.height == 480));
        REQUIRE(close_to(camera.focal_x, 3200.0 * 0.16) && close_to(camera.focal_y, 3210.0 * 0.16));
        REQUIRE(close_to(camera.centre_x, 2010.0 * 0.16) && close_to(camera.centre_y, 1490.0 * 0.16));
        // [k1 k2 p1 p2 k3 k4 k5 k6] from the device's [k1 k2 k3] radial and [p1 p2] tangential.
        REQUIRE(close_to(camera.distortion[0], 0.1) && close_to(camera.distortion[1], -0.2) && close_to(camera.distortion[2], 0.004) && close_to(camera.distortion[3], -0.005) && close_to(camera.distortion[4], 0.3));
        REQUIRE((camera.distortion[5] == 0.0) && (camera.distortion[6] == 0.0) && (camera.distortion[7] == 0.0));

        // A wider stream is a band across the middle of the region: 4000 x 2250 scaled to 1280 x 720.
        REQUIRE(mobile::stream_camera(characteristics, 1280, 720, camera));
        REQUIRE(close_to(camera.focal_x, 3200.0 * 0.32) && close_to(camera.focal_y, 3210.0 * 0.32));
        REQUIRE(close_to(camera.centre_x, 2010.0 * 0.32) && close_to(camera.centre_y, (1490.0 - 375.0) * 0.32));

        // A narrower one is a column: 3000 x 3000 scaled to 600 x 600, starting 500 pixels in.
        REQUIRE(mobile::stream_camera(characteristics, 600, 600, camera));
        REQUIRE(close_to(camera.focal_x, 3200.0 * 0.2) && close_to(camera.centre_x, (2010.0 - 500.0) * 0.2) && close_to(camera.centre_y, 1490.0 * 0.2));

        // A region offset into the array moves the principal point with it.
        characteristics.region[0] = 8.0;
        characteristics.region[1] = 6.0;
        REQUIRE(mobile::stream_camera(characteristics, 640, 480, camera));
        REQUIRE(close_to(camera.centre_x, (2010.0 - 8.0) * 0.16) && close_to(camera.centre_y, (1490.0 - 6.0) * 0.16));

        // Without a calibration the focal length comes from the lens and the sensor, the centre is the region's centre and
        // there is no distortion.
        characteristics.calibrated = false;
        characteristics.focal_length_millimetres = 4.0;
        characteristics.physical_size_millimetres[0] = 5.0;
        characteristics.physical_size_millimetres[1] = 3.75;
        characteristics.pixel_array_size[0] = 4016;
        characteristics.pixel_array_size[1] = 3012;
        REQUIRE(mobile::stream_camera(characteristics, 640, 480, camera));
        REQUIRE(close_to(camera.focal_x, 4.0 * 4016.0 / 5.0 * 0.16) && close_to(camera.focal_y, 4.0 * 3012.0 / 3.75 * 0.16));
        REQUIRE(close_to(camera.centre_x, 320.0) && close_to(camera.centre_y, 240.0));
        REQUIRE(camera.distortion[0] == 0.0);

        // Nothing to go on.
        characteristics.focal_length_millimetres = 0.0;
        REQUIRE(!mobile::stream_camera(characteristics, 640, 480, camera));
        std::printf("stream_camera: passed\n");
    }

    // A corridor of hashed tiles, as in the api demonstrations, seen from camera_x along it, into rows stride bytes apart.
    void render_corridor(std::vector<unsigned char>& pixels, const int stride, const zeroslam_sensor_parameters_camera_struct& camera, const double camera_x) {
        constexpr static const double wall_depth = 6.0;
        for (int row = 0; row < camera.height; ++row) {
            const double ray_y = ((static_cast<double>(row) + 0.5) - camera.centre_y) / camera.focal_y;
            const bool wall = (std::abs(ray_y) * wall_depth) <= 1.0;
            const double depth = wall ? wall_depth : (1.0 / std::abs(ray_y));
            const unsigned long long int surface = wall ? 1ull : ((ray_y > 0.0) ? 2ull : 3ull);
            const long long int cell_v = static_cast<long long int>(std::floor(4.0 * (wall ? (ray_y * depth) : depth)));
            for (int column = 0; column < camera.width; ++column) {
                const double ray_x = ((static_cast<double>(column) + 0.5) - camera.centre_x) / camera.focal_x;
                const long long int cell_u = static_cast<long long int>(std::floor(4.0 * (camera_x + (ray_x * depth))));
                unsigned long long int hash = (static_cast<unsigned long long int>(cell_u) * 73856093ull) ^ (static_cast<unsigned long long int>(cell_v) * 19349663ull) ^ (surface * 83492791ull);
                hash ^= hash >> 13;
                hash *= 0x5bd1e995ull;
                hash ^= hash >> 15;
                pixels[static_cast<std::size_t>((row * stride) + column)] = static_cast<unsigned char>(40ull + (hash % 176ull));
            }
        }
    }

    std::size_t count_lines(const std::string& path) {
        std::string text;
        if (!dataset::read_text_file(path, text)) {
            return 0;
        }
        std::size_t lines = 0;
        for (const char character : text) {
            lines += (character == '\n') ? 1u : 0u;
        }
        return lines;
    }

    std::string only_entry(const std::string& directory) {
        std::vector<std::string> names;
        if (!gtl::directory::list_directory(directory, names) || (names.size() != 1)) {
            return "";
        }
        return directory + "/" + names[0];
    }

    void test_session(const std::string& output_directory) {
        zeroslam_sensor_parameters_camera_struct camera{};
        camera.width = 320;
        camera.height = 240;
        camera.focal_x = 262.5;
        camera.focal_y = 262.5;
        camera.centre_x = 160.0;
        camera.centre_y = 120.0;
        // The camera's rows are padded, as a camera's often are.
        const int stride = camera.width + 64;
        std::vector<unsigned char> pixels(static_cast<std::size_t>(stride * camera.height), 0);

        mobile::session session(output_directory + "/runs");
        std::string error;
        REQUIRE(!session.is_running());
        REQUIRE(!session.start(camera, "detector=unknown\n", error));
        REQUIRE(!error.empty());
        REQUIRE(session.start(camera, "verbosity=0\n", error));
        REQUIRE(session.is_running());
        REQUIRE(session.update());
        REQUIRE(session.current().frames_processed == 0);
        REQUIRE(session.current().state == mobile::tracking_state::initialising);

        session.toggle_recording();
        constexpr static const int frame_count = 30;
        constexpr static const long long frame_interval = 33333333ll;
        for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
            render_corridor(pixels, stride, camera, 0.05 * frame_index);
            mobile::frame_view frame;
            frame.pixels = pixels.data();
            frame.width = camera.width;
            frame.height = camera.height;
            frame.row_stride = stride;
            frame.timestamp_nanoseconds = 1000000000ll + (frame_interval * frame_index);
            session.submit(frame);
            session.wait_until_idle();
        }
        // A frame of another size, or one that is not later than the last, is not processed.
        mobile::frame_view wrong;
        wrong.pixels = pixels.data();
        wrong.width = camera.width - 1;
        wrong.height = camera.height;
        wrong.row_stride = stride;
        wrong.timestamp_nanoseconds = 5000000000ll;
        session.submit(wrong);
        session.wait_until_idle();
        session.toggle_recording();
        session.wait_until_idle();
        REQUIRE(session.update());
        const mobile::snapshot& tracked = session.current();
        std::printf("session: %s after %lld frames (%lld received, %lld dropped), %zu landmarks, %zu keyframes, %.1f ms a frame\n", mobile::tracking_state_name(tracked.state), tracked.frames_processed, tracked.frames_received, tracked.frames_dropped, tracked.landmarks.size() / 3, tracked.keyframes.size() / 3, tracked.average_process_milliseconds);
        REQUIRE(tracked.frames_received == frame_count);
        REQUIRE(tracked.frames_processed == frame_count);
        REQUIRE(tracked.frames_dropped == 0);
        REQUIRE(tracked.state == mobile::tracking_state::tracking);
        REQUIRE(tracked.posed);
        REQUIRE(tracked.landmarks.size() >= 3 * 50);
        REQUIRE(tracked.keyframes.size() >= 3 * 2);
        REQUIRE((tracked.landmarks.size() % 3) == 0);
        REQUIRE(tracked.image.size() == static_cast<std::size_t>(camera.width * camera.height));
        // The image is the last frame without its padding.
        REQUIRE(tracked.image[static_cast<std::size_t>(camera.width)] == pixels[static_cast<std::size_t>(stride)]);
        REQUIRE(tracked.timestamp_nanoseconds == 1000000000ll + (frame_interval * (frame_count - 1)));
        // The camera moves along +x, the corridor's length, which the map's first frame looks across.
        REQUIRE(tracked.pose[0] > 0.0);
        REQUIRE(tracked.message.find("Recorded 30 frames") == 0);

        // The recording is a scene in the directory form, a frame and a calibration line each.
        const std::string recording = only_entry(output_directory + "/runs");
        REQUIRE(!recording.empty());
        std::vector<std::string> frames;
        REQUIRE(gtl::directory::list_directory(recording + "/scene/sensor/image_01", frames));
        REQUIRE(frames.size() == static_cast<std::size_t>(frame_count));
        REQUIRE(frames[0] == "00000000001000000000.pnm");
        REQUIRE(count_lines(recording + "/scene/sensor/image_01.txt") == static_cast<std::size_t>(frame_count));
        dataset::pnm_format format = dataset::pnm_format::ppm;
        unsigned int width = 0;
        unsigned int height = 0;
        std::vector<unsigned char> stored;
        REQUIRE(dataset::read_pnm(recording + "/scene/sensor/image_01/" + frames.back(), format, width, height, stored));
        REQUIRE((format == dataset::pnm_format::pgm8) && (width == 320) && (height == 240) && (stored == tracked.image));

        // Saving writes a pose a posed frame and every landmark.
        session.save();
        session.wait_until_idle();
        REQUIRE(session.update());
        const mobile::snapshot& saved = session.current();
        std::printf("session: %s\n", saved.message.c_str());
        REQUIRE(saved.message.find("Saved ") == 0);
        const std::string run = saved.message.substr(saved.message.rfind(' ') + 1);
        const std::size_t poses = count_lines(run + "/trajectory.txt");
        REQUIRE((poses >= 20) && (poses <= static_cast<std::size_t>(frame_count)));
        REQUIRE(count_lines(run + "/map.ply") == 10 + (saved.landmarks.size() / 3));

        // A reset starts the map again.
        session.reset();
        session.wait_until_idle();
        REQUIRE(session.update());
        const mobile::snapshot& reset = session.current();
        REQUIRE(reset.message == "Reset");
        REQUIRE(reset.state == mobile::tracking_state::initialising);
        REQUIRE(!reset.posed && reset.landmarks.empty() && reset.keyframes.empty());

        session.stop();
        REQUIRE(!session.is_running());
        // Nothing is taken once stopped.
        session.submit(wrong);
        std::printf("session: passed\n");
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage %s [empty output directory]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const std::string output_directory = argv[1];
    std::vector<std::string> names;
    REQUIRE(gtl::directory::make_directories(output_directory));
    REQUIRE(gtl::directory::list_directory(output_directory, names) && names.empty());
    test_choose_stream_size();
    test_stream_camera();
    test_session(output_directory);
    return EXIT_SUCCESS;
}
