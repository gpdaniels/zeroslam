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

#include "directory.hpp"
#include "file.hpp"
#include "import.hpp"
#include "json.hpp"
#include "paths.hpp"
#include "rotation.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    struct camera_calibration {
        double width = 0.0;
        double height = 0.0;
        double intrinsics[4] = {};
        double translation[3] = {};
        double quaternion_xyzw[4] = { 0.0, 0.0, 0.0, 1.0 };
    };

    struct frame_reference {
        long long timestamp_nanoseconds = 0;
        std::string filename;
    };

    struct imu_sample {
        long long timestamp_nanoseconds = 0;
        double angular_velocity[3] = {};
        double linear_acceleration[3] = {};
    };

    using rigid = rotation::rigid;

    // The glasses hold both cameras on their side, the image x axis pointing down, so every frame is turned 90 degrees
    // clockwise upright. A pixel-centre position (u, v) of a height h image moves to (h - v, u): the upright camera's x
    // axis is the original's -y and its y axis the original's x, a -90 degree turn about the optical axis.
    constexpr static const double upright_translation[3] = { 0.0, 0.0, 0.0 };
    constexpr static const double upright_quaternion_xyzw[4] = { 0.0, 0.0, -0.70710678118654752, 0.70710678118654752 };

    void rotate_intrinsics_upright(camera_calibration& camera) {
        const double fx = camera.intrinsics[0];
        const double fy = camera.intrinsics[1];
        const double cx = camera.intrinsics[2];
        const double cy = camera.intrinsics[3];
        camera.intrinsics[0] = fy;
        camera.intrinsics[1] = fx;
        camera.intrinsics[2] = camera.height - cy;
        camera.intrinsics[3] = cx;
        std::swap(camera.width, camera.height);
    }

    const gtl::json::value* find_member(const gtl::json::value& object, const char* key) {
        if (!object.is<gtl::json::value::object_type>()) {
            return nullptr;
        }
        const gtl::json::value::object_type& members = object.as<gtl::json::value::object_type>();
        const gtl::json::value::object_type::const_iterator iterator = members.find(key);
        return (iterator == members.end()) ? nullptr : &iterator->second;
    }

    bool read_numbers(const gtl::json::value* array, double* values, const std::size_t count) {
        if ((array == nullptr) || !array->is<gtl::json::value::array_type>()) {
            return false;
        }
        const gtl::json::value::array_type& items = array->as<gtl::json::value::array_type>();
        if (items.size() != count) {
            return false;
        }
        for (std::size_t index = 0; index < count; ++index) {
            if (!items[index].is<gtl::json::value::number_type>()) {
                return false;
            }
            values[index] = items[index].as<gtl::json::value::number_type>();
        }
        return true;
    }

    // A 'T_b_s' entry: the sensor's pose in the body (the right imu), a scalar-last 'qvec' and a 'tvec'.
    bool read_body_from_sensor(const gtl::json::value& sensor, double translation[3], double quaternion_xyzw[4]) {
        const gtl::json::value* transform = find_member(sensor, "T_b_s");
        return (transform != nullptr) && read_numbers(find_member(*transform, "qvec"), quaternion_xyzw, 4) && read_numbers(find_member(*transform, "tvec"), translation, 3);
    }

    bool read_pinhole_calibration(const std::string& path, camera_calibration& cam0, camera_calibration& cam1, double imu_translation[3], double imu_quaternion_xyzw[4], std::string& error) {
        std::string text;
        if (!dataset::read_text_file(path, text)) {
            error = "cannot read '" + path + "'";
            return false;
        }
        gtl::json document;
        if (!document.parse(text)) {
            error = "'" + path + "' is not valid json";
            return false;
        }
        const gtl::json::value& root = document.document();
        const char* const camera_names[2] = { "cam0", "cam1" };
        camera_calibration* const cameras[2] = { &cam0, &cam1 };
        for (int index = 0; index < 2; ++index) {
            const gtl::json::value* camera = find_member(root, camera_names[index]);
            if (camera == nullptr) {
                error = "'" + path + "' has no '" + camera_names[index] + "'";
                return false;
            }
            const gtl::json::value* model = find_member(*camera, "model");
            if ((model == nullptr) || !model->is<gtl::json::value::string_type>() || (model->as<gtl::json::value::string_type>() != "PINHOLE")) {
                error = "'" + path + "' " + camera_names[index] + " is not a 'PINHOLE' camera (use the pinhole calibration, not the aria one)";
                return false;
            }
            if (!read_numbers(find_member(*camera, "params"), &cameras[index]->intrinsics[0], 4)) {
                error = "'" + path + "' " + camera_names[index] + " has no 4 element 'params'";
                return false;
            }
            const gtl::json::value* resolution = find_member(*camera, "resolution");
            const gtl::json::value* width = (resolution == nullptr) ? nullptr : find_member(*resolution, "width");
            const gtl::json::value* height = (resolution == nullptr) ? nullptr : find_member(*resolution, "height");
            if ((width == nullptr) || !width->is<gtl::json::value::number_type>() || (height == nullptr) || !height->is<gtl::json::value::number_type>()) {
                error = "'" + path + "' " + camera_names[index] + " has no 'resolution' width and height";
                return false;
            }
            cameras[index]->width = width->as<gtl::json::value::number_type>();
            cameras[index]->height = height->as<gtl::json::value::number_type>();
            if (!read_body_from_sensor(*camera, cameras[index]->translation, cameras[index]->quaternion_xyzw)) {
                error = "'" + path + "' " + camera_names[index] + " has no 'T_b_s' qvec and tvec";
                return false;
            }
            // No pixel_centre_principal_point shift: these intrinsics come out of colmap's image_undistorter, and colmap
            // already puts the centre of pixel (0, 0) at (0.5, 0.5), which is this project's pixel-centre frame.
        }
        const gtl::json::value* imu = find_member(root, "imu0");
        if ((imu == nullptr) || !read_body_from_sensor(*imu, imu_translation, imu_quaternion_xyzw)) {
            error = "'" + path + "' has no 'imu0' with a 'T_b_s' qvec and tvec";
            return false;
        }
        return true;
    }

    bool read_frame_csv(const std::string& path, std::vector<frame_reference>& frames, std::string& error) {
        return import::for_each_line(path, error, [&frames](const char* line) {
            long long timestamp = 0;
            char filename[256] = {};
            if (std::sscanf(line, "%lld,%255[^,\r\n]", &timestamp, &filename[0]) == 2) {
                frames.push_back({ timestamp, std::string(&filename[0]) });
            }
        });
    }

    // The pseudo dense ground truth, '[timestamp] [tx] [ty] [tz] [qx] [qy] [qz] [qw]' per keyframe, a world_from_sensor
    // pose. The documentation calls the sensor the left camera, but the poses turn with the imu's gyroscope, not with
    // cam0 (whose axes are about 105 degrees apart), so they are the imu's (the body's) and are carried onto cam0.
    bool read_ground_truth_file(const std::string& path, std::vector<import::ground_truth_sample>& samples, std::string& error) {
        return import::for_each_line(path, error, [&samples](const char* line) {
            import::ground_truth_sample sample;
            double timestamp = 0.0;
            if (std::sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf", &timestamp, &sample.position[0], &sample.position[1], &sample.position[2], &sample.quaternion_xyzw[0], &sample.quaternion_xyzw[1], &sample.quaternion_xyzw[2], &sample.quaternion_xyzw[3]) == 8) {
                sample.timestamp_nanoseconds = std::llround(timestamp);
                samples.push_back(sample);
            }
        });
    }

    bool read_imu_csv(const std::string& path, std::vector<imu_sample>& samples, std::string& error) {
        return import::for_each_line(path, error, [&samples](const char* line) {
            imu_sample sample;
            if (std::sscanf(line, "%lld,%lf,%lf,%lf,%lf,%lf,%lf", &sample.timestamp_nanoseconds, &sample.angular_velocity[0], &sample.angular_velocity[1], &sample.angular_velocity[2], &sample.linear_acceleration[0], &sample.linear_acceleration[1], &sample.linear_acceleration[2]) == 7) {
                samples.push_back(sample);
            }
        });
    }

    // A sequence holds tens of thousands of frames each converted by a separate process, so the conversions run in parallel.
    bool convert_images(const std::vector<std::pair<std::string, std::string>>& conversions, std::string& error) {
        std::atomic<std::size_t> next(0);
        std::atomic<bool> failed(false);
        std::mutex error_mutex;
        const auto worker = [&]() {
            for (std::size_t index = next++; (index < conversions.size()) && !failed; index = next++) {
                std::string conversion_error;
                // The 'pgm:' prefix keeps a single colour frame (the overexposed start of a sequence) 8-bit greyscale:
                // left to pick the '.pnm' format itself, convert would write such a frame as a 1-bit bitmap.
                if (!import::convert_image(conversions[index].first, "-colorspace Gray -rotate 90", "pgm:" + conversions[index].second, conversion_error)) {
                    const std::lock_guard<std::mutex> lock(error_mutex);
                    if (!failed.exchange(true)) {
                        error = conversion_error;
                    }
                }
                if ((index + 1) % 2000 == 0) {
                    std::printf("    converted %zu of %zu frames...\n", index + 1, conversions.size());
                    std::fflush(stdout);
                }
            }
        };
        const unsigned int thread_count = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> threads;
        for (unsigned int index = 0; index < thread_count; ++index) {
            threads.emplace_back(worker);
        }
        for (std::thread& thread : threads) {
            thread.join();
        }
        return !failed;
    }

    void print_usage(const char* argv0) {
        std::printf("Usage %s [asl-dir] [pinhole.json] [output.mcap] [options...]\n", argv0);
        std::printf("    asl-dir        - An already extracted LaMAria ASL folder (the 'aria/' directory holding\n");
        std::printf("                     cam0/, cam1/, and imu0/, or the directory containing it; unzip the\n");
        std::printf("                     'ASL (Pinhole)' download first; this tool does not read zips).\n");
        std::printf("    pinhole.json   - The sequence's 'Pinhole calibration' download (not the aria one: the\n");
        std::printf("                     ASL images are already undistorted to it).\n");
        std::printf("    output.mcap    - Where to write the packed scene; the directory form is written\n");
        std::printf("                     alongside it (output.mcap's path with the extension dropped),\n");
        std::printf("                     and left in place for inspection, exactly as 'dataset expand' does.\n");
        std::printf("    options:\n");
        std::printf("        --ground-truth [txt] - The sequence's 'Pseudo-dense GT' download. Without one (the\n");
        std::printf("                               test sequences, and training sequences that have none)\n");
        std::printf("                               the scene carries no ground truth and keeps every frame.\n");
        std::printf("        --tools-dir [dir]    - Directory containing 'zeroslam-dataset' (default: next to\n");
        std::printf("                               this tool), used to pack and validate the result.\n");
        std::printf("Produces two cameras (image_01 = cam0, the left camera, image_02 = cam1) and one imu\n");
        std::printf("(imu_01, the right 1 kHz imu), all keeping the dataset's own timestamps, trimmed to the\n");
        std::printf("ground truth's coverage when there is one. The frames are turned 90 degrees clockwise out of\n");
        std::printf("the sideways orientation the glasses capture them in. The scene's ego frame is the upright\n");
        std::printf("cam0: the ground truth (the imu's pose) is carried onto it, and cam1 and the imu are posed\n");
        std::printf("relative to it through the calibration's imu body.\n");
    }
}

int main(int argc, char* argv[]) {
    std::string asl_directory;
    std::string calibration_path;
    std::string output_path;
    std::string ground_truth_path;
    std::string tools_directory_override;

    for (int i = 1; i < argc; ++i) {
        const auto matches = [&](const char* name) {
            return std::strcmp(argv[i], name) == 0;
        };
        if (matches("--help") || matches("-h")) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        else if (matches("--tools-dir") || matches("--ground-truth")) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for option: %s\n", argv[i]);
                return EXIT_FAILURE;
            }
            std::string& value = matches("--tools-dir") ? tools_directory_override : ground_truth_path;
            value = argv[++i];
        }
        else if (argv[i][0] == '-') {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
        else if (asl_directory.empty()) {
            asl_directory = argv[i];
        }
        else if (calibration_path.empty()) {
            calibration_path = argv[i];
        }
        else if (output_path.empty()) {
            output_path = argv[i];
        }
        else {
            std::fprintf(stderr, "Unexpected argument: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }
    if (output_path.empty()) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    // Accept the zip's top level directory as well as the 'aria/' directory inside it.
    if (!gtl::paths::is_directory(asl_directory + "/cam0") && gtl::paths::is_directory(asl_directory + "/aria/cam0")) {
        asl_directory += "/aria";
    }
    if (!gtl::paths::is_directory(asl_directory + "/cam0") || !gtl::paths::is_directory(asl_directory + "/cam1") || !gtl::paths::is_directory(asl_directory + "/imu0")) {
        std::fprintf(stderr, "'%s' is not a LaMAria ASL folder: it has no cam0/, cam1/, and imu0/ directories.\n", asl_directory.c_str());
        return EXIT_FAILURE;
    }

    std::string error;

    std::printf("Reading the pinhole calibration...\n");
    camera_calibration cam0;
    camera_calibration cam1;
    double imu_translation[3] = {};
    double imu_quaternion_xyzw[4] = {};
    if (!read_pinhole_calibration(calibration_path, cam0, cam1, imu_translation, imu_quaternion_xyzw, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }

    std::printf("Reading frame timestamps...\n");
    std::vector<frame_reference> cam0_frames;
    std::vector<frame_reference> cam1_frames;
    if (!read_frame_csv(asl_directory + "/cam0/data.csv", cam0_frames, error) || !read_frame_csv(asl_directory + "/cam1/data.csv", cam1_frames, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }
    const auto by_timestamp = [](const frame_reference& a, const frame_reference& b) {
        return a.timestamp_nanoseconds < b.timestamp_nanoseconds;
    };
    std::sort(cam0_frames.begin(), cam0_frames.end(), by_timestamp);
    std::sort(cam1_frames.begin(), cam1_frames.end(), by_timestamp);
    std::vector<std::pair<std::size_t, std::size_t>> matched_frames;
    {
        std::size_t cam0_index = 0;
        std::size_t cam1_index = 0;
        while ((cam0_index < cam0_frames.size()) && (cam1_index < cam1_frames.size())) {
            const long long timestamp_0 = cam0_frames[cam0_index].timestamp_nanoseconds;
            const long long timestamp_1 = cam1_frames[cam1_index].timestamp_nanoseconds;
            if (timestamp_0 == timestamp_1) {
                matched_frames.push_back({ cam0_index, cam1_index });
                ++cam0_index;
                ++cam1_index;
            }
            else if (timestamp_0 < timestamp_1) {
                ++cam0_index;
            }
            else {
                ++cam1_index;
            }
        }
    }
    std::printf("    %zu of cam0's %zu and cam1's %zu frames are present on both sides.\n", matched_frames.size(), cam0_frames.size(), cam1_frames.size());
    if (matched_frames.size() < 2) {
        std::fprintf(stderr, "Fewer than 2 frames are present on both cam0 and cam1.\n");
        return EXIT_FAILURE;
    }

    std::vector<import::ground_truth_sample> ground_truth;
    const bool has_ground_truth = !ground_truth_path.empty();
    if (!has_ground_truth) {
        std::printf("No ground truth given, so the scene will carry none and cannot be benchmarked against.\n");
    }
    else {
        std::printf("Reading ground truth...\n");
        if (!read_ground_truth_file(ground_truth_path, ground_truth, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return EXIT_FAILURE;
        }
        if (ground_truth.size() < 2) {
            std::fprintf(stderr, "'%s' holds fewer than 2 ground truth samples.\n", ground_truth_path.c_str());
            return EXIT_FAILURE;
        }
        std::sort(ground_truth.begin(), ground_truth.end(), [](const import::ground_truth_sample& a, const import::ground_truth_sample& b) {
            return a.timestamp_nanoseconds < b.timestamp_nanoseconds;
        });
    }
    const auto covered = [&](const long long timestamp_nanoseconds) {
        return !has_ground_truth || import::ground_truth_covers(ground_truth, timestamp_nanoseconds);
    };

    std::printf("Reading imu samples...\n");
    std::vector<imu_sample> imu_samples;
    if (!read_imu_csv(asl_directory + "/imu0/data.csv", imu_samples, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }
    std::sort(imu_samples.begin(), imu_samples.end(), [](const imu_sample& a, const imu_sample& b) {
        return a.timestamp_nanoseconds < b.timestamp_nanoseconds;
    });

    std::vector<std::pair<std::size_t, std::size_t>> retained_frames;
    for (const std::pair<std::size_t, std::size_t>& match : matched_frames) {
        if (covered(cam0_frames[match.first].timestamp_nanoseconds)) {
            retained_frames.push_back(match);
        }
    }
    if (retained_frames.size() < 2) {
        std::fprintf(stderr, "Fewer than 2 frames fall inside the ground truth's time coverage.\n");
        return EXIT_FAILURE;
    }
    if (has_ground_truth) {
        std::printf("    %zu of %zu matched frames are inside the ground truth's coverage.\n", retained_frames.size(), matched_frames.size());
    }

    const std::string output_directory = gtl::paths::path_stem(output_path);
    if (!gtl::directory::make_directories(output_directory + "/sensor/image_01") || !gtl::directory::make_directories(output_directory + "/sensor/image_02")) {
        std::fprintf(stderr, "Failed to create: %s\n", output_directory.c_str());
        return EXIT_FAILURE;
    }

    // The calibration poses every sensor on the right imu (the body): re-pose cam1 and the imu on the upright cam0, the ego.
    const rigid sideways_from_upright = rigid::from_pose(upright_translation, upright_quaternion_xyzw);
    const rigid body_from_cam0 = rigid::from_pose(cam0.translation, cam0.quaternion_xyzw) * sideways_from_upright;
    const rigid body_from_cam1 = rigid::from_pose(cam1.translation, cam1.quaternion_xyzw) * sideways_from_upright;
    const rigid body_from_imu = rigid::from_pose(imu_translation, imu_quaternion_xyzw);
    const rigid cam0_from_body = body_from_cam0.inverse();
    rotate_intrinsics_upright(cam0);
    rotate_intrinsics_upright(cam1);
    const double translation_01[3] = { 0.0, 0.0, 0.0 };
    const double quaternion_01[4] = { 0.0, 0.0, 0.0, 1.0 };
    double translation_02[3];
    double quaternion_02[4];
    (cam0_from_body * body_from_cam1).to_pose(translation_02, quaternion_02);
    double translation_imu[3];
    double quaternion_imu[4];
    (cam0_from_body * body_from_imu).to_pose(translation_imu, quaternion_imu);
    std::printf("Ego frame is cam0: cam1 at (%.4f, %.4f, %.4f) m, imu at (%.4f, %.4f, %.4f) m.\n", translation_02[0], translation_02[1], translation_02[2], translation_imu[0], translation_imu[1], translation_imu[2]);

    std::printf("Writing the calibration%s of %zu frames...\n", has_ground_truth ? " and the ground truth trajectory" : "", retained_frames.size());
    gtl::file calibration_01((output_directory + "/sensor/image_01.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    gtl::file calibration_02((output_directory + "/sensor/image_02.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    gtl::file trajectory;
    if (has_ground_truth) {
        trajectory.open((output_directory + "/trajectory.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    }
    else {
        import::remove_absent_scene_file(output_directory + "/trajectory.txt");
    }
    if (!calibration_01.is_open() || !calibration_02.is_open() || (has_ground_truth && !trajectory.is_open())) {
        std::fprintf(stderr, "Failed to create the calibration or trajectory files in '%s'.\n", output_directory.c_str());
        return EXIT_FAILURE;
    }
    std::vector<std::pair<std::string, std::string>> conversions;
    conversions.reserve(retained_frames.size() * 2);
    for (const std::pair<std::size_t, std::size_t>& index : retained_frames) {
        const long long timestamp = cam0_frames[index.first].timestamp_nanoseconds;
        if (has_ground_truth) {
            double body_position[3];
            double body_quaternion_xyzw[4];
            if (!import::interpolate_ground_truth(ground_truth, timestamp, body_position, body_quaternion_xyzw)) {
                std::fprintf(stderr, "Internal error: frame at %lld ns has no ground truth despite being inside its coverage.\n", timestamp);
                return EXIT_FAILURE;
            }
            double position[3];
            double quaternion_xyzw[4];
            (rigid::from_pose(body_position, body_quaternion_xyzw) * body_from_cam0).to_pose(position, quaternion_xyzw);
            if (!dataset::write_pose_line(trajectory, timestamp, position, quaternion_xyzw)) {
                std::fprintf(stderr, "Failed to write trajectory.\n");
                return EXIT_FAILURE;
            }
        }
        if (!import::write_calibration_line(calibration_01, timestamp, translation_01, quaternion_01, "plumb_bob", cam0.intrinsics, nullptr, 0) || !import::write_calibration_line(calibration_02, timestamp, translation_02, quaternion_02, "plumb_bob", cam1.intrinsics, nullptr, 0)) {
            std::fprintf(stderr, "Failed to write the calibration.\n");
            return EXIT_FAILURE;
        }
        const std::string frame_name = dataset::frame_filename(static_cast<unsigned long long>(timestamp));
        conversions.push_back({ asl_directory + "/cam0/data/" + cam0_frames[index.first].filename, output_directory + "/sensor/image_01/" + frame_name });
        conversions.push_back({ asl_directory + "/cam1/data/" + cam1_frames[index.second].filename, output_directory + "/sensor/image_02/" + frame_name });
    }
    calibration_01.close();
    calibration_02.close();
    trajectory.close();

    std::printf("Converting %zu frames...\n", conversions.size());
    std::fflush(stdout);
    if (!convert_images(conversions, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }

    std::printf("Writing imu samples...\n");
    std::size_t imu_covered = 0;
    for (const imu_sample& sample : imu_samples) {
        if (covered(sample.timestamp_nanoseconds)) {
            ++imu_covered;
        }
    }
    if (imu_covered == 0) {
        import::remove_absent_scene_file(output_directory + "/sensor/imu_01.txt");
        std::printf("    None of %zu imu samples are inside the ground truth's coverage, so the scene carries no imu.\n", imu_samples.size());
    }
    else {
        gtl::file imu_handle((output_directory + "/sensor/imu_01.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
        if (!imu_handle.is_open()) {
            std::fprintf(stderr, "Failed to create: %s/sensor/imu_01.txt\n", output_directory.c_str());
            return EXIT_FAILURE;
        }
        for (const imu_sample& sample : imu_samples) {
            if (!covered(sample.timestamp_nanoseconds)) {
                continue;
            }
            const double values[13] = { translation_imu[0], translation_imu[1], translation_imu[2], quaternion_imu[0], quaternion_imu[1], quaternion_imu[2], quaternion_imu[3], sample.angular_velocity[0], sample.angular_velocity[1], sample.angular_velocity[2], sample.linear_acceleration[0], sample.linear_acceleration[1], sample.linear_acceleration[2] };
            if (!dataset::write_sample_line(imu_handle, sample.timestamp_nanoseconds, &values[0], 13)) {
                std::fprintf(stderr, "Failed to write imu sample.\n");
                return EXIT_FAILURE;
            }
        }
        std::printf("    %zu of %zu imu samples are %s.\n", imu_covered, imu_samples.size(), has_ground_truth ? "inside the ground truth's coverage" : "written");
    }

    if (!import::collapse_scene(tools_directory_override, output_directory, output_path, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
