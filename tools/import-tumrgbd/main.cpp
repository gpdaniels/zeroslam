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
#include "paths.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    struct frame_reference {
        long long timestamp_nanoseconds = 0;
        std::string filename;
    };

    struct accelerometer_sample {
        long long timestamp_nanoseconds = 0;
        double linear_acceleration[3] = {};
    };

    constexpr static const double origin[3] = { 0.0, 0.0, 0.0 };
    constexpr static const double camera_mounting[4] = { 0.0, 0.0, 0.0, 1.0 };
    constexpr static const double accelerometer_mounting[4] = { 0.0, 0.0, 1.0, 0.0 };

    struct freiburg_calibration {
        const char* name;
        double intrinsics[4];
        double distortion[5];
        std::size_t distortion_count;
    };

    constexpr static const freiburg_calibration freiburg_calibrations[3] = {
        { "freiburg1", { 517.3, 516.5, 318.6, 255.3 }, { 0.2624, -0.9531, -0.0054, 0.0026, 1.1633 }, 5 },
        { "freiburg2", { 520.9, 521.0, 325.1, 249.7 }, { 0.2312, -0.7849, -0.0033, -0.0001, 0.9172 }, 5 },
        { "freiburg3", { 535.4, 539.2, 320.1, 247.6 }, { 0.0, 0.0, 0.0, 0.0, 0.0 }, 0 }
    };

    const freiburg_calibration* match_freiburg(const std::string& text) {
        for (const freiburg_calibration& calibration : freiburg_calibrations) {
            if (text.find(calibration.name) != std::string::npos) {
                return &calibration;
            }
        }
        return nullptr;
    }

    bool read_frame_file(const std::string& path, std::vector<frame_reference>& frames, std::string& error) {
        return import::for_each_line(path, error, [&frames](char* line) {
            char timestamp[64] = {};
            char filename[256] = {};
            long long timestamp_nanoseconds = 0;
            if ((std::sscanf(line, "%63s %255s", &timestamp[0], &filename[0]) == 2) && dataset::parse_timestamp_nanoseconds(&timestamp[0], timestamp_nanoseconds)) {
                frames.push_back({ timestamp_nanoseconds, std::string(&filename[0]) });
            }
        });
    }

    bool read_ground_truth_file(const std::string& path, std::vector<import::ground_truth_sample>& samples, std::string& error) {
        std::vector<double> values;
        return import::for_each_line(path, error, [&samples, &values](char* line) {
            import::ground_truth_sample sample;
            if (dataset::parse_sample_line(line, sample.timestamp_nanoseconds, values) && (values.size() == 7)) {
                std::copy(values.begin(), values.begin() + 3, &sample.position[0]);
                std::copy(values.begin() + 3, values.end(), &sample.quaternion_xyzw[0]);
                samples.push_back(sample);
            }
        });
    }

    bool read_accelerometer_file(const std::string& path, std::vector<accelerometer_sample>& samples, std::string& error) {
        std::vector<double> values;
        return import::for_each_line(path, error, [&samples, &values](char* line) {
            accelerometer_sample sample;
            if (dataset::parse_sample_line(line, sample.timestamp_nanoseconds, values) && (values.size() == 3)) {
                std::copy(values.begin(), values.end(), &sample.linear_acceleration[0]);
                samples.push_back(sample);
            }
        });
    }

    void print_usage(const char* argv0) {
        std::printf("Usage %s [tumrgbd-dir] [output.mcap] [options...]\n", argv0);
        std::printf("    tumrgbd-dir   - An extracted TUM RGB-D dataset directory (containing\n");
        std::printf("                     rgb.txt, depth.txt, groundtruth.txt, accelerometer.txt,\n");
        std::printf("                     rgb/, depth/).\n");
        std::printf("    output.mcap   - Where to write the packed scene; the directory form is written\n");
        std::printf("                     alongside it (output.mcap's path with the extension dropped),\n");
        std::printf("                     and left in place for inspection, exactly as 'dataset expand' does.\n");
        std::printf("    options:\n");
        std::printf("        --tools-dir [dir] - Directory containing 'zeroslam-dataset' (default: next to\n");
        std::printf("                            this tool), used to pack and validate the result.\n");
        std::printf("        --freiburg [1-3]  - Which camera the sequence was recorded on, setting the\n");
        std::printf("                            intrinsics and distortion written (default: read from the\n");
        std::printf("                            dataset's own rgb.txt header, then the directory name).\n");
        std::printf("Produces two cameras (image_01 = the rgb camera as rgb8, depth_01 = the depth camera in\n");
        std::printf("metres as 32-bit float) and one accelerometer (accelerometer_01, linear acceleration\n");
        std::printf("only), all keeping the dataset's own timestamps, trimmed to the ground truth's coverage.\n");
    }
}

int main(int argc, char* argv[]) {
    std::string tumrgbd_directory;
    std::string output_path;
    std::string tools_directory_override;
    const freiburg_calibration* calibration_override = nullptr;

    for (int i = 1; i < argc; ++i) {
        const auto matches = [&](const char* name) {
            return std::strcmp(argv[i], name) == 0;
        };
        if (matches("--help") || matches("-h")) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        else if (matches("--tools-dir")) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for option: --tools-dir\n");
                return EXIT_FAILURE;
            }
            tools_directory_override = argv[++i];
        }
        else if (matches("--freiburg")) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for option: --freiburg\n");
                return EXIT_FAILURE;
            }
            const char* const value = argv[++i];
            if ((std::strlen(value) != 1) || (value[0] < '1') || (value[0] > '3')) {
                std::fprintf(stderr, "Invalid value for --freiburg: '%s', expected 1, 2 or 3.\n", value);
                return EXIT_FAILURE;
            }
            calibration_override = &freiburg_calibrations[value[0] - '1'];
        }
        else if (argv[i][0] == '-') {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
        else if (tumrgbd_directory.empty()) {
            tumrgbd_directory = argv[i];
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

    std::string error;

    std::printf("Reading the rgb and depth frame lists...\n");
    std::vector<frame_reference> rgb_frames;
    std::vector<frame_reference> depth_frames;
    if (!read_frame_file(tumrgbd_directory + "/rgb.txt", rgb_frames, error) || !read_frame_file(tumrgbd_directory + "/depth.txt", depth_frames, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }
    std::printf("    %zu rgb frames, %zu depth frames.\n", rgb_frames.size(), depth_frames.size());

    const freiburg_calibration* calibration = calibration_override;
    const char* calibration_source = "--freiburg";
    if (calibration == nullptr) {
        std::string header;
        if (dataset::read_text_file(tumrgbd_directory + "/rgb.txt", header)) {
            calibration = match_freiburg(header.substr(0, (header.size() < 512) ? header.size() : 512));
            calibration_source = "the dataset's own rgb.txt header";
        }
    }
    if (calibration == nullptr) {
        calibration = match_freiburg(tumrgbd_directory);
        calibration_source = "the sequence directory's name";
    }
    if (calibration == nullptr) {
        std::fprintf(stderr, "Cannot tell which freiburg camera '%s' was recorded on: neither its rgb.txt header nor its path names one, and the three cameras' calibrations differ too much to guess. Pass --freiburg [1|2|3].\n", tumrgbd_directory.c_str());
        return EXIT_FAILURE;
    }
    double intrinsics[4] = { calibration->intrinsics[0], calibration->intrinsics[1], calibration->intrinsics[2], calibration->intrinsics[3] };
    import::pixel_centre_principal_point(&intrinsics[0]);
    std::printf("    %s calibration (from %s): [fx fy cx cy] = [%.4f %.4f %.4f %.4f]", calibration->name, calibration_source, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]);
    if (calibration->distortion_count == 0) {
        std::printf(", published already undistorted.\n");
    }
    else {
        std::printf(", [k1 k2 p1 p2 k3] = [%.4f %.4f %.4f %.4f %.4f].\n", calibration->distortion[0], calibration->distortion[1], calibration->distortion[2], calibration->distortion[3], calibration->distortion[4]);
    }
    if (depth_frames.empty()) {
        std::fprintf(stderr, "'%s' lists no depth frames.\n", (tumrgbd_directory + "/depth.txt").c_str());
        return EXIT_FAILURE;
    }

    std::vector<std::pair<std::size_t, std::size_t>> matched_frames;
    {
        std::size_t depth_index = 0;
        for (std::size_t rgb_index = 0; rgb_index < rgb_frames.size(); ++rgb_index) {
            const long long rgb_timestamp = rgb_frames[rgb_index].timestamp_nanoseconds;
            while ((depth_index + 1 < depth_frames.size()) && (depth_frames[depth_index + 1].timestamp_nanoseconds <= rgb_timestamp)) {
                ++depth_index;
            }
            long long best_delta = std::abs(rgb_timestamp - depth_frames[depth_index].timestamp_nanoseconds);
            std::size_t best_index = depth_index;
            if (depth_index + 1 < depth_frames.size()) {
                const long long delta_next = std::abs(rgb_timestamp - depth_frames[depth_index + 1].timestamp_nanoseconds);
                if (delta_next < best_delta) {
                    best_delta = delta_next;
                    best_index = depth_index + 1;
                }
            }
            if (best_delta <= 100000000LL) {
                matched_frames.push_back({ rgb_index, best_index });
            }
        }
    }
    std::printf("    %zu rgb frames have a depth frame within 0.1 s.\n", matched_frames.size());

    std::vector<import::ground_truth_sample> ground_truth;
    const std::string ground_truth_path = tumrgbd_directory + "/groundtruth.txt";
    const bool has_ground_truth = gtl::paths::is_regular_file(ground_truth_path);
    if (!has_ground_truth) {
        std::printf("No ground truth at '%s', so the scene will carry none and cannot be benchmarked against.\n", ground_truth_path.c_str());
    }
    else {
        std::printf("Reading ground truth...\n");
        if (!read_ground_truth_file(ground_truth_path, ground_truth, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return EXIT_FAILURE;
        }
        if (ground_truth.size() < 2) {
            std::fprintf(stderr, "'%s' holds fewer than 2 ground truth samples; remove the file if the sequence has none.\n", ground_truth_path.c_str());
            return EXIT_FAILURE;
        }
        std::sort(ground_truth.begin(), ground_truth.end(), [](const import::ground_truth_sample& a, const import::ground_truth_sample& b) {
            return a.timestamp_nanoseconds < b.timestamp_nanoseconds;
        });
    }

    std::printf("Reading accelerometer samples...\n");
    std::vector<accelerometer_sample> accelerometer_samples;
    if (!read_accelerometer_file(tumrgbd_directory + "/accelerometer.txt", accelerometer_samples, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }

    std::vector<std::pair<std::size_t, std::size_t>> retained_frames;
    for (const std::pair<std::size_t, std::size_t>& match : matched_frames) {
        if (!has_ground_truth || import::ground_truth_covers(ground_truth, rgb_frames[match.first].timestamp_nanoseconds)) {
            retained_frames.push_back(match);
        }
    }
    if (retained_frames.size() < 2) {
        std::fprintf(stderr, "%s\n", has_ground_truth ? "Fewer than 2 matched frames fall inside the ground truth's time coverage." : "Fewer than 2 matched frames.");
        return EXIT_FAILURE;
    }
    if (has_ground_truth) {
        std::printf("    %zu of %zu matched frames are inside the ground truth's coverage.\n", retained_frames.size(), matched_frames.size());
    }

    const std::string output_directory = gtl::paths::path_stem(output_path);
    if (!gtl::directory::make_directories(output_directory + "/sensor/image_01") || !gtl::directory::make_directories(output_directory + "/sensor/depth_01")) {
        std::fprintf(stderr, "Failed to create: %s\n", output_directory.c_str());
        return EXIT_FAILURE;
    }

    std::printf("Writing %zu frames%s...\n", retained_frames.size(), has_ground_truth ? " and the ground truth trajectory" : "");
    gtl::file calibration_rgb((output_directory + "/sensor/image_01.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    gtl::file calibration_depth((output_directory + "/sensor/depth_01.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    gtl::file trajectory;
    if (has_ground_truth) {
        trajectory.open((output_directory + "/trajectory.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    }
    else {
        import::remove_absent_scene_file(output_directory + "/trajectory.txt");
    }
    if (!calibration_rgb.is_open() || !calibration_depth.is_open() || (has_ground_truth && !trajectory.is_open())) {
        std::fprintf(stderr, "Failed to create the calibration or trajectory files in '%s'.\n", output_directory.c_str());
        return EXIT_FAILURE;
    }
    for (const std::pair<std::size_t, std::size_t>& index : retained_frames) {
        const long long timestamp = rgb_frames[index.first].timestamp_nanoseconds;
        if (has_ground_truth) {
            double position[3];
            double quaternion_xyzw[4];
            if (!import::interpolate_ground_truth(ground_truth, timestamp, position, quaternion_xyzw)) {
                std::fprintf(stderr, "Internal error: frame at %lld ns has no ground truth despite being inside its coverage.\n", timestamp);
                return EXIT_FAILURE;
            }
            if (!dataset::write_pose_line(trajectory, timestamp, position, quaternion_xyzw)) {
                std::fprintf(stderr, "Failed to write trajectory.\n");
                return EXIT_FAILURE;
            }
        }
        const std::string frame_name = dataset::frame_filename(static_cast<unsigned long long>(timestamp));
        const std::string rgb_path = tumrgbd_directory + "/" + rgb_frames[index.first].filename;
        if (!import::convert_image(rgb_path, "-type TrueColor", output_directory + "/sensor/image_01/" + frame_name, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return EXIT_FAILURE;
        }
        if (!import::write_calibration_line(calibration_rgb, timestamp, origin, camera_mounting, "plumb_bob", intrinsics, &calibration->distortion[0], calibration->distortion_count)) {
            std::fprintf(stderr, "Failed to write the rgb calibration.\n");
            return EXIT_FAILURE;
        }
        const std::string depth_path = tumrgbd_directory + "/" + depth_frames[index.second].filename;
        if (!import::convert_depth_image(depth_path, output_directory + "/sensor/depth_01/" + frame_name, 5000.0, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return EXIT_FAILURE;
        }
        if (!import::write_calibration_line(calibration_depth, timestamp, origin, camera_mounting, "plumb_bob", intrinsics, &calibration->distortion[0], calibration->distortion_count)) {
            std::fprintf(stderr, "Failed to write the depth calibration.\n");
            return EXIT_FAILURE;
        }
    }
    calibration_rgb.close();
    calibration_depth.close();
    trajectory.close();

    std::printf("Writing accelerometer samples...\n");
    std::size_t accelerometer_covered = 0;
    for (const accelerometer_sample& sample : accelerometer_samples) {
        if (!has_ground_truth || import::ground_truth_covers(ground_truth, sample.timestamp_nanoseconds)) {
            ++accelerometer_covered;
        }
    }
    if (accelerometer_covered == 0) {
        import::remove_absent_scene_file(output_directory + "/sensor/accelerometer_01.txt");
        if (accelerometer_samples.empty()) {
            std::printf("    'accelerometer.txt' holds no samples, so the scene carries no accelerometer.\n");
        }
        else {
            std::printf("    None of %zu accelerometer samples are inside the ground truth's coverage, so the scene carries no accelerometer.\n", accelerometer_samples.size());
        }
    }
    else {
        std::size_t accelerometer_written = 0;
        gtl::file handle((output_directory + "/sensor/accelerometer_01.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
        if (!handle.is_open()) {
            std::fprintf(stderr, "Failed to create: %s/sensor/accelerometer_01.txt\n", output_directory.c_str());
            return EXIT_FAILURE;
        }
        for (const accelerometer_sample& sample : accelerometer_samples) {
            if (has_ground_truth && !import::ground_truth_covers(ground_truth, sample.timestamp_nanoseconds)) {
                continue;
            }
            const double values[10] = { origin[0], origin[1], origin[2], accelerometer_mounting[0], accelerometer_mounting[1], accelerometer_mounting[2], accelerometer_mounting[3], sample.linear_acceleration[0], sample.linear_acceleration[1], sample.linear_acceleration[2] };
            if (!dataset::write_sample_line(handle, sample.timestamp_nanoseconds, &values[0], 10)) {
                std::fprintf(stderr, "Failed to write accelerometer sample.\n");
                return EXIT_FAILURE;
            }
            ++accelerometer_written;
        }
        std::printf("    %zu of %zu accelerometer samples are inside the ground truth's coverage.\n", accelerometer_written, accelerometer_samples.size());
    }

    if (!import::collapse_scene(tools_directory_override, output_directory, output_path, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
