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
#include "rotation.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
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

    struct imu_sample {
        long long timestamp_nanoseconds = 0;
        double angular_velocity[3] = {};
        double linear_acceleration[3] = {};
    };

    constexpr static const double origin[3] = { 0.0, 0.0, 0.0 };
    constexpr static const double identity_rotation[4] = { 0.0, 0.0, 0.0, 1.0 };

    bool read_calibration(const std::string& path, double intrinsics[4], std::string& error) {
        bool found = false;
        if (!import::for_each_line(path, error, [&intrinsics, &found](const char* line) {
                found = found || (std::sscanf(line, "%lf %lf %lf %lf", &intrinsics[0], &intrinsics[1], &intrinsics[2], &intrinsics[3]) == 4);
            })) {
            return false;
        }
        if (!found) {
            error = "'" + path + "' has no 'fx fy cx cy' line";
            return false;
        }
        import::pixel_centre_principal_point(intrinsics);
        return true;
    }

    bool read_frame_list(const std::string& path, std::vector<frame_reference>& frames, std::string& error) {
        return import::for_each_line(path, error, [&frames](const char* line) {
            char timestamp[64] = {};
            char filename[256] = {};
            long long timestamp_nanoseconds = 0;
            if ((std::sscanf(line, "%63s %255s", &timestamp[0], &filename[0]) == 2) && dataset::parse_timestamp_nanoseconds(&timestamp[0], timestamp_nanoseconds)) {
                const char* base = &filename[0];
                for (const char* character = base; *character != '\0'; ++character) {
                    if ((*character == '/') || (*character == '\\')) {
                        base = character + 1;
                    }
                }
                frames.push_back({ timestamp_nanoseconds, std::string(base) });
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

    bool read_imu_file(const std::string& path, std::vector<imu_sample>& samples, std::string& error) {
        std::vector<double> values;
        return import::for_each_line(path, error, [&samples, &values](char* line) {
            imu_sample sample;
            if (dataset::parse_sample_line(line, sample.timestamp_nanoseconds, values) && (values.size() == 6)) {
                std::copy(values.begin(), values.begin() + 3, &sample.angular_velocity[0]);
                std::copy(values.begin() + 3, values.end(), &sample.linear_acceleration[0]);
                samples.push_back(sample);
            }
        });
    }

    bool read_extrinsics_1_2(const std::string& path, double transform[3][4], std::string& error) {
        std::vector<double> values;
        if (!import::for_each_line(path, error, [&values](const char* line) {
                double numbers[4];
                const int matched = std::sscanf(line, "%lf %lf %lf %lf", &numbers[0], &numbers[1], &numbers[2], &numbers[3]);
                values.insert(values.end(), &numbers[0], &numbers[0] + ((matched > 0) ? matched : 0));
            })) {
            return false;
        }
        if (values.size() < 12) {
            error = "'" + path + "' has no 3x4 rigid transform";
            return false;
        }
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 4; ++column) {
                transform[row][column] = values[static_cast<std::size_t>((row * 4) + column)];
            }
        }
        return true;
    }

    void extrinsic_from_transform(const double transform[3][4], double translation[3], double quaternion_xyzw[4]) {
        double rotation[3][3];
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                rotation[row][column] = transform[row][column];
            }
            translation[row] = transform[row][3];
        }
        rotation::matrix_to_quaternion(&rotation[0][0], quaternion_xyzw);
    }

    bool write_image_frame(const std::string& source_path, const std::string& destination_directory, const std::string& frame_name, const double intrinsics[4], const double translation[3], const double quaternion_xyzw[4], const long long timestamp_nanoseconds, gtl::file& calibration_handle, std::string& error) {
        if (!import::convert_image(source_path, "-colorspace Gray", destination_directory + "/" + frame_name, error)) {
            return false;
        }
        if (!import::write_calibration_line(calibration_handle, timestamp_nanoseconds, translation, quaternion_xyzw, "plumb_bob", intrinsics, nullptr, 0)) {
            error = "Failed to write the calibration for: " + source_path;
            return false;
        }
        return true;
    }

    void print_usage(const char* argv0) {
        std::printf("Usage %s [eth3d-dir] [output.mcap] [options...]\n", argv0);
        std::printf("    eth3d-dir   - An already extracted ETH3D SLAM dataset directory\n");
        std::printf("                  (https://www.eth3d.net/slam_documentation) holding the monocular\n");
        std::printf("                  part: calibration.txt, rgb.txt, rgb/ and, for the half of the\n");
        std::printf("                  benchmark that publishes one, groundtruth.txt. The\n");
        std::printf("                  optional stereo (rgb2/), depth (depth/), and imu (imu.txt) parts\n");
        std::printf("                  are imported when found inside this directory, or may be pointed\n");
        std::printf("                  at with --stereo, --rgbd, and --imu when the dataset's per-part\n");
        std::printf("                  zips were extracted separately.\n");
        std::printf("    output.mcap - Where to write the packed scene; the directory form is written\n");
        std::printf("                  alongside it (output.mcap's path with the extension dropped),\n");
        std::printf("                  and left in place for inspection, exactly as 'dataset expand' does.\n");
        std::printf("    options:\n");
        std::printf("        --tools-dir [dir] - Directory containing 'zeroslam-dataset' (default: next to\n");
        std::printf("                            this tool), used to pack and validate the result.\n");
        std::printf("        --stereo [dir]     - The extracted stereo part (calibration2.txt,\n");
        std::printf("                            extrinsics_1_2.txt, rgb2/), relative to the same base time\n");
        std::printf("                            and matching the rgb frames by name.\n");
        std::printf("        --rgbd [dir]       - The extracted depth part (depth.txt, depth/).\n");
        std::printf("        --imu [dir]        - The extracted imu part (imu.txt and any imu2.txt,\n");
        std::printf("                            imu3.txt, imu4.txt).\n");
        std::printf("Produces image_01 (the rgb camera), and, when present, image_02 (the rgb2 camera),\n");
        std::printf("depth_01 (the depth camera), and the imu_01 to imu_04 imu streams, all keeping the\n");
        std::printf("dataset's own timestamps, trimmed to the ground truth's coverage: the rgb and rgb2\n");
        std::printf("frames as 8-bit greyscale, the depth scaled by 5000 into metres as 32-bit float.\n");
    }
}

int main(int argc, char* argv[]) {
    std::string eth3d_directory;
    std::string output_path;
    std::string tools_directory_override;
    std::string stereo_directory_override;
    std::string rgbd_directory_override;
    std::string imu_directory_override;

    for (int i = 1; i < argc; ++i) {
        const auto matches = [&](const char* name) {
            return std::strcmp(argv[i], name) == 0;
        };
        const auto option_value = [&](const char* name, std::string& value) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for option: %s\n", name);
                return false;
            }
            value = argv[++i];
            return true;
        };
        if (matches("--help") || matches("-h")) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        else if (matches("--tools-dir") || matches("--stereo") || matches("--rgbd") || matches("--imu")) {
            std::string& value = matches("--tools-dir") ? tools_directory_override : (matches("--stereo") ? stereo_directory_override : (matches("--rgbd") ? rgbd_directory_override : imu_directory_override));
            if (!option_value(argv[i], value)) {
                return EXIT_FAILURE;
            }
        }
        else if (argv[i][0] == '-') {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
        else if (eth3d_directory.empty()) {
            eth3d_directory = argv[i];
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

    const std::string stereo_directory = stereo_directory_override.empty() ? eth3d_directory : stereo_directory_override;
    const std::string rgbd_directory = rgbd_directory_override.empty() ? eth3d_directory : rgbd_directory_override;
    const std::string imu_directory = imu_directory_override.empty() ? eth3d_directory : imu_directory_override;
    const bool stereo_enabled = gtl::paths::is_directory(stereo_directory + "/rgb2") && gtl::paths::is_regular_file(stereo_directory + "/calibration2.txt") && gtl::paths::is_regular_file(stereo_directory + "/extrinsics_1_2.txt");
    const bool rgbd_enabled = gtl::paths::is_directory(rgbd_directory + "/depth") && gtl::paths::is_regular_file(rgbd_directory + "/depth.txt");
    const bool imu_enabled = gtl::paths::is_regular_file(imu_directory + "/imu.txt");
    if (!stereo_directory_override.empty() && !stereo_enabled) {
        std::fprintf(stderr, "--stereo '%s' has no calibration2.txt, extrinsics_1_2.txt, and rgb2/ files.\n", stereo_directory.c_str());
        return EXIT_FAILURE;
    }
    if (!rgbd_directory_override.empty() && !rgbd_enabled) {
        std::fprintf(stderr, "--rgbd '%s' has no depth.txt and depth/ files.\n", rgbd_directory.c_str());
        return EXIT_FAILURE;
    }
    if (!imu_directory_override.empty() && !imu_enabled) {
        std::fprintf(stderr, "--imu '%s' has no imu.txt file.\n", imu_directory.c_str());
        return EXIT_FAILURE;
    }

    std::printf("Reading the rgb camera calibration...\n");
    double intrinsics_01[4] = {};
    if (!read_calibration(eth3d_directory + "/calibration.txt", intrinsics_01, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }

    double intrinsics_02[4] = { intrinsics_01[0], intrinsics_01[1], intrinsics_01[2], intrinsics_01[3] };
    double translation_02[3] = { 0.0, 0.0, 0.0 };
    double quaternion_02[4] = { 0.0, 0.0, 0.0, 1.0 };
    if (stereo_enabled) {
        std::printf("Reading the rgb2 camera calibration...\n");
        double transform[3][4] = {};
        if (!read_calibration(stereo_directory + "/calibration2.txt", intrinsics_02, error) || !read_extrinsics_1_2(stereo_directory + "/extrinsics_1_2.txt", transform, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return EXIT_FAILURE;
        }
        extrinsic_from_transform(transform, translation_02, quaternion_02);
    }

    std::printf("Reading the rgb frame list...\n");
    std::vector<frame_reference> rgb_frames;
    if (!read_frame_list(eth3d_directory + "/rgb.txt", rgb_frames, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }
    std::printf("    %zu rgb frames.\n", rgb_frames.size());

    std::unordered_map<std::string, long long> depth_by_name;
    if (rgbd_enabled) {
        std::printf("Reading the depth frame list...\n");
        std::vector<frame_reference> depth_frames;
        if (!read_frame_list(rgbd_directory + "/depth.txt", depth_frames, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return EXIT_FAILURE;
        }
        for (const frame_reference& depth : depth_frames) {
            depth_by_name[depth.filename] = depth.timestamp_nanoseconds;
        }
        std::printf("    %zu depth frames.\n", depth_frames.size());
    }

    std::vector<import::ground_truth_sample> ground_truth;
    const std::string ground_truth_path = eth3d_directory + "/groundtruth.txt";
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

    struct imu_source {
        const char* filename;
        const char* sensor_name;
        double extrinsic_quaternion_xyzw[4];
        double extrinsic_translation[3];
    };

    const imu_source imu_sources[] = {
        { "imu.txt", "imu_01", { +0.003722, -0.000400, +0.999993, -0.000692 }, { +0.003462, +0.022029, -0.001361 } },
        { "imu2.txt", "imu_02", { -0.004570, -0.002131, +0.003339, +0.999982 }, { -0.085130, -0.005079, -0.001918 } },
        { "imu3.txt", "imu_03", { -0.000457, +0.001860, +0.999971, -0.007430 }, { +0.003683, +0.007967, -0.001332 } },
        { "imu4.txt", "imu_04", { -0.006178, +0.001185, +0.004843, +0.999968 }, { -0.084923, +0.008505, -0.001303 } },
    };

    std::vector<std::size_t> retained_frames;
    std::size_t rgb2_skipped = 0;
    std::size_t depth_skipped = 0;
    for (std::size_t i = 0; i < rgb_frames.size(); ++i) {
        const frame_reference& frame = rgb_frames[i];
        if (has_ground_truth && !import::ground_truth_covers(ground_truth, frame.timestamp_nanoseconds)) {
            continue;
        }
        if (stereo_enabled && !gtl::paths::is_regular_file(stereo_directory + "/rgb2/" + frame.filename)) {
            ++rgb2_skipped;
            continue;
        }
        if (rgbd_enabled) {
            const auto depth = depth_by_name.find(frame.filename);
            if ((depth == depth_by_name.end()) || (depth->second != frame.timestamp_nanoseconds)) {
                ++depth_skipped;
                continue;
            }
        }
        retained_frames.push_back(i);
    }
    std::printf("    %zu of %zu rgb frames are inside the ground truth's coverage (skipped %zu without rgb2, %zu without depth).\n", retained_frames.size(), rgb_frames.size(), rgb2_skipped, depth_skipped);
    if (retained_frames.size() < 2) {
        std::fprintf(stderr, "Fewer than 2 frames fall inside the ground truth's time coverage.\n");
        return EXIT_FAILURE;
    }

    const std::string output_directory = gtl::paths::path_stem(output_path);
    if (!gtl::directory::make_directories(output_directory + "/sensor/image_01") || (stereo_enabled && !gtl::directory::make_directories(output_directory + "/sensor/image_02")) || (rgbd_enabled && !gtl::directory::make_directories(output_directory + "/sensor/depth_01"))) {
        std::fprintf(stderr, "Failed to create: %s\n", output_directory.c_str());
        return EXIT_FAILURE;
    }

    std::printf("Writing %zu frames and the ground truth trajectory...\n", retained_frames.size());
    gtl::file calibration_01((output_directory + "/sensor/image_01.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    gtl::file calibration_02;
    gtl::file calibration_depth;
    if (stereo_enabled) {
        calibration_02.open((output_directory + "/sensor/image_02.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    }
    if (rgbd_enabled) {
        calibration_depth.open((output_directory + "/sensor/depth_01.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    }
    gtl::file trajectory;
    if (has_ground_truth) {
        trajectory.open((output_directory + "/trajectory.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    }
    else {
        import::remove_absent_scene_file(output_directory + "/trajectory.txt");
    }
    if (!calibration_01.is_open() || (stereo_enabled && !calibration_02.is_open()) || (rgbd_enabled && !calibration_depth.is_open()) || (has_ground_truth && !trajectory.is_open())) {
        std::fprintf(stderr, "Failed to create the calibration or trajectory files in '%s'.\n", output_directory.c_str());
        return EXIT_FAILURE;
    }
    for (const std::size_t index : retained_frames) {
        const long long timestamp = rgb_frames[index].timestamp_nanoseconds;
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

        if (!write_image_frame(eth3d_directory + "/rgb/" + rgb_frames[index].filename, output_directory + "/sensor/image_01", frame_name, intrinsics_01, origin, identity_rotation, timestamp, calibration_01, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return EXIT_FAILURE;
        }
        if (stereo_enabled && !write_image_frame(stereo_directory + "/rgb2/" + rgb_frames[index].filename, output_directory + "/sensor/image_02", frame_name, intrinsics_02, translation_02, quaternion_02, timestamp, calibration_02, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return EXIT_FAILURE;
        }
        if (rgbd_enabled) {
            if (!import::convert_depth_image(rgbd_directory + "/depth/" + rgb_frames[index].filename, output_directory + "/sensor/depth_01/" + frame_name, 5000.0, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                return EXIT_FAILURE;
            }
            if (!import::write_calibration_line(calibration_depth, timestamp, origin, identity_rotation, "plumb_bob", intrinsics_01, nullptr, 0)) {
                std::fprintf(stderr, "Failed to write the depth calibration.\n");
                return EXIT_FAILURE;
            }
        }
    }
    calibration_01.close();
    calibration_02.close();
    calibration_depth.close();
    trajectory.close();

    if (imu_enabled) {
        std::printf("Writing imu samples...\n");
        for (const imu_source& source : imu_sources) {
            const std::string imu_path = imu_directory + "/" + source.filename;
            if (!gtl::paths::is_regular_file(imu_path)) {
                import::remove_absent_scene_file(output_directory + "/sensor/" + source.sensor_name + ".txt");
                continue;
            }
            std::vector<imu_sample> samples;
            if (!read_imu_file(imu_path, samples, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                return EXIT_FAILURE;
            }
            std::size_t imu_covered = 0;
            for (const imu_sample& sample : samples) {
                if (!has_ground_truth || import::ground_truth_covers(ground_truth, sample.timestamp_nanoseconds)) {
                    ++imu_covered;
                }
            }
            if (imu_covered == 0) {
                import::remove_absent_scene_file(output_directory + "/sensor/" + source.sensor_name + ".txt");
                std::printf("    %s: none of %zu samples from %s are inside the ground truth's coverage, so the scene carries no %s.\n", source.sensor_name, samples.size(), source.filename, source.sensor_name);
                continue;
            }
            gtl::file imu_handle((output_directory + "/sensor/" + source.sensor_name + ".txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
            if (!imu_handle.is_open()) {
                std::fprintf(stderr, "Failed to create: %s/sensor/%s.txt\n", output_directory.c_str(), source.sensor_name);
                return EXIT_FAILURE;
            }
            std::size_t imu_written = 0;
            for (const imu_sample& sample : samples) {
                if (has_ground_truth && !import::ground_truth_covers(ground_truth, sample.timestamp_nanoseconds)) {
                    continue;
                }
                const double values[13] = { source.extrinsic_translation[0], source.extrinsic_translation[1], source.extrinsic_translation[2], source.extrinsic_quaternion_xyzw[0], source.extrinsic_quaternion_xyzw[1], source.extrinsic_quaternion_xyzw[2], source.extrinsic_quaternion_xyzw[3], sample.angular_velocity[0], sample.angular_velocity[1], sample.angular_velocity[2], sample.linear_acceleration[0], sample.linear_acceleration[1], sample.linear_acceleration[2] };
                if (!dataset::write_sample_line(imu_handle, sample.timestamp_nanoseconds, &values[0], 13)) {
                    std::fprintf(stderr, "Failed to write imu sample.\n");
                    return EXIT_FAILURE;
                }
                ++imu_written;
            }
            std::printf("    %s: %zu of %zu samples from %s are inside the ground truth's coverage.\n", source.sensor_name, imu_written, samples.size(), source.filename);
        }
    }

    if (!import::collapse_scene(tools_directory_override, output_directory, output_path, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
