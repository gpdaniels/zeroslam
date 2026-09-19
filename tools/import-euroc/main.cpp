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
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    struct camera_calibration {
        double t_bs[16] = {};
        double intrinsics[4] = {};
        double distortion[4] = {};
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

    bool parse_bracketed_doubles(const std::string& text, const std::string& key, std::vector<double>& values) {
        const std::size_t key_position = text.find(key + ":");
        if (key_position == std::string::npos) {
            return false;
        }
        const std::size_t open = text.find('[', key_position);
        const std::size_t close = (open == std::string::npos) ? std::string::npos : text.find(']', open);
        if ((open == std::string::npos) || (close == std::string::npos)) {
            return false;
        }
        const std::string inner = text.substr(open + 1, close - open - 1);
        std::size_t start = 0;
        while (start <= inner.size()) {
            const std::size_t comma = inner.find(',', start);
            const std::string token = inner.substr(start, ((comma == std::string::npos) ? inner.size() : comma) - start);
            if (token.find_first_not_of(" \t\r\n") != std::string::npos) {
                char* end = nullptr;
                const double value = std::strtod(token.c_str(), &end);
                if (end == token.c_str()) {
                    return false;
                }
                values.push_back(value);
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
        return true;
    }

    bool read_camera_yaml(const std::string& path, camera_calibration& calibration, std::string& error) {
        std::string text;
        if (!dataset::read_text_file(path, text)) {
            error = "cannot read '" + path + "'";
            return false;
        }
        if (text.find("camera_model: pinhole") == std::string::npos) {
            error = "'" + path + "' is not a 'pinhole' camera";
            return false;
        }
        if (text.find("distortion_model: radial-tangential") == std::string::npos) {
            error = "'" + path + "' is not a 'radial-tangential' distortion model";
            return false;
        }
        std::vector<double> t_bs;
        if (!parse_bracketed_doubles(text, "data", t_bs) || (t_bs.size() != 16)) {
            error = "'" + path + "' has no 4x4 'T_BS' data";
            return false;
        }
        std::copy(t_bs.begin(), t_bs.end(), &calibration.t_bs[0]);
        std::vector<double> intrinsics;
        if (!parse_bracketed_doubles(text, "intrinsics", intrinsics) || (intrinsics.size() != 4)) {
            error = "'" + path + "' has no 4 element 'intrinsics'";
            return false;
        }
        std::copy(intrinsics.begin(), intrinsics.end(), &calibration.intrinsics[0]);
        import::pixel_centre_principal_point(&calibration.intrinsics[0]);
        std::vector<double> distortion;
        if (!parse_bracketed_doubles(text, "distortion_coefficients", distortion) || (distortion.size() != 4)) {
            error = "'" + path + "' has no 4 element 'distortion_coefficients'";
            return false;
        }
        std::copy(distortion.begin(), distortion.end(), &calibration.distortion[0]);
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

    bool read_ground_truth_csv(const std::string& path, std::vector<import::ground_truth_sample>& samples, std::string& error) {
        return import::for_each_line(path, error, [&samples](const char* line) {
            import::ground_truth_sample sample;
            double quaternion_w = 0.0;
            if (std::sscanf(line, "%lld,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &sample.timestamp_nanoseconds, &sample.position[0], &sample.position[1], &sample.position[2], &quaternion_w, &sample.quaternion_xyzw[0], &sample.quaternion_xyzw[1], &sample.quaternion_xyzw[2]) == 8) {
                sample.quaternion_xyzw[3] = quaternion_w;
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

    bool write_camera_frame(const std::string& source_path, const std::string& destination_directory, const std::string& frame_name, const camera_calibration& calibration, const double translation[3], const double quaternion_xyzw[4], const long long timestamp_nanoseconds, gtl::file& calibration_handle, std::string& error) {
        if (!import::convert_image(source_path, "", destination_directory + "/" + frame_name, error)) {
            return false;
        }
        if (!import::write_calibration_line(calibration_handle, timestamp_nanoseconds, translation, quaternion_xyzw, "plumb_bob", calibration.intrinsics, calibration.distortion, 4)) {
            error = "Failed to write the calibration for: " + source_path;
            return false;
        }
        return true;
    }

    using rigid = rotation::rigid;

    void print_usage(const char* argv0) {
        std::printf("Usage %s [mav0-dir] [groundtruth.csv] [output.mcap] [options...]\n", argv0);
        std::printf("    mav0-dir       - An already extracted EuRoC-MAV ASL 'mav0/' directory (unzip the\n");
        std::printf("                     official dataset zip first; this tool does not read zips).\n");
        std::printf("    groundtruth.csv - A corrected ground truth trajectory, '#time(ns),px,py,pz,qw,qx,\n");
        std::printf("                     qy,qz,...' (this project uses github.com/rpng/open_vins's\n");
        std::printf("                     ov_data/euroc_mav/[sequence].csv, more accurate than the\n");
        std::printf("                     dataset's own state_groundtruth_estimate0 on the V1 sequences).\n");
        std::printf("    output.mcap    - Where to write the packed scene; the directory form is written\n");
        std::printf("                     alongside it (output.mcap's path with the extension dropped),\n");
        std::printf("                     and left in place for inspection, exactly as 'dataset expand' does.\n");
        std::printf("    options:\n");
        std::printf("        --tools-dir [dir] - Directory containing 'zeroslam-dataset' (default: next to\n");
        std::printf("                            this tool), used to pack and validate the result.\n");
        std::printf("Produces two cameras (image_01 = cam0, image_02 = cam1) and one imu (imu_01), all\n");
        std::printf("keeping the dataset's own timestamps, trimmed to the ground truth's coverage. The\n");
        std::printf("scene's ego frame is cam0 (x right, y down, z forward): the ground truth is re-expressed\n");
        std::printf("as cam0's pose in the z-up Vicon world and cam1 and the imu are posed relative to cam0.\n");
    }
}

int main(int argc, char* argv[]) {
    std::string mav0_directory;
    std::string ground_truth_path;
    std::string output_path;
    std::string tools_directory_override;

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
        else if (argv[i][0] == '-') {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
        else if (mav0_directory.empty()) {
            mav0_directory = argv[i];
        }
        else if (ground_truth_path.empty()) {
            ground_truth_path = argv[i];
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

    std::printf("Reading camera calibration...\n");
    camera_calibration cam0;
    camera_calibration cam1;
    if (!read_camera_yaml(mav0_directory + "/cam0/sensor.yaml", cam0, error) || !read_camera_yaml(mav0_directory + "/cam1/sensor.yaml", cam1, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }

    std::printf("Reading frame timestamps...\n");
    std::vector<frame_reference> cam0_frames;
    std::vector<frame_reference> cam1_frames;
    if (!read_frame_csv(mav0_directory + "/cam0/data.csv", cam0_frames, error) || !read_frame_csv(mav0_directory + "/cam1/data.csv", cam1_frames, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }
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

    std::printf("Reading ground truth...\n");
    std::vector<import::ground_truth_sample> ground_truth;
    if (!read_ground_truth_csv(ground_truth_path, ground_truth, error)) {
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

    std::printf("Reading imu samples...\n");
    std::vector<imu_sample> imu_samples;
    if (!read_imu_csv(mav0_directory + "/imu0/data.csv", imu_samples, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }

    std::vector<std::pair<std::size_t, std::size_t>> retained_frames;
    for (const std::pair<std::size_t, std::size_t>& match : matched_frames) {
        if (import::ground_truth_covers(ground_truth, cam0_frames[match.first].timestamp_nanoseconds)) {
            retained_frames.push_back(match);
        }
    }
    if (retained_frames.size() < 2) {
        std::fprintf(stderr, "Fewer than 2 frames fall inside the ground truth's time coverage.\n");
        return EXIT_FAILURE;
    }
    std::printf("    %zu of %zu matched frames are inside the ground truth's coverage.\n", retained_frames.size(), matched_frames.size());

    const std::string output_directory = gtl::paths::path_stem(output_path);
    if (!gtl::directory::make_directories(output_directory + "/sensor/image_01") || !gtl::directory::make_directories(output_directory + "/sensor/image_02")) {
        std::fprintf(stderr, "Failed to create: %s\n", output_directory.c_str());
        return EXIT_FAILURE;
    }

    const rigid body_from_cam0 = rigid::from_matrix(cam0.t_bs);
    const rigid body_from_cam1 = rigid::from_matrix(cam1.t_bs);
    const rigid cam0_from_body = body_from_cam0.inverse();
    double translation_01[3];
    double quaternion_01[4];
    rigid().to_pose(translation_01, quaternion_01);
    double translation_02[3];
    double quaternion_02[4];
    (cam0_from_body * body_from_cam1).to_pose(translation_02, quaternion_02);
    double translation_imu[3];
    double quaternion_imu[4];
    cam0_from_body.to_pose(translation_imu, quaternion_imu);
    std::printf("Ego frame is cam0: cam1 at (%.4f, %.4f, %.4f) m, imu at (%.4f, %.4f, %.4f) m.\n", translation_02[0], translation_02[1], translation_02[2], translation_imu[0], translation_imu[1], translation_imu[2]);

    std::printf("Writing %zu frames and the ground truth trajectory...\n", retained_frames.size());
    gtl::file calibration_01((output_directory + "/sensor/image_01.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    gtl::file calibration_02((output_directory + "/sensor/image_02.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    gtl::file trajectory((output_directory + "/trajectory.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    if ((!calibration_01.is_open()) || (!calibration_02.is_open()) || (!trajectory.is_open())) {
        std::fprintf(stderr, "Failed to create the calibration or trajectory files in '%s'.\n", output_directory.c_str());
        return EXIT_FAILURE;
    }
    for (const std::pair<std::size_t, std::size_t>& index : retained_frames) {
        const long long timestamp = cam0_frames[index.first].timestamp_nanoseconds;

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

        const std::string frame_name = dataset::frame_filename(static_cast<unsigned long long>(timestamp));
        if (!write_camera_frame(mav0_directory + "/cam0/data/" + cam0_frames[index.first].filename, output_directory + "/sensor/image_01", frame_name, cam0, translation_01, quaternion_01, timestamp, calibration_01, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return EXIT_FAILURE;
        }
        if (!write_camera_frame(mav0_directory + "/cam1/data/" + cam1_frames[index.second].filename, output_directory + "/sensor/image_02", frame_name, cam1, translation_02, quaternion_02, timestamp, calibration_02, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return EXIT_FAILURE;
        }
    }
    calibration_01.close();
    calibration_02.close();
    trajectory.close();

    std::printf("Writing imu samples...\n");
    std::size_t imu_covered = 0;
    for (const imu_sample& sample : imu_samples) {
        if (import::ground_truth_covers(ground_truth, sample.timestamp_nanoseconds)) {
            ++imu_covered;
        }
    }
    if (imu_covered == 0) {
        import::remove_absent_scene_file(output_directory + "/sensor/imu_01.txt");
        std::printf("    None of %zu imu samples are inside the ground truth's coverage, so the scene carries no imu.\n", imu_samples.size());
    }
    else {
        std::size_t imu_written = 0;
        gtl::file imu_handle((output_directory + "/sensor/imu_01.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
        if (!imu_handle.is_open()) {
            std::fprintf(stderr, "Failed to create: %s/sensor/imu_01.txt\n", output_directory.c_str());
            return EXIT_FAILURE;
        }
        for (const imu_sample& sample : imu_samples) {
            if (!import::ground_truth_covers(ground_truth, sample.timestamp_nanoseconds)) {
                continue;
            }
            const double values[13] = { translation_imu[0], translation_imu[1], translation_imu[2], quaternion_imu[0], quaternion_imu[1], quaternion_imu[2], quaternion_imu[3], sample.angular_velocity[0], sample.angular_velocity[1], sample.angular_velocity[2], sample.linear_acceleration[0], sample.linear_acceleration[1], sample.linear_acceleration[2] };
            if (!dataset::write_sample_line(imu_handle, sample.timestamp_nanoseconds, &values[0], 13)) {
                std::fprintf(stderr, "Failed to write imu sample.\n");
                return EXIT_FAILURE;
            }
            ++imu_written;
        }
        std::printf("    %zu of %zu imu samples are inside the ground truth's coverage.\n", imu_written, imu_samples.size());
    }

    if (!import::collapse_scene(tools_directory_override, output_directory, output_path, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
