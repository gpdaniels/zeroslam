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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using rigid = rotation::rigid;

    constexpr static const int camera_count = 4;

    constexpr static const double world_rotation[9] = { 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0 };

    struct projection {
        bool present = false;
        double matrix[12] = {};
    };

    struct scene_camera {
        int index = 0;
        std::string sensor_name;
        double intrinsics[4] = {};
        double translation[3] = {};
        double quaternion_xyzw[4] = { 0.0, 0.0, 0.0, 1.0 };
    };

    bool read_projections(const std::string& path, projection* const projections, std::string& error) {
        return import::for_each_line(path, error, [projections](const char* line) {
            char key[8] = {};
            double values[12] = {};
            if (std::sscanf(line, "%7[^:]:%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &key[0], &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7], &values[8], &values[9], &values[10], &values[11]) != 13) {
                return;
            }
            if ((std::strlen(&key[0]) != 2) || (key[0] != 'P') || (key[1] < '0') || (key[1] >= ('0' + camera_count))) {
                return;
            }
            projection& target = projections[key[1] - '0'];
            std::copy(&values[0], &values[0] + 12, &target.matrix[0]);
            target.present = true;
        });
    }

    bool read_timestamps(const std::string& path, std::vector<long long>& timestamps, std::string& error) {
        return import::for_each_line(path, error, [&timestamps](const char* line) {
            char* end = nullptr;
            const double seconds = std::strtod(line, &end);
            if (end != line) {
                timestamps.push_back(std::llround(seconds * 1000000000.0));
            }
        });
    }

    bool read_poses(const std::string& path, std::vector<rigid>& poses, std::string& error) {
        return import::for_each_line(path, error, [&poses](const char* line) {
            double values[12] = {};
            if (std::sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7], &values[8], &values[9], &values[10], &values[11]) == 12) {
                poses.push_back(rigid::from_matrix(&values[0]));
            }
        });
    }

    bool is_rectified_projection(const double matrix[12]) {
        const int zeros[3] = { 1, 4, 8 };
        for (const int entry : zeros) {
            if (std::fabs(matrix[entry]) > 1e-6) {
                return false;
            }
        }
        return (std::fabs(matrix[9]) <= 1e-6) && (std::fabs(matrix[10] - 1.0) <= 1e-6);
    }

    void decompose_projection(const double matrix[12], double intrinsics[4], double offset[3]) {
        intrinsics[0] = matrix[0];
        intrinsics[1] = matrix[5];
        intrinsics[2] = matrix[2];
        intrinsics[3] = matrix[6];
        offset[0] = (matrix[3] - (intrinsics[2] * matrix[11])) / intrinsics[0];
        offset[1] = (matrix[7] - (intrinsics[3] * matrix[11])) / intrinsics[1];
        offset[2] = matrix[11];
    }

    std::string camera_directory(const int index) {
        return "image_" + std::to_string(index);
    }

    std::string source_frame_filename(const std::size_t index) {
        char filename[32];
        std::snprintf(&filename[0], sizeof(filename), "%06zu.png", index);
        return &filename[0];
    }

    void print_usage(const char* argv0) {
        std::printf("Usage %s [sequence-dir] ([poses.txt]) [output.mcap] [options...]\n", argv0);
        std::printf("    sequence-dir  - An extracted KITTI odometry sequence directory ('dataset/\n");
        std::printf("                    sequences/[xx]', holding calib.txt, times.txt and the image_[0-3]\n");
        std::printf("                    frame directories; the greyscale, colour and calibration downloads\n");
        std::printf("                    all extract into the same tree).\n");
        std::printf("    poses.txt     - The sequence's ground truth poses ('dataset/poses/[xx].txt', one 3x4\n");
        std::printf("                    row major matrix per frame), optional: the benchmark publishes them\n");
        std::printf("                    for sequences 00 to 10 only, and a sequence imported without them\n");
        std::printf("                    gives a scene that can be run but not benchmarked against.\n");
        std::printf("    output.mcap   - Where to write the packed scene; the directory form is written\n");
        std::printf("                    alongside it (output.mcap's path with the extension dropped),\n");
        std::printf("                    and left in place for inspection, exactly as 'dataset expand' does.\n");
        std::printf("    options:\n");
        std::printf("        --cameras [list]  - Which of the dataset's cameras to import, a comma separated\n");
        std::printf("                            list of 0 (left greyscale), 1 (right greyscale), 2 (left\n");
        std::printf("                            colour) and 3 (right colour) (default: every one present).\n");
        std::printf("        --tools-dir [dir] - Directory containing 'zeroslam-dataset' (default: next to\n");
        std::printf("                            this tool), used to pack and validate the result.\n");
        std::printf("Produces one camera per imported KITTI camera, in the dataset's own order (image_01 =\n");
        std::printf("the first imported, which is the scene's ego frame), all undistorted and rectified, and\n");
        std::printf("a ground truth pose per frame when the poses were given, keeping the sequence's own\n");
        std::printf("timestamps (seconds from the start of the sequence). The velodyne scans are not\n");
        std::printf("imported.\n");
    }

    bool parse_camera_list(const char* const list, bool* const requested, std::string& error) {
        const char* character = list;
        for (;;) {
            if ((*character < '0') || (*character >= ('0' + camera_count))) {
                error = "Invalid camera list '" + std::string(list) + "': expected a comma separated list of 0 to " + std::to_string(camera_count - 1) + ".";
                return false;
            }
            if (requested[*character - '0']) {
                error = "Invalid camera list '" + std::string(list) + "': camera " + std::string(1, *character) + " is named more than once.";
                return false;
            }
            requested[*character - '0'] = true;
            ++character;
            if (*character == '\0') {
                return true;
            }
            if (*character != ',') {
                error = "Invalid camera list '" + std::string(list) + "': expected a comma separated list of 0 to " + std::to_string(camera_count - 1) + ".";
                return false;
            }
            ++character;
        }
    }

    bool check_frames(const std::string& frames_directory, const std::size_t count, std::string& error) {
        for (std::size_t frame = 0; frame < count; ++frame) {
            const std::string frame_path = frames_directory + "/" + source_frame_filename(frame);
            if (!gtl::paths::is_regular_file(frame_path)) {
                error = "Missing frame: '" + frame_path + "' (times.txt lists " + std::to_string(count) + " frames).";
                return false;
            }
        }
        const std::string beyond_path = frames_directory + "/" + source_frame_filename(count);
        if (gtl::paths::is_regular_file(beyond_path)) {
            error = "'" + frames_directory + "' holds more frames than the " + std::to_string(count) + " times.txt lists ('" + beyond_path + "' exists); the frame directory and the timestamps are from different sequences.";
            return false;
        }
        return true;
    }
}

int main(int argc, char* argv[]) {
    std::string sequence_directory;
    std::string poses_path;
    std::string output_path;
    std::string tools_directory_override;
    std::vector<std::string> positionals;
    bool requested[camera_count] = {};
    bool any_requested = false;
    std::string parse_error;

    for (int i = 1; i < argc; ++i) {
        const auto matches = [&](const char* name) {
            return std::strcmp(argv[i], name) == 0;
        };
        if (matches("--help") || matches("-h")) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        else if (matches("--cameras")) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for option: --cameras\n");
                return EXIT_FAILURE;
            }
            if (!parse_camera_list(argv[++i], &requested[0], parse_error)) {
                std::fprintf(stderr, "%s\n", parse_error.c_str());
                return EXIT_FAILURE;
            }
            any_requested = true;
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
        else if (positionals.size() < 3) {
            positionals.push_back(argv[i]);
        }
        else {
            std::fprintf(stderr, "Unexpected argument: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }
    if (positionals.size() < 2) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    sequence_directory = positionals.front();
    output_path = positionals.back();
    if (positionals.size() == 3) {
        poses_path = positionals[1];
    }
    if (gtl::paths::is_regular_file(output_path) && (gtl::paths::path_extension(output_path) != ".mcap")) {
        std::fprintf(stderr, "Refusing to write the scene over '%s', an existing file that is not an mcap.%s\n", output_path.c_str(), (positionals.size() == 2) ? " A ground truth goes between the sequence directory and the output ('[sequence-dir] [poses.txt] [output.mcap]')." : "");
        return EXIT_FAILURE;
    }
    if (!any_requested) {
        std::fill(&requested[0], &requested[0] + camera_count, true);
    }

    std::string error;

    std::printf("Reading camera calibration...\n");
    projection projections[camera_count];
    if (!read_projections(sequence_directory + "/calib.txt", &projections[0], error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }

    std::printf("Reading frame timestamps...\n");
    std::vector<long long> timestamps;
    if (!read_timestamps(sequence_directory + "/times.txt", timestamps, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }
    if (timestamps.size() < 2) {
        std::fprintf(stderr, "'%s' holds fewer than 2 timestamps.\n", (sequence_directory + "/times.txt").c_str());
        return EXIT_FAILURE;
    }
    for (std::size_t index = 1; index < timestamps.size(); ++index) {
        if (timestamps[index] <= timestamps[index - 1]) {
            std::fprintf(stderr, "'%s' holds timestamps that do not increase, %lld then %lld at line %zu.\n", (sequence_directory + "/times.txt").c_str(), timestamps[index - 1], timestamps[index], index + 1);
            return EXIT_FAILURE;
        }
    }
    std::printf("    %zu frames over %.2f s.\n", timestamps.size(), static_cast<double>(timestamps.back() - timestamps.front()) / 1000000000.0);

    std::vector<rigid> poses;
    if (poses_path.empty()) {
        std::printf("No ground truth given, so the scene will carry none and cannot be benchmarked against.\n");
    }
    else {
        std::printf("Reading ground truth...\n");
        if (!read_poses(poses_path, poses, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return EXIT_FAILURE;
        }
        if (poses.size() != timestamps.size()) {
            std::fprintf(stderr, "'%s' holds %zu poses for the %zu frames of '%s' (the odometry benchmark publishes a pose per frame, for sequences 00 to 10 only).\n", poses_path.c_str(), poses.size(), timestamps.size(), (sequence_directory + "/times.txt").c_str());
            return EXIT_FAILURE;
        }
    }

    std::vector<scene_camera> cameras;
    rigid reference_from_primary;
    {
        std::vector<rigid> reference_from_camera;
        for (int index = 0; index < camera_count; ++index) {
            if (!requested[index]) {
                continue;
            }
            const std::string frames_directory = sequence_directory + "/" + camera_directory(index);
            if (!gtl::paths::is_directory(frames_directory)) {
                if (any_requested) {
                    std::fprintf(stderr, "Camera %d was requested but '%s' does not exist (the greyscale and colour images are separate downloads).\n", index, frames_directory.c_str());
                    return EXIT_FAILURE;
                }
                continue;
            }
            if (!projections[index].present) {
                std::fprintf(stderr, "'%s' holds no 'P%d' projection matrix for the frames in '%s'.\n", (sequence_directory + "/calib.txt").c_str(), index, frames_directory.c_str());
                return EXIT_FAILURE;
            }
            if (!check_frames(frames_directory, timestamps.size(), error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                return EXIT_FAILURE;
            }
            if (!is_rectified_projection(&projections[index].matrix[0])) {
                std::fprintf(stderr, "'P%d' of '%s' is not a rectified projection K [I | t]; this importer carries no rotation or skew from a projection matrix.\n", index, (sequence_directory + "/calib.txt").c_str());
                return EXIT_FAILURE;
            }
            scene_camera camera;
            camera.index = index;
            double offset[3] = {};
            decompose_projection(&projections[index].matrix[0], &camera.intrinsics[0], &offset[0]);
            if ((camera.intrinsics[0] <= 0.0) || (camera.intrinsics[1] <= 0.0)) {
                std::fprintf(stderr, "'P%d' of '%s' has no usable focal length.\n", index, (sequence_directory + "/calib.txt").c_str());
                return EXIT_FAILURE;
            }
            import::pixel_centre_principal_point(&camera.intrinsics[0]);
            rigid pose_in_reference;
            for (int axis = 0; axis < 3; ++axis) {
                pose_in_reference.translation[axis] = -offset[axis];
            }
            cameras.push_back(camera);
            reference_from_camera.push_back(pose_in_reference);
        }
        if (cameras.empty()) {
            std::fprintf(stderr, "'%s' holds none of the image_[0-%d] frame directories.\n", sequence_directory.c_str(), camera_count - 1);
            return EXIT_FAILURE;
        }
        reference_from_primary = reference_from_camera.front();
        const rigid primary_from_reference = reference_from_primary.inverse();
        for (std::size_t index = 0; index < cameras.size(); ++index) {
            char sensor_name[32];
            std::snprintf(&sensor_name[0], sizeof(sensor_name), "image_%02zu", index + 1);
            cameras[index].sensor_name = &sensor_name[0];
            (primary_from_reference * reference_from_camera[index]).to_pose(&cameras[index].translation[0], &cameras[index].quaternion_xyzw[0]);
        }
    }
    std::printf("Ego frame is the dataset's %s:\n", camera_directory(cameras.front().index).c_str());
    for (const scene_camera& camera : cameras) {
        std::printf("    %s = %s, %s, at (%.4f, %.4f, %.4f) m, [fx fy cx cy] = [%.4f %.4f %.4f %.4f].\n", camera.sensor_name.c_str(), camera_directory(camera.index).c_str(), (camera.index < 2) ? "greyscale" : "colour", camera.translation[0], camera.translation[1], camera.translation[2], camera.intrinsics[0], camera.intrinsics[1], camera.intrinsics[2], camera.intrinsics[3]);
    }

    const std::string output_directory = gtl::paths::path_stem(output_path);
    for (const scene_camera& camera : cameras) {
        if (!gtl::directory::make_directories(output_directory + "/sensor/" + camera.sensor_name)) {
            std::fprintf(stderr, "Failed to create: %s\n", (output_directory + "/sensor/" + camera.sensor_name).c_str());
            return EXIT_FAILURE;
        }
    }

    if (poses.empty()) {
        import::remove_absent_scene_file(output_directory + "/trajectory.txt");
    }
    else {
        std::printf("Writing the ground truth trajectory...\n");
        rigid world_from_kitti;
        std::copy(&world_rotation[0], &world_rotation[0] + 9, &world_from_kitti.rotation[0]);
        gtl::file trajectory((output_directory + "/trajectory.txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
        if (!trajectory.is_open()) {
            std::fprintf(stderr, "Failed to create: %s/trajectory.txt\n", output_directory.c_str());
            return EXIT_FAILURE;
        }
        double previous_xyzw[4] = { 0.0, 0.0, 0.0, 1.0 };
        for (std::size_t index = 0; index < timestamps.size(); ++index) {
            double position[3];
            double quaternion_xyzw[4];
            (world_from_kitti * poses[index] * reference_from_primary).to_pose(&position[0], &quaternion_xyzw[0]);
            double dot = 0.0;
            for (int element = 0; element < 4; ++element) {
                dot += quaternion_xyzw[element] * previous_xyzw[element];
            }
            if (dot < 0.0) {
                for (int element = 0; element < 4; ++element) {
                    quaternion_xyzw[element] = -quaternion_xyzw[element];
                }
            }
            std::copy(&quaternion_xyzw[0], &quaternion_xyzw[0] + 4, &previous_xyzw[0]);
            if (!dataset::write_pose_line(trajectory, timestamps[index], &position[0], &quaternion_xyzw[0])) {
                std::fprintf(stderr, "Failed to write trajectory.\n");
                return EXIT_FAILURE;
            }
        }
    }

    for (const scene_camera& camera : cameras) {
        std::printf("Writing %zu frames of %s (the dataset's %s)...\n", timestamps.size(), camera.sensor_name.c_str(), camera_directory(camera.index).c_str());
        std::fflush(stdout);
        const std::string frames_directory = sequence_directory + "/" + camera_directory(camera.index);
        const std::string destination_directory = output_directory + "/sensor/" + camera.sensor_name;
        gtl::file calibration((output_directory + "/sensor/" + camera.sensor_name + ".txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
        if (!calibration.is_open()) {
            std::fprintf(stderr, "Failed to create: %s/sensor/%s.txt\n", output_directory.c_str(), camera.sensor_name.c_str());
            return EXIT_FAILURE;
        }
        for (std::size_t index = 0; index < timestamps.size(); ++index) {
            const std::string source_path = frames_directory + "/" + source_frame_filename(index);
            const std::string destination_path = destination_directory + "/" + dataset::frame_filename(static_cast<unsigned long long>(timestamps[index]));
            if (!import::convert_image(source_path, (camera.index < 2) ? "-type Grayscale" : "-type TrueColor", destination_path, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                return EXIT_FAILURE;
            }
            if (!import::write_calibration_line(calibration, timestamps[index], &camera.translation[0], &camera.quaternion_xyzw[0], "plumb_bob", &camera.intrinsics[0], nullptr, 0)) {
                std::fprintf(stderr, "Failed to write the calibration for: %s\n", source_path.c_str());
                return EXIT_FAILURE;
            }
            if (((index + 1) % 500) == 0) {
                std::printf("    %zu of %zu frames.\n", index + 1, timestamps.size());
                std::fflush(stdout);
            }
        }
    }

    if (!import::collapse_scene(tools_directory_override, output_directory, output_path, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
