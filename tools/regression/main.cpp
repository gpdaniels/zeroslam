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

#include "dataset.hpp"
#include "directory.hpp"
#include "file.hpp"
#include "paths.hpp"
#include "process.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    struct error_statistics {
        double maximum = 0.0;
        double minimum = 0.0;
        double mean = 0.0;
        double median = 0.0;
        double rmse = 0.0;
        double standard_deviation = 0.0;
    };

    bool parse_error_statistics(const std::string& output, error_statistics& statistics) {
        for (std::size_t line_start = 0; line_start < output.size();) {
            std::size_t line_end = output.find('\n', line_start);
            if (line_end == std::string::npos) {
                line_end = output.size();
            }
            const std::string line = output.substr(line_start, line_end - line_start);
            line_start = line_end + 1;
            std::size_t poses = 0;
            if (std::sscanf(line.c_str(), "statistics: rmse=%lf mean=%lf max=%lf median=%lf min=%lf std=%lf poses=%zu", &statistics.rmse, &statistics.mean, &statistics.maximum, &statistics.median, &statistics.minimum, &statistics.standard_deviation, &poses) == 7) {
                return true;
            }
        }
        return false;
    }

    // The commit recorded in the log: the git HEAD of the tools directory, falling back to
    // the current working directory, so results are attributed to the code that was built.
    std::string discover_commit_hash(const std::string& tools_directory) {
        const std::string directories[2] = { tools_directory, std::string(".") };
        for (const std::string& directory : directories) {
            if (directory.empty()) {
                continue;
            }
            std::string output;
            int exit_code = -1;
            const std::string command = "git -C " + platform::quote_for_shell(directory) + " rev-parse --short=12 HEAD 2>&1";
            if (!platform::run_command(command, false, output, exit_code) || (exit_code != 0)) {
                continue;
            }
            std::string hash;
            for (const char character : output) {
                const bool hexadecimal = ((character >= '0') && (character <= '9')) || ((character >= 'a') && (character <= 'f'));
                if (!hexadecimal) {
                    break;
                }
                hash += character;
            }
            if (hash.size() >= 7) {
                return hash;
            }
        }
        return "unknown";
    }

    std::string make_absolute(const std::string& path) {
        if (path.empty()) {
            return path;
        }
        if (path[0] == '/' || path[0] == '\\' || (path.size() >= 2 && path[1] == ':')) {
            return path;
        }
        std::string cwd;
        if (!gtl::paths::get_current_working_directory(cwd)) {
            return path;
        }
        return cwd + "/" + path;
    }

    void print_usage(const char* argv0) {
        std::printf("Usage %s [scene] [options...]\n", argv0);
        std::printf("    scene - The path of a scene mcap file (datasets/[dataset]/[scene].mcap) holding raw\n");
        std::printf("            mono8 or rgb8 image messages, the camera intrinsics, and the ground truth as\n");
        std::printf("            the root -> ego -> sensor frame tree on /tf; or a bare '[dataset]/[scene]' name,\n");
        std::printf("            looked up in the datasets directory next to the tools. A scene that does not\n");
        std::printf("            exist locally is first fetched with the dataset tool.\n");
        std::printf("    options:\n");
        std::printf("        --ground-truth [file]  - Override the ground truth with a TUM trajectory file.\n");
        std::printf("        --tools-dir [dir]      - Directory containing the zeroslam tools (default: next to this tool).\n");
        std::printf("        --work-dir [dir]       - Directory for run outputs (default: ./zeroslam-regression-work/[name]).\n");
        std::printf("        --log [file]           - Log file to append the results to (default: ./regression.log).\n");
        std::printf("        --name [name]          - Dataset name recorded in the log (default: from the scene path).\n");
        std::printf("        --commit [hash]        - Commit hash recorded in the log (default: the git HEAD of the tools directory).\n");
        std::printf("        --frames [count]       - Process only the first [count] frames of the scene.\n");
        std::printf("        --config [key=value]   - Slam configuration setting forwarded to the process tool (repeatable).\n");
        std::printf("        --first                - Anchor the evaluation alignment on the first pose pair (the default).\n");
        std::printf("        --centroid             - Anchor the evaluation alignment on the two centroids instead.\n");
        std::printf("The tool validates the scene, runs the SLAM system on it, evaluates the recorded trajectory\n");
        std::printf("against the ground truth, and appends one line of results to the log file. The recorded metrics\n");
        std::printf("never fail the run: the exit code only reflects operational failures.\n");
    }
}

int main(int argc, char* argv[]) {
    std::string scene_path;
    std::string ground_truth_override;
    std::string tools_directory_override;
    std::string work_directory_override;
    std::string log_path = "regression.log";
    std::string dataset_name_override;
    std::string commit_override;
    std::string frames_override;
    std::vector<std::string> config_settings;
    bool overlap_first_pose = true;

    for (int i = 1; i < argc; ++i) {
        const auto matches = [&](const char* name) {
            return std::strcmp(argv[i], name) == 0;
        };
        const auto take_value = [&](std::string& destination) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for option: %s\n", argv[i]);
                return false;
            }
            destination = argv[++i];
            return true;
        };
        if (matches("--help") || matches("-h")) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        else if (matches("--ground-truth")) {
            if (!take_value(ground_truth_override))
                return EXIT_FAILURE;
        }
        else if (matches("--tools-dir")) {
            if (!take_value(tools_directory_override))
                return EXIT_FAILURE;
        }
        else if (matches("--work-dir")) {
            if (!take_value(work_directory_override))
                return EXIT_FAILURE;
        }
        else if (matches("--log")) {
            if (!take_value(log_path))
                return EXIT_FAILURE;
        }
        else if (matches("--name")) {
            if (!take_value(dataset_name_override))
                return EXIT_FAILURE;
        }
        else if (matches("--commit")) {
            if (!take_value(commit_override))
                return EXIT_FAILURE;
        }
        else if (matches("--frames")) {
            if (!take_value(frames_override))
                return EXIT_FAILURE;
        }
        else if (matches("--config")) {
            std::string setting;
            if (!take_value(setting))
                return EXIT_FAILURE;
            config_settings.push_back(setting);
        }
        else if (matches("--first") || matches("-f")) {
            overlap_first_pose = true;
        }
        else if (matches("--centroid") || matches("-c")) {
            overlap_first_pose = false;
        }
        else if (argv[i][0] == '-') {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
        else if (scene_path.empty()) {
            scene_path = argv[i];
        }
        else {
            std::fprintf(stderr, "Unexpected argument: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }

    if (scene_path.empty()) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    std::size_t frame_limit = 0;
    if (!frames_override.empty()) {
        char* end_pointer = nullptr;
        const unsigned long value = std::strtoul(frames_override.c_str(), &end_pointer, 10);
        if ((end_pointer == frames_override.c_str()) || (*end_pointer != 0) || (value < 2)) {
            std::fprintf(stderr, "Invalid value for --frames: '%s' (at least 2 frames are needed).\n", frames_override.c_str());
            return EXIT_FAILURE;
        }
        frame_limit = static_cast<std::size_t>(value);
    }

    // The tools default to the directory containing this tool.
    std::string executable_directory;
    std::string default_tools_directory = ".";
    if (gtl::paths::get_executable_directory(executable_directory)) {
        default_tools_directory = executable_directory;
    }
    const std::string tools_directory = tools_directory_override.empty() ? default_tools_directory : tools_directory_override;
#if defined(_WIN32)
    const char* const executable_extension = ".exe";
#else
    const char* const executable_extension = "";
#endif
    const std::string process_binary = tools_directory + "/zeroslam-process" + executable_extension;
    const std::string evaluate_binary = tools_directory + "/zeroslam-evaluate" + executable_extension;
    for (const std::string& binary : { process_binary, evaluate_binary }) {
        if (!gtl::paths::is_regular_file(binary)) {
            std::fprintf(stderr, "Tool not found: '%s' (build all tools, or provide --tools-dir).\n", binary.c_str());
            return EXIT_FAILURE;
        }
    }

    if (!gtl::paths::is_regular_file(scene_path) && (gtl::paths::path_extension(scene_path) != ".mcap")) {
        scene_path = tools_directory + "/datasets/" + scene_path + ".mcap";
    }

    // The recorded name is "[dataset]/[scene]", the last two path components without the
    // mcap extension.
    std::string scene_name = scene_path;
    {
        constexpr static const char extension[] = ".mcap";
        constexpr static const std::size_t extension_length = sizeof(extension) - 1;
        if ((scene_name.size() > extension_length) && (scene_name.compare(scene_name.size() - extension_length, extension_length, extension) == 0)) {
            scene_name.resize(scene_name.size() - extension_length);
        }
        std::vector<std::string> components;
        std::string component;
        for (const char character : scene_name + "/") {
            if ((character == '/') || (character == '\\')) {
                if (!component.empty() && (component != ".") && (component != "..")) {
                    components.push_back(component);
                }
                component.clear();
            }
            else {
                component += character;
            }
        }
        if (components.size() >= 2) {
            scene_name = components[components.size() - 2] + "/" + components.back();
        }
        else if (!components.empty()) {
            scene_name = components.back();
        }
    }

    // When the scene file does not exist locally, fetch it with the dataset tool.
    {
        const std::string dataset_binary = tools_directory + "/zeroslam-dataset" + executable_extension;
        if (!gtl::paths::is_regular_file(scene_path) && gtl::paths::is_regular_file(dataset_binary)) {
            // The datasets root sits above the "[dataset]/[scene].mcap" components.
            const std::string trimmed = scene_path;
            const std::size_t last_separator = trimmed.find_last_of("/\\");
            const std::size_t previous_separator = ((last_separator == std::string::npos) || (last_separator == 0)) ? std::string::npos : trimmed.find_last_of("/\\", last_separator - 1);
            const std::string root_directory = (previous_separator == std::string::npos) ? std::string(".") : trimmed.substr(0, previous_separator);
            std::printf("Scene not found locally, fetching '%s' with the dataset tool...\n", scene_name.c_str());
            std::fflush(stdout);
            const std::string fetch_command = platform::quote_for_shell(make_absolute(dataset_binary)) + " get " + platform::quote_for_shell(scene_name) + " --datasets " + platform::quote_for_shell(make_absolute(root_directory)) + " 2>&1";
            std::string fetch_output;
            int fetch_exit_code = -1;
            if (!platform::run_command(fetch_command, true, fetch_output, fetch_exit_code) || (fetch_exit_code != 0)) {
                std::fprintf(stderr, "Failed to fetch the scene (exit code %d).\n", fetch_exit_code);
                return EXIT_FAILURE;
            }
            std::printf("\n");
        }
    }

    std::printf("Validating scene...\n");
    std::fflush(stdout);
    mcap reader;
    std::string scene_error;
    if (!reader.open(scene_path, scene_error)) {
        std::fprintf(stderr, "Invalid scene: %s: %s.\n", scene_path.c_str(), scene_error.c_str());
        return EXIT_FAILURE;
    }
    dataset::mcap_scene_information scene;
    if (!dataset::inspect_mcap_scene(reader, scene, scene_error, false)) {
        std::fprintf(stderr, "Invalid scene: %s: %s.\n", scene_path.c_str(), scene_error.c_str());
        return EXIT_FAILURE;
    }
    const dataset::camera_information& camera = scene.primary();
    if (camera.frames < 2) {
        std::fprintf(stderr, "Invalid scene: at least two frames are needed.\n");
        return EXIT_FAILURE;
    }
    if (!camera.camera_info || (camera.fx <= 0.0) || (camera.fy <= 0.0)) {
        std::fprintf(stderr, "Invalid scene: no usable camera intrinsics.\n");
        return EXIT_FAILURE;
    }
    if ((camera.encoding != "mono8") && (camera.encoding != "rgb8")) {
        std::fprintf(stderr, "Invalid scene: the primary camera holds '%s' frames; the process tool consumes mono8 or rgb8.\n", camera.encoding.c_str());
        return EXIT_FAILURE;
    }
    if (ground_truth_override.empty() && (scene.poses < 3)) {
        std::fprintf(stderr, "Invalid scene: at least 3 ground truth poses are needed (or provide --ground-truth).\n");
        return EXIT_FAILURE;
    }
    std::size_t frames = camera.frames;
    if ((frame_limit > 0) && (frame_limit < frames)) {
        frames = frame_limit;
    }

    std::printf("    scene:        %s\n", scene_name.c_str());
    std::printf("    camera:       %s (pinhole [%.10g %.10g %.10g %.10g], %zu frames of %ux%u)\n", camera.camera_name.c_str(), camera.fx, camera.fy, camera.cx, camera.cy, camera.frames, camera.width, camera.height);
    if (frames < camera.frames) {
        std::printf("    limit:        processing the first %zu of %zu frames\n", frames, camera.frames);
    }
    else if (frame_limit > camera.frames) {
        std::printf("    limit:        --frames %zu requested, the scene has only %zu frames\n", frame_limit, camera.frames);
    }
    std::printf("    ground truth: %zu poses%s\n", scene.poses, ground_truth_override.empty() ? "" : " (overridden)");

    // The log file is space separated, keep the recorded fields to single tokens.
    std::string dataset_name = dataset_name_override.empty() ? scene_name : dataset_name_override;
    for (char& character : dataset_name) {
        if ((character == ' ') || (character == '\t')) {
            character = '_';
        }
    }

    const std::string work_directory = work_directory_override.empty() ? ("zeroslam-regression-work/" + dataset_name) : work_directory_override;
    if (!gtl::directory::make_directories(work_directory)) {
        std::fprintf(stderr, "Failed to create the work directory: %s\n", work_directory.c_str());
        return EXIT_FAILURE;
    }

    // The ground truth for the evaluation: an override file, or the pose messages
    // extracted into the work directory as a nanosecond-timestamped TUM-style trajectory.
    std::string ground_truth_path = ground_truth_override;
    if (ground_truth_path.empty()) {
        ground_truth_path = work_directory + "/groundtruth.txt";
        std::size_t poses = 0;
        if (!dataset::write_trajectory(scene, ground_truth_path, poses)) {
            std::fprintf(stderr, "Failed to write the ground truth: %s\n", ground_truth_path.c_str());
            return EXIT_FAILURE;
        }
    }
    else if (dataset::count_trajectory_poses(ground_truth_path) < 3) {
        std::fprintf(stderr, "Invalid ground truth override: '%s' needs at least 3 nanosecond-timestamped poses.\n", ground_truth_path.c_str());
        return EXIT_FAILURE;
    }

    // The processing tool writes its outputs into the current working directory,
    // run it from the work directory with absolute paths to everything else.
    std::printf("\nRunning the SLAM system...\n");
#if defined(_WIN32)
    const std::string change_directory = "cd /d " + platform::quote_for_shell(work_directory) + " && ";
#else
    const std::string change_directory = "cd " + platform::quote_for_shell(work_directory) + " && ";
#endif
    std::string process_command = change_directory + platform::quote_for_shell(make_absolute(process_binary)) + " " + platform::quote_for_shell(make_absolute(scene_path));
    if (frames < camera.frames) {
        process_command += " --frames " + std::to_string(frames);
    }
    for (const std::string& setting : config_settings) {
        process_command += " --config " + platform::quote_for_shell(setting);
    }
    process_command += " 2>&1";

    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::string process_output;
    int process_exit_code = -1;
    if (!platform::run_command(process_command, true, process_output, process_exit_code) || (process_exit_code != 0)) {
        std::fprintf(stderr, "The processing tool failed (exit code %d).\n", process_exit_code);
        return EXIT_FAILURE;
    }
    const std::chrono::duration<double> duration = std::chrono::steady_clock::now() - start;

    const std::string trajectory_path = work_directory + "/trajectory.txt";
    if (dataset::count_trajectory_poses(trajectory_path) < 3) {
        std::fprintf(stderr, "The processing tool did not produce a usable trajectory: %s\n", trajectory_path.c_str());
        return EXIT_FAILURE;
    }

    std::printf("\nEvaluating the trajectory...\n");
    std::string evaluate_command = platform::quote_for_shell(make_absolute(evaluate_binary)) + " " + platform::quote_for_shell(make_absolute(ground_truth_path)) + " " + platform::quote_for_shell(make_absolute(trajectory_path));
    evaluate_command += overlap_first_pose ? " --first" : " --centroid";
    evaluate_command += " 2>&1";
    std::string evaluate_output;
    int evaluate_exit_code = -1;
    if (!platform::run_command(evaluate_command, true, evaluate_output, evaluate_exit_code) || (evaluate_exit_code != 0)) {
        std::fprintf(stderr, "The evaluation tool failed (exit code %d).\n", evaluate_exit_code);
        return EXIT_FAILURE;
    }
    error_statistics statistics;
    if (!parse_error_statistics(evaluate_output, statistics)) {
        std::fprintf(stderr, "Failed to parse the error statistics from the evaluation tool output.\n");
        return EXIT_FAILURE;
    }

    std::string commit = commit_override.empty() ? discover_commit_hash(tools_directory) : commit_override;
    for (char& character : commit) {
        if ((character == ' ') || (character == '\t')) {
            character = '_';
        }
    }

    char timestamp[32] = "unknown";
    const std::time_t now = std::time(nullptr);
    const std::tm* const utc = std::gmtime(&now);
    if (utc != nullptr) {
        std::strftime(&timestamp[0], sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", utc);
    }

    gtl::file log_handle(log_path.c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::end_of_file);
    if (!log_handle.is_open()) {
        std::fprintf(stderr, "Failed to open the log file: %s\n", log_path.c_str());
        return EXIT_FAILURE;
    }
    {
        gtl::file::size_type log_size = 0;
        log_handle.get_size(log_size);
        if (log_size == 0) {
            const char header[] = "# commit dataset frames ate_rmse_m ate_mean_m ate_max_m ate_median_m ate_min_m ate_std_m duration_s timestamp\n";
            gtl::file::size_type header_length = sizeof(header) - 1;
            log_handle.write(header, header_length);
        }
    }
    char line[512];
    const int line_length = std::snprintf(
        &line[0],
        sizeof(line),
        "%s %s %zu %.6f %.6f %.6f %.6f %.6f %.6f %.2f %s",
        commit.c_str(),
        dataset_name.c_str(),
        frames,
        statistics.rmse,
        statistics.mean,
        statistics.maximum,
        statistics.median,
        statistics.minimum,
        statistics.standard_deviation,
        duration.count(),
        &timestamp[0]
    );
    if (line_length > 0) {
        gtl::file::size_type write_length = static_cast<gtl::file::size_type>(line_length);
        log_handle.write(&line[0], write_length);
        const char newline = '\n';
        gtl::file::size_type newline_length = 1;
        log_handle.write(&newline, newline_length);
    }

    std::printf("\nAppended to %s:\n", log_path.c_str());
    std::printf("    %s\n", &line[0]);
    return EXIT_SUCCESS;
}
