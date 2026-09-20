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
#include "metrics.hpp"
#include "plot.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    void print_usage(const char* const program_name) {
        std::printf("Usage: %s <ground_truth.txt> [estimate.txt] [estimate.txt] [--first|--centroid] [--max-rmse <m>] [--plot <axes>]...\n", program_name);
        std::printf("  ground_truth.txt: the reference trajectory in the TUM format\n");
        std::printf("  estimate.txt:     up to two estimated trajectories to align and compare, paired with the\n");
        std::printf("                    reference by timestamp (nearest within %.3f s)\n", static_cast<double>(metrics::default_association_tolerance_nanoseconds) * 1.0e-9);
        std::printf("  --first:          constrain the first pose pair to overlap (the default)\n");
        std::printf("  --centroid:       align about the two centroids instead, the least-squares best fit\n");
        std::printf("  --max-rmse:       exit with failure if any aligned ATE rmse exceeds this bound (metres)\n");
        std::printf("  --plot:           write trajectory_[axis].ppm for each axis of 'x', 'y', 'z' (e.g. 'xyz', 'xy', 'z')\n");
    }

    void print_report(const char* const estimated_filename, const metrics::result& evaluation, const bool overlapped_first_pose, const std::size_t ground_truth_poses, const std::size_t estimated_poses) {
        const metrics::similarity& transform = evaluation.transform;
        std::printf("Alignment: %s\n", estimated_filename);
        std::printf("  Anchor:      %s\n", overlapped_first_pose ? "first pose" : "centroid");
        std::printf("  Scale:       { % 3.5f }\n", transform.scale);
        for (int row = 0; row < 3; ++row) {
            std::printf("  Rotation:    { % 3.5f, % 3.5f, % 3.5f }\n", transform.rotation[row][0], transform.rotation[row][1], transform.rotation[row][2]);
        }
        std::printf("  Translation: { % 3.5f, % 3.5f, % 3.5f }\n", transform.translation[0], transform.translation[1], transform.translation[2]);
        if (evaluation.alignment_degenerate) {
            std::printf("  Conditioning: %.3g, a straight path; the rotation about the direction of travel is not observable from the positions and is taken from the trajectories' own orientations.\n", evaluation.alignment_conditioning);
        }

        std::printf("Errors (m):\n");
        std::printf("  max:    %f\n", evaluation.ate_maximum);
        std::printf("  min:    %f\n", evaluation.ate_minimum);
        std::printf("  mean:   %f\n", evaluation.ate_mean);
        std::printf("  median: %f\n", evaluation.ate_median);
        std::printf("  rmse:   %f\n", evaluation.ate_rmse);
        std::printf("  sse:    %f\n", evaluation.ate_sum_squared);
        std::printf("  std:    %f\n", evaluation.ate_standard_deviation);
        std::printf("statistics: rmse=%.9g mean=%.9g max=%.9g median=%.9g min=%.9g std=%.9g poses=%zu\n", evaluation.ate_rmse, evaluation.ate_mean, evaluation.ate_maximum, evaluation.ate_median, evaluation.ate_minimum, evaluation.ate_standard_deviation, evaluation.pairs.size());

        std::printf("Trajectory Distances (m, over the %zu associated poses):\n", evaluation.pairs.size());
        std::printf("  Ground Truth: %f (%zu poses in the file)\n", evaluation.distance_ground_truth, ground_truth_poses);
        std::printf("  Estimated:    %f (%zu poses in the file)\n", evaluation.distance_estimated, estimated_poses);

        std::printf("Relative Displacement Errors (m, aligned, over pose-index intervals):\n");
        for (const metrics::interval_result& relative : evaluation.relative) {
            if (relative.segments == 0) {
                std::printf("  interval %2zu: (not enough poses)\n", relative.interval);
                continue;
            }
            std::printf("  interval %2zu: rmse: %f mean: %f max: %f (%zu segments)\n", relative.interval, relative.rmse, relative.mean, relative.maximum, relative.segments);
        }
        if (evaluation.per_metre.segments != 0) {
            std::printf("Relative Pose Error per metre of ground truth path (%zu segments):\n", evaluation.per_metre.segments);
            std::printf("  translation drift: rmse %.2f%% median %.2f%%\n", evaluation.per_metre.translation_percent_rmse, evaluation.per_metre.translation_percent_median);
            std::printf("  rotation:          rmse %.3f deg median %.3f deg\n", evaluation.per_metre.rotation_degrees_rmse, evaluation.per_metre.rotation_degrees_median);
        }
        std::printf("Per-Segment Scale Drift (aligned est/gt segment length):\n");
        for (const metrics::interval_result& relative : evaluation.relative) {
            if (relative.segments == 0) {
                std::printf("  interval %2zu: (not enough poses)\n", relative.interval);
                continue;
            }
            if (relative.measurable_segments == 0) {
                std::printf("  interval %2zu: (no measurable ground-truth segments)\n", relative.interval);
                continue;
            }
            std::printf("  interval %2zu: median: %f worst: %f (drift %+.2f%% median, %+.2f%% worst, %zu segments)\n", relative.interval, relative.scale_median, relative.scale_worst, (relative.scale_median - 1.0) * 100.0, (relative.scale_worst - 1.0) * 100.0, relative.measurable_segments);
        }
    }
}

int main(const int argc, char* argv[]) {
    std::string ground_truth_filename;
    std::vector<std::string> estimated_filenames;
    bool should_overlap_first_pose = true;
    bool rmse_gate_enabled = false;
    double rmse_gate_maximum = 0.0;
    std::vector<std::string> requested_plot_axes;

    for (int i = 1; i < argc; ++i) {
        const auto matches = [&](const char* name) {
            return std::strcmp(argv[i], name) == 0;
        };
        const auto take_value = [&](const char*& value) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for option: %s\n", argv[i]);
                return false;
            }
            value = argv[++i];
            return true;
        };
        if (matches("--help") || matches("-h")) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        else if (matches("--first") || matches("-f")) {
            should_overlap_first_pose = true;
        }
        else if (matches("--centroid") || matches("-c")) {
            should_overlap_first_pose = false;
        }
        else if (matches("--max-rmse")) {
            const char* text = nullptr;
            if (!take_value(text)) {
                return EXIT_FAILURE;
            }
            char* parse_end = nullptr;
            rmse_gate_maximum = std::strtod(text, &parse_end);
            if ((parse_end == text) || (*parse_end != 0) || !(rmse_gate_maximum > 0.0)) {
                std::fprintf(stderr, "--max-rmse requires a positive number, got '%s'\n", text);
                return EXIT_FAILURE;
            }
            rmse_gate_enabled = true;
        }
        else if (matches("--plot") || matches("-p")) {
            const char* axes = nullptr;
            if (!take_value(axes)) {
                return EXIT_FAILURE;
            }
            if ((*axes == '\0') || (std::strspn(axes, "xyz") != std::strlen(axes))) {
                std::fprintf(stderr, "--plot requires a combination of 'x', 'y', and 'z', got '%s'\n", axes);
                return EXIT_FAILURE;
            }
            requested_plot_axes.push_back(axes);
        }
        else if (argv[i][0] == '-') {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
        else if (ground_truth_filename.empty()) {
            ground_truth_filename = argv[i];
        }
        else if (estimated_filenames.size() < 2) {
            estimated_filenames.push_back(argv[i]);
        }
        else {
            std::fprintf(stderr, "Unexpected argument: %s (at most two estimated trajectories)\n", argv[i]);
            return EXIT_FAILURE;
        }
    }

    if (ground_truth_filename.empty()) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::vector<dataset::trajectory_pose> ground_truth_trajectory;
    if (!dataset::load_trajectory(ground_truth_filename, ground_truth_trajectory)) {
        std::fprintf(stderr, "Failed to load ground truth: %s\n", ground_truth_filename.c_str());
        return EXIT_FAILURE;
    }
    std::printf("Loaded ground truth: %s (%zu poses)\n", ground_truth_filename.c_str(), ground_truth_trajectory.size());

    std::vector<std::vector<double>> aligned_trajectories_x;
    std::vector<std::vector<double>> aligned_trajectories_y;
    std::vector<std::vector<double>> aligned_trajectories_z;
    bool rmse_gate_failed = false;

    for (const std::string& estimated_filename : estimated_filenames) {
        std::printf("\n");

        std::vector<dataset::trajectory_pose> estimated_trajectory;
        if (!dataset::load_trajectory(estimated_filename, estimated_trajectory)) {
            std::fprintf(stderr, "Failed to load: %s\n", estimated_filename.c_str());
            return EXIT_FAILURE;
        }
        std::printf("Loaded: %s (%zu poses)\n", estimated_filename.c_str(), estimated_trajectory.size());

        const metrics::result evaluation = metrics::evaluate(ground_truth_trajectory, estimated_trajectory, should_overlap_first_pose);
        std::printf("Associated: %zu of %zu estimated poses with a ground truth pose (%zu of %zu, within %.3f s)\n", evaluation.pairs.size(), estimated_trajectory.size(), evaluation.pairs.size(), ground_truth_trajectory.size(), static_cast<double>(metrics::default_association_tolerance_nanoseconds) * 1.0e-9);
        if (evaluation.pairs.size() < 3) {
            std::fprintf(stderr, "Need at least 3 associated poses\n");
            return EXIT_FAILURE;
        }
        if (!evaluation.valid) {
            std::fprintf(stderr, "Alignment failed\n");
            return EXIT_FAILURE;
        }
        print_report(estimated_filename.c_str(), evaluation, should_overlap_first_pose, ground_truth_trajectory.size(), estimated_trajectory.size());

        if (rmse_gate_enabled) {
            const bool gate_passed = (evaluation.ate_rmse <= rmse_gate_maximum);
            std::printf("RMSE gate: %f <= %f -> %s\n", evaluation.ate_rmse, rmse_gate_maximum, gate_passed ? "PASS" : "FAIL");
            rmse_gate_failed = rmse_gate_failed || !gate_passed;
        }

        aligned_trajectories_x.push_back(evaluation.aligned_x);
        aligned_trajectories_y.push_back(evaluation.aligned_y);
        aligned_trajectories_z.push_back(evaluation.aligned_z);
    }

    if (estimated_filenames.empty()) {
        if (rmse_gate_enabled) {
            std::fprintf(stderr, "--max-rmse was given but no estimated trajectory was provided\n");
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
    if (requested_plot_axes.empty()) {
        return rmse_gate_failed ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    std::vector<double> ground_truth_xs;
    std::vector<double> ground_truth_ys;
    std::vector<double> ground_truth_zs;
    for (const dataset::trajectory_pose& pose : ground_truth_trajectory) {
        ground_truth_xs.push_back(pose.x_coordinate);
        ground_truth_ys.push_back(pose.y_coordinate);
        ground_truth_zs.push_back(pose.z_coordinate);
    }

    const int image_width = 800;
    const int image_height = 600;
    const int plot_margin = 50;
    const unsigned char plot_colors[3][3] = { { 255, 0, 0 }, { 0, 128, 0 }, { 0, 0, 255 } };

    for (const std::string& axis_string : requested_plot_axes) {
        for (const char axis_char : axis_string) {
            bool plotted = true;
            if (axis_char == 'x') {
                plotted = draw_plot(ground_truth_ys, ground_truth_zs, estimated_filenames, aligned_trajectories_y, aligned_trajectories_z, "y", "z", "trajectory_x.ppm", image_width, image_height, plot_margin, plot_colors);
            }
            else if (axis_char == 'y') {
                plotted = draw_plot(ground_truth_xs, ground_truth_zs, estimated_filenames, aligned_trajectories_x, aligned_trajectories_z, "x", "z", "trajectory_y.ppm", image_width, image_height, plot_margin, plot_colors);
            }
            else if (axis_char == 'z') {
                plotted = draw_plot(ground_truth_xs, ground_truth_ys, estimated_filenames, aligned_trajectories_x, aligned_trajectories_y, "x", "y", "trajectory_z.ppm", image_width, image_height, plot_margin, plot_colors);
            }
            if (!plotted) {
                std::fprintf(stderr, "Failed to write the '%c' plot\n", axis_char);
                return EXIT_FAILURE;
            }
        }
    }

    return rmse_gate_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
