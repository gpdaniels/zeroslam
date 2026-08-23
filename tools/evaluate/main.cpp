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

#include "absolute_orientation.hpp"
#include "plot.hpp"

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

struct pose {
    double timestamp;
    double x_coordinate;
    double y_coordinate;
    double z_coordinate;
    double quaternion_x;
    double quaternion_y;
    double quaternion_z;
    double quaternion_w;
};

bool load_trajectory(const char* const filename, std::vector<pose>& trajectory) {
    std::FILE* const file_handle = std::fopen(filename, "r");
    if (file_handle == nullptr) {
        std::fprintf(stderr, "Failed to open file: %s\n", filename);
        return false;
    }
    char line_buffer[256];
    while (std::fgets(line_buffer, sizeof(line_buffer), file_handle) != nullptr) {
        pose point;
        const int items_scanned = std::sscanf(
            line_buffer,
            "%lf %lf %lf %lf %lf %lf %lf %lf",
            &point.timestamp,
            &point.x_coordinate,
            &point.y_coordinate,
            &point.z_coordinate,
            &point.quaternion_x,
            &point.quaternion_y,
            &point.quaternion_z,
            &point.quaternion_w
        );
        if (items_scanned == 8) {
            trajectory.push_back(point);
        }
    }
    std::fclose(file_handle);
    return true;
}

void print_usage(const char* const program_name) {
    const char* const usage_string = "Usage: %s <ground_truth.txt> [est1.txt] [est2.txt] [--first] [--plot|-p <axes>]...\n";
    std::printf(usage_string, program_name);
    std::printf("  First file is ground truth (reference)\n");
    std::printf("  Subsequent files are estimated trajectories to align and compare\n");
    std::printf("  --first: constrain the first pose of estimated and ground truth to overlap\n");
    std::printf("  axes: any combination of 'x', 'y', 'z' (e.g., 'xyz', 'xy', 'z')\n");
}

void print_alignment_report(
    const char* const estimated_filename,
    const double final_scale,
    const double final_rotation[3][3],
    const double final_translation[3]
) {
    std::printf("Alignment: %s\n", estimated_filename);
    std::printf("  Scale:       { % 3.5f }\n", final_scale);
    std::printf("  Rotation:    { % 3.5f, % 3.5f, % 3.5f }\n", final_rotation[0][0], final_rotation[0][1], final_rotation[0][2]);
    std::printf("  Rotation:    { % 3.5f, % 3.5f, % 3.5f }\n", final_rotation[1][0], final_rotation[1][1], final_rotation[1][2]);
    std::printf("  Rotation:    { % 3.5f, % 3.5f, % 3.5f }\n", final_rotation[2][0], final_rotation[2][1], final_rotation[2][2]);
    std::printf("  Translation: { % 3.5f, % 3.5f, % 3.5f }\n", final_translation[0], final_translation[1], final_translation[2]);
}

void print_error_statistics(
    const double maximum_error,
    const double minimum_error,
    const double mean_error,
    const double median_error,
    const double root_mean_square_error,
    const double sum_squared_error,
    const double standard_deviation_error
) {
    std::printf("Errors (m):\n");
    std::printf("  max:    %f\n", maximum_error);
    std::printf("  min:    %f\n", minimum_error);
    std::printf("  mean:   %f\n", mean_error);
    std::printf("  median: %f\n", median_error);
    std::printf("  rmse:   %f\n", root_mean_square_error);
    std::printf("  sse:    %f\n", sum_squared_error);
    std::printf("  std:    %f\n", standard_deviation_error);
}

void print_distance_statistics(
    const double ground_truth_distance,
    const double estimated_distance,
    const size_t alignment_pose_count,
    const size_t ground_truth_pose_count,
    const size_t estimated_trajectory_pose_count
) {
    std::printf("Trajectory Distances (m):\n");
    std::printf("  Ground Truth: %f (%zu/%zu poses)\n", ground_truth_distance, alignment_pose_count, ground_truth_pose_count);
    std::printf("  Estimated:    %f (%zu/%zu poses)\n", estimated_distance, alignment_pose_count, estimated_trajectory_pose_count);
}


int main(const int argc, char* argv[]) {
    std::string ground_truth_filename;
    std::string estimated_filename_1;
    std::string estimated_filename_2;
    bool should_overlap_first_pose = false;
    std::vector<std::string> requested_plot_axes;
    
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') {
            if (ground_truth_filename.empty()) {
                ground_truth_filename = argv[i];
            } else if (estimated_filename_1.empty()) {
                estimated_filename_1 = argv[i];
            } else if (estimated_filename_2.empty()) {
                estimated_filename_2 = argv[i];
            }
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else if (std::strcmp(argv[i], "--first") == 0 || std::strcmp(argv[i], "-f") == 0) {
            should_overlap_first_pose = true;
        } else if (std::strcmp(argv[i], "--plot") == 0 || std::strcmp(argv[i], "-p") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                requested_plot_axes.push_back(argv[++i]);
            }
        }
    }
    
    if (ground_truth_filename.empty()) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::vector<pose> ground_truth_trajectory;
    if (!load_trajectory(ground_truth_filename.c_str(), ground_truth_trajectory)) {
        std::fprintf(stderr, "Failed to load ground truth: %s\n", ground_truth_filename.c_str());
        return EXIT_FAILURE;
    }
    std::printf("Loaded ground truth: %s (%zu poses)\n", ground_truth_filename.c_str(), ground_truth_trajectory.size());
    
    const char* estimated_filenames[2];
    estimated_filenames[0] = estimated_filename_1.empty() ? nullptr : estimated_filename_1.c_str();
    estimated_filenames[1] = estimated_filename_2.empty() ? nullptr : estimated_filename_2.c_str();

    const int estimated_file_count = (estimated_filename_1.empty() ? 0 : 1) + (estimated_filename_2.empty() ? 0 : 1);

    std::vector<std::string> aligned_trajectory_names;
    std::vector<std::vector<double>> aligned_trajectories_x;
    std::vector<std::vector<double>> aligned_trajectories_y;
    std::vector<std::vector<double>> aligned_trajectories_z;
    
    for (int f = 0; f < estimated_file_count; ++f) {
        std::printf("\n");

        std::vector<pose> estimated_trajectory;
        if (!load_trajectory(estimated_filenames[f], estimated_trajectory)) {
            std::fprintf(stderr, "Failed to load: %s\n", estimated_filenames[f]);
            return EXIT_FAILURE;
        }
        std::printf("Loaded: %s (%zu poses)\n", estimated_filenames[f], estimated_trajectory.size());
        
        const size_t alignment_pose_count = std::min(estimated_trajectory.size(), ground_truth_trajectory.size());
        if (alignment_pose_count < 3) {
            std::fprintf(stderr, "Need at least 3 poses\n");
            return EXIT_FAILURE;
        }
        
        std::vector<double> estimated_xs;
        std::vector<double> estimated_ys;
        std::vector<double> estimated_zs;
        std::vector<double> ground_truth_xs;
        std::vector<double> ground_truth_ys;
        std::vector<double> ground_truth_zs;

        for (size_t i = 0; i < alignment_pose_count; ++i) {
            estimated_xs.push_back(estimated_trajectory[i].x_coordinate);
            estimated_ys.push_back(estimated_trajectory[i].y_coordinate);
            estimated_zs.push_back(estimated_trajectory[i].z_coordinate);
            ground_truth_xs.push_back(ground_truth_trajectory[i].x_coordinate);
            ground_truth_ys.push_back(ground_truth_trajectory[i].y_coordinate);
            ground_truth_zs.push_back(ground_truth_trajectory[i].z_coordinate);
        }
        
        double final_scale = 1.0;
        double final_rotation[3][3];
        double final_translation[3];
        
        if (!absolute_orientation(
            ground_truth_xs.data(),
            ground_truth_ys.data(),
            ground_truth_zs.data(),
            alignment_pose_count,
            estimated_xs.data(),
            estimated_ys.data(),
            estimated_zs.data(),
            alignment_pose_count,
            final_scale,
            final_rotation,
            final_translation,
            should_overlap_first_pose
        )) {
            std::fprintf(stderr, "Alignment failed\n");
            return EXIT_FAILURE;
        }
        
        print_alignment_report(estimated_filenames[f], final_scale, final_rotation, final_translation);
        
        std::vector<double> aligned_xs;
        std::vector<double> aligned_ys;
        std::vector<double> aligned_zs;

        for (size_t i = 0; i < alignment_pose_count; ++i) {
            const double original_x = estimated_trajectory[i].x_coordinate;
            const double original_y = estimated_trajectory[i].y_coordinate;
            const double original_z = estimated_trajectory[i].z_coordinate;

            const double aligned_x = (final_rotation[0][0] * original_x + final_rotation[0][1] * original_y + final_rotation[0][2] * original_z) * final_scale + final_translation[0];
            const double aligned_y = (final_rotation[1][0] * original_x + final_rotation[1][1] * original_y + final_rotation[1][2] * original_z) * final_scale + final_translation[1];
            const double aligned_z = (final_rotation[2][0] * original_x + final_rotation[2][1] * original_y + final_rotation[2][2] * original_z) * final_scale + final_translation[2];
            
            aligned_xs.push_back(aligned_x);
            aligned_ys.push_back(aligned_y);
            aligned_zs.push_back(aligned_z);
        }
        
        std::vector<double> error_magnitudes;
        double sum_of_errors = 0.0;
        double sum_of_squared_errors = 0.0;

        for (size_t i = 0; i < alignment_pose_count; ++i) {
            const double delta_x = aligned_xs[i] - ground_truth_trajectory[i].x_coordinate;
            const double delta_y = aligned_ys[i] - ground_truth_trajectory[i].y_coordinate;
            const double delta_z = aligned_zs[i] - ground_truth_trajectory[i].z_coordinate;
            const double error_magnitude = std::sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
            error_magnitudes.push_back(error_magnitude);
            sum_of_errors += error_magnitude;
            sum_of_squared_errors += (error_magnitude * error_magnitude);
        }

        std::sort(error_magnitudes.begin(), error_magnitudes.end());
        const double minimum_error = error_magnitudes.front();
        const double maximum_error = error_magnitudes.back();
        const double median_error = error_magnitudes[error_magnitudes.size() / 2];
        const double mean_error = sum_of_errors / static_cast<double>(alignment_pose_count);
        const double root_mean_square_error = std::sqrt(sum_of_squared_errors / static_cast<double>(alignment_pose_count));
        const double standard_deviation_error = std::sqrt(std::max(0.0, sum_of_squared_errors / static_cast<double>(alignment_pose_count) - mean_error * mean_error));
        
        print_error_statistics(maximum_error, minimum_error, mean_error, median_error, root_mean_square_error, sum_of_squared_errors, standard_deviation_error);

        double total_distance_estimated = 0.0;
        double total_distance_ground_truth = 0.0;
        for (size_t i = 0; i + 1 < alignment_pose_count; ++i) {
            const double delta_x_est = aligned_xs[i + 1] - aligned_xs[i];
            const double delta_y_est = aligned_ys[i + 1] - aligned_ys[i];
            const double delta_z_est = aligned_zs[i + 1] - aligned_zs[i];
            total_distance_estimated += std::sqrt(delta_x_est * delta_x_est + delta_y_est * delta_y_est + delta_z_est * delta_z_est);

            const double delta_x_gt = ground_truth_trajectory[i + 1].x_coordinate - ground_truth_trajectory[i].x_coordinate;
            const double delta_y_gt = ground_truth_trajectory[i + 1].y_coordinate - ground_truth_trajectory[i].y_coordinate;
            const double delta_z_gt = ground_truth_trajectory[i + 1].z_coordinate - ground_truth_trajectory[i].z_coordinate;
            total_distance_ground_truth += std::sqrt(delta_x_gt * delta_x_gt + delta_y_gt * delta_y_gt + delta_z_gt * delta_z_gt);
        }

        print_distance_statistics(total_distance_ground_truth, total_distance_estimated, alignment_pose_count, ground_truth_trajectory.size(), estimated_trajectory.size());
        
        aligned_trajectory_names.push_back(estimated_filenames[f]);
        aligned_trajectories_x.push_back(aligned_xs);
        aligned_trajectories_y.push_back(aligned_ys);
        aligned_trajectories_z.push_back(aligned_zs);
    }
    
    if (estimated_file_count == 0) {
        return EXIT_SUCCESS;
    }
    if (requested_plot_axes.empty()) {
        return EXIT_SUCCESS;
    }
    
    std::vector<double> ground_truth_xs;
    std::vector<double> ground_truth_ys;
    std::vector<double> ground_truth_zs;
    for (size_t i = 0; i < ground_truth_trajectory.size(); ++i) {
        ground_truth_xs.push_back(ground_truth_trajectory[i].x_coordinate);
        ground_truth_ys.push_back(ground_truth_trajectory[i].y_coordinate);
        ground_truth_zs.push_back(ground_truth_trajectory[i].z_coordinate);
    }
    
    const int image_width = 800;
    const int image_height = 600;
    const int plot_margin = 50;
    const unsigned char plot_colors[3][3] = { {255, 0, 0}, {0, 128, 0}, {0, 0, 255} };
        
    for (const std::string& axis_string : requested_plot_axes) {
        for (const char axis_char : axis_string) {
            if (axis_char == 'x') {
                draw_plot(ground_truth_ys, ground_truth_zs, aligned_trajectory_names, aligned_trajectories_y, aligned_trajectories_z, "y", "z", "trajectory_x.ppm", image_width, image_height, plot_margin, plot_colors);
            } else if (axis_char == 'y') {
                draw_plot(ground_truth_xs, ground_truth_zs, aligned_trajectory_names, aligned_trajectories_x, aligned_trajectories_z, "x", "z", "trajectory_y.ppm", image_width, image_height, plot_margin, plot_colors);
            } else if (axis_char == 'z') {
                draw_plot(ground_truth_xs, ground_truth_ys, aligned_trajectory_names, aligned_trajectories_x, aligned_trajectories_y, "x", "y", "trajectory_z.ppm", image_width, image_height, plot_margin, plot_colors);
            }
        }
    }
    
    return EXIT_SUCCESS;
}
