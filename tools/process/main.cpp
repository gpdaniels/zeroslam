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

#include "cdr.hpp"
#include "dataset.hpp"
#include "file.hpp"
#include "mcap.hpp"
#include "metrics.hpp"
#include "rotation.hpp"
#include "zeroslam/zeroslam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace {
    std::atomic<bool> shutdown_requested{ false };

    void signal_handler(int) {
        std::signal(SIGINT, SIG_DFL);
        shutdown_requested = true;
    }

    int verbosity = 1;
    const char* truth_path = nullptr;

#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    void log(const int level, const char* const format, ...) {
        if (level > verbosity) {
            return;
        }
        FILE* const stream = (level == 1) ? stderr : stdout;
        va_list arguments;
        va_start(arguments, format);
        std::vfprintf(stream, format, arguments);
        va_end(arguments);
        std::fputc('\n', stream);
    }

    struct camera_pose {
        long long timestamp;
        double centre[3];
        double quaternion_xyzw[4];
    };
}

inline bool print_evaluation(const char* const trajectory_path, const std::vector<camera_pose>& poses, const size_t frames_ingested) {
    std::vector<dataset::trajectory_pose> ground_truth;
    if (!dataset::load_trajectory(trajectory_path, ground_truth)) {
        std::fprintf(stderr, "Failed to load the ground truth trajectory: %s\n", trajectory_path);
        return false;
    }
    std::vector<dataset::trajectory_pose> estimated;
    estimated.reserve(poses.size());
    for (const camera_pose& pose : poses) {
        dataset::trajectory_pose converted;
        converted.timestamp_nanoseconds = pose.timestamp;
        converted.x_coordinate = pose.centre[0];
        converted.y_coordinate = pose.centre[1];
        converted.z_coordinate = pose.centre[2];
        converted.quaternion_x = pose.quaternion_xyzw[0];
        converted.quaternion_y = pose.quaternion_xyzw[1];
        converted.quaternion_z = pose.quaternion_xyzw[2];
        converted.quaternion_w = pose.quaternion_xyzw[3];
        estimated.push_back(converted);
    }
    const metrics::result evaluation = metrics::evaluate(ground_truth, estimated, true);
    std::printf("Evaluation against %s:\n", trajectory_path);
    std::printf("  Poses:     %zu of %zu frames posed, %zu paired with the ground truth\n", poses.size(), frames_ingested, evaluation.pairs.size());
    if (!evaluation.valid) {
        std::printf("  (too few paired poses, or a degenerate alignment)\n");
        return true;
    }
    std::printf("  Absolute:  rmse %.6f m, max %.6f m (first pose anchored alignment, scale %.4f)\n", evaluation.ate_rmse, evaluation.ate_maximum, evaluation.transform.scale);
    std::printf("  Distance:  estimated %.3f m over a ground truth %.3f m (ratio %.4f)\n", evaluation.distance_estimated, evaluation.distance_ground_truth, (evaluation.distance_ground_truth > 0.0) ? (evaluation.distance_estimated / evaluation.distance_ground_truth) : 0.0);
    if (evaluation.per_metre.segments != 0) {
        std::printf("  Per metre: translation drift %.2f%% rmse (%.2f%% median), rotation %.3f deg rmse (%.3f deg median), %zu segments\n", evaluation.per_metre.translation_percent_rmse, evaluation.per_metre.translation_percent_median, evaluation.per_metre.rotation_degrees_rmse, evaluation.per_metre.rotation_degrees_median, evaluation.per_metre.segments);
    }
    for (const metrics::interval_result& relative : evaluation.relative) {
        if (relative.segments == 0) {
            continue;
        }
        std::printf("  Relative:  interval %2zu: rmse %.6f m, max %.6f m", relative.interval, relative.rmse, relative.maximum);
        if (relative.measurable_segments != 0) {
            std::printf("; segment scale median %.4f, worst %.4f", relative.scale_median, relative.scale_worst);
        }
        std::printf(" (%zu segments)\n", relative.segments);
    }
    return true;
}

inline bool save_trajectory_as_txt(const char* path, const std::vector<camera_pose>& poses) {
    gtl::file output(path, gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    if (!output.is_open()) {
        return false;
    }
    for (const camera_pose& pose : poses) {
        if (!dataset::write_pose_line(output, pose.timestamp, &pose.centre[0], &pose.quaternion_xyzw[0])) {
            return false;
        }
    }
    return true;
}

inline bool save_trajectory_and_map_as_ply(
    const char* path,
    const zeroslam_sensor_parameters_camera_struct& camera,
    const std::vector<zeroslam_point_struct>& landmarks,
    const std::vector<camera_pose>& poses
) {
    // Camera colour (blue for trajectory).
    constexpr static const unsigned char cam_r = 0;
    constexpr static const unsigned char cam_g = 255;
    constexpr static const unsigned char cam_b = 255;
    // Camera frustum parameters.
    constexpr static const double frustum_scale = 1.0;
    // Vertices and edges.
    constexpr static const size_t vertices_per_camera = 5; // 1 centre + 4 corners
    constexpr static const size_t edges_per_camera = 8;    // 4 from centre to corners + 4 rectangle edges

    gtl::file output(path, gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    if (!output.is_open()) {
        return false;
    }

    const size_t num_landmarks = landmarks.size();
    const size_t num_cameras = poses.size();
    const size_t total_vertices = num_landmarks + num_cameras * vertices_per_camera;
    const size_t total_edges = num_cameras * edges_per_camera;

    // Write PLY header.
    char header[512];
    const int characters = std::snprintf(
        header,
        sizeof(header),
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created using ZeroSLAM by Geoffrey Daniels\n"
        "element vertex %zu\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "element edge %zu\n"
        "property int vertex1\n"
        "property int vertex2\n"
        "end_header\n",
        total_vertices,
        total_edges
    );
    if ((characters <= 0) || (static_cast<size_t>(characters) >= sizeof(header))) {
        return false;
    }
    gtl::file::size_type length = static_cast<gtl::file::size_type>(characters);
    if (!output.write(header, length)) {
        return false;
    }

    // Helper lambda to write vertex.
    constexpr static const auto write_vertex = [](const gtl::file& file, float x, float y, float z, unsigned char r, unsigned char g, unsigned char b) {
        gtl::file::size_type size = sizeof(float);
        file.write(reinterpret_cast<const char*>(&x), size);
        size = sizeof(float);
        file.write(reinterpret_cast<const char*>(&y), size);
        size = sizeof(float);
        file.write(reinterpret_cast<const char*>(&z), size);
        size = sizeof(unsigned char);
        file.write(reinterpret_cast<const char*>(&r), size);
        size = sizeof(unsigned char);
        file.write(reinterpret_cast<const char*>(&g), size);
        size = sizeof(unsigned char);
        file.write(reinterpret_cast<const char*>(&b), size);
    };

    // Helper lambda to write edge.
    constexpr static const auto write_edge = [](const gtl::file& file, int v1, int v2) {
        gtl::file::size_type size = sizeof(int);
        file.write(reinterpret_cast<const char*>(&v1), size);
        size = sizeof(int);
        file.write(reinterpret_cast<const char*>(&v2), size);
    };

    constexpr static const auto to_byte = [](const float channel) {
        const double scaled = static_cast<double>(channel) * 255.0;
        return static_cast<unsigned char>(scaled < 0.0 ? 0.0 : (scaled > 255.0 ? 255.0 : scaled));
    };

    for (const zeroslam_point_struct& landmark : landmarks) {
        write_vertex(output, landmark.x, landmark.y, landmark.z, to_byte(landmark.r), to_byte(landmark.g), to_byte(landmark.b));
    }

    const double image_corners[4][2] = {
        { 0, 0 },
        { static_cast<double>(camera.width), 0 },
        { static_cast<double>(camera.width), static_cast<double>(camera.height) },
        { 0, static_cast<double>(camera.height) }
    };
    for (size_t index = 0; index < num_cameras; ++index) {
        const camera_pose& pose = poses[index];
        const double ramp = static_cast<double>(index) / static_cast<double>(num_cameras);
        const unsigned char shade = static_cast<unsigned char>(cam_g * ramp);
        double camera_to_world[9];
        rotation::quaternion_to_matrix(&pose.quaternion_xyzw[0], &camera_to_world[0]);

        // Write camera centre.
        write_vertex(output, static_cast<float>(pose.centre[0]), static_cast<float>(pose.centre[1]), static_cast<float>(pose.centre[2]), cam_r, shade, cam_b);

        // Write frustum corners.
        for (int i = 0; i < 4; ++i) {
            const double corner[3] = {
                frustum_scale * (image_corners[i][0] - camera.centre_x) / camera.focal_x,
                frustum_scale * (image_corners[i][1] - camera.centre_y) / camera.focal_y,
                frustum_scale
            };
            double world_corner[3];
            for (int row = 0; row < 3; ++row) {
                world_corner[row] = pose.centre[row];
                for (int column = 0; column < 3; ++column) {
                    world_corner[row] += camera_to_world[3 * row + column] * corner[column];
                }
            }
            write_vertex(output, static_cast<float>(world_corner[0]), static_cast<float>(world_corner[1]), static_cast<float>(world_corner[2]), cam_r, shade, cam_b);
        }
    }

    // Write camera frustum edges.
    for (size_t i = 0; i < num_cameras; ++i) {
        const int camera_index = static_cast<int>(num_landmarks + i * vertices_per_camera);
        // Edges from centre to corners.
        for (int j = 0; j < 4; ++j) {
            write_edge(output, camera_index, camera_index + 1 + j);
        }
        // Rectangle edges connecting corners.
        for (int j = 0; j < 4; ++j) {
            write_edge(output, camera_index + 1 + j, camera_index + 1 + ((j + 1) % 4));
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    const char* scene_path = nullptr;
    size_t frame_limit = static_cast<size_t>(-1);
    size_t skip_count = 0;
    bool live = false;
    std::vector<const char*> settings;
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
        const auto take_count = [&](size_t& count, const unsigned long minimum, const char* const requirement) {
            const char* text = nullptr;
            if (!take_value(text)) {
                return false;
            }
            char* end_pointer = nullptr;
            const unsigned long value = std::strtoul(text, &end_pointer, 10);
            if ((end_pointer == text) || (*end_pointer != 0) || (value < minimum)) {
                std::fprintf(stderr, "Invalid value for %s: '%s' (%s).\n", argv[i - 1], text, requirement);
                return false;
            }
            count = static_cast<size_t>(value);
            return true;
        };
        if (matches("--help") || matches("-h")) {
            scene_path = nullptr;
            break;
        }
        else if (matches("--config")) {
            const char* setting = nullptr;
            if (!take_value(setting)) {
                return EXIT_FAILURE;
            }
            settings.push_back(setting);
        }
        else if (matches("--frames")) {
            if (!take_count(frame_limit, 2, "at least 2 frames are needed")) {
                return EXIT_FAILURE;
            }
        }
        else if (matches("--skip")) {
            if (!take_count(skip_count, 0, "expected a non-negative frame count")) {
                return EXIT_FAILURE;
            }
        }
        else if (matches("--truth")) {
            if (!take_value(truth_path)) {
                return EXIT_FAILURE;
            }
        }
        else if (matches("--live")) {
            live = true;
        }
        else if (matches("--verbose")) {
            size_t level = 0;
            if (!take_count(level, 0, "must be 0 to 5")) {
                return EXIT_FAILURE;
            }
            if (level > 5) {
                std::fprintf(stderr, "Invalid value for --verbose: '%s' (must be 0 to 5).\n", argv[i]);
                return EXIT_FAILURE;
            }
            verbosity = static_cast<int>(level);
        }
        else if (argv[i][0] == '-') {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
        else if (scene_path == nullptr) {
            scene_path = argv[i];
        }
        else {
            std::fprintf(stderr, "Unexpected argument: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }

    if (scene_path == nullptr) {
        std::printf("Usage %s [scene] [--frames count] [--skip count] [--verbose level] [--truth trajectory.txt] [--live] [--config key=value]...\n", argv[0]);
        std::printf("    scene     - Scene mcap file path.\n");
        std::printf("    --frames  - Optional limit, process only the first [count] frames.\n");
        std::printf("    --skip    - Optional offset, ignore the first [count] frames (camera info is still consumed).\n");
        std::printf("    --verbose - Optional log verbosity (0=silent, 1=errors, 2=warnings, 3=notes, 4=progress, 5=debug), default is 1.\n");
        std::printf("    --truth   - Optional ground truth trajectory (TUM format); the run ends with absolute, relative and scale drift figures against it.\n");
        std::printf("    --live    - Skip the final global adjustment: the trajectory is the one tracked live, as the viewer shows it.\n");
        std::printf("    --config  - Optional slam configuration setting (repeatable), see the library's configuration keys.\n");
        return EXIT_SUCCESS;
    }

    log(4, "Scene: %s", scene_path);
    if (frame_limit != static_cast<size_t>(-1)) {
        log(4, "Frame limit: %zu frames", frame_limit);
    }
    if (skip_count > 0) {
        log(4, "Skipping the first %zu frames", skip_count);
    }
    log(4, "Loading scene...");
    mcap reader;
    std::string error;
    if (!reader.open(scene_path, error)) {
        std::fprintf(stderr, "Failed to open the scene file '%s' as mcap: %s.\n", scene_path, error.c_str());
        return EXIT_FAILURE;
    }
    const mcap::channel_type* image_channel = nullptr;
    const mcap::channel_type* info_channel = nullptr;
    dataset::find_camera_channels(reader, image_channel, info_channel);
    if (image_channel == nullptr) {
        std::fprintf(stderr, "The scene has no raw image channel.\n");
        return EXIT_FAILURE;
    }
    if (info_channel == nullptr) {
        std::fprintf(stderr, "The scene has no camera info channel for the intrinsics.\n");
        return EXIT_FAILURE;
    }

    log(4, "Loading slam system...");
    std::signal(SIGINT, signal_handler);
    zeroslam::system slam;
    if (!slam.is_valid()) {
        std::fprintf(stderr, "Failed to create the slam system.\n");
        return EXIT_FAILURE;
    }
    char verbosity_setting[32];
    std::snprintf(verbosity_setting, sizeof(verbosity_setting), "verbosity=%d", verbosity);
    settings.insert(settings.begin(), verbosity_setting);
    for (const char* const setting : settings) {
        if (slam.set_configuration(setting, static_cast<int>(std::strlen(setting))) != zeroslam_return_success) {
            std::fprintf(stderr, "Invalid value for --config: '%s' (expected a known key=value setting).\n", setting);
            return EXIT_FAILURE;
        }
    }

    log(4, "Ready.");
    log(4, "Processing frames...");
    size_t frames = 0;
    std::vector<long long> timestamps;
    constexpr static const int sensor_id = 1;
    zeroslam_sensor_parameters_camera_struct camera{};
    bool rig_set = false;

    std::vector<size_t> image_messages;
    long long current_timestamp = -1;

    const std::vector<mcap::message_index_type>& message_index = reader.get_message_index();
    const size_t total_messages = message_index.size();
    size_t processed_messages = 0;

    auto process_buffered_images = [&]() {
        for (const size_t image_index : image_messages) {
            if (frames >= frame_limit) {
                break;
            }
            mcap::message_type buffered;
            if (!reader.read_message(image_index, buffered)) {
                std::fprintf(stderr, "Failed to read an image message: %s.\n", reader.get_read_error().c_str());
                return false;
            }
            const mcap::message_type* const img_msg = &buffered;
            if (skip_count > 0) {
                --skip_count;
                continue;
            }
            if ((camera.focal_x <= 0.0) || (camera.focal_y <= 0.0)) {
                std::fprintf(stderr, "Image message %zu arrived before valid camera intrinsics.\n", frames);
                return false;
            }
            cdr::image image;
            if (!cdr::read_image(img_msg->data, img_msg->length, image)) {
                std::fprintf(stderr, "Image message %zu does not decode.\n", frames);
                return false;
            }
            const bool dimensions_valid = (image.width > 0) && (image.height > 0) && (image.width <= 4096) && (image.height <= 4096);
            const bool mono = (image.encoding == "mono8") && (image.step == image.width) && (image.data.size() == static_cast<std::size_t>(image.width) * image.height);
            const bool colour = (image.encoding == "rgb8") && (image.step == 3u * image.width) && (image.data.size() == 3u * static_cast<std::size_t>(image.width) * image.height);
            const bool consistent = dimensions_valid && (mono || colour);
            const bool matching = (frames == 0) || ((static_cast<int>(image.width) == camera.width) && (static_cast<int>(image.height) == camera.height));
            if (!consistent || !matching) {
                std::fprintf(stderr, "Image message %zu is not a consistent mono8 or rgb8 image.\n", frames);
                return false;
            }
            if (!rig_set) {
                camera.width = static_cast<int>(image.width);
                camera.height = static_cast<int>(image.height);
                zeroslam_sensor_rig_struct rig{};
                rig.type = zeroslam_sensor_camera;
                rig.sensor_id = sensor_id;
                rig.parameters_length = static_cast<int>(sizeof(camera));
                rig.parameters_data = &camera;
                if (slam.set_sensor_rig(&rig, 1) != zeroslam_return_success) {
                    std::fprintf(stderr, "The library rejected the camera rig.\n");
                    return false;
                }
                rig_set = true;
            }
            std::vector<unsigned char> greyscale;
            if (colour) {
                const std::size_t pixels = static_cast<std::size_t>(image.width) * image.height;
                greyscale.resize(pixels);
                for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                    const unsigned int red = image.data[3 * pixel + 0];
                    const unsigned int green = image.data[3 * pixel + 1];
                    const unsigned int blue = image.data[3 * pixel + 2];
                    greyscale[pixel] = static_cast<unsigned char>(((77 * red) + (150 * green) + (29 * blue)) >> 8);
                }
            }
            timestamps.push_back(static_cast<long long>(img_msg->log_time));
            log(4, "Starting frame %zu", frames + 1);
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            zeroslam_sensor_data_struct measurement{};
            measurement.timestamp = timestamps.back();
            measurement.sensor_id = sensor_id;
            measurement.measurement_length = static_cast<int>(image.width * image.height);
            measurement.measurement_data = colour ? greyscale.data() : image.data.data();
            const zeroslam_return_enum result = slam.set_sensor_data(&measurement, 1);
            if (result != zeroslam_return_success) {
                std::fprintf(stderr, "The library rejected frame %zu: %s.\n", frames + 1, zeroslam_return_enum_to_string(result));
                return false;
            }
            std::chrono::duration<double> frame_duration = std::chrono::steady_clock::now() - start;
            log(4, "Finished frame %zu, took: %f seconds", frames + 1, frame_duration.count());
            ++frames;
        }
        image_messages.clear();
        return true;
    };

    size_t progress_percent = static_cast<size_t>(-1);
    const auto log_progress = [&]() {
        if (frame_limit != static_cast<size_t>(-1)) {
            const size_t percent = (frames * 100) / frame_limit;
            if (percent != progress_percent) {
                progress_percent = percent;
                log(4, "Processing: %3zu%% (%zu / %zu frames)", percent, frames, frame_limit);
            }
        }
        else {
            const size_t percent = (processed_messages * 100) / (total_messages == 0 ? 1 : total_messages);
            if (percent != progress_percent) {
                progress_percent = percent;
                log(4, "Processing: %3zu%% (%zu / %zu messages)", percent, processed_messages, total_messages);
            }
        }
    };
    for (size_t message_number = 0; message_number < total_messages; ++message_number) {
        const mcap::message_index_type& entry = message_index[message_number];
        ++processed_messages;
        log_progress();

        if (shutdown_requested) {
            log(3, "Interrupt received, stopping...");
            break;
        }

        const long long message_time = static_cast<long long>(entry.log_time);
        if ((current_timestamp != -1) && (message_time != current_timestamp)) {
            if (!process_buffered_images()) {
                return EXIT_FAILURE;
            }
            if (frames >= frame_limit) {
                break;
            }
        }
        current_timestamp = message_time;

        if (entry.channel_id == info_channel->id) {
            mcap::message_type message;
            cdr::camera_info information;
            if (reader.read_message(message_number, message) && cdr::read_camera_info(message.data, message.length, information)) {
                double parameters[12] = {};
                bool distortion_model_recognised = true;
                dataset::camera_info_to_parameters(information, &parameters[0], 12, distortion_model_recognised);
                if (!distortion_model_recognised) {
                    std::fprintf(stderr, "Unsupported distortion model '%s': proceeding as an undistorted pinhole.\n", information.distortion_model.c_str());
                }
                if (rig_set) {
                    bool changed = false;
                    changed |= (parameters[0] != camera.focal_x) || (parameters[1] != camera.focal_y);
                    changed |= (parameters[2] != camera.centre_x) || (parameters[3] != camera.centre_y);
                    for (size_t index = 0; index < 8; ++index) {
                        changed |= (parameters[4 + index] != camera.distortion[index]);
                    }
                    if (changed) {
                        log(2, "Camera intrinsics changed after frame %zu, the first intrinsics are kept.", frames);
                    }
                }
                else {
                    camera.focal_x = parameters[0];
                    camera.focal_y = parameters[1];
                    camera.centre_x = parameters[2];
                    camera.centre_y = parameters[3];
                    for (size_t index = 0; index < 8; ++index) {
                        camera.distortion[index] = parameters[4 + index];
                    }
                }
            }
        }
        else if (entry.channel_id == image_channel->id) {
            image_messages.push_back(message_number);
        }
    }

    if (!process_buffered_images()) {
        return EXIT_FAILURE;
    }

    log_progress();

    if (frames < 2) {
        std::fprintf(stderr, "At least two frames must be provided to create a map.\n");
        return EXIT_FAILURE;
    }

    if (!live) {
        log(4, "Finalising the map...");
        if (slam.finalise() != zeroslam_return_success) {
            std::fprintf(stderr, "Failed to finalise the map.\n");
            return EXIT_FAILURE;
        }
    }
    log(4, "Saving map and camera trajectory...");

    std::vector<camera_pose> poses;
    for (const long long timestamp : timestamps) {
        zeroslam_pose_struct reported{};
        if (slam.get_pose_at_timestamp(&reported, timestamp) != zeroslam_return_success) {
            continue;
        }
        camera_pose pose;
        pose.timestamp = reported.timestamp;
        for (int axis = 0; axis < 3; ++axis) {
            pose.centre[axis] = reported.pose[axis];
        }
        for (int component = 0; component < 4; ++component) {
            pose.quaternion_xyzw[component] = reported.pose[3 + component];
        }
        poses.push_back(pose);
    }
    std::vector<zeroslam_point_struct> landmarks;
    zeroslam_map_chunk_struct chunk{};
    zeroslam_return_enum chunk_result = slam.get_map_chunk(0.0f, 0.0f, 0.0f, &chunk);
    if (chunk_result == zeroslam_return_failure_insufficient_data_length) {
        landmarks.resize(static_cast<size_t>(chunk.points_length));
        chunk.points = landmarks.data();
        chunk_result = slam.get_map_chunk(0.0f, 0.0f, 0.0f, &chunk);
    }
    if (chunk_result != zeroslam_return_success) {
        std::fprintf(stderr, "Failed to read the map: %s.\n", zeroslam_return_enum_to_string(chunk_result));
        return EXIT_FAILURE;
    }
    landmarks.resize(static_cast<size_t>(chunk.points_length));

    // Note: This can be easily plotted using evo: `evo_traj tum trajectory.txt -p`.
    if (!save_trajectory_as_txt("trajectory.txt", poses)) {
        std::fprintf(stderr, "Failed to save camera trajectory to txt file.\n");
    }
    // Note: This can be easily visualised using meshlab.
    if (!save_trajectory_and_map_as_ply("map.ply", camera, landmarks, poses)) {
        std::fprintf(stderr, "Failed to save map and camera trajectory to ply file.\n");
    }

    if (truth_path != nullptr) {
        if (!print_evaluation(truth_path, poses, timestamps.size())) {
            return EXIT_FAILURE;
        }
    }

    log(4, "Done.");
}
