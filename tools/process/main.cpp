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
#include "file.hpp"
#include "mcap.hpp"
#include "slam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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
}

inline unsigned char* file_load(const char* path, std::size_t& length) {
    const gtl::file input(path);
    if (!input.is_open()) {
        return nullptr;
    }
    gtl::file::size_type size = 0;
    if ((!input.get_size(size)) || (size <= 0)) {
        return nullptr;
    }
    unsigned char* data = new unsigned char[size];
    if (!data) {
        return nullptr;
    }
    gtl::file::size_type index = 0;
    while (index < size) {
        gtl::file::size_type delta = size - index;
        if ((!input.read(reinterpret_cast<char*>(&data[index]), delta)) || (delta == 0)) {
            delete[] data;
            return nullptr;
        }
        index += delta;
    }
    length = static_cast<std::size_t>(size);
    return data;
}

inline bool save_trajectory_as_txt(const char* path, const map::map& reconstruction, const std::vector<long long>& timestamps) {
    gtl::file output(path, gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
    if (!output.is_open()) {
        return false;
    }

    const std::map<int, frame::frame> ordered_frames(reconstruction.frames.begin(), reconstruction.frames.end());

    // Write camera trajectory in TUM format.
    for (const auto& [id, frame] : ordered_frames) {
        const auto& R = frame.rotation;
        const auto& t = frame.translation;

        // Camera centre in world coordinates.
        matrix::matrix<double, 3, 1> centre = -matrix::transpose(R) * t;
        matrix::matrix<double, 4, 1> rotation = lie::so3<double>(R).get_quaternion();

        // Write camera pose.
        if ((id < 0) || (static_cast<size_t>(id) >= timestamps.size())) {
            continue;
        }
        const long long timestamp = timestamps[static_cast<std::size_t>(id)];
        const cdr::time stamp = cdr::time::from_nanoseconds(timestamp);
        char line[512];
        // Note: Format is one pose per line as "timestamp x y z q_x q_y q_z q_w\n"
        const int characters = std::snprintf(line, sizeof(line), "%d.%09u %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", stamp.sec, stamp.nanosec, centre[0], centre[1], centre[2], rotation[1], rotation[2], rotation[3], rotation[0]);
        if ((characters <= 0) || (static_cast<size_t>(characters) >= sizeof(line))) {
            return false;
        }
        gtl::file::size_type length = static_cast<gtl::file::size_type>(characters);
        if (!output.write(line, length)) {
            return false;
        }
    }

    return true;
}

inline bool save_trajectory_and_map_as_ply(const char* path, int image_width, int image_height, const map::map& reconstruction) {
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

    const size_t num_landmarks = reconstruction.landmarks.size();
    const size_t num_cameras = reconstruction.frames.size();
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
    constexpr static const auto write_vertex = [](const gtl::file& output, float x, float y, float z, unsigned char r, unsigned char g, unsigned char b) {
        gtl::file::size_type length = sizeof(float);
        output.write(reinterpret_cast<const char*>(&x), length);
        length = sizeof(float);
        output.write(reinterpret_cast<const char*>(&y), length);
        length = sizeof(float);
        output.write(reinterpret_cast<const char*>(&z), length);
        length = sizeof(unsigned char);
        output.write(reinterpret_cast<const char*>(&r), length);
        length = sizeof(unsigned char);
        output.write(reinterpret_cast<const char*>(&g), length);
        length = sizeof(unsigned char);
        output.write(reinterpret_cast<const char*>(&b), length);
    };

    // Helper lambda to write edge.
    constexpr static const auto write_edge = [](const gtl::file& output, int v1, int v2) {
        gtl::file::size_type length = sizeof(int);
        output.write(reinterpret_cast<const char*>(&v1), length);
        length = sizeof(int);
        output.write(reinterpret_cast<const char*>(&v2), length);
    };

    // Write landmark vertices.
    for (const auto& [id, landmark] : reconstruction.landmarks) {
        const auto& pos = landmark.location;
        const auto& colour = landmark.colour;
        write_vertex(
            output,
            static_cast<float>(pos[0]),
            static_cast<float>(pos[1]),
            static_cast<float>(pos[2]),
            static_cast<unsigned char>(math::max(0.0, math::min(colour[0] * 255.0, 255.0))),
            static_cast<unsigned char>(math::max(0.0, math::min(colour[1] * 255.0, 255.0))),
            static_cast<unsigned char>(math::max(0.0, math::min(colour[2] * 255.0, 255.0)))
        );
    }

    // Write camera frustum vertices
    for (const auto& [id, frame] : reconstruction.frames) {
        const auto& R = frame.rotation;
        const auto& t = frame.translation;
        const double image_corners[4][2] = {
            { 0, 0 },
            { static_cast<double>(image_width), 0 },
            { static_cast<double>(image_width), static_cast<double>(image_height) },
            { 0, static_cast<double>(image_height) }
        };
        matrix::matrix<double, 3, 1> corners[4];
        for (int i = 0; i < 4; ++i) {
            frame.camera.unproject(&image_corners[i][0], corners[i].data());
            corners[i] = corners[i] * frustum_scale;
        }

        // Camera centre in world coordinates.
        matrix::matrix<double, 3, 1> centre = -matrix::transpose(R) * t;
        matrix::matrix<double, 3, 1> world_corners[4];
        for (int i = 0; i < 4; ++i) {
            world_corners[i] = matrix::transpose(R) * corners[i] + centre;
        }

        // Write camera centre.
        write_vertex(
            output,
            static_cast<float>(centre[0]),
            static_cast<float>(centre[1]),
            static_cast<float>(centre[2]),
            cam_r,
            static_cast<unsigned char>(cam_g * (static_cast<float>(id) / static_cast<float>(reconstruction.frames.size()))),
            cam_b
        );

        // Write frustum corners.
        for (int i = 0; i < 4; ++i) {
            write_vertex(
                output,
                static_cast<float>(world_corners[i][0]),
                static_cast<float>(world_corners[i][1]),
                static_cast<float>(world_corners[i][2]),
                cam_r,
                static_cast<unsigned char>(cam_g * (static_cast<float>(id) / static_cast<float>(reconstruction.frames.size()))),
                cam_b
            );
        }
    }

    // Write camera frustum edges.
    for (size_t i = 0; i < reconstruction.frames.size(); ++i) {
        const int camera_index = num_landmarks + i * vertices_per_camera;
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
    // Extract the optional --frames and --verbose limits, leaving the positional arguments in place.
    size_t frame_limit = static_cast<size_t>(-1);
    int verbose = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--frames") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for option: --frames\n");
                return EXIT_FAILURE;
            }
            char* end_pointer = nullptr;
            const unsigned long value = std::strtoul(argv[i + 1], &end_pointer, 10);
            if ((end_pointer == argv[i + 1]) || (*end_pointer != 0) || (value < 2)) {
                std::fprintf(stderr, "Invalid value for --frames: '%s' (at least 2 frames are needed).\n", argv[i + 1]);
                return EXIT_FAILURE;
            }
            frame_limit = static_cast<size_t>(value);
            for (int j = i; j + 2 < argc; ++j) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            --i;
        }
        else if (std::strcmp(argv[i], "--verbose") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for option: --verbose\n");
                return EXIT_FAILURE;
            }
            char* end_pointer = nullptr;
            const long value = std::strtol(argv[i + 1], &end_pointer, 10);
            if ((end_pointer == argv[i + 1]) || (*end_pointer != 0) || (value < 0) || (value > 2)) {
                std::fprintf(stderr, "Invalid value for --verbose: '%s' (must be 0, 1, or 2).\n", argv[i + 1]);
                return EXIT_FAILURE;
            }
            verbose = static_cast<int>(value);
            for (int j = i; j + 2 < argc; ++j) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            --i;
        }
    }

    if (argc < 2) {
        std::printf("Usage %s [scene] [--frames count] [--verbose level]\n", argv[0]);
        std::printf("    scene     - Scene mcap file path.\n");
        std::printf("    --frames  - Optional limit, process only the first [count] frames.\n");
        std::printf("    --verbose - Optional verbosity level (0=quiet, 1=progress, 2=detailed), default is 0.\n");
        return EXIT_SUCCESS;
    }

    if (verbose >= 2) {
        std::printf("Provided arguments...\n");
        std::printf("    scene:    %s\n", argv[1]);
        if (frame_limit != static_cast<size_t>(-1)) {
            std::printf("    limit:    %zu frames\n", frame_limit);
        }
        std::printf("Loading scene...\n");
    }
    std::size_t file_length = 0;
    unsigned char* file_data = file_load(argv[1], file_length);
    if (file_data == nullptr) {
        std::fprintf(stderr, "Failed to load the scene file '%s', the input must be an mcap file.\n", argv[1]);
        return EXIT_FAILURE;
    }
    mcap reader;
    std::string error;
    if (!reader.parse(file_data, file_length, error)) {
        std::fprintf(stderr, "Failed to parse the scene file '%s' as mcap: %s.\n", argv[1], error.c_str());
        delete[] file_data;
        return EXIT_FAILURE;
    }
    const mcap::channel_type* image_channel = nullptr;
    const mcap::channel_type* info_channel = nullptr;
    for (const mcap::channel_type& channel : reader.get_channels()) {
        const mcap::schema_type* schema = reader.find_schema(channel.schema_id);
        if (schema == nullptr) {
            continue;
        }
        if ((schema->name == "sensor_msgs/msg/Image") && (image_channel == nullptr)) {
            image_channel = &channel;
        }
        else if ((schema->name == "sensor_msgs/msg/CameraInfo") && (info_channel == nullptr)) {
            info_channel = &channel;
        }
    }
    if (image_channel == nullptr) {
        std::fprintf(stderr, "The scene has no raw image channel.\n");
        delete[] file_data;
        return EXIT_FAILURE;
    }
    if (info_channel == nullptr) {
        std::fprintf(stderr, "The scene has no camera info channel for the intrinsics.\n");
        delete[] file_data;
        return EXIT_FAILURE;
    }

    if (verbose >= 2) {
        std::printf("Loading slam system...\n");
    }
    std::signal(SIGINT, signal_handler);
    slam slam;

    if (verbose >= 2) {
        std::printf("Ready.\n");
        std::printf("\n");
        std::printf("Processing frames...\n");
    }
    size_t frames = 0;
    size_t rows = 0;
    size_t cols = 0;
    std::vector<long long> timestamps;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;

    std::vector<const mcap::message_type*> image_messages;
    long long current_timestamp = -1;

    const size_t total_messages = reader.get_messages().size();
    size_t processed_messages = 0;

    auto process_buffered_images = [&]() {
        for (const mcap::message_type* img_msg : image_messages) {
            if (frames >= frame_limit) {
                break;
            }
            if ((fx <= 0.0) || (fy <= 0.0)) {
                std::fprintf(stderr, "Image message %zu arrived before valid camera intrinsics.\n", frames);
                return false;
            }
            cdr::image image;
            if (!cdr::read_image(img_msg->data, img_msg->length, image)) {
                std::fprintf(stderr, "Image message %zu does not decode.\n", frames);
                return false;
            }
            const bool consistent = (image.encoding == "mono8") && (image.width > 0) && (image.height > 0) && (image.width <= 4096) && (image.height <= 4096) && (image.step == image.width) && (image.data.size() == static_cast<std::size_t>(image.width) * image.height);
            const bool matching = (frames == 0) || ((image.width == cols) && (image.height == rows));
            if (!consistent || !matching) {
                std::fprintf(stderr, "Image message %zu is not a consistent mono8 image.\n", frames);
                return false;
            }
            cols = image.width;
            rows = image.height;
            timestamps.push_back(static_cast<long long>(img_msg->log_time));
            if (verbose >= 2) {
                std::printf("\n");
                std::printf("Starting frame %zu\n", frames + 1);
            }
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            matrix::matrix<double, 3, 3> intrinsic = { { { fx, 0.0, cx },
                                                         { 0.0, fy, cy },
                                                         { 0.0, 0.0, 1.0 } } };
            image::image frame(rows, cols, image.data.data());
            slam.process_frame(intrinsic, frame);
            std::chrono::duration<double> frame_duration = std::chrono::steady_clock::now() - start;
            if (verbose >= 2) {
                std::printf("Finished frame %zu, took: %f seconds\n", frames + 1, frame_duration.count());
                std::fflush(stdout);
            }
            ++frames;
        }
        image_messages.clear();
        return true;
    };

    for (const mcap::message_type& message : reader.get_messages()) {
        ++processed_messages;
        if (verbose == 1) {
            if (frame_limit != static_cast<size_t>(-1)) {
                std::printf("\rProcessing: %3zu%% (%zu / %zu frames)", (frames * 100) / frame_limit, frames, frame_limit);
            }
            else {
                std::printf("\rProcessing: %3zu%% (%zu / %zu messages)", (processed_messages * 100) / (total_messages == 0 ? 1 : total_messages), processed_messages, total_messages);
            }
            std::fflush(stdout);
        }

        if (shutdown_requested) {
            if (verbose >= 1)
                std::printf("\nInterrupt received, stopping...\n");
            break;
        }

        const long long message_time = static_cast<long long>(message.log_time);
        if ((current_timestamp != -1) && (message_time != current_timestamp)) {
            if (!process_buffered_images()) {
                delete[] file_data;
                return EXIT_FAILURE;
            }
            if (frames >= frame_limit) {
                break;
            }
        }
        current_timestamp = message_time;

        if (message.channel_id == info_channel->id) {
            cdr::camera_info information;
            if (cdr::read_camera_info(message.data, message.length, information)) {
                fx = information.k[0];
                fy = information.k[4];
                cx = information.k[2];
                cy = information.k[5];
            }
        }
        else if (message.channel_id == image_channel->id) {
            image_messages.push_back(&message);
        }
    }

    if (!process_buffered_images()) {
        delete[] file_data;
        return EXIT_FAILURE;
    }
    delete[] file_data;

    if (verbose == 1) {
        if (frame_limit != static_cast<size_t>(-1)) {
            std::printf("\rProcessing: %3zu%% (%zu / %zu frames)", (frames * 100) / frame_limit, frames, frame_limit);
        }
        std::printf("\n");
    }

    if (frames < 2) {
        std::fprintf(stderr, "At least two frames must be provided to create a map.\n");
        return EXIT_FAILURE;
    }

    if (verbose >= 2) {
        std::printf("\n");
        std::printf("Saving map and camera trajectory...\n");
    }

    // Note: This can be easily plotted using evo: `evo_traj tum trajectory.txt -p`.
    if (!save_trajectory_as_txt("trajectory.txt", slam.reconstruction, timestamps)) {
        std::fprintf(stderr, "Failed to save camera trajectory to txt file.\n");
    }
    // Note: This can be easily visualised using meshlab.
    if (!save_trajectory_and_map_as_ply("map.ply", cols, rows, slam.reconstruction)) {
        std::fprintf(stderr, "Failed to save map and camera trajectory to ply file.\n");
    }

    if (verbose >= 2) {
        std::printf("Done.\n");
    }
}
