/*
Copyright (C) 2025 Geoffrey Daniels. https://gpdaniels.com/

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

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace {
    std::atomic<bool> shutdown_requested{false};
    void signal_handler(int) {
        std::signal(SIGINT, SIG_DFL);
        shutdown_requested = true;
    }
}

inline unsigned char* pgm_load(const char* path, std::size_t& rows, std::size_t& cols) {
    std::FILE* handle = std::fopen(path, "rb");
    if (!handle) {
        return nullptr;
    }
    char magic[2]{};
    if (std::fread(&magic[0], sizeof(char), 2, handle) != 2) {
        std::fclose(handle);
        return nullptr;
    }
    if ((magic[0] != 'P') || (magic[1] != '5')) {
        std::fclose(handle);
        return nullptr;
    }
    for (int c = std::fgetc(handle); true; c = std::fgetc(handle)) {
        if ((c == EOF) || std::feof(handle) || std::ferror(handle)) {
            std::fclose(handle);
            return nullptr;
        }
        if ((c != ' ') && (c != '\t') && (c != '\n') && (c != '\r')) {
            std::ungetc(c, handle);
            break;
        }
    }
    if (std::fscanf(handle, "%zu %zu", &cols, &rows) != 2) {
        std::fclose(handle);
        return nullptr;
    }
    if ((cols > 4096) || (rows > 4096)) {
        std::fclose(handle);
        return nullptr;
    }
    for (int c = std::fgetc(handle); true; c = std::fgetc(handle)) {
        if ((c == EOF) || std::feof(handle) || std::ferror(handle)) {
            std::fclose(handle);
            return nullptr;
        }
        if ((c != ' ') && (c != '\t') && (c != '\n') && (c != '\r')) {
            std::ungetc(c, handle);
            break;
        }
    }
    int maximum;
    if (std::fscanf(handle, "%d", &maximum) != 1) {
        std::fclose(handle);
        return nullptr;
    }
    const std::size_t size = cols * rows;
    if (size > 4096 * 4096) {
        std::fclose(handle);
        return nullptr;
    }
    int c = std::fgetc(handle);
    if ((c == EOF) || std::feof(handle) || std::ferror(handle) || (c != ' ' && c != '\t' && c != '\n' && c != '\r')) {
        std::fclose(handle);
        return nullptr;
    }
    unsigned char* pixels = new unsigned char[size];
    std::size_t index = 0;
    std::size_t attempts = 10;
    while (index < size) {
        const std::size_t delta = std::fread(&pixels[index], sizeof(char), size - index, handle);
        if ((delta == 0) || std::feof(handle) || std::ferror(handle)) {
            if (std::feof(handle) || std::ferror(handle) || (attempts-- == 0)) {
                std::fclose(handle);
                delete[] pixels;
                return nullptr;
            }
        }
        index += delta;
    }
    if (index != size) {
        std::fclose(handle);
        delete[] pixels;
        return nullptr;
    }
    std::fclose(handle);
    return pixels;
}

inline bool save_trajectory_as_txt(const char* path, const map::map& reconstruction) {
    std::FILE* handle = std::fopen(path, "wb");
    if (!handle) {
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
        // Note: Format is one pose per line as "timestamp x y z q_x q_y q_z q_w\n"
        std::fprintf(handle, "%d %f %f %f %f %f %f %f\n", id, centre[0], centre[1], centre[2], rotation[1], rotation[2], rotation[3], rotation[0]);
    }

    std::fclose(handle);
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

    std::FILE* handle = std::fopen(path, "wb");
    if (!handle) {
        return false;
    }

    const size_t num_landmarks = reconstruction.landmarks.size();
    const size_t num_cameras = reconstruction.frames.size();
    const size_t total_vertices = num_landmarks + num_cameras * vertices_per_camera;
    const size_t total_edges = num_cameras * edges_per_camera;

    // Write PLY header.
    std::fprintf(handle, "ply\n");
    std::fprintf(handle, "format binary_little_endian 1.0\n");
    std::fprintf(handle, "comment Created using ZeroSLAM by Geoffrey Daniels\n");
    std::fprintf(handle, "element vertex %zu\n", total_vertices);
    std::fprintf(handle, "property float x\n");
    std::fprintf(handle, "property float y\n");
    std::fprintf(handle, "property float z\n");
    std::fprintf(handle, "property uchar red\n");
    std::fprintf(handle, "property uchar green\n");
    std::fprintf(handle, "property uchar blue\n");
    std::fprintf(handle, "element edge %zu\n", total_edges);
    std::fprintf(handle, "property int vertex1\n");
    std::fprintf(handle, "property int vertex2\n");
    std::fprintf(handle, "end_header\n");

    // Helper lambda to write vertex.
    constexpr static const auto write_vertex = [](std::FILE* handle, float x, float y, float z, unsigned char r, unsigned char g, unsigned char b) {
        std::fwrite(&x, sizeof(float), 1, handle);
        std::fwrite(&y, sizeof(float), 1, handle);
        std::fwrite(&z, sizeof(float), 1, handle);
        std::fwrite(&r, sizeof(unsigned char), 1, handle);
        std::fwrite(&g, sizeof(unsigned char), 1, handle);
        std::fwrite(&b, sizeof(unsigned char), 1, handle);
    };

    // Helper lambda to write edge.
    constexpr static const auto write_edge = [](std::FILE* handle, int v1, int v2) {
        std::fwrite(&v1, sizeof(int), 1, handle);
        std::fwrite(&v2, sizeof(int), 1, handle);
    };

    // Write landmark vertices.
    for (const auto& [id, landmark] : reconstruction.landmarks) {
        const auto& pos = landmark.location;
        const auto& colour = landmark.colour;
        write_vertex(
            handle,
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
            handle,
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
                handle,
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
            write_edge(handle, camera_index, camera_index + 1 + j);
        }
        // Rectangle edges connecting corners.
        for (int j = 0; j < 4; ++j) {
            write_edge(handle, camera_index + 1 + j, camera_index + 1 + ((j + 1) % 4));
        }
    }

    std::fclose(handle);
    return true;
}

int main(int argc, char* argv[]) {
    std::printf(".-----------------------------------------------.\n");
    std::printf("|   _____             _____ __    _____ _____   |\n");
    std::printf("|  |__   |___ ___ ___|   __|  |  |  _  |     |  |\n");
    std::printf("|  |   __| -_|  _| . |__   |  |__|     | | | |  |\n");
    std::printf("|  |_____|___|_| |___|_____|_____|__|__|_|_|_|  |\n");
    std::printf("|                                               |\n");
    std::printf("| This software is a:                           |\n");
    std::printf("|  |- simple                                    |\n");
    std::printf("|  |- minimal                                   |\n");
    std::printf("|  |- indirect                                  |\n");
    std::printf("|  |- monocular                                 |\n");
    std::printf("|  |- factor-graph                              |\n");
    std::printf("|  |- deterministic                             |\n");
    std::printf("|  |- dependency-free                           |\n");
    std::printf("|  '- visual SLAM system written in pure C++.   |\n");
    std::printf("|                                               |\n");
    std::printf("| No external libraries. No frills. Just SLAM.  |\n");
    std::printf("|                                               |\n");
    std::printf("| >   https://github.com/gpdaniels/zeroslam   < |\n");
    std::printf("|                                               |\n");
    std::printf("| Licensed under GPLv3                          |\n");
    std::printf("| Get in touch for commercial licensing.        |\n");
    std::printf("'-----------------------------------------------'\n");
    std::printf("\n");
    std::fflush(stdout);

    if (argc < 6) {
        std::printf("Usage %s [video] [fx] [fy] [cx] [cy]\n", argv[0]);
        std::printf("    video - A directory path containing same-sized undistorted greyscale frames in pgm format named: 000.pgm, 001.pgm, ...\n");
        std::printf("    fx    - The horizontal focal length of the camera in pixels.\n");
        std::printf("    fy    - The vertical focal length of the camera in pixels.\n");
        std::printf("    cx    - The horizontal centre pixel location in pixels.\n");
        std::printf("    cy    - The vertical centre pixel location in pixels.\n");
        return EXIT_SUCCESS;
    }

    std::printf("Provided arguments...\n");
    std::printf("    video:    %s\n", argv[1]);
    std::printf("    fx:       %s\n", argv[2]);
    std::printf("    fy:       %s\n", argv[3]);
    std::printf("    cx:       %s\n", argv[4]);
    std::printf("    cy:       %s\n", argv[5]);

    std::printf("Loading video...\n");
    char path[2048];
    const size_t start_filename = std::strlen(argv[1]);
    if (start_filename >= 2000) {
        std::fprintf(stderr, "Invalid video directory path, too long.\n");
        return EXIT_FAILURE;
    }
    std::memcpy(&path[0], argv[1], start_filename + 1);

    size_t frames = 0;
    size_t rows = 0;
    size_t cols = 0;
    unsigned char** data = new unsigned char*[1000];
    for (size_t i = 0; i < 1000; ++i) {
        path[start_filename + 0] = '/';
        path[start_filename + 1] = '0' + ((i / 100) % 10);
        path[start_filename + 2] = '0' + ((i / 10) % 10);
        path[start_filename + 3] = '0' + ((i / 1) % 10);
        path[start_filename + 4] = '.';
        path[start_filename + 5] = 'p';
        path[start_filename + 6] = 'g';
        path[start_filename + 7] = 'm';
        path[start_filename + 8] = 0;
        size_t temp_rows = 0;
        size_t temp_cols = 0;
        data[i] = pgm_load(path, temp_rows, temp_cols);
        if (data[i] == nullptr) {
            frames = i;
            break;
        }
        if (i == 0) {
            rows = temp_rows;
            cols = temp_cols;
        }
        else {
            if ((rows != temp_rows) || (cols != temp_cols)) {
                std::fprintf(stderr, "Mismatching image sizes, image %zu is different from ones before it.\n", i);
                delete[] data;
                return EXIT_FAILURE;
            }
        }
    }
    if (frames == 0) {
        std::fprintf(stderr, "Failed to load any frames.\n");
        delete[] data;
        return EXIT_FAILURE;
    }
    if (frames < 2) {
        std::fprintf(stderr, "At least two frames must be provided to create a map.\n");
        delete[] data;
        return EXIT_FAILURE;
    }
    std::printf("    frames:   %zu (000.pgm -> %03zu.pgm)\n", frames, frames - 1);
    std::printf("    width:    %zu\n", cols);
    std::printf("    height:   %zu\n", rows);

    const double fx = std::strtod(argv[2], nullptr);
    const double fy = std::strtod(argv[3], nullptr);
    const double cx = std::strtod(argv[4], nullptr);
    const double cy = std::strtod(argv[5], nullptr);
    matrix::matrix<double, 3, 3> intrinsic = { { { fx, 0.0, cx },
                                                 { 0.0, fy, cy },
                                                 { 0.0, 0.0, 1.0 } } };

    std::printf("Loading slam system...\n");
    std::signal(SIGINT, signal_handler);
    slam slam;

    std::printf("Converting images...\n");
    // Convert raw data into image type.
    image::image images[1000];
    for (size_t i = 0; i < frames; ++i) {
        images[i] = image::image(rows, cols, data[i]);
        delete[] data[i];
        data[i] = nullptr;
    }
    delete[] data;

    std::printf("Ready.\n");
    std::printf("\n");

    std::printf("Processing frames...\n");
    for (size_t i = 0; i < frames; ++i) {
        if (shutdown_requested) {
            std::printf("Interrupt received, stopping...\n");
            break;
        }
        std::printf("\n");
        std::printf("Starting frame %zu/%zu\n", i + 1, frames);
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        slam.process_frame(intrinsic, images[i]);
        std::chrono::duration<double> frame_duration = std::chrono::steady_clock::now() - start;
        std::printf("Finished frame %zu/%zu, took: %f seconds\n", i + 1, frames, frame_duration.count());
        std::fflush(stdout);
    }

    std::printf("Saving map and camera trajectory...\n");
    // Note: This can be easily plotted using evo: `evo_traj tum trajectory.txt -p`.
    if (!save_trajectory_as_txt("trajectory.txt", slam.reconstruction)) {
        std::fprintf(stderr, "Failed to save camera trajectory to txt file.\n");
    }
    // Note: This can be easily visualised using meshlab.
    if (!save_trajectory_and_map_as_ply("map.ply", cols, rows, slam.reconstruction)) {
        std::fprintf(stderr, "Failed to save map and camera trajectory to ply file.\n");
    }

    std::printf("Done.\n");
}
