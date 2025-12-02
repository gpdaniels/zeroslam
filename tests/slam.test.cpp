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

#include <cmath>
#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

class random_pcg final {
private:
    unsigned long long int state;
    unsigned long long int increment;

public:
    random_pcg()
        : state(0x853C49E6748FEA9Bull)
        , increment(0xDA3E39CB94B95BDBull) {
    }

    unsigned int get_random_raw() {
        unsigned long long int state_previous = this->state;
        this->state = state_previous * 0x5851F42D4C957F2Dull + this->increment;
        unsigned int state_shift_xor_shift = static_cast<unsigned int>(((state_previous >> 18u) ^ state_previous) >> 27u);
        int rotation = state_previous >> 59u;
        return (state_shift_xor_shift >> rotation) | (state_shift_xor_shift << ((-rotation) & 31));
    }
};

class world {
private:
    struct triangle {
        matrix::matrix<double, 3, 1> v0, v1, v2;
        unsigned char intensity;
    };

private:
    std::vector<triangle> scene;
    int image_width;
    int image_height;

public:
    world(int width, int height)
        : image_width(width)
        , image_height(height) {
        random_pcg rng;
        add_cube(-4.0, -2.0, 10.0, 1.5, static_cast<unsigned char>(rng.get_random_raw() % 256));
        add_cube(4.0, -2.0, 10.0, 1.5, static_cast<unsigned char>(rng.get_random_raw() % 256));
        add_cube(0.0, 3.0, 12.0, 2.5, static_cast<unsigned char>(rng.get_random_raw() % 256));
        add_cube(-3.0, 0.0, 6.0, 1.0, static_cast<unsigned char>(rng.get_random_raw() % 256));
        add_cube(3.0, 0.0, 6.0, 1.0, static_cast<unsigned char>(rng.get_random_raw() % 256));
        double floor_y = 3.0;
        for (int i = -3; i <= 3; i++) {
            for (int j = 0; j <= 6; j++) {
                const double x = i * 2.0;
                const double z = j * 2.0 + 4.0;
                scene.push_back({ { { x, floor_y, z } }, { { x + 2.0, floor_y, z } }, { { x + 2.0, floor_y, z + 2.0 } }, static_cast<unsigned char>(rng.get_random_raw() % 256) });
                scene.push_back({ { { x, floor_y, z } }, { { x + 2.0, floor_y, z + 2.0 } }, { { x, floor_y, z + 2.0 } }, static_cast<unsigned char>(rng.get_random_raw() % 256) });
            }
        }
        double wall_z = 18.0;
        for (int i = -3; i <= 3; i++) {
            for (int j = 0; j <= 6; j++) {
                const double x = i * 2.0;
                const double y = 1.0 - j * 2.0;
                scene.push_back({ { { x, y, wall_z } }, { { x + 2.0, y, wall_z } }, { { x + 2.0, y + 2.0, wall_z } }, static_cast<unsigned char>(rng.get_random_raw() % 256) });
                scene.push_back({ { { x, y, wall_z } }, { { x + 2.0, y + 2.0, wall_z } }, { { x, y + 2.0, wall_z } }, static_cast<unsigned char>(rng.get_random_raw() % 256) });
            }
        }
    }

private:
    static bool project_point(const matrix::matrix<double, 3, 3>& intrinsics, const matrix::matrix<double, 3, 1>& point, double& u, double& v) {
        if (point[2] <= 0.01) {
            return false;
        }
        u = intrinsics[0][0] * (point[0] / point[2]) + intrinsics[0][2];
        v = intrinsics[1][1] * (point[1] / point[2]) + intrinsics[1][2];
        return true;
    }

    void add_cube(double cx, double cy, double cz, double size, unsigned char base_intensity) {
        const double s = size / 2.0;
        matrix::matrix<double, 3, 1> v[8] = {
            { { cx - s, cy - s, cz - s } },
            { { cx + s, cy - s, cz - s } },
            { { cx + s, cy + s, cz - s } },
            { { cx - s, cy + s, cz - s } },
            { { cx - s, cy - s, cz + s } },
            { { cx + s, cy - s, cz + s } },
            { { cx + s, cy + s, cz + s } },
            { { cx - s, cy + s, cz + s } }
        };
        scene.push_back(triangle{ v[4], v[5], v[6], static_cast<unsigned char>(base_intensity - 0) });
        scene.push_back(triangle{ v[4], v[6], v[7], static_cast<unsigned char>(base_intensity - 0) });
        scene.push_back(triangle{ v[0], v[2], v[1], static_cast<unsigned char>(base_intensity - 40) });
        scene.push_back(triangle{ v[0], v[3], v[2], static_cast<unsigned char>(base_intensity - 40) });
        scene.push_back(triangle{ v[1], v[2], v[6], static_cast<unsigned char>(base_intensity - 20) });
        scene.push_back(triangle{ v[1], v[6], v[5], static_cast<unsigned char>(base_intensity - 20) });
        scene.push_back(triangle{ v[0], v[4], v[7], static_cast<unsigned char>(base_intensity - 30) });
        scene.push_back(triangle{ v[0], v[7], v[3], static_cast<unsigned char>(base_intensity - 30) });
        scene.push_back(triangle{ v[3], v[7], v[6], static_cast<unsigned char>(base_intensity - 10) });
        scene.push_back(triangle{ v[3], v[6], v[2], static_cast<unsigned char>(base_intensity - 10) });
        scene.push_back(triangle{ v[0], v[1], v[5], static_cast<unsigned char>(base_intensity - 50) });
        scene.push_back(triangle{ v[0], v[5], v[4], static_cast<unsigned char>(base_intensity - 50) });
    }

    void fill_triangle(
        unsigned char* data,
        double* depth_buffer,
        int x0,
        int y0,
        double z0,
        int x1,
        int y1,
        double z1,
        int x2,
        int y2,
        double z2,
        unsigned char intensity
    ) {
        if (y0 > y1) {
            std::swap(y0, y1);
            std::swap(x0, x1);
            std::swap(z0, z1);
        }
        if (y0 > y2) {
            std::swap(y0, y2);
            std::swap(x0, x2);
            std::swap(z0, z2);
        }
        if (y1 > y2) {
            std::swap(y1, y2);
            std::swap(x1, x2);
            std::swap(z1, z2);
        }

        auto fill_scanline = [&](int y, int xa, int xb, double za, double zb) {
            if (y < 0 || y >= image_height)
                return;
            if (xa > xb) {
                std::swap(xa, xb);
                std::swap(za, zb);
            }
            xa = std::max(0, xa);
            xb = std::min(image_width - 1, xb);

            for (int x = xa; x <= xb; x++) {
                double t = (xb == xa) ? 0.0 : (x - xa) / (double)(xb - xa);
                double z = za + t * (zb - za);

                int idx = y * image_width + x;
                if (z < depth_buffer[idx]) {
                    depth_buffer[idx] = z;
                    data[idx] = intensity;
                }
            }
        };

        if (y1 == y2) {
            for (int y = y0; y <= y1; y++) {
                double t = (y - y0) / (double)(y1 - y0 + 1e-6);
                int xa = x0 + t * (x1 - x0);
                int xb = x0 + t * (x2 - x0);
                double za = z0 + t * (z1 - z0);
                double zb = z0 + t * (z2 - z0);
                fill_scanline(y, xa, xb, za, zb);
            }
        }
        else if (y0 == y1) {
            for (int y = y0; y <= y2; y++) {
                double t = (y - y0) / (double)(y2 - y0 + 1e-6);
                int xa = x0 + t * (x2 - x0);
                int xb = x1 + t * (x2 - x1);
                double za = z0 + t * (z2 - z0);
                double zb = z1 + t * (z2 - z1);
                fill_scanline(y, xa, xb, za, zb);
            }
        }
        else {
            int x_mid = x0 + (y1 - y0) * (x2 - x0) / (double)(y2 - y0 + 1e-6);
            double z_mid = z0 + (y1 - y0) * (z2 - z0) / (double)(y2 - y0 + 1e-6);

            for (int y = y0; y <= y1; y++) {
                double t = (y - y0) / (double)(y1 - y0 + 1e-6);
                int xa = x0 + t * (x1 - x0);
                int xb = x0 + t * (x_mid - x0);
                double za = z0 + t * (z1 - z0);
                double zb = z0 + t * (z_mid - z0);
                fill_scanline(y, xa, xb, za, zb);
            }

            for (int y = y1; y <= y2; y++) {
                double t = (y - y1) / (double)(y2 - y1 + 1e-6);
                int xa = x1 + t * (x2 - x1);
                int xb = x_mid + t * (x2 - x_mid);
                double za = z1 + t * (z2 - z1);
                double zb = z_mid + t * (z2 - z_mid);
                fill_scanline(y, xa, xb, za, zb);
            }
        }
    }

public:
    void render_frame(const lie::se3<double>& pose, const matrix::matrix<double, 3, 3>& intrinsics, image::image& img) {
        unsigned char* data = img.get_data();
        std::fill(data, data + image_width * image_height, 0);
        std::vector<double> depth_buffer(image_width * image_height, std::numeric_limits<double>::max());
        for (const triangle& triangle : scene) {
            const matrix::matrix<double, 3, 1> v0_cam = pose * triangle.v0;
            const matrix::matrix<double, 3, 1> v1_cam = pose * triangle.v1;
            const matrix::matrix<double, 3, 1> v2_cam = pose * triangle.v2;
            if ((v0_cam[2] <= 0.01) && (v1_cam[2] <= 0.01) && (v2_cam[2] <= 0.01)) {
                continue;
            }
            double u0, v0;
            if (!project_point(intrinsics, v0_cam, u0, v0)) {
                continue;
            }
            double u1, v1;
            if (!project_point(intrinsics, v1_cam, u1, v1)) {
                continue;
            }
            double u2, v2;
            if (!project_point(intrinsics, v2_cam, u2, v2)) {
                continue;
            }
            fill_triangle(
                data,
                depth_buffer.data(),
                static_cast<int>(u0 + 0.5),
                static_cast<int>(v0 + 0.5),
                v0_cam[2],
                static_cast<int>(u1 + 0.5),
                static_cast<int>(v1 + 0.5),
                v1_cam[2],
                static_cast<int>(u2 + 0.5),
                static_cast<int>(v2 + 0.5),
                v2_cam[2],
                triangle.intensity
            );
        }
    }
};

inline bool ppm_save(const std::string& filepath, std::size_t width, std::size_t height, const unsigned char* pixels) {
    std::FILE* handle = std::fopen(filepath.c_str(), "wb");
    if (!handle) {
        return false;
    }
    if (std::fprintf(handle, "P5\n") != 3) {
        std::fclose(handle);
        return false;
    }
    if (std::fprintf(handle, "%zu %zu\n", width, height) <= 0) {
        std::fclose(handle);
        return false;
    }
    if (std::fprintf(handle, "%d\n", 255) <= 0) {
        std::fclose(handle);
        return false;
    }
    const std::size_t size = width * height;
    std::size_t index = 0;
    while (index < size) {
        const std::size_t delta = std::fwrite(&pixels[index], sizeof(char), size - index, handle);
        if (delta == 0) {
            break;
        }
        index += delta;
    }
    if (index != size) {
        return false;
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
    static_cast<void>(argc);
    static_cast<void>(argv);

    const int num_frames = 10;
    const int width = 640;
    const int height = 480;

    matrix::matrix<double, 3, 3> intrinsics;
    intrinsics[0][0] = 500;
    intrinsics[0][1] = 0;
    intrinsics[0][2] = width / 2;
    intrinsics[1][0] = 0;
    intrinsics[1][1] = 500;
    intrinsics[1][2] = height / 2;
    intrinsics[2][0] = 0;
    intrinsics[2][1] = 0;
    intrinsics[2][2] = 1;

    std::vector<lie::se3<double>> trajectory;
    for (int i = 0; i < 10; i++) {
        trajectory.push_back({ lie::so3<double>::rotation(0, 0, 0), { { i * 0.015, 0, 0 } } });
    }

    world renderer(width, height);
    slam system;

    for (const lie::se3<double>& pose : trajectory) {
        image::image frame(width, height);
        renderer.render_frame(pose, intrinsics, frame);
        system.process_frame(intrinsics, frame);
        std::fflush(stdout);
    }

    if (!save_trajectory_and_map_as_ply("map.ply", width, height, system.reconstruction)) {
        std::fprintf(stderr, "Failed to save map and camera trajectory to ply file.\n");
    }

    REQUIRE(system.reconstruction.frames.size() == num_frames);
    REQUIRE(system.reconstruction.landmarks.size() > 100);

    for (const auto& [id, f] : system.reconstruction.frames) {
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                REQUIRE(!std::isnan(f.rotation[r][c]));
            }
        }
        for (int r = 0; r < 3; r++) {
            REQUIRE(!std::isnan(f.translation[r]));
        }
    }

    const frame::frame& f0 = system.reconstruction.frames[0];
    const frame::frame& f9 = system.reconstruction.frames[9];
    const double dz = f9.translation[2] - f0.translation[2];
    const double dy = f9.translation[1] - f0.translation[1];
    const double dx = f9.translation[0] - f0.translation[0];
    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    REQUIRE(distance > 0.05);

    return EXIT_SUCCESS;
}