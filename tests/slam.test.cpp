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

#include "slam.hpp"

#include "core/random_pcg.hpp"
#include "mapping/loop_closure.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

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

class world {
private:
    struct triangle {
        math::matrix<double, 3, 1> v0, v1, v2;
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
        core::random_pcg rng;
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
    static bool project_point(const math::matrix<double, 3, 3>& intrinsics, const math::matrix<double, 3, 1>& point, double& u, double& v) {
        if (point[2] <= 0.01) {
            return false;
        }
        u = intrinsics[0][0] * (point[0] / point[2]) + intrinsics[0][2];
        v = intrinsics[1][1] * (point[1] / point[2]) + intrinsics[1][2];
        return true;
    }

    void add_cube(double cx, double cy, double cz, double size, unsigned char base_intensity) {
        const double s = size / 2.0;
        math::matrix<double, 3, 1> v[8] = {
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
                double t = (xb == xa) ? 0.0 : static_cast<double>(x - xa) / static_cast<double>(xb - xa);
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
                double t = static_cast<double>(y - y0) / (y1 - y0 + 1e-6);
                int xa = static_cast<int>(x0 + t * (x1 - x0));
                int xb = static_cast<int>(x0 + t * (x2 - x0));
                double za = z0 + t * (z1 - z0);
                double zb = z0 + t * (z2 - z0);
                fill_scanline(y, xa, xb, za, zb);
            }
        }
        else if (y0 == y1) {
            for (int y = y0; y <= y2; y++) {
                double t = static_cast<double>(y - y0) / (y2 - y0 + 1e-6);
                int xa = static_cast<int>(x0 + t * (x2 - x0));
                int xb = static_cast<int>(x1 + t * (x2 - x1));
                double za = z0 + t * (z2 - z0);
                double zb = z1 + t * (z2 - z1);
                fill_scanline(y, xa, xb, za, zb);
            }
        }
        else {
            int x_mid = static_cast<int>(x0 + (static_cast<double>(y1 - y0) * (x2 - x0)) / (y2 - y0 + 1e-6));
            double z_mid = z0 + (y1 - y0) * (z2 - z0) / (y2 - y0 + 1e-6);

            for (int y = y0; y <= y1; y++) {
                double t = static_cast<double>(y - y0) / (y1 - y0 + 1e-6);
                int xa = static_cast<int>(x0 + t * (x1 - x0));
                int xb = static_cast<int>(x0 + t * (x_mid - x0));
                double za = z0 + t * (z1 - z0);
                double zb = z0 + t * (z_mid - z0);
                fill_scanline(y, xa, xb, za, zb);
            }

            for (int y = y1; y <= y2; y++) {
                double t = static_cast<double>(y - y1) / (y2 - y1 + 1e-6);
                int xa = static_cast<int>(x1 + t * (x2 - x1));
                int xb = static_cast<int>(x_mid + t * (x2 - x_mid));
                double za = z1 + t * (z2 - z1);
                double zb = z_mid + t * (z2 - z_mid);
                fill_scanline(y, xa, xb, za, zb);
            }
        }
    }

public:
    void render_frame(const math::se3<double>& pose, const math::matrix<double, 3, 3>& intrinsics, image::image& img) {
        unsigned char* data = img.get_data();
        std::fill(data, data + image_width * image_height, static_cast<unsigned char>(0));
        std::vector<double> depth_buffer(static_cast<size_t>(image_width * image_height), std::numeric_limits<double>::max());
        for (const triangle& face : scene) {
            const math::matrix<double, 3, 1> v0_cam = pose * face.v0;
            const math::matrix<double, 3, 1> v1_cam = pose * face.v1;
            const math::matrix<double, 3, 1> v2_cam = pose * face.v2;
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
                static_cast<int>(std::floor(u0)),
                static_cast<int>(std::floor(v0)),
                v0_cam[2],
                static_cast<int>(std::floor(u1)),
                static_cast<int>(std::floor(v1)),
                v1_cam[2],
                static_cast<int>(std::floor(u2)),
                static_cast<int>(std::floor(v2)),
                v2_cam[2],
                face.intensity
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

inline bool save_trajectory_and_map_as_ply(const char* path, int image_width, int image_height, const mapping::map& reconstruction) {
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
    constexpr static const auto write_vertex = [](std::FILE* file_handle, float x, float y, float z, unsigned char r, unsigned char g, unsigned char b) {
        std::fwrite(&x, sizeof(float), 1, file_handle);
        std::fwrite(&y, sizeof(float), 1, file_handle);
        std::fwrite(&z, sizeof(float), 1, file_handle);
        std::fwrite(&r, sizeof(unsigned char), 1, file_handle);
        std::fwrite(&g, sizeof(unsigned char), 1, file_handle);
        std::fwrite(&b, sizeof(unsigned char), 1, file_handle);
    };

    // Helper lambda to write edge.
    constexpr static const auto write_edge = [](std::FILE* file_handle, int v1, int v2) {
        std::fwrite(&v1, sizeof(int), 1, file_handle);
        std::fwrite(&v2, sizeof(int), 1, file_handle);
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
        math::matrix<double, 3, 1> corners[4];
        for (int i = 0; i < 4; ++i) {
            frame.camera.unproject(&image_corners[i][0], corners[i].data());
            corners[i] = corners[i] * frustum_scale;
        }

        // Camera centre in world coordinates.
        math::matrix<double, 3, 1> centre = -math::transpose(R) * t;
        math::matrix<double, 3, 1> world_corners[4];
        for (int i = 0; i < 4; ++i) {
            world_corners[i] = math::transpose(R) * corners[i] + centre;
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
        const int camera_index = static_cast<int>(num_landmarks + i * vertices_per_camera);
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

static void test_ratio_test() {
    {
        std::vector<match::pair> matches;
        slam::ratio_test(matches);
        REQUIRE(matches.empty());
    }
    {
        std::vector<match::pair> matches = { { 0, 5, 10.0f } };
        slam::ratio_test(matches);
        REQUIRE(matches.size() == 1);
        REQUIRE(matches[0].rhs_index == 5);
    }
    {
        std::vector<match::pair> matches = { { 0, 5, 10.0f }, { 0, 6, 20.0f } };
        slam::ratio_test(matches);
        REQUIRE(matches.size() == 1);
        REQUIRE(matches[0].rhs_index == 5);
        REQUIRE(matches[0].score == 10.0f);
    }
    {
        std::vector<match::pair> matches = { { 0, 5, 15.0f }, { 0, 6, 20.0f } };
        slam::ratio_test(matches);
        REQUIRE(matches.size() == 1);
        REQUIRE(matches[0].rhs_index == 5);
    }
    {
        std::vector<match::pair> matches = { { 0, 5, 30.0f }, { 0, 6, 31.0f } };
        slam::ratio_test(matches);
        REQUIRE(matches.empty());
    }
    {
        std::vector<match::pair> matches = { { 0, 5, 0.0f }, { 0, 6, 0.0f } };
        slam::ratio_test(matches);
        REQUIRE(matches.empty());
    }
    {
        std::vector<match::pair> matches = { { 0, 5, 0.0f }, { 0, 6, 20.0f } };
        slam::ratio_test(matches);
        REQUIRE(matches.size() == 1);
        REQUIRE(matches[0].rhs_index == 5);
    }
    {
        std::vector<match::pair> matches = { { 0, 6, 20.0f }, { 0, 5, 10.0f } };
        slam::ratio_test(matches);
        REQUIRE(matches.size() == 1);
        REQUIRE(matches[0].rhs_index == 5);
    }
    {
        std::vector<match::pair> matches = { { 0, 5, 10.0f }, { 0, 6, 20.0f }, { 0, 7, 30.0f } };
        slam::ratio_test(matches);
        REQUIRE(matches.empty());
    }
    {
        std::vector<match::pair> matches = {
            { 0, 5, 10.0f },
            { 0, 6, 20.0f },
            { 1, 7, 30.0f },
            { 1, 8, 31.0f },
            { 2, 9, 12.0f },
            { 3, 4, 0.0f },
            { 3, 5, 0.0f },
            { 4, 6, 6.0f },
            { 4, 7, 16.0f }
        };
        slam::ratio_test(matches);
        REQUIRE(matches.size() == 3);
        REQUIRE((matches[0].lhs_index == 0) && (matches[0].rhs_index == 5));
        REQUIRE((matches[1].lhs_index == 2) && (matches[1].rhs_index == 9));
        REQUIRE((matches[2].lhs_index == 4) && (matches[2].rhs_index == 6));
    }
}

static void test_model_selection_score() {
    const float residuals[5] = { 0.0f, 0.4e-5f, 0.9e-5f, 1.0e-5f, 2.0e-5f };
    REQUIRE(std::abs(slam::model_selection_score(residuals, 5, 1.0e-5f) - 1.7e-5) < 1.0e-9);
    REQUIRE(slam::model_selection_score(residuals, 0, 1.0e-5f) == 0.0);
    const float high[2] = { 5.0e-5f, 9.0e-5f };
    REQUIRE(slam::model_selection_score(high, 2, 1.0e-5f) == 0.0);
}

static void test_parallax_angle() {
    const math::matrix<double, 3, 1> origin = { { 0.0, 0.0, 0.0 } };
    const double pi = 3.14159265358979323846;
    {
        const math::matrix<double, 3, 1> point = { { 0.0, 0.0, 5.0 } };
        const math::matrix<double, 3, 1> centre = { { 5.0, 0.0, 0.0 } };
        REQUIRE(std::abs(slam::parallax_angle(point, origin, centre) - (pi / 4.0)) < 1.0e-9);
    }
    {
        const math::matrix<double, 3, 1> point = { { 0.0, 0.0, 5.0 } };
        const math::matrix<double, 3, 1> centre = { { 0.01, 0.0, 0.0 } };
        REQUIRE((slam::parallax_angle(point, origin, centre) * 180.0 / pi) < 1.0);
    }
    {
        const math::matrix<double, 3, 1> point = { { 0.0, 0.0, 5.0 } };
        const math::matrix<double, 3, 1> centre = { { 2.0, 0.0, 0.0 } };
        REQUIRE((slam::parallax_angle(point, origin, centre) * 180.0 / pi) > 1.0);
    }
    {
        const math::matrix<double, 3, 1> point = { { 0.0, 0.0, 5.0 } };
        REQUIRE(slam::parallax_angle(point, point, origin) == 0.0);
    }
}

static void generate_init_correspondences(bool planar, std::vector<estimation::correspondence_2d_2d<double>>& out) {
    out.clear();
    core::random_pcg rng;
    const double translation_x = 0.5;
    for (int i = 0; i < 200; ++i) {
        const double x = (static_cast<double>(rng.get_random_raw() % 6000) / 1000.0) - 3.0;
        const double y = (static_cast<double>(rng.get_random_raw() % 6000) / 1000.0) - 3.0;
        const double z = planar ? 8.0 : (4.0 + (static_cast<double>(rng.get_random_raw() % 10000) / 1000.0));
        const double rhs_z = z;
        if ((z <= 0.01) || (rhs_z <= 0.01)) {
            continue;
        }
        out.push_back({ { { x / z, y / z } }, { { (x + translation_x) / rhs_z, y / rhs_z } } });
    }
}

static void test_model_selection() {
    constexpr static const auto compute_ratio_homography = [](bool planar) -> double {
        std::vector<estimation::correspondence_2d_2d<double>> correspondences;
        generate_init_correspondences(planar, correspondences);
        const size_t count = correspondences.size();
        std::vector<float> essential_residuals(count);
        std::vector<size_t> essential_inliers(count);
        size_t essential_inlier_count = count;
        estimation::robust::estimate::essential<double>::model model_essential;
        const bool essential_ok = estimation::robust::solver::essential<double>::solve(correspondences.data(), count, essential_residuals.data(), essential_inliers.data(), essential_inlier_count, model_essential);
        std::vector<float> homography_residuals(count);
        std::vector<size_t> homography_inliers(count);
        size_t homography_inlier_count = count;
        estimation::robust::estimate::homography<double>::model model_homography;
        const bool homography_ok = estimation::robust::solver::homography<double>::solve(correspondences.data(), count, homography_residuals.data(), homography_inliers.data(), homography_inlier_count, model_homography);
        const double score_essential = essential_ok ? slam::model_selection_score(essential_residuals.data(), count, 1.0e-5f) : 0.0;
        const double score_homography = homography_ok ? slam::model_selection_score(homography_residuals.data(), count, 1.0e-5f) : 0.0;
        const double score_total = score_essential + score_homography;
        return (score_total > 0.0) ? (score_homography / score_total) : 0.0;
    };
    REQUIRE(compute_ratio_homography(false) <= 0.45);
    REQUIRE(compute_ratio_homography(true) > 0.45);
}

static void test_predict_constant_velocity() {
    const auto yaw = [](double theta) -> math::matrix<double, 3, 3> {
        const double c = std::cos(theta);
        const double s = std::sin(theta);
        return math::matrix<double, 3, 3>({ { { c, -s, 0.0 }, { s, c, 0.0 }, { 0.0, 0.0, 1.0 } } });
    };
    const math::matrix<double, 3, 3> identity = math::matrix<double, 3, 3>::identity();

    {
        const math::matrix<double, 3, 3> rotation = identity;
        const math::matrix<double, 3, 1> translation = { { 1.0, 2.0, 3.0 } };
        math::matrix<double, 3, 3> rotation_predicted;
        math::matrix<double, 3, 1> translation_predicted;
        slam::predict_constant_velocity(rotation, translation, rotation, translation, rotation_predicted, translation_predicted);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                REQUIRE(rotation_predicted[i][j] == rotation[i][j]);
            }
            REQUIRE(translation_predicted[i] == translation[i]);
        }
    }

    {
        const double d = 0.5;
        const math::matrix<double, 3, 1> translation_previous = { { 0.0, 0.0, 0.0 } };
        const math::matrix<double, 3, 1> translation_last = { { d, 0.0, 0.0 } };
        math::matrix<double, 3, 3> rotation_predicted;
        math::matrix<double, 3, 1> translation_predicted;
        slam::predict_constant_velocity(identity, translation_previous, identity, translation_last, rotation_predicted, translation_predicted);
        REQUIRE(std::abs(translation_predicted[0] - (2.0 * d)) < 1.0e-12);
        REQUIRE(std::abs(translation_predicted[1] - 0.0) < 1.0e-12);
        REQUIRE(std::abs(translation_predicted[2] - 0.0) < 1.0e-12);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                REQUIRE(std::abs(rotation_predicted[i][j] - identity[i][j]) < 1.0e-12);
            }
        }
    }

    {
        const double d = 0.05;
        const math::matrix<double, 3, 1> zero = { { 0.0, 0.0, 0.0 } };
        const math::matrix<double, 3, 3> rotation_previous = identity;
        const math::matrix<double, 3, 3> rotation_last = yaw(d);
        math::matrix<double, 3, 3> rotation_predicted;
        math::matrix<double, 3, 1> translation_predicted;
        slam::predict_constant_velocity(rotation_previous, zero, rotation_last, zero, rotation_predicted, translation_predicted);
        const math::matrix<double, 3, 3> rotation_expected = yaw(2.0 * d);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                REQUIRE(std::abs(rotation_predicted[i][j] - rotation_expected[i][j]) < 1.0e-12);
            }
            REQUIRE(std::abs(translation_predicted[i] - 0.0) < 1.0e-12);
        }
        const math::matrix<double, 3, 3> should_be_identity = rotation_predicted * math::transpose(rotation_predicted);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                REQUIRE(std::abs(should_be_identity[i][j] - identity[i][j]) < 1.0e-12);
            }
        }
        const double determinant =
            rotation_predicted[0][0] * (rotation_predicted[1][1] * rotation_predicted[2][2] - rotation_predicted[1][2] * rotation_predicted[2][1]) -
            rotation_predicted[0][1] * (rotation_predicted[1][0] * rotation_predicted[2][2] - rotation_predicted[1][2] * rotation_predicted[2][0]) +
            rotation_predicted[0][2] * (rotation_predicted[1][0] * rotation_predicted[2][1] - rotation_predicted[1][1] * rotation_predicted[2][0]);
        REQUIRE(std::abs(determinant - 1.0) < 1.0e-12);
    }
}

static void test_loop_closure(const int width, const int height, const math::matrix<double, 3, 3>& intrinsics, world& renderer, const mapping::frame::settings::association_kind association = mapping::frame::settings::association_kind::klt) {
    std::vector<math::se3<double>> trajectory;
    for (int i = 0; i < 24; i++) {
        trajectory.push_back({ math::so3<double>::rotation(0, 0, 0), { { i * 0.5, 0, 0 } } });
    }
    for (int i = 23; i >= 0; i--) {
        trajectory.push_back({ math::so3<double>::rotation(0, 0, 0), { { i * 0.5, 0, 0 } } });
    }

    slam system;
    system.frontend.association = association;
    system.frontend.track_collision_distance = 0.0f;
    for (const math::se3<double>& pose : trajectory) {
        image::image frame(static_cast<size_t>(width), static_cast<size_t>(height));
        renderer.render_frame(pose, intrinsics, frame);
        system.process_frame(intrinsics, frame);
    }

    REQUIRE(system.state() == slam::tracking_state::tracking);
    REQUIRE(!system.verified_loops.empty());

    system.finalise();
    double furthest = 0.0;
    for (const auto& [frame_id, frame_record] : system.reconstruction.frames) {
        static_cast<void>(frame_id);
        for (size_t r = 0; r < 3; r++) {
            REQUIRE(!std::isnan(frame_record.translation[r]));
        }
        furthest = std::max(furthest, std::sqrt(frame_record.translation.get_length_squared()));
    }
    const int last_frame_id = static_cast<int>(trajectory.size()) - 1;
    REQUIRE(system.reconstruction.frames.count(last_frame_id) != 0);
    REQUIRE(std::sqrt(system.reconstruction.frames.at(last_frame_id).translation.get_length_squared()) < 0.15 * furthest);
}

static void test_relocalisation(const int width, const int height, const math::matrix<double, 3, 3>& intrinsics, world& renderer) {
    std::vector<math::se3<double>> outward;
    for (int i = 0; i < 24; i++) {
        outward.push_back({ math::so3<double>::rotation(0, 0, 0), { { i * 0.3, 0, 0 } } });
    }
    const std::vector<unsigned char> black(static_cast<size_t>(width) * static_cast<size_t>(height), 0);

    slam system;
    REQUIRE(system.state() == slam::tracking_state::initialising);
    for (const math::se3<double>& pose : outward) {
        image::image frame(static_cast<size_t>(width), static_cast<size_t>(height));
        renderer.render_frame(pose, intrinsics, frame);
        system.process_frame(intrinsics, frame);
    }
    REQUIRE(system.state() == slam::tracking_state::tracking);
    const size_t frames_before_blackout = system.reconstruction.frames.size();
    for (int gap = 0; gap < 60; ++gap) {
        image::image frame(static_cast<size_t>(width), static_cast<size_t>(height), black.data());
        system.process_frame(intrinsics, frame);
    }
    REQUIRE(system.state() == slam::tracking_state::initialising);
    REQUIRE(system.reconstruction.frames.size() <= frames_before_blackout + 10);

    const int relocalised_frame_id = 24 + 60;
    for (int i = 23; i >= 0; i--) {
        image::image frame(static_cast<size_t>(width), static_cast<size_t>(height));
        renderer.render_frame(outward[static_cast<size_t>(i)], intrinsics, frame);
        system.process_frame(intrinsics, frame);
        if (i == 23) {
            REQUIRE(system.state() == slam::tracking_state::tracking);
            REQUIRE(system.reconstruction.frames.count(relocalised_frame_id) != 0);
            const mapping::frame& relocalised = system.reconstruction.frames.at(relocalised_frame_id);
            const mapping::frame& last_outward = system.reconstruction.frames.at(23);
            const math::matrix<double, 3, 1> relocalised_centre = -math::transpose(relocalised.rotation) * relocalised.translation;
            const math::matrix<double, 3, 1> last_outward_centre = -math::transpose(last_outward.rotation) * last_outward.translation;
            const math::matrix<double, 3, 1> first_centre = -math::transpose(system.reconstruction.frames.at(0).rotation) * system.reconstruction.frames.at(0).translation;
            REQUIRE(std::sqrt((relocalised_centre - last_outward_centre).get_length_squared()) < 0.1 * std::sqrt((last_outward_centre - first_centre).get_length_squared()));
        }
    }
    REQUIRE(system.state() == slam::tracking_state::tracking);

    system.finalise();
    double furthest = 0.0;
    for (const auto& [frame_id, frame_record] : system.reconstruction.frames) {
        static_cast<void>(frame_id);
        for (size_t r = 0; r < 3; r++) {
            REQUIRE(!std::isnan(frame_record.translation[r]));
        }
        furthest = std::max(furthest, std::sqrt(frame_record.translation.get_length_squared()));
    }
    const int last_frame_id = 24 + 60 + 24 - 1;
    REQUIRE(system.reconstruction.frames.count(last_frame_id) != 0);
    REQUIRE(std::sqrt(system.reconstruction.frames.at(last_frame_id).translation.get_length_squared()) < 0.1 * furthest);
    for (const auto& [landmark_id, landmark_observations] : system.reconstruction.observations) {
        REQUIRE(system.reconstruction.landmarks.count(landmark_id) != 0);
        for (size_t index = 1; index < landmark_observations.size(); ++index) {
            REQUIRE(landmark_observations[index - 1].frame_id < landmark_observations[index].frame_id);
        }
    }
}

static void test_submap_join(const int width, const int height, const math::matrix<double, 3, 3>& intrinsics, world& renderer) {
    std::vector<math::se3<double>> outward;
    for (int i = 0; i < 24; i++) {
        outward.push_back({ math::so3<double>::rotation(0, 0, 0), { { i * 0.5, 0, 0 } } });
    }
    std::vector<math::se3<double>> back;
    for (int i = -12; i < 24; i++) {
        back.push_back({ math::so3<double>::rotation(0, 0, 0), { { i * 0.5, 0, 0 } } });
    }
    const std::vector<unsigned char> black(static_cast<size_t>(width) * static_cast<size_t>(height), 0);

    slam system;
    system.frontend.track_collision_distance = 0.0f;
    for (const math::se3<double>& pose : outward) {
        image::image frame(static_cast<size_t>(width), static_cast<size_t>(height));
        renderer.render_frame(pose, intrinsics, frame);
        system.process_frame(intrinsics, frame);
    }
    REQUIRE(system.state() == slam::tracking_state::tracking);
    for (int gap = 0; gap < 60; ++gap) {
        image::image frame(static_cast<size_t>(width), static_cast<size_t>(height), black.data());
        system.process_frame(intrinsics, frame);
    }
    REQUIRE(system.state() == slam::tracking_state::initialising);
    const size_t keyframes_before = system.keyframe_ids().size();

    for (const math::se3<double>& pose : back) {
        image::image frame(static_cast<size_t>(width), static_cast<size_t>(height));
        renderer.render_frame(pose, intrinsics, frame);
        system.process_frame(intrinsics, frame);
    }
    REQUIRE(system.state() == slam::tracking_state::tracking);
    REQUIRE(system.submaps().size() == 1);
    REQUIRE(system.submaps().front().joined);
    REQUIRE(system.keyframe_ids().size() > keyframes_before);
    REQUIRE(!system.verified_loops.empty());

    system.finalise();
    const int last_frame_id = 24 + 60 + static_cast<int>(back.size()) - 1;
    REQUIRE(system.reconstruction.frames.count(last_frame_id) != 0);
    REQUIRE(system.reconstruction.frames.count(23) != 0);
    const mapping::frame& last = system.reconstruction.frames.at(last_frame_id);
    const mapping::frame& outward_end = system.reconstruction.frames.at(23);
    const mapping::frame& outward_start = system.reconstruction.frames.at(0);
    const math::matrix<double, 3, 1> last_centre = -math::transpose(last.rotation) * last.translation;
    const math::matrix<double, 3, 1> end_centre = -math::transpose(outward_end.rotation) * outward_end.translation;
    const math::matrix<double, 3, 1> start_centre = -math::transpose(outward_start.rotation) * outward_start.translation;
    for (size_t r = 0; r < 3; r++) {
        REQUIRE(!std::isnan(last_centre[r]));
    }
    REQUIRE(std::sqrt((last_centre - end_centre).get_length_squared()) < 0.15 * std::sqrt((end_centre - start_centre).get_length_squared()));
}

static void test_rotation_only(const int width, const int height, const math::matrix<double, 3, 3>& intrinsics, world& renderer) {
    std::vector<math::se3<double>> trajectory;
    for (int i = 0; i < 10; i++) {
        trajectory.push_back({ math::so3<double>::rotation(0, 0, 0), { { i * 0.3, 0, 0 } } });
    }
    const int turning_frames = 90;
    for (int i = 1; i <= turning_frames; i++) {
        const math::so3<double> rotation = math::so3<double>::rotation(0, i * (3.14159265358979323846 / 4.0) / turning_frames, 0);
        trajectory.push_back({ rotation, rotation * math::matrix<double, 3, 1>({ 9 * 0.3, 0, 0 }) });
    }

    slam system;
    system.frontend.inverse_depth = true;
    for (const math::se3<double>& pose : trajectory) {
        image::image frame(static_cast<size_t>(width), static_cast<size_t>(height));
        renderer.render_frame(pose, intrinsics, frame);
        system.process_frame(intrinsics, frame);
    }
    REQUIRE(system.state() == slam::tracking_state::tracking);
    system.finalise();

    REQUIRE(system.admitted_at_infinity > 20);
    const mapping::frame& turn = system.reconstruction.frames.at(9);
    const mapping::frame& start = system.reconstruction.frames.at(0);
    const math::matrix<double, 3, 1> start_centre = -math::transpose(start.rotation) * start.translation;
    const math::matrix<double, 3, 1> turn_centre = -math::transpose(turn.rotation) * turn.translation;
    const double leg = std::sqrt((turn_centre - start_centre).get_length_squared());
    double worst_heading = 0.0;
    for (int i = 1; i <= turning_frames; i++) {
        const int frame_id = 9 + i;
        REQUIRE(system.reconstruction.frames.count(frame_id) != 0);
        const mapping::frame& turning = system.reconstruction.frames.at(frame_id);
        const math::matrix<double, 3, 3> estimated_relative = turning.rotation * math::transpose(turn.rotation);
        const math::matrix<double, 3, 3> true_relative = trajectory[static_cast<size_t>(frame_id)].rotation().get_matrix() * math::transpose(trajectory[9].rotation().get_matrix());
        const math::matrix<double, 3, 3> difference = estimated_relative * math::transpose(true_relative);
        const double cosine = std::max(-1.0, std::min(1.0, (difference[0][0] + difference[1][1] + difference[2][2] - 1.0) / 2.0));
        const double turned_degrees = static_cast<double>(i) * 45.0 / static_cast<double>(turning_frames);
        worst_heading = std::max(worst_heading, (std::acos(cosine) * 180.0 / 3.14159265358979323846) - 0.75 * turned_degrees);
        const math::matrix<double, 3, 1> centre = -math::transpose(turning.rotation) * turning.translation;
        REQUIRE(std::sqrt((centre - turn_centre).get_length_squared()) < 5.0 * leg);
    }
    REQUIRE(worst_heading < 8.0);
}

static void test_lines(const int width, const int height, const math::matrix<double, 3, 3>& intrinsics, world& renderer) {
    std::vector<math::se3<double>> trajectory;
    for (int i = 0; i < 12; i++) {
        trajectory.push_back({ math::so3<double>::rotation(0, 0, 0), { { i * 0.3, 0, 0 } } });
    }

    slam system;
    system.frontend.lines = true;
    for (const math::se3<double>& pose : trajectory) {
        image::image frame(static_cast<size_t>(width), static_cast<size_t>(height));
        renderer.render_frame(pose, intrinsics, frame);
        system.process_frame(intrinsics, frame);
    }
    system.finalise();
    system.reconstruction.cull();

    REQUIRE(system.reconstruction.landmarks.size() > 50);
    REQUIRE(system.reconstruction.line_landmarks.size() > 5);
    size_t observations = 0;
    size_t accurate = 0;
    for (const auto& [line_id, line_record] : system.reconstruction.line_landmarks) {
        REQUIRE(system.reconstruction.line_observations.count(line_id) != 0);
        const std::vector<mapping::map::line_observation>& line_observations = system.reconstruction.line_observations.at(line_id);
        REQUIRE(line_observations.size() >= 2);
        REQUIRE(std::abs(line_record.plucker_line.direction.get_length_squared() - 1.0) < 1.0e-9);
        REQUIRE(std::abs(geometry::plucker::dot(line_record.plucker_line.moment, line_record.plucker_line.direction)) < 1.0e-9);
        double error = 0.0;
        for (const mapping::map::line_observation& observation : line_observations) {
            REQUIRE(system.reconstruction.frames.count(observation.frame_id) != 0);
            error += mapping::map::line_reprojection_error(system.reconstruction.frames.at(observation.frame_id), line_record, observation);
            ++observations;
        }
        error /= static_cast<double>(line_observations.size());
        REQUIRE(error < 5.991);
        accurate += (error < 2.0);
    }
    REQUIRE(observations > system.reconstruction.line_landmarks.size());
    REQUIRE(accurate * 2 > system.reconstruction.line_landmarks.size());
    for (const auto& [frame_id, frame_record] : system.reconstruction.frames) {
        static_cast<void>(frame_id);
        for (size_t r = 0; r < 3; r++) {
            REQUIRE(!std::isnan(frame_record.translation[r]));
        }
    }
}

static void test_map_reacquisition(const int width, const int height, const math::matrix<double, 3, 3>& intrinsics, world& renderer) {
    std::vector<math::se3<double>> trajectory;
    for (int i = 0; i < 12; i++) {
        trajectory.push_back({ math::so3<double>::rotation(0, 0, 0), { { i * 0.3, 0, 0 } } });
    }
    for (int i = 11; i >= 0; i--) {
        trajectory.push_back({ math::so3<double>::rotation(0, 0, 0), { { i * 0.3, 0, 0 } } });
    }

    const auto run = [&](slam& system) {
        for (const math::se3<double>& pose : trajectory) {
            image::image frame(static_cast<size_t>(width), static_cast<size_t>(height));
            renderer.render_frame(pose, intrinsics, frame);
            system.process_frame(intrinsics, frame);
        }
    };
    slam system_a;
    run(system_a);
    slam system_b;
    run(system_b);

    REQUIRE(system_a.reacquired_by_epipolar > 0);

    for (const auto& [landmark_id, landmark_observations] : system_a.reconstruction.observations) {
        REQUIRE(system_a.reconstruction.landmarks.count(landmark_id) != 0);
        REQUIRE(!landmark_observations.empty());
        for (size_t index = 1; index < landmark_observations.size(); ++index) {
            if (!(landmark_observations[index - 1].frame_id < landmark_observations[index].frame_id)) {
                std::fprintf(stderr, "DEBUG landmark %d frames:", landmark_id);
                for (const auto& o : landmark_observations) {
                    std::fprintf(stderr, " %d", o.frame_id);
                }
                std::fprintf(stderr, "\n");
            }
            REQUIRE(landmark_observations[index - 1].frame_id < landmark_observations[index].frame_id);
        }
    }
    for (const auto& [frame_id, frame_record] : system_a.reconstruction.frames) {
        static_cast<void>(frame_id);
        for (size_t r = 0; r < 3; r++) {
            for (size_t c = 0; c < 3; c++) {
                REQUIRE(!std::isnan(frame_record.rotation[r][c]));
            }
            REQUIRE(!std::isnan(frame_record.translation[r]));
        }
    }

    REQUIRE(system_a.reacquired_by_epipolar == system_b.reacquired_by_epipolar);
    REQUIRE(system_a.reconstruction.frames.size() == system_b.reconstruction.frames.size());
    REQUIRE(system_a.reconstruction.landmarks.size() == system_b.reconstruction.landmarks.size());
    for (const auto& [frame_id, frame_record] : system_a.reconstruction.frames) {
        const auto other_it = system_b.reconstruction.frames.find(frame_id);
        REQUIRE(other_it != system_b.reconstruction.frames.end());
        for (size_t r = 0; r < 3; r++) {
            for (size_t c = 0; c < 3; c++) {
                REQUIRE(frame_record.rotation[r][c] == other_it->second.rotation[r][c]);
            }
            REQUIRE(frame_record.translation[r] == other_it->second.translation[r]);
        }
    }
    for (const auto& [landmark_id, landmark_record] : system_a.reconstruction.landmarks) {
        const auto other_it = system_b.reconstruction.landmarks.find(landmark_id);
        REQUIRE(other_it != system_b.reconstruction.landmarks.end());
        for (size_t r = 0; r < 3; r++) {
            REQUIRE(landmark_record.location[r] == other_it->second.location[r]);
        }
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    test_ratio_test();
    test_model_selection_score();
    test_parallax_angle();
    test_model_selection();
    test_predict_constant_velocity();

    const int num_frames = 10;
    const int width = 640;
    const int height = 480;

    math::matrix<double, 3, 3> intrinsics;
    intrinsics[0][0] = 500;
    intrinsics[0][1] = 0;
    intrinsics[0][2] = width / 2;
    intrinsics[1][0] = 0;
    intrinsics[1][1] = 500;
    intrinsics[1][2] = height / 2;
    intrinsics[2][0] = 0;
    intrinsics[2][1] = 0;
    intrinsics[2][2] = 1;

    std::vector<math::se3<double>> trajectory;
    for (int i = 0; i < 10; i++) {
        trajectory.push_back({ math::so3<double>::rotation(0, 0, 0), { { i * 0.3, 0, 0 } } });
    }

    world renderer(width, height);
    slam system;

    for (const math::se3<double>& pose : trajectory) {
        image::image frame(static_cast<size_t>(width), static_cast<size_t>(height));
        renderer.render_frame(pose, intrinsics, frame);
        system.process_frame(intrinsics, frame);
        std::fflush(stdout);
    }

    if (!save_trajectory_and_map_as_ply("map.ply", width, height, system.reconstruction)) {
        std::fprintf(stderr, "Failed to save map and camera trajectory to ply file.\n");
    }

    REQUIRE(system.reconstruction.frames.size() == num_frames - 2);
    REQUIRE(system.reconstruction.frames.find(1) == system.reconstruction.frames.end());
    REQUIRE(system.reconstruction.frames.find(2) == system.reconstruction.frames.end());
    REQUIRE(system.reconstruction.frames.find(3) != system.reconstruction.frames.end());
    REQUIRE(system.reconstruction.landmarks.size() > 50);

    for (const auto& [id, f] : system.reconstruction.frames) {
        for (size_t r = 0; r < 3; r++) {
            for (size_t c = 0; c < 3; c++) {
                REQUIRE(!std::isnan(f.rotation[r][c]));
            }
        }
        for (size_t r = 0; r < 3; r++) {
            REQUIRE(!std::isnan(f.translation[r]));
        }
    }

    const mapping::frame& f0 = system.reconstruction.frames[0];
    const mapping::frame& f9 = system.reconstruction.frames[9];
    const double dz = f9.translation[2] - f0.translation[2];
    const double dy = f9.translation[1] - f0.translation[1];
    const double dx = f9.translation[0] - f0.translation[0];
    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    REQUIRE(distance > 0.05);

    test_map_reacquisition(width, height, intrinsics, renderer);
    test_loop_closure(width, height, intrinsics, renderer);
    test_loop_closure(width, height, intrinsics, renderer, mapping::frame::settings::association_kind::match);
    test_loop_closure(width, height, intrinsics, renderer, mapping::frame::settings::association_kind::both);
    test_relocalisation(width, height, intrinsics, renderer);
    test_submap_join(width, height, intrinsics, renderer);
    test_lines(width, height, intrinsics, renderer);
    test_rotation_only(width, height, intrinsics, renderer);

    return EXIT_SUCCESS;
}
