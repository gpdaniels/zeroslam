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

#include "slam_depth.hpp"

#include "image/image.hpp"
#include "math/math.hpp"
#include "sensor/camera.hpp"
#include "sensor/depth.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

namespace {
    constexpr static const int width = 320;
    constexpr static const int height = 240;
    constexpr static const double focal = 262.5;
    constexpr static const double wall_depth = 6.0;

    // A corridor of hashed tiles, a wall ahead and a floor and ceiling a metre either side of the camera at x along it, and
    // its depth along the optical axis.
    void render_corridor(const double camera_x, unsigned char* const pixels, float* const metres) {
        for (int row = 0; row < height; ++row) {
            const double ray_y = (static_cast<double>(row) + 0.5 - (0.5 * height)) / focal;
            const bool wall = (std::abs(ray_y) * wall_depth) <= 1.0;
            const double depth = wall ? wall_depth : (1.0 / std::abs(ray_y));
            for (int column = 0; column < width; ++column) {
                const double ray_x = (static_cast<double>(column) + 0.5 - (0.5 * width)) / focal;
                const long long int cell_u = static_cast<long long int>(std::floor(4.0 * (camera_x + (ray_x * depth))));
                const long long int cell_v = static_cast<long long int>(std::floor(4.0 * (wall ? (ray_y * depth) : depth)));
                unsigned long long int hash = (static_cast<unsigned long long int>(cell_u) * 73856093ull) ^ (static_cast<unsigned long long int>(cell_v) * 19349663ull) ^ ((wall ? 1ull : ((ray_y > 0.0) ? 2ull : 3ull)) * 83492791ull);
                hash ^= hash >> 13;
                hash *= 0x5bd1e995ull;
                hash ^= hash >> 15;
                pixels[static_cast<size_t>((row * width) + column)] = static_cast<unsigned char>(40ull + (hash % 176ull));
                metres[static_cast<size_t>((row * width) + column)] = static_cast<float>(depth);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const double parameters[4] = { focal, focal, 0.5 * width, 0.5 * height };
    const sensor::model camera(sensor::pinhole(&parameters[0], 4));
    sensor::depth::settings settings;
    settings.minimum = 0.1;
    settings.maximum = 10.0;
    std::vector<unsigned char> pixels(static_cast<size_t>(width * height));
    std::vector<float> metres(static_cast<size_t>(width * height));

    {
        // A metric map with depth starts at the first frame, alone, and the frames moved 5 cm each along the corridor come
        // out at that distance; only the keyframes keep their depth images.
        slam system;
        system.reconstruction.metric_scale = true;
        constexpr static const int frames = 20;
        for (int frame = 0; frame < frames; ++frame) {
            render_corridor(0.05 * static_cast<double>(frame), pixels.data(), metres.data());
            const image::image grey(static_cast<size_t>(height), static_cast<size_t>(width), pixels.data());
            system.process_frame(camera, grey, sensor::depth(static_cast<size_t>(height), static_cast<size_t>(width), metres.data(), settings));
        }
        REQUIRE(system.state() == slam::tracking_state::tracking);
        REQUIRE(system.keyframe_ids().count(0) != 0);
        REQUIRE(system.landmarks_from_depth > 0);
        const mapping::frame& last = system.reconstruction.frames.at(frames - 1);
        const math::matrix<double, 3, 1> centre = -(math::transpose(last.rotation) * last.translation);
        std::fprintf(stdout, "Corridor: the last frame at (%.4f, %.4f, %.4f), the camera at (%.4f, 0, 0).\n", centre[0], centre[1], centre[2], 0.05 * (frames - 1));
        REQUIRE(std::abs(centre[0] - (0.05 * (frames - 1))) < 0.02);
        REQUIRE(std::abs(centre[1]) < 0.02);
        REQUIRE(std::abs(centre[2]) < 0.02);
        size_t kept = 0;
        for (const auto& [frame_id, frame] : system.reconstruction.frames) {
            const bool keyframe = system.keyframe_ids().count(frame_id) != 0;
            // The newest frame keeps its depth until the next arrives.
            REQUIRE(keyframe || (frame_id == (frames - 1)) || frame.depth.empty());
            kept += frame.depth.empty() ? 0u : 1u;
        }
        REQUIRE(kept >= 1);
    }

    {
        // Without metric scale a depth image is held but not used: the map starts from two views as a monocular one.
        slam system;
        for (int frame = 0; frame < 3; ++frame) {
            render_corridor(0.05 * static_cast<double>(frame), pixels.data(), metres.data());
            const image::image grey(static_cast<size_t>(height), static_cast<size_t>(width), pixels.data());
            system.process_frame(camera, grey, sensor::depth(static_cast<size_t>(height), static_cast<size_t>(width), metres.data(), settings));
        }
        REQUIRE(system.landmarks_from_depth == 0);
        REQUIRE(system.keyframe_ids().count(0) == 0);
    }

    return EXIT_SUCCESS;
}
