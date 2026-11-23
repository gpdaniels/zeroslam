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

#include "api.hpp"

#include "math/math.hpp"
#include "zeroslam/zeroslam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static void render_corridor(const double camera_x, const int width, const int height, const zeroslam_sensor_parameters_camera_struct& camera, unsigned char* const pixels) {
    constexpr static const double wall_depth = 6.0;
    for (int row = 0; row < height; ++row) {
        const double ray_y = (static_cast<double>(row) + 0.5 - camera.centre_y) / camera.focal_y;
        for (int column = 0; column < width; ++column) {
            const double ray_x = (static_cast<double>(column) + 0.5 - camera.centre_x) / camera.focal_x;
            const bool wall = (math::abs(ray_y) * wall_depth) <= 1.0;
            const double depth = wall ? wall_depth : (1.0 / math::abs(ray_y));
            const long long int cell_u = static_cast<long long int>(math::floor(4.0 * (camera_x + (ray_x * depth))));
            const long long int cell_v = static_cast<long long int>(math::floor(4.0 * (wall ? (ray_y * depth) : depth)));
            unsigned long long int hash = (static_cast<unsigned long long int>(cell_u) * 73856093ull) ^ (static_cast<unsigned long long int>(cell_v) * 19349663ull) ^ ((wall ? 1ull : ((ray_y > 0.0) ? 2ull : 3ull)) * 83492791ull);
            hash ^= hash >> 13;
            hash *= 0x5bd1e995ull;
            hash ^= hash >> 15;
            pixels[static_cast<size_t>((row * width) + column)] = static_cast<unsigned char>(40ull + (hash % 176ull));
        }
    }
}

// The corridor's depth along the optical axis, which is the depth the ray of unit z reaches.
static void render_corridor_depth(const int width, const int height, const zeroslam_sensor_parameters_camera_struct& camera, float* const metres) {
    constexpr static const double wall_depth = 6.0;
    for (int row = 0; row < height; ++row) {
        const double ray_y = (static_cast<double>(row) + 0.5 - camera.centre_y) / camera.focal_y;
        const bool wall = (math::abs(ray_y) * wall_depth) <= 1.0;
        const double depth = wall ? wall_depth : (1.0 / math::abs(ray_y));
        for (int column = 0; column < width; ++column) {
            metres[static_cast<size_t>((row * width) + column)] = static_cast<float>(depth);
        }
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        REQUIRE(std::strcmp(zeroslam_return_enum_to_string(zeroslam_return_success), "success") == 0);
        REQUIRE(std::strcmp(zeroslam_return_enum_to_string(zeroslam_return_failure_invalid_sensor_data), "failure: invalid sensor data") == 0);
        REQUIRE(std::strcmp(zeroslam_sensor_enum_to_string(zeroslam_sensor_local_angular), "local angular") == 0);
        REQUIRE(std::strcmp(zeroslam_sensor_enum_to_string(zeroslam_sensor_camera), "camera") == 0);
    }

    {
        zeroslam::system wrapped;
        REQUIRE(wrapped.is_valid());
        long long int timestamp = -1;
        REQUIRE(wrapped.get_timestamp(&timestamp) == zeroslam_return_success);
        REQUIRE(timestamp == 0);
        zeroslam::system moved(static_cast<zeroslam::system&&>(wrapped));
        REQUIRE(!wrapped.is_valid());
        REQUIRE(moved.is_valid());
        REQUIRE(wrapped.get_timestamp(&timestamp) == zeroslam_return_failure_invalid_system);
        REQUIRE(moved.get_timestamp(&timestamp) == zeroslam_return_success);
        zeroslam::system assigned;
        assigned = static_cast<zeroslam::system&&>(moved);
        REQUIRE(!moved.is_valid());
        REQUIRE(assigned.is_valid());
    }

    {
        REQUIRE(zeroslam_create(nullptr) == zeroslam_return_failure_invalid_argument);
        zeroslam_system* system = nullptr;
        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_failure_invalid_system);
        REQUIRE(zeroslam_create(&system) == zeroslam_return_success);
        REQUIRE(system != nullptr);
        long long int timestamp = -1;
        REQUIRE(zeroslam_get_timestamp(nullptr, &timestamp) == zeroslam_return_failure_invalid_system);
        REQUIRE(zeroslam_get_timestamp(system, nullptr) == zeroslam_return_failure_invalid_argument);
        REQUIRE(zeroslam_get_timestamp(system, &timestamp) == zeroslam_return_success);
        REQUIRE(timestamp == 0);
        // Finalising asks the map to settle; a system with nothing in it settles at once, and a second time over too.
        REQUIRE(zeroslam_finalise(nullptr) == zeroslam_return_failure_invalid_system);
        REQUIRE(zeroslam_finalise(system) == zeroslam_return_success);
        REQUIRE(zeroslam_finalise(system) == zeroslam_return_success);
        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_success);
        REQUIRE(system == nullptr);
    }

    {
        zeroslam_system* system = nullptr;
        REQUIRE(zeroslam_create(&system) == zeroslam_return_success);
        int length = 0;
        REQUIRE(zeroslam_get_configuration(system, nullptr, &length) == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(length > 0);
        std::vector<char> buffer(static_cast<size_t>(length));
        int capacity = length;
        REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
        REQUIRE(capacity == length - 1);
        REQUIRE(std::strstr(buffer.data(), "verbosity=1") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "association=both\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "collisions=2\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "extrinsics=declared\n") != nullptr);

        const char* const delta = "verbosity=3\n";
        REQUIRE(zeroslam_set_configuration(system, delta, static_cast<int>(std::strlen(delta))) == zeroslam_return_success);
        capacity = length;
        REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
        REQUIRE(std::strstr(buffer.data(), "verbosity=3") != nullptr);

        const char* const frontend = "detector=harris\ndetector_sigma=2\nrefiner=rohr\nrefiner_sigma=1.5\n";
        REQUIRE(zeroslam_set_configuration(system, frontend, static_cast<int>(std::strlen(frontend))) == zeroslam_return_success);
        capacity = static_cast<int>(buffer.size());
        REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
        REQUIRE(std::strstr(buffer.data(), "detector=harris\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "detector_sigma=2\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "refiner=rohr\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "refiner_sigma=1.5\n") != nullptr);
        const char* const budget = "budget=fixed\n";
        REQUIRE(zeroslam_set_configuration(system, budget, static_cast<int>(std::strlen(budget))) == zeroslam_return_success);
        capacity = static_cast<int>(buffer.size());
        REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
        REQUIRE(std::strstr(buffer.data(), "budget=fixed\n") != nullptr);
        const char* const bad_budget = "budget=lavish\n";
        REQUIRE(zeroslam_set_configuration(system, bad_budget, static_cast<int>(std::strlen(bad_budget))) == zeroslam_return_failure_invalid_configuration);
        const char* const tracker = "tracker=extrema\n";
        REQUIRE(zeroslam_set_configuration(system, tracker, static_cast<int>(std::strlen(tracker))) == zeroslam_return_success);
        length = 0;
        REQUIRE(zeroslam_get_configuration(system, nullptr, &length) == zeroslam_return_failure_insufficient_data_length);
        buffer.resize(static_cast<size_t>(length));
        capacity = length;
        REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
        REQUIRE(std::strstr(buffer.data(), "tracker=extrema\n") != nullptr);
        const char* const association = "association=both\n";
        REQUIRE(zeroslam_set_configuration(system, association, static_cast<int>(std::strlen(association))) == zeroslam_return_success);
        length = 0;
        REQUIRE(zeroslam_get_configuration(system, nullptr, &length) == zeroslam_return_failure_insufficient_data_length);
        buffer.resize(static_cast<size_t>(length));
        capacity = length;
        REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
        REQUIRE(std::strstr(buffer.data(), "association=both\n") != nullptr);
        const char* const bad_association = "association=flow\n";
        REQUIRE(zeroslam_set_configuration(system, bad_association, static_cast<int>(std::strlen(bad_association))) == zeroslam_return_failure_invalid_configuration);
        const char* const descriptor = "descriptor=teblid\n";
        REQUIRE(zeroslam_set_configuration(system, descriptor, static_cast<int>(std::strlen(descriptor))) == zeroslam_return_success);
        const char* const mser = "detector=mser\n";
        REQUIRE(zeroslam_set_configuration(system, mser, static_cast<int>(std::strlen(mser))) == zeroslam_return_success);
        const char* const bsift = "descriptor=bsift\n";
        REQUIRE(zeroslam_set_configuration(system, bsift, static_cast<int>(std::strlen(bsift))) == zeroslam_return_success);
        const char* const affine = "affine=on\nblur=on\n";
        REQUIRE(zeroslam_set_configuration(system, affine, static_cast<int>(std::strlen(affine))) == zeroslam_return_success);
        const char* const bad_affine = "affine=maybe\n";
        REQUIRE(zeroslam_set_configuration(system, bad_affine, static_cast<int>(std::strlen(bad_affine))) == zeroslam_return_failure_invalid_configuration);
        const char* const bad_descriptor = "descriptor=sift\n";
        REQUIRE(zeroslam_set_configuration(system, bad_descriptor, static_cast<int>(std::strlen(bad_descriptor))) == zeroslam_return_failure_invalid_configuration);
        const char* const lines = "lines=on\n";
        REQUIRE(zeroslam_set_configuration(system, lines, static_cast<int>(std::strlen(lines))) == zeroslam_return_success);
        const char* const damping = "damping=on\n";
        REQUIRE(zeroslam_set_configuration(system, damping, static_cast<int>(std::strlen(damping))) == zeroslam_return_success);
        const char* const bad_damping = "damping=some\n";
        REQUIRE(zeroslam_set_configuration(system, bad_damping, static_cast<int>(std::strlen(bad_damping))) == zeroslam_return_failure_invalid_configuration);
        const char* const collisions = "collisions=3.5\n";
        REQUIRE(zeroslam_set_configuration(system, collisions, static_cast<int>(std::strlen(collisions))) == zeroslam_return_success);
        const char* const bad_collisions = "collisions=near\n";
        REQUIRE(zeroslam_set_configuration(system, bad_collisions, static_cast<int>(std::strlen(bad_collisions))) == zeroslam_return_failure_invalid_configuration);
        const char* const outliers = "outliers=3\n";
        REQUIRE(zeroslam_set_configuration(system, outliers, static_cast<int>(std::strlen(outliers))) == zeroslam_return_success);
        const char* const bad_outliers = "outliers=many\n";
        REQUIRE(zeroslam_set_configuration(system, bad_outliers, static_cast<int>(std::strlen(bad_outliers))) == zeroslam_return_failure_invalid_configuration);
        const char* const anchor = "anchor=affine_illumination\nanchor_refresh=0.75\n";
        REQUIRE(zeroslam_set_configuration(system, anchor, static_cast<int>(std::strlen(anchor))) == zeroslam_return_success);
        const char* const bad_anchor = "anchor=homography\n";
        REQUIRE(zeroslam_set_configuration(system, bad_anchor, static_cast<int>(std::strlen(bad_anchor))) == zeroslam_return_failure_invalid_configuration);
        const char* const bad_anchor_refresh = "anchor_refresh=loose\n";
        REQUIRE(zeroslam_set_configuration(system, bad_anchor_refresh, static_cast<int>(std::strlen(bad_anchor_refresh))) == zeroslam_return_failure_invalid_configuration);
        length = 0;
        REQUIRE(zeroslam_get_configuration(system, nullptr, &length) == zeroslam_return_failure_insufficient_data_length);
        buffer.resize(static_cast<size_t>(length));
        capacity = length;
        REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
        REQUIRE(std::strstr(buffer.data(), "lines=on\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "damping=on\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "collisions=3.5\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "outliers=3\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "anchor=affine_illumination\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "anchor_refresh=0.75\n") != nullptr);
        const char* const wavelet = "flow=wavelet\nwavelet_window=3\nwavelet_levels=5\nwavelet_robust=on\nwavelet_undecimated=on\nwavelet_seed=klt_fallback\nline_pose=on\nline_angle=25\nsolver=square_root\nsolver_precision=single\nglobal_adjustment=10\nadjustment=relative\n";
        REQUIRE(zeroslam_set_configuration(system, wavelet, static_cast<int>(std::strlen(wavelet))) == zeroslam_return_success);
        REQUIRE(zeroslam_get_configuration(system, nullptr, &length) == zeroslam_return_failure_insufficient_data_length);
        buffer.assign(static_cast<size_t>(length), '\0');
        capacity = length;
        REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
        REQUIRE(std::strstr(buffer.data(), "flow=wavelet\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "wavelet_window=3\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "wavelet_levels=5\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "wavelet_robust=on\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "wavelet_undecimated=on\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "wavelet_seed=klt_fallback\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "line_pose=on\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "line_angle=25\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "solver=square_root\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "solver_precision=single\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "global_adjustment=10\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "\nadjustment=relative\n") != nullptr);
        const char* const bad_wavelet[10] = { "flow=phase\n", "wavelet_window=0\n", "wavelet_levels=9\n", "wavelet_seed=maybe\n", "line_pose=yes\n", "line_angle=90\n", "solver=sparse\n", "solver_precision=half\n", "global_adjustment=0\n", "adjustment=global\n" };
        for (int index = 0; index < 10; ++index) {
            REQUIRE(zeroslam_set_configuration(system, bad_wavelet[index], static_cast<int>(std::strlen(bad_wavelet[index]))) == zeroslam_return_failure_invalid_configuration);
        }
        const char* const wavelet_off = "flow=intensity\nwavelet_robust=off\nwavelet_undecimated=off\nwavelet_seed=rest\nline_pose=off\nline_angle=off\nsolver=dense_schur\nsolver_precision=double\nglobal_adjustment=off\nadjustment=absolute\n";
        REQUIRE(zeroslam_set_configuration(system, wavelet_off, static_cast<int>(std::strlen(wavelet_off))) == zeroslam_return_success);
        capacity = length;
        REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
        REQUIRE(std::strstr(buffer.data(), "flow=intensity\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "wavelet_seed=rest\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "line_pose=off\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "line_angle=off\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "solver=dense_schur\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "solver_precision=double\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "global_adjustment=off\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "\nadjustment=absolute\n") != nullptr);

        // Each calibration level reads back, off by default, and an unknown one is rejected.
        REQUIRE(std::strstr(buffer.data(), "\ncalibration=off\n") != nullptr);
        const char* const calibrations[5] = { "calibration=focal\n", "calibration=focal_centre\n", "calibration=intrinsics\n", "calibration=all\n", "calibration=off\n" };
        buffer.assign(static_cast<size_t>(length) + 64u, '\0');
        for (int index = 0; index < 5; ++index) {
            REQUIRE(zeroslam_set_configuration(system, calibrations[index], static_cast<int>(std::strlen(calibrations[index]))) == zeroslam_return_success);
            capacity = static_cast<int>(buffer.size());
            REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
            REQUIRE(std::strstr(buffer.data(), calibrations[index]) != nullptr);
        }
        const char* const bad_calibration = "calibration=distortion\n";
        REQUIRE(zeroslam_set_configuration(system, bad_calibration, static_cast<int>(std::strlen(bad_calibration))) == zeroslam_return_failure_invalid_configuration);

        // Each local map source reads back, and an unknown one is rejected.
        const char* const local_maps[4] = { "local_map=covisible\n", "local_map=voxels\n", "local_map=both\n", "local_map=fallback\n" };
        for (int index = 0; index < 4; ++index) {
            REQUIRE(zeroslam_set_configuration(system, local_maps[index], static_cast<int>(std::strlen(local_maps[index]))) == zeroslam_return_success);
            REQUIRE(zeroslam_get_configuration(system, nullptr, &length) == zeroslam_return_failure_insufficient_data_length);
            buffer.assign(static_cast<size_t>(length), '\0');
            capacity = length;
            REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
            REQUIRE(std::strstr(buffer.data(), local_maps[index]) != nullptr);
        }
        const char* const bad_local_map = "local_map=octree\n";
        REQUIRE(zeroslam_set_configuration(system, bad_local_map, static_cast<int>(std::strlen(bad_local_map))) == zeroslam_return_failure_invalid_configuration);
        // Each place recognition reads back, and an unknown one is rejected.
        const char* const place_recognitions[2] = { "place_recognition=ibow\n", "place_recognition=hbst\n" };
        for (int index = 0; index < 2; ++index) {
            REQUIRE(zeroslam_set_configuration(system, place_recognitions[index], static_cast<int>(std::strlen(place_recognitions[index]))) == zeroslam_return_success);
            REQUIRE(zeroslam_get_configuration(system, nullptr, &length) == zeroslam_return_failure_insufficient_data_length);
            buffer.assign(static_cast<size_t>(length), '\0');
            capacity = length;
            REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
            REQUIRE(std::strstr(buffer.data(), place_recognitions[index]) != nullptr);
        }
        const char* const bad_place_recognition = "place_recognition=dbow\n";
        REQUIRE(zeroslam_set_configuration(system, bad_place_recognition, static_cast<int>(std::strlen(bad_place_recognition))) == zeroslam_return_failure_invalid_configuration);
        const char* const loop_revisits[2] = { "loop_revisits=off\n", "loop_revisits=on\n" };
        for (int index = 0; index < 2; ++index) {
            REQUIRE(zeroslam_set_configuration(system, loop_revisits[index], static_cast<int>(std::strlen(loop_revisits[index]))) == zeroslam_return_success);
            REQUIRE(zeroslam_get_configuration(system, nullptr, &length) == zeroslam_return_failure_insufficient_data_length);
            buffer.assign(static_cast<size_t>(length), '\0');
            capacity = length;
            REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
            REQUIRE(std::strstr(buffer.data(), loop_revisits[index]) != nullptr);
        }
        const char* const bad_loop_revisits = "loop_revisits=sometimes\n";
        REQUIRE(zeroslam_set_configuration(system, bad_loop_revisits, static_cast<int>(std::strlen(bad_loop_revisits))) == zeroslam_return_failure_invalid_configuration);
        const char* const loop_confirmations[2] = { "loop_confirmations=3\n", "loop_confirmations=1\n" };
        for (int index = 0; index < 2; ++index) {
            REQUIRE(zeroslam_set_configuration(system, loop_confirmations[index], static_cast<int>(std::strlen(loop_confirmations[index]))) == zeroslam_return_success);
            REQUIRE(zeroslam_get_configuration(system, nullptr, &length) == zeroslam_return_failure_insufficient_data_length);
            buffer.assign(static_cast<size_t>(length), '\0');
            capacity = length;
            REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
            REQUIRE(std::strstr(buffer.data(), loop_confirmations[index]) != nullptr);
        }
        const char* const bad_loop_confirmations[2] = { "loop_confirmations=0\n", "loop_confirmations=three\n" };
        for (int index = 0; index < 2; ++index) {
            REQUIRE(zeroslam_set_configuration(system, bad_loop_confirmations[index], static_cast<int>(std::strlen(bad_loop_confirmations[index]))) == zeroslam_return_failure_invalid_configuration);
        }

        const char* const collisions_off = "collisions=off\nanchor=off\nanchor_refresh=off\n";
        REQUIRE(zeroslam_set_configuration(system, collisions_off, static_cast<int>(std::strlen(collisions_off))) == zeroslam_return_success);
        length = 0;
        REQUIRE(zeroslam_get_configuration(system, nullptr, &length) == zeroslam_return_failure_insufficient_data_length);
        buffer.resize(static_cast<size_t>(length));
        capacity = length;
        REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
        REQUIRE(std::strstr(buffer.data(), "collisions=off\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "anchor=off\n") != nullptr);
        REQUIRE(std::strstr(buffer.data(), "anchor_refresh=off\n") != nullptr);
        const char* const bad_lines = "lines=maybe\n";
        REQUIRE(zeroslam_set_configuration(system, bad_lines, static_cast<int>(std::strlen(bad_lines))) == zeroslam_return_failure_invalid_configuration);
        const char* const depth = "depth=inverse\n";
        REQUIRE(zeroslam_set_configuration(system, depth, static_cast<int>(std::strlen(depth))) == zeroslam_return_success);
        length = 0;
        REQUIRE(zeroslam_get_configuration(system, nullptr, &length) == zeroslam_return_failure_insufficient_data_length);
        buffer.resize(static_cast<size_t>(length));
        capacity = length;
        REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
        REQUIRE(std::strstr(buffer.data(), "depth=inverse\n") != nullptr);
        const char* const bad_depth = "depth=metric\n";
        REQUIRE(zeroslam_set_configuration(system, bad_depth, static_cast<int>(std::strlen(bad_depth))) == zeroslam_return_failure_invalid_configuration);
        const char* const bad_tracker = "tracker=orb\n";
        REQUIRE(zeroslam_set_configuration(system, bad_tracker, static_cast<int>(std::strlen(bad_tracker))) == zeroslam_return_failure_invalid_configuration);
        const char* const reset = "detector=fast\nrefiner=subpixel\ntracker=klt\nassociation=klt\ndescriptor=orb\nlines=off\ndepth=xyz\nbudget=free\n";
        REQUIRE(zeroslam_set_configuration(system, reset, static_cast<int>(std::strlen(reset))) == zeroslam_return_success);
        const char* const bad_measure = "detector=warp\n";
        REQUIRE(zeroslam_set_configuration(system, bad_measure, static_cast<int>(std::strlen(bad_measure))) == zeroslam_return_failure_invalid_configuration);
        const char* const bad_sigma = "refiner_sigma=9\n";
        REQUIRE(zeroslam_set_configuration(system, bad_sigma, static_cast<int>(std::strlen(bad_sigma))) == zeroslam_return_failure_invalid_configuration);
        const char* const bad_decimal = "detector_sigma=1.5.2\n";
        REQUIRE(zeroslam_set_configuration(system, bad_decimal, static_cast<int>(std::strlen(bad_decimal))) == zeroslam_return_failure_invalid_configuration);

        const char* const bad = "warp_speed=9\nverbosity=1\n";
        REQUIRE(zeroslam_set_configuration(system, bad, static_cast<int>(std::strlen(bad))) == zeroslam_return_failure_invalid_configuration);
        capacity = length;
        REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
        REQUIRE(std::strstr(buffer.data(), "verbosity=3") != nullptr);
        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_success);
    }

    {
        zeroslam_system* system = nullptr;
        REQUIRE(zeroslam_create(&system) == zeroslam_return_success);
        zeroslam_sensor_parameters_camera_struct camera_parameters{};
        camera_parameters.width = 64;
        camera_parameters.height = 48;
        camera_parameters.focal_x = 50.0;
        camera_parameters.focal_y = 50.0;
        camera_parameters.centre_x = 32.0;
        camera_parameters.centre_y = 24.0;

        zeroslam_sensor_rig_struct rig{};
        rig.type = zeroslam_sensor_camera;
        rig.sensor_id = 0;
        rig.parameters_length = static_cast<int>(sizeof(camera_parameters));
        rig.parameters_data = &camera_parameters;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_failure_invalid_rig_sensor);

        rig.sensor_id = 1;
        rig.type = zeroslam_sensor_local_linear;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_failure_invalid_rig_sensor);

        rig.type = zeroslam_sensor_camera;
        rig.parameters_length = 4;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_failure_invalid_rig_sensor);

        rig.parameters_length = static_cast<int>(sizeof(camera_parameters));
        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_success);

        // A rig of several cameras, each with its own id.
        zeroslam_sensor_rig_struct two[2] = { rig, rig };
        REQUIRE(zeroslam_set_sensor_rig(system, &two[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
        two[1].sensor_id = 2;
        REQUIRE(zeroslam_set_sensor_rig(system, &two[0], 2) == zeroslam_return_success);

        // Extrinsics of the rig's sensors only, each once, with a rotation and a known refinement.
        zeroslam_sensor_extrinsics_struct extrinsics[2] = {};
        extrinsics[0].sensor_id = 1;
        extrinsics[0].pose[6] = 1.0;
        extrinsics[1].sensor_id = 2;
        extrinsics[1].pose[0] = 0.1;
        extrinsics[1].pose[6] = 1.0;
        extrinsics[1].refinement = zeroslam_extrinsic_refine;
        REQUIRE(zeroslam_set_sensor_extrinsics(nullptr, &extrinsics[0], 2) == zeroslam_return_failure_invalid_system);
        REQUIRE(zeroslam_set_sensor_extrinsics(system, nullptr, 2) == zeroslam_return_failure_invalid_argument);
        extrinsics[1].sensor_id = 7;
        REQUIRE(zeroslam_set_sensor_extrinsics(system, &extrinsics[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
        extrinsics[1].sensor_id = 1;
        REQUIRE(zeroslam_set_sensor_extrinsics(system, &extrinsics[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
        extrinsics[1].sensor_id = 2;
        extrinsics[1].pose[6] = 0.0;
        REQUIRE(zeroslam_set_sensor_extrinsics(system, &extrinsics[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
        extrinsics[1].pose[6] = 1.0;
        extrinsics[1].refinement = 5;
        REQUIRE(zeroslam_set_sensor_extrinsics(system, &extrinsics[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
        extrinsics[1].refinement = zeroslam_extrinsic_refine;
        REQUIRE(zeroslam_set_sensor_extrinsics(system, &extrinsics[0], 2) == zeroslam_return_success);
        {
            // The extrinsics come back as they were given, one per camera.
            int held_length = 0;
            REQUIRE(zeroslam_get_sensor_extrinsics(system, nullptr, nullptr) == zeroslam_return_failure_invalid_argument);
            REQUIRE(zeroslam_get_sensor_extrinsics(system, nullptr, &held_length) == zeroslam_return_failure_insufficient_data_length);
            REQUIRE(held_length == 2);
            zeroslam_sensor_extrinsics_struct held[2] = {};
            REQUIRE(zeroslam_get_sensor_extrinsics(system, &held[0], &held_length) == zeroslam_return_success);
            REQUIRE((held[0].sensor_id == 1) && (held[1].sensor_id == 2));
            REQUIRE(held[1].pose[0] == 0.1);
            REQUIRE(held[1].pose[6] == 1.0);
            REQUIRE(held[1].refinement == zeroslam_extrinsic_refine);
            REQUIRE(held[0].refinement == zeroslam_extrinsic_fixed);
        }

        // The extrinsics setting.
        const char* const refine = "extrinsics=refine\n";
        REQUIRE(zeroslam_set_configuration(system, refine, static_cast<int>(std::strlen(refine))) == zeroslam_return_success);
        const char* const unknown = "extrinsics=sometimes\n";
        REQUIRE(zeroslam_set_configuration(system, unknown, static_cast<int>(std::strlen(unknown))) == zeroslam_return_failure_invalid_configuration);

        // The cameras of a frame share its timestamp: a camera twice, or a newer frame before the frame is whole, is refused,
        // and a frame may arrive over several calls.
        std::vector<unsigned char> pixels(static_cast<size_t>(camera_parameters.width * camera_parameters.height), 128);
        zeroslam_sensor_data_struct frame[3] = {};
        for (int i = 0; i < 3; ++i) {
            frame[i].measurement_length = camera_parameters.width * camera_parameters.height;
            frame[i].measurement_data = pixels.data();
            frame[i].timestamp = 1000;
        }
        frame[0].sensor_id = 1;
        frame[1].sensor_id = 1;
        REQUIRE(zeroslam_set_sensor_data(system, &frame[0], 2) == zeroslam_return_failure_invalid_sensor_data);
        frame[1].timestamp = 2000;
        REQUIRE(zeroslam_set_sensor_data(system, &frame[0], 2) == zeroslam_return_failure_invalid_sensor_data);
        frame[1].sensor_id = 2;
        frame[1].timestamp = 1000;
        frame[2].sensor_id = 2;
        frame[2].timestamp = 2000;
        REQUIRE(zeroslam_set_sensor_data(system, &frame[0], 3) == zeroslam_return_success);
        long long int timestamp = 0;
        REQUIRE(zeroslam_get_timestamp(system, &timestamp) == zeroslam_return_success);
        REQUIRE(timestamp == 1000);
        // The frame at 2000 waits for its first camera.
        frame[0].timestamp = 3000;
        REQUIRE(zeroslam_set_sensor_data(system, &frame[0], 1) == zeroslam_return_failure_invalid_sensor_data);
        frame[0].timestamp = 2000;
        REQUIRE(zeroslam_set_sensor_data(system, &frame[0], 1) == zeroslam_return_success);
        REQUIRE(zeroslam_get_timestamp(system, &timestamp) == zeroslam_return_success);
        REQUIRE(timestamp == 2000);
        REQUIRE(zeroslam_set_sensor_extrinsics(system, &extrinsics[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_success);
    }

    {
        // Any camera model through the model struct, each checked: its count, its parameters' range, finiteness, the size.
        zeroslam_system* system = nullptr;
        REQUIRE(zeroslam_create(&system) == zeroslam_return_success);
        REQUIRE(zeroslam_camera_model_parameter_count(zeroslam_camera_model_radial_tangential) == 12);
        REQUIRE(zeroslam_camera_model_parameter_count(zeroslam_camera_model_invalid) == 0);
        REQUIRE(std::strcmp(zeroslam_camera_model_enum_to_string(zeroslam_camera_model_double_sphere), "double sphere") == 0);
        zeroslam_sensor_parameters_camera_model_struct camera{};
        camera.width = 64;
        camera.height = 48;
        camera.model = zeroslam_camera_model_kannala_brandt;
        camera.parameters_length = 8;
        const double fisheye[8] = { 30.0, 30.0, 32.0, 24.0, 0.01, -0.002, 0.0, 0.0 };
        std::memcpy(&camera.parameters[0], &fisheye[0], sizeof(fisheye));
        zeroslam_sensor_rig_struct rig{};
        rig.type = zeroslam_sensor_camera;
        rig.sensor_id = 1;
        rig.parameters_length = static_cast<int>(sizeof(camera));
        rig.parameters_data = &camera;
        camera.parameters_length = 7;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_failure_invalid_rig_sensor);
        camera.parameters_length = 8;
        camera.parameters[0] = -30.0;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_failure_invalid_rig_sensor);
        camera.parameters[0] = std::numeric_limits<double>::quiet_NaN();
        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_failure_invalid_rig_sensor);
        camera.parameters[0] = 30.0;
        const int unknown_model = 99;
        std::memcpy(&camera.model, &unknown_model, sizeof(unknown_model));
        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_failure_invalid_rig_sensor);
        camera.model = zeroslam_camera_model_kannala_brandt;
        camera.width = 0;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_failure_invalid_rig_sensor);
        camera.width = 64;
        rig.parameters_length = static_cast<int>(sizeof(camera)) - 8;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_failure_invalid_rig_sensor);
        rig.parameters_length = static_cast<int>(sizeof(camera));
        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_success);
        std::vector<unsigned char> pixels(64 * 48, 128);
        zeroslam_sensor_data_struct data{};
        data.timestamp = 1;
        data.sensor_id = 1;
        data.measurement_length = 64 * 48;
        data.measurement_data = pixels.data();
        REQUIRE(zeroslam_set_sensor_data(system, &data, 1) == zeroslam_return_success);
        data.measurement_length = 64 * 47;
        data.timestamp = 2;
        REQUIRE(zeroslam_set_sensor_data(system, &data, 1) == zeroslam_return_failure_invalid_sensor_data);
        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_success);
    }

    {
        zeroslam_system* system = nullptr;
        REQUIRE(zeroslam_create(&system) == zeroslam_return_success);

        constexpr static const int width = 320;
        constexpr static const int height = 240;
        zeroslam_sensor_parameters_camera_struct camera_parameters{};
        camera_parameters.width = width;
        camera_parameters.height = height;
        camera_parameters.focal_x = 262.5;
        camera_parameters.focal_y = 262.5;
        camera_parameters.centre_x = 160.0;
        camera_parameters.centre_y = 120.0;
        zeroslam_sensor_rig_struct rig{};
        rig.type = zeroslam_sensor_camera;
        rig.sensor_id = 3;
        rig.parameters_length = static_cast<int>(sizeof(camera_parameters));
        rig.parameters_data = &camera_parameters;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_success);

        const char* const lines_on = "lines=on\n";
        REQUIRE(zeroslam_set_configuration(system, lines_on, static_cast<int>(std::strlen(lines_on))) == zeroslam_return_success);

        // The calibration of a camera of the rig is the rig's before any frame, and nothing for a sensor it does not hold.
        zeroslam_sensor_parameters_camera_model_struct calibrated{};
        const auto calibration_is_rig = [&calibrated, &camera_parameters]() {
            const double expected[12] = { camera_parameters.focal_x, camera_parameters.focal_y, camera_parameters.centre_x, camera_parameters.centre_y, camera_parameters.distortion[0], camera_parameters.distortion[1], camera_parameters.distortion[2], camera_parameters.distortion[3], camera_parameters.distortion[4], camera_parameters.distortion[5], camera_parameters.distortion[6], camera_parameters.distortion[7] };
            bool same = (calibrated.width == camera_parameters.width) && (calibrated.height == camera_parameters.height) && (calibrated.model == zeroslam_camera_model_radial_tangential) && (calibrated.parameters_length == 12);
            for (size_t index = 0; index < 12; ++index) {
                same = same && (calibrated.parameters[index] == expected[index]);
            }
            return same;
        };
        REQUIRE(zeroslam_get_camera_calibration(nullptr, 3, &calibrated) == zeroslam_return_failure_invalid_system);
        REQUIRE(zeroslam_get_camera_calibration(system, 3, nullptr) == zeroslam_return_failure_invalid_argument);
        REQUIRE(zeroslam_get_camera_calibration(system, 9, &calibrated) == zeroslam_return_failure_invalid_argument);
        REQUIRE(zeroslam_get_camera_calibration(system, 3, &calibrated) == zeroslam_return_success);
        REQUIRE(calibration_is_rig());

        std::vector<unsigned char> pixels(static_cast<size_t>(width * height));
        zeroslam_sensor_data_struct data{};
        data.sensor_id = 3;
        data.measurement_length = width * height;
        data.measurement_data = pixels.data();

        render_corridor(0.0, width, height, camera_parameters, pixels.data());
        data.timestamp = 1000;
        data.sensor_id = 9;
        REQUIRE(zeroslam_set_sensor_data(system, &data, 1) == zeroslam_return_failure_invalid_sensor_data);
        data.sensor_id = 3;
        data.measurement_length = 16;
        REQUIRE(zeroslam_set_sensor_data(system, &data, 1) == zeroslam_return_failure_invalid_sensor_data);
        data.measurement_length = width * height;

        REQUIRE(zeroslam_set_sensor_data(system, &data, 1) == zeroslam_return_success);
        REQUIRE(zeroslam_set_sensor_data(system, &data, 1) == zeroslam_return_failure_invalid_sensor_data);

        constexpr static const int frame_count = 20;
        for (int frame = 1; frame < frame_count; ++frame) {
            render_corridor(0.05 * static_cast<double>(frame), width, height, camera_parameters, pixels.data());
            data.timestamp = 1000 * (frame + 1);
            REQUIRE(zeroslam_set_sensor_data(system, &data, 1) == zeroslam_return_success);
        }

        long long int timestamp = 0;
        REQUIRE(zeroslam_get_timestamp(system, &timestamp) == zeroslam_return_success);
        REQUIRE(timestamp == 1000 * frame_count);

        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_failure_invalid_rig_sensor);

        // With the calibration off it stays the rig's after the frames too.
        REQUIRE(zeroslam_get_camera_calibration(system, 3, &calibrated) == zeroslam_return_success);
        REQUIRE(calibration_is_rig());

        zeroslam_pose_struct pose{};
        REQUIRE(zeroslam_get_pose(system, &pose) == zeroslam_return_success);
        REQUIRE(pose.timestamp == 1000 * frame_count);
        REQUIRE((pose.pose[0] != 0.0) || (pose.pose[1] != 0.0) || (pose.pose[2] != 0.0));
        zeroslam_pose_struct pose_first{};
        REQUIRE(zeroslam_get_pose_at_timestamp(system, &pose_first, 1000) == zeroslam_return_success);
        REQUIRE(pose_first.pose[0] == 0.0);
        REQUIRE(pose_first.pose[1] == 0.0);
        REQUIRE(pose_first.pose[2] == 0.0);
        REQUIRE(pose_first.pose[6] == 1.0);
        REQUIRE(zeroslam_get_pose_at_timestamp(system, &pose_first, 12345) == zeroslam_return_failure_invalid_argument);

        zeroslam_map_chunk_struct chunk{};
        chunk.points = nullptr;
        chunk.points_length = 0;
        REQUIRE(zeroslam_get_map_chunk(system, 0.0f, 0.0f, 0.0f, &chunk) == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(chunk.points_length > 0);
        std::vector<zeroslam_point_struct> points(static_cast<size_t>(chunk.points_length));
        chunk.points = points.data();
        chunk.points_length = static_cast<int>(points.size()) - 1;
        REQUIRE(zeroslam_get_map_chunk(system, 0.0f, 0.0f, 0.0f, &chunk) == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(chunk.points_length == static_cast<int>(points.size()));
        REQUIRE(zeroslam_get_map_chunk(system, 0.0f, 0.0f, 0.0f, &chunk) == zeroslam_return_success);
        REQUIRE(chunk.points_length == static_cast<int>(points.size()));
        REQUIRE(chunk.timestamp == 1000 * frame_count);
        for (const zeroslam_point_struct& point : points) {
            REQUIRE((point.x >= chunk.min_x) && (point.x <= chunk.max_x));
            REQUIRE((point.z >= chunk.min_z) && (point.z <= chunk.max_z));
        }

        zeroslam_map_lines_struct lines{};
        REQUIRE(zeroslam_get_map_lines(system, &lines) == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(lines.lines_length > 0);
        std::vector<zeroslam_line_struct> line_buffer(static_cast<size_t>(lines.lines_length));
        lines.lines = line_buffer.data();
        REQUIRE(zeroslam_get_map_lines(system, &lines) == zeroslam_return_success);
        REQUIRE(lines.lines_length == static_cast<int>(line_buffer.size()));

        zeroslam_map_edges_struct edges{};
        REQUIRE(zeroslam_get_map_edges(system, &edges) == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(edges.edges_length > 0);
        std::vector<zeroslam_edge_struct> edge_buffer(static_cast<size_t>(edges.edges_length));
        edges.edges = edge_buffer.data();
        REQUIRE(zeroslam_get_map_edges(system, &edges) == zeroslam_return_success);
        REQUIRE(edges.edges_length == static_cast<int>(edge_buffer.size()));
        for (const zeroslam_edge_struct& edge : edge_buffer) {
            REQUIRE((edge.type == zeroslam_edge_covisibility) || (edge.type == zeroslam_edge_loop));
            REQUIRE(edge.weight > 0);
            REQUIRE(edge.timestamp_a != edge.timestamp_b);
            zeroslam_pose_struct stored_pose{};
            REQUIRE(zeroslam_get_pose_at_timestamp(system, &stored_pose, edge.timestamp_a) == zeroslam_return_success);
            REQUIRE(zeroslam_get_pose_at_timestamp(system, &stored_pose, edge.timestamp_b) == zeroslam_return_success);
        }

        zeroslam_map_keyframes_struct keyframes{};
        REQUIRE(zeroslam_get_map_keyframes(system, &keyframes) == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(keyframes.keyframes_length >= 2);
        std::vector<long long int> keyframe_buffer(static_cast<size_t>(keyframes.keyframes_length));
        keyframes.keyframes = keyframe_buffer.data();
        REQUIRE(zeroslam_get_map_keyframes(system, &keyframes) == zeroslam_return_success);
        REQUIRE(keyframes.keyframes_length == static_cast<int>(keyframe_buffer.size()));
        for (size_t i = 0; i < keyframe_buffer.size(); ++i) {
            REQUIRE((i == 0) || (keyframe_buffer[i] > keyframe_buffer[i - 1]));
            zeroslam_pose_struct stored_pose{};
            REQUIRE(zeroslam_get_pose_at_timestamp(system, &stored_pose, keyframe_buffer[i]) == zeroslam_return_success);
        }
        REQUIRE(zeroslam_get_map_lines(nullptr, &lines) == zeroslam_return_failure_invalid_system);
        REQUIRE(zeroslam_get_map_edges(system, nullptr) == zeroslam_return_failure_invalid_argument);

        // The voxel map holds the map's finite landmarks near the newest keyframe in sorted, distinct, occupied voxels, each
        // holding the landmarks inside its cube.
        zeroslam_map_voxels_struct voxels{};
        REQUIRE(zeroslam_get_map_voxels(system, &voxels) == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(voxels.voxels_length > 0);
        std::vector<zeroslam_voxel_struct> voxel_buffer(static_cast<size_t>(voxels.voxels_length));
        voxels.voxels = voxel_buffer.data();
        REQUIRE(zeroslam_get_map_voxels(system, &voxels) == zeroslam_return_success);
        REQUIRE(voxels.voxels_length == static_cast<int>(voxel_buffer.size()));
        REQUIRE(voxels.voxel_size > 0.0f);
        int voxel_points = 0;
        for (size_t i = 0; i < voxel_buffer.size(); ++i) {
            REQUIRE(voxel_buffer[i].points > 0);
            voxel_points += voxel_buffer[i].points;
            if (i > 0) {
                const zeroslam_voxel_struct& previous = voxel_buffer[i - 1];
                const zeroslam_voxel_struct& current = voxel_buffer[i];
                REQUIRE((previous.x < current.x) || ((previous.x == current.x) && ((previous.y < current.y) || ((previous.y == current.y) && (previous.z < current.z)))));
            }
        }
        zeroslam_map_chunk_struct landmark_count{};
        REQUIRE(zeroslam_get_map_chunk(system, 0.0f, 0.0f, 0.0f, &landmark_count) == zeroslam_return_failure_insufficient_data_length);
        REQUIRE((voxel_points > 0) && (voxel_points <= landmark_count.points_length));
        std::vector<zeroslam_point_struct> landmark_buffer(static_cast<size_t>(landmark_count.points_length));
        landmark_count.points = landmark_buffer.data();
        REQUIRE(zeroslam_get_map_chunk(system, 0.0f, 0.0f, 0.0f, &landmark_count) == zeroslam_return_success);
        int inside = 0;
        for (const zeroslam_point_struct& point : landmark_buffer) {
            for (const zeroslam_voxel_struct& voxel : voxel_buffer) {
                // Note: The api's corners and points are floats, so allow a hair at each face.
                const float slack = 1.0e-4f * voxels.voxel_size;
                if ((point.x >= voxel.x - slack) && (point.x < voxel.x + voxels.voxel_size + slack) && (point.y >= voxel.y - slack) && (point.y < voxel.y + voxels.voxel_size + slack) && (point.z >= voxel.z - slack) && (point.z < voxel.z + voxels.voxel_size + slack)) {
                    ++inside;
                    break;
                }
            }
        }
        REQUIRE(inside >= voxel_points);
        zeroslam_map_voxels_struct short_voxels{};
        short_voxels.voxels = voxel_buffer.data();
        short_voxels.voxels_length = voxels.voxels_length - 1;
        REQUIRE((voxels.voxels_length < 2) || (zeroslam_get_map_voxels(system, &short_voxels) == zeroslam_return_failure_insufficient_data_length));
        REQUIRE(zeroslam_get_map_voxels(nullptr, &voxels) == zeroslam_return_failure_invalid_system);
        REQUIRE(zeroslam_get_map_voxels(system, nullptr) == zeroslam_return_failure_invalid_argument);

        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_success);
    }

    {
        zeroslam_system* system = nullptr;
        REQUIRE(zeroslam_create(&system) == zeroslam_return_success);

        constexpr static const int width = 320;
        constexpr static const int height = 240;
        zeroslam_sensor_parameters_camera_struct camera_parameters{};
        camera_parameters.width = width;
        camera_parameters.height = height;
        camera_parameters.focal_x = 262.5;
        camera_parameters.focal_y = 262.5;
        camera_parameters.centre_x = 160.0;
        camera_parameters.centre_y = 120.0;
        zeroslam_sensor_rig_struct rig{};
        rig.type = zeroslam_sensor_camera;
        rig.sensor_id = 3;
        rig.parameters_length = static_cast<int>(sizeof(camera_parameters));
        rig.parameters_data = &camera_parameters;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_success);

        constexpr static const int frame_count = 21;
        std::vector<std::vector<unsigned char>> frames(static_cast<size_t>(frame_count), std::vector<unsigned char>(static_cast<size_t>(width * height), 128));
        for (int frame = 1; frame < frame_count; ++frame) {
            render_corridor(0.05 * static_cast<double>(frame - 1), width, height, camera_parameters, frames[static_cast<size_t>(frame)].data());
        }
        std::vector<zeroslam_sensor_data_struct> batch(static_cast<size_t>(frame_count));
        for (int frame = 0; frame < frame_count; ++frame) {
            zeroslam_sensor_data_struct& data = batch[static_cast<size_t>(frame)];
            data.sensor_id = 3;
            data.timestamp = 1000 * (frame + 1);
            data.measurement_length = width * height;
            data.measurement_data = frames[static_cast<size_t>(frame)].data();
        }

        batch[2].measurement_length = 16;
        REQUIRE(zeroslam_set_sensor_data(system, batch.data(), 3) == zeroslam_return_failure_invalid_sensor_data);
        batch[2].measurement_length = width * height;
        batch[2].timestamp = batch[1].timestamp;
        REQUIRE(zeroslam_set_sensor_data(system, batch.data(), 3) == zeroslam_return_failure_invalid_sensor_data);
        batch[2].timestamp = 3000;
        long long int timestamp = -1;
        REQUIRE(zeroslam_get_timestamp(system, &timestamp) == zeroslam_return_success);
        REQUIRE(timestamp == 0);
        zeroslam_pose_struct pose{};
        REQUIRE(zeroslam_get_pose(system, &pose) == zeroslam_return_failure_invalid_argument);

        REQUIRE(zeroslam_set_sensor_data(system, batch.data(), frame_count) == zeroslam_return_success);
        REQUIRE(zeroslam_get_timestamp(system, &timestamp) == zeroslam_return_success);
        REQUIRE(timestamp == 1000 * frame_count);

        zeroslam_map_keyframes_struct keyframes{};
        REQUIRE(zeroslam_get_map_keyframes(system, &keyframes) == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(keyframes.keyframes_length >= 2);

        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_success);
    }

    {
        // A stereo rig, the second camera 20 cm to the right of the first, initialises from its first frame and poses the
        // camera in metres: the corridor goes by at 5 cm a frame, and the primary camera's poses are what is reported.
        constexpr static const int width = 320;
        constexpr static const int height = 240;
        constexpr static const double baseline = 0.2;
        zeroslam::system system;
        REQUIRE(system.is_valid());
        zeroslam_sensor_parameters_camera_struct camera_parameters{};
        camera_parameters.width = width;
        camera_parameters.height = height;
        camera_parameters.focal_x = 262.5;
        camera_parameters.focal_y = 262.5;
        camera_parameters.centre_x = 160.0;
        camera_parameters.centre_y = 120.0;
        zeroslam_sensor_rig_struct rig[2] = {};
        for (int i = 0; i < 2; ++i) {
            rig[i].type = zeroslam_sensor_camera;
            rig[i].sensor_id = 1 + i;
            rig[i].parameters_length = static_cast<int>(sizeof(camera_parameters));
            rig[i].parameters_data = &camera_parameters;
        }
        REQUIRE(system.set_sensor_rig(&rig[0], 2) == zeroslam_return_success);
        zeroslam_sensor_extrinsics_struct extrinsics[2] = {};
        extrinsics[0].sensor_id = 1;
        extrinsics[0].pose[6] = 1.0;
        extrinsics[1].sensor_id = 2;
        extrinsics[1].pose[0] = baseline;
        extrinsics[1].pose[6] = 1.0;
        REQUIRE(system.set_sensor_extrinsics(&extrinsics[0], 2) == zeroslam_return_success);
        std::vector<unsigned char> left(static_cast<size_t>(width * height));
        std::vector<unsigned char> right(static_cast<size_t>(width * height));
        constexpr static const int frame_count = 20;
        for (int frame = 0; frame < frame_count; ++frame) {
            const double position = 0.05 * static_cast<double>(frame);
            render_corridor(position, width, height, camera_parameters, left.data());
            render_corridor(position + baseline, width, height, camera_parameters, right.data());
            zeroslam_sensor_data_struct data[2] = {};
            data[0].timestamp = 1000 * (frame + 1);
            data[0].sensor_id = 1;
            data[0].measurement_length = width * height;
            data[0].measurement_data = left.data();
            data[1] = data[0];
            data[1].sensor_id = 2;
            data[1].measurement_data = right.data();
            REQUIRE(system.set_sensor_data(&data[0], 2) == zeroslam_return_success);
        }
        zeroslam_pose_struct first{};
        REQUIRE(system.get_pose_at_timestamp(&first, 1000) == zeroslam_return_success);
        REQUIRE(first.pose[0] == 0.0);
        REQUIRE(first.pose[6] == 1.0);
        zeroslam_pose_struct last{};
        REQUIRE(system.get_pose(&last) == zeroslam_return_success);
        REQUIRE(last.timestamp == 1000 * frame_count);
        const double travelled = 0.05 * static_cast<double>(frame_count - 1);
        REQUIRE(math::abs(last.pose[0] - travelled) < 0.05 * travelled);
        REQUIRE(math::abs(last.pose[1]) < 0.02);
        REQUIRE(math::abs(last.pose[2]) < 0.02);
    }

    {
        // A depth camera in the rig: validated against its camera, and its frames paired with the camera's by time.
        zeroslam_system* system = nullptr;
        REQUIRE(zeroslam_create(&system) == zeroslam_return_success);

        constexpr static const int width = 320;
        constexpr static const int height = 240;
        zeroslam_sensor_parameters_camera_struct camera_parameters{};
        camera_parameters.width = width;
        camera_parameters.height = height;
        camera_parameters.focal_x = 262.5;
        camera_parameters.focal_y = 262.5;
        camera_parameters.centre_x = 160.0;
        camera_parameters.centre_y = 120.0;
        zeroslam_sensor_parameters_depth_struct depth_parameters{};
        depth_parameters.camera_id = 3;
        depth_parameters.encoding = zeroslam_depth_encoding_float_32;
        depth_parameters.measure = zeroslam_depth_measure_axial;
        depth_parameters.registration = zeroslam_depth_registration_camera;
        depth_parameters.scale = 1.0;
        depth_parameters.minimum_depth = 0.1;
        depth_parameters.maximum_depth = 10.0;
        depth_parameters.noise_quadratic = 1.425e-3;
        depth_parameters.pairing_tolerance = 400;
        zeroslam_sensor_rig_struct rig[2]{};
        rig[0].type = zeroslam_sensor_camera;
        rig[0].sensor_id = 3;
        rig[0].parameters_length = static_cast<int>(sizeof(camera_parameters));
        rig[0].parameters_data = &camera_parameters;
        rig[1].type = zeroslam_sensor_depth;
        rig[1].sensor_id = 4;
        rig[1].parameters_length = static_cast<int>(sizeof(depth_parameters));
        rig[1].parameters_data = &depth_parameters;

        const auto rejected = [&](zeroslam_sensor_parameters_depth_struct changed) {
            const zeroslam_sensor_parameters_depth_struct kept = depth_parameters;
            depth_parameters = changed;
            const bool result = zeroslam_set_sensor_rig(system, &rig[0], 2) == zeroslam_return_failure_invalid_rig_sensor;
            depth_parameters = kept;
            return result;
        };
        zeroslam_sensor_parameters_depth_struct changed = depth_parameters;
        changed.camera_id = 9;
        REQUIRE(rejected(changed));
        changed = depth_parameters;
        changed.encoding = 7;
        REQUIRE(rejected(changed));
        changed = depth_parameters;
        changed.measure = 2;
        REQUIRE(rejected(changed));
        changed = depth_parameters;
        changed.registration = 1;
        REQUIRE(rejected(changed));
        changed = depth_parameters;
        changed.encoding = zeroslam_depth_encoding_unsigned_16;
        changed.scale = 0.0;
        REQUIRE(rejected(changed));
        changed = depth_parameters;
        changed.maximum_depth = changed.minimum_depth;
        REQUIRE(rejected(changed));
        changed = depth_parameters;
        changed.noise_quadratic = 0.0;
        REQUIRE(rejected(changed));
        changed = depth_parameters;
        changed.noise_linear = -1.0;
        REQUIRE(rejected(changed));
        changed = depth_parameters;
        changed.pairing_tolerance = -1;
        REQUIRE(rejected(changed));
        rig[1].parameters_length = 8;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
        rig[1].parameters_length = static_cast<int>(sizeof(depth_parameters));
        rig[1].sensor_id = 3;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
        rig[1].sensor_id = 4;
        // A depth camera alone is no rig, and two depth cameras on one camera are one too many.
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[1], 1) == zeroslam_return_failure_invalid_rig_sensor);
        zeroslam_sensor_rig_struct three[3] = { rig[0], rig[1], rig[1] };
        three[2].sensor_id = 5;
        REQUIRE(zeroslam_set_sensor_rig(system, &three[0], 3) == zeroslam_return_failure_invalid_rig_sensor);
        // A depth camera goes with a single camera: a rig of several cameras does not take one yet.
        zeroslam_sensor_rig_struct stereo[3] = { rig[0], rig[0], rig[1] };
        stereo[1].sensor_id = 6;
        REQUIRE(zeroslam_set_sensor_rig(system, &stereo[0], 3) == zeroslam_return_failure_invalid_rig_sensor);
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 2) == zeroslam_return_success);

        constexpr static const int frame_count = 20;
        std::vector<unsigned char> pixels(static_cast<size_t>(width * height));
        std::vector<float> metres(static_cast<size_t>(width * height));
        render_corridor_depth(width, height, camera_parameters, metres.data());
        zeroslam_sensor_data_struct data[2]{};
        data[0].sensor_id = 4;
        data[0].measurement_length = width * height * static_cast<int>(sizeof(float));
        data[0].measurement_data = metres.data();
        data[1].sensor_id = 3;
        data[1].measurement_length = width * height;
        data[1].measurement_data = pixels.data();

        // A depth frame is the camera's size in its encoding.
        data[0].timestamp = 1000;
        data[1].timestamp = 1000;
        data[0].measurement_length = width * height;
        REQUIRE(zeroslam_set_sensor_data(system, &data[0], 2) == zeroslam_return_failure_invalid_sensor_data);
        data[0].measurement_length = width * height * static_cast<int>(sizeof(float));

        for (int frame = 0; frame < frame_count; ++frame) {
            render_corridor(0.05 * static_cast<double>(frame), width, height, camera_parameters, pixels.data());
            // The depth frames come a little before or after their camera frames, within the pairing tolerance; given with
            // the camera frame, either order pairs.
            data[0].timestamp = (1000 * (frame + 1)) + (((frame % 2) == 0) ? 150 : -150);
            data[1].timestamp = 1000 * (frame + 1);
            if ((frame % 3) == 0) {
                const zeroslam_sensor_data_struct swapped[2] = { data[1], data[0] };
                REQUIRE(zeroslam_set_sensor_data(system, &swapped[0], 2) == zeroslam_return_success);
            }
            else {
                REQUIRE(zeroslam_set_sensor_data(system, &data[0], 2) == zeroslam_return_success);
            }
        }
        // Each sensor's timestamps increase on their own: a depth frame cannot repeat one.
        data[0].timestamp = (1000 * frame_count) - 150;
        REQUIRE(zeroslam_set_sensor_data(system, &data[0], 1) == zeroslam_return_failure_invalid_sensor_data);

        long long int timestamp = 0;
        REQUIRE(zeroslam_get_timestamp(system, &timestamp) == zeroslam_return_success);
        REQUIRE(timestamp == 1000 * frame_count);

        // The depth gives the map metric scale: the camera moved 5 cm a frame along x, and the map has it there.
        zeroslam_pose_struct pose{};
        REQUIRE(zeroslam_get_pose_at_timestamp(system, &pose, 1000) == zeroslam_return_success);
        REQUIRE((pose.pose[0] == 0.0) && (pose.pose[1] == 0.0) && (pose.pose[2] == 0.0));
        REQUIRE(zeroslam_get_pose(system, &pose) == zeroslam_return_success);
        REQUIRE(pose.timestamp == 1000 * frame_count);
        const double travelled = 0.05 * static_cast<double>(frame_count - 1);
        std::fprintf(stdout, "RGB-D corridor: the last frame at (%.4f, %.4f, %.4f), the camera at (%.4f, 0, 0).\n", pose.pose[0], pose.pose[1], pose.pose[2], travelled);
        REQUIRE(math::abs(pose.pose[0] - travelled) < 0.02);
        REQUIRE(math::abs(pose.pose[1]) < 0.02);
        REQUIRE(math::abs(pose.pose[2]) < 0.02);

        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_success);
    }

    {
        // Sixteen bit depth, and depth frames given ahead of their camera frames: the nearest within the tolerance pairs,
        // and a camera frame with no depth frame near it is processed without one.
        zeroslam_system* system = nullptr;
        REQUIRE(zeroslam_create(&system) == zeroslam_return_success);

        constexpr static const int width = 320;
        constexpr static const int height = 240;
        zeroslam_sensor_parameters_camera_struct camera_parameters{};
        camera_parameters.width = width;
        camera_parameters.height = height;
        camera_parameters.focal_x = 262.5;
        camera_parameters.focal_y = 262.5;
        camera_parameters.centre_x = 160.0;
        camera_parameters.centre_y = 120.0;
        zeroslam_sensor_parameters_depth_struct depth_parameters{};
        depth_parameters.camera_id = 3;
        depth_parameters.encoding = zeroslam_depth_encoding_unsigned_16;
        depth_parameters.measure = zeroslam_depth_measure_axial;
        depth_parameters.registration = zeroslam_depth_registration_camera;
        depth_parameters.scale = 1.0 / 5000.0;
        depth_parameters.minimum_depth = 0.1;
        depth_parameters.maximum_depth = 10.0;
        depth_parameters.noise_quadratic = 1.425e-3;
        depth_parameters.pairing_tolerance = 200;
        zeroslam_sensor_rig_struct rig[2]{};
        rig[0].type = zeroslam_sensor_camera;
        rig[0].sensor_id = 3;
        rig[0].parameters_length = static_cast<int>(sizeof(camera_parameters));
        rig[0].parameters_data = &camera_parameters;
        rig[1].type = zeroslam_sensor_depth;
        rig[1].sensor_id = 4;
        rig[1].parameters_length = static_cast<int>(sizeof(depth_parameters));
        rig[1].parameters_data = &depth_parameters;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 2) == zeroslam_return_success);

        std::vector<float> metres(static_cast<size_t>(width * height));
        render_corridor_depth(width, height, camera_parameters, metres.data());
        std::vector<unsigned short> values(metres.size());
        for (size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<unsigned short>((5000.0 * static_cast<double>(metres[index])) + 0.5);
        }
        std::vector<unsigned char> pixels(static_cast<size_t>(width * height));
        constexpr static const int frame_count = 20;
        for (int frame = 0; frame < frame_count; ++frame) {
            render_corridor(0.05 * static_cast<double>(frame), width, height, camera_parameters, pixels.data());
            // Two depth frames come ahead of each camera frame, the second nearer it; the fifth camera frame has none near.
            zeroslam_sensor_data_struct depth_data[2]{};
            for (int copy = 0; copy < 2; ++copy) {
                depth_data[copy].sensor_id = 4;
                depth_data[copy].timestamp = (1000 * (frame + 1)) - ((frame == 4) ? 700 : 0) - ((copy == 0) ? 300 : 100);
                depth_data[copy].measurement_length = width * height * static_cast<int>(sizeof(unsigned short));
                depth_data[copy].measurement_data = values.data();
            }
            REQUIRE(zeroslam_set_sensor_data(system, &depth_data[0], 2) == zeroslam_return_success);
            zeroslam_sensor_data_struct data{};
            data.sensor_id = 3;
            data.timestamp = 1000 * (frame + 1);
            data.measurement_length = width * height;
            data.measurement_data = pixels.data();
            REQUIRE(zeroslam_set_sensor_data(system, &data, 1) == zeroslam_return_success);
        }
        zeroslam_pose_struct pose{};
        REQUIRE(zeroslam_get_pose(system, &pose) == zeroslam_return_success);
        const double travelled = 0.05 * static_cast<double>(frame_count - 1);
        REQUIRE(math::abs(pose.pose[0] - travelled) < 0.02);

        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_success);
    }

    {
        // A camera with a depth camera, an imu and a wheel odometer: each takes its own id, and no two sensors share one.
        const int width = 64;
        const int height = 48;
        zeroslam_sensor_parameters_camera_struct camera{};
        camera.width = width;
        camera.height = height;
        camera.focal_x = 50.0;
        camera.focal_y = 50.0;
        camera.centre_x = 32.0;
        camera.centre_y = 24.0;
        zeroslam_sensor_parameters_depth_struct depth = zeroslam::registered_depth(1, zeroslam_depth_encoding_float_32, 1.0);
        zeroslam_sensor_parameters_imu_struct imu{ 1.6968e-04, 2.0e-03, 1.9393e-05, 3.0e-03, 200.0, 0.0, 0.0 };
        zeroslam_sensor_parameters_wheel_odometry_struct odometer{};
        odometer.model = zeroslam_odometry_differential_drive;
        odometer.parameters[0] = 0.1;
        odometer.parameters[1] = 0.1;
        odometer.parameters[2] = 0.5;
        const double noise[8] = { 0.01, 0.002, 0.005, 0.002, 0.01, 0.005, 0.001, 0.0005 };
        for (size_t i = 0; i < 8; ++i) {
            odometer.noise[i] = noise[i];
        }
        zeroslam_sensor_rig_struct rig[4] = {
            { zeroslam_sensor_camera, 1, static_cast<int>(sizeof(camera)), &camera },
            { zeroslam_sensor_depth, 2, static_cast<int>(sizeof(depth)), &depth },
            { zeroslam_sensor_imu, 3, static_cast<int>(sizeof(imu)), &imu },
            { zeroslam_sensor_wheel_odometry, 4, static_cast<int>(sizeof(odometer)), &odometer }
        };
        zeroslam_system* system = nullptr;
        REQUIRE(zeroslam_create(&system) == zeroslam_return_success);
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 4) == zeroslam_return_success);
        const int shared[][2] = { { 2, 3 }, { 2, 4 }, { 3, 4 }, { 4, 1 }, { 3, 1 } };
        for (const int (&pair)[2] : shared) {
            const int kept = rig[pair[0] - 1].sensor_id;
            rig[pair[0] - 1].sensor_id = rig[pair[1] - 1].sensor_id;
            REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 4) == zeroslam_return_failure_invalid_rig_sensor);
            rig[pair[0] - 1].sensor_id = kept;
        }

        // Without the depth camera. The imu's and the odometer's extrinsics may come alone, after the camera's, and leave it be.
        rig[1] = rig[0];
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[1], 3) == zeroslam_return_success);
        zeroslam_sensor_extrinsics_struct extrinsics[3] = {};
        for (int i = 0; i < 3; ++i) {
            extrinsics[i].sensor_id = (i == 0) ? 1 : (i + 2);
            extrinsics[i].refinement = zeroslam_extrinsic_fixed;
            extrinsics[i].pose[i] = 0.25;
            extrinsics[i].pose[6] = 1.0;
        }
        REQUIRE(zeroslam_set_sensor_extrinsics(system, &extrinsics[0], 1) == zeroslam_return_success);
        REQUIRE(zeroslam_set_sensor_extrinsics(system, &extrinsics[1], 1) == zeroslam_return_success);
        REQUIRE(zeroslam_set_sensor_extrinsics(system, &extrinsics[2], 1) == zeroslam_return_success);
        zeroslam_sensor_extrinsics_struct reported[2] = {};
        int reported_length = 2;
        REQUIRE(zeroslam_get_sensor_extrinsics(system, &reported[0], &reported_length) == zeroslam_return_success);
        REQUIRE((reported_length == 2) && (reported[0].sensor_id == 1) && (math::abs(reported[0].pose[0] - 0.25) < 1.0e-12) && (reported[1].sensor_id == 3) && (math::abs(reported[1].pose[1] - 0.25) < 1.0e-12));
        zeroslam_odometry_calibration_struct calibration{};
        REQUIRE(zeroslam_get_odometry_calibration(system, &calibration) == zeroslam_return_success);
        REQUIRE((calibration.sensor_id == 4) && (math::abs(calibration.pose[2] - 0.25) < 1.0e-12));

        // The odometer's samples are checked whole with a batch, as the imu's are: none of a batch is taken when one is refused.
        zeroslam_sensor_measurement_imu_struct still{};
        still.linear_acceleration[1] = -9.81;
        double wheels[3] = { 0.0, 0.0, 0.0 };
        std::vector<unsigned char> pixels(static_cast<size_t>(width * height), 128);
        zeroslam_sensor_data_struct batch[3] = {
            { 1000, 3, static_cast<int>(sizeof(still)), &still },
            { 1000, 4, 16, &wheels[0] },
            { 2000, 1, width * height, pixels.data() }
        };
        REQUIRE(zeroslam_set_sensor_data(system, &batch[0], 3) == zeroslam_return_failure_invalid_sensor_data);
        batch[1].measurement_length = static_cast<int>(sizeof(wheels));
        for (int frame = 0; frame < 5; ++frame) {
            REQUIRE(zeroslam_set_sensor_data(system, &batch[0], 3) == zeroslam_return_success);
            for (zeroslam_sensor_data_struct& entry : batch) {
                entry.timestamp += 2000;
            }
        }

        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_success);
    }

    return EXIT_SUCCESS;
}
