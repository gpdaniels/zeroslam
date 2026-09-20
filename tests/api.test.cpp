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

#include "simulation/scene.hpp"
#include "zeroslam/zeroslam.hpp"

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

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

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
        const char* const anchor = "anchor=affine_illumination\nanchor_refresh=30\n";
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
        REQUIRE(std::strstr(buffer.data(), "anchor_refresh=30\n") != nullptr);
        const char* const wavelet = "flow=wavelet\nwavelet_window=3\nwavelet_levels=5\nwavelet_robust=on\nwavelet_undecimated=on\nwavelet_seed=klt_fallback\nline_pose=on\nline_angle=25\nsolver=square_root\nsolver_precision=single\nglobal_adjustment=10\n";
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
        const char* const bad_wavelet[9] = { "flow=phase\n", "wavelet_window=0\n", "wavelet_levels=9\n", "wavelet_seed=maybe\n", "line_pose=yes\n", "line_angle=90\n", "solver=sparse\n", "solver_precision=half\n", "global_adjustment=0\n" };
        for (int index = 0; index < 9; ++index) {
            REQUIRE(zeroslam_set_configuration(system, bad_wavelet[index], static_cast<int>(std::strlen(bad_wavelet[index]))) == zeroslam_return_failure_invalid_configuration);
        }
        const char* const wavelet_off = "flow=intensity\nwavelet_robust=off\nwavelet_undecimated=off\nwavelet_seed=rest\nline_pose=off\nline_angle=off\nsolver=dense_schur\nsolver_precision=double\nglobal_adjustment=off\n";
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

        zeroslam_sensor_rig_struct two[2] = { rig, rig };
        two[1].sensor_id = 2;
        REQUIRE(zeroslam_set_sensor_rig(system, &two[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
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

        simulation::random_stream random(2026);
        std::vector<unsigned char> pixels(static_cast<size_t>(width * height));
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                const double value = 127.0 + 60.0 * math::sin(0.11 * static_cast<double>(column)) * math::cos(0.13 * static_cast<double>(row)) + 40.0 * random.uniform();
                pixels[static_cast<size_t>(row * width + column)] = static_cast<unsigned char>(math::max(0.0, math::min(255.0, value)));
            }
        }

        zeroslam_sensor_data_struct data{};
        data.sensor_id = 3;
        data.measurement_length = width * height;
        data.measurement_data = pixels.data();

        data.timestamp = 1000;
        data.sensor_id = 9;
        REQUIRE(zeroslam_set_sensor_data(system, &data, 1) == zeroslam_return_failure_invalid_sensor_data);
        data.sensor_id = 3;
        data.measurement_length = 16;
        REQUIRE(zeroslam_set_sensor_data(system, &data, 1) == zeroslam_return_failure_invalid_sensor_data);
        data.measurement_length = width * height;

        REQUIRE(zeroslam_set_sensor_data(system, &data, 1) == zeroslam_return_success);
        data.timestamp = 2000;
        REQUIRE(zeroslam_set_sensor_data(system, &data, 1) == zeroslam_return_success);
        data.timestamp = 2000;
        REQUIRE(zeroslam_set_sensor_data(system, &data, 1) == zeroslam_return_failure_invalid_sensor_data);

        long long int timestamp = 0;
        REQUIRE(zeroslam_get_timestamp(system, &timestamp) == zeroslam_return_success);
        REQUIRE(timestamp == 2000);

        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_failure_invalid_rig_sensor);

        zeroslam_pose_struct pose{};
        REQUIRE(zeroslam_get_pose(system, &pose) == zeroslam_return_success);
        REQUIRE((pose.timestamp == 1000) || (pose.timestamp == 2000));
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
        const zeroslam_return_enum chunk_result = zeroslam_get_map_chunk(system, 0.0f, 0.0f, 0.0f, &chunk);
        if (chunk_result == zeroslam_return_success) {
            REQUIRE(chunk.points_length == 0);
        }
        else {
            REQUIRE(chunk_result == zeroslam_return_failure_insufficient_data_length);
            REQUIRE(chunk.points_length > 0);
            std::vector<zeroslam_point_struct> points(static_cast<size_t>(chunk.points_length));
            chunk.points = points.data();
            chunk.points_length = static_cast<int>(points.size());
            REQUIRE(zeroslam_get_map_chunk(system, 0.0f, 0.0f, 0.0f, &chunk) == zeroslam_return_success);
            REQUIRE(chunk.points_length == static_cast<int>(points.size()));
        }

        zeroslam_map_lines_struct lines{};
        const zeroslam_return_enum lines_result = zeroslam_get_map_lines(system, &lines);
        if (lines_result == zeroslam_return_success) {
            REQUIRE(lines.lines_length == 0);
        }
        else {
            REQUIRE(lines_result == zeroslam_return_failure_insufficient_data_length);
            std::vector<zeroslam_line_struct> line_buffer(static_cast<size_t>(lines.lines_length));
            lines.lines = line_buffer.data();
            REQUIRE(zeroslam_get_map_lines(system, &lines) == zeroslam_return_success);
            REQUIRE(lines.lines_length == static_cast<int>(line_buffer.size()));
        }
        zeroslam_map_edges_struct edges{};
        const zeroslam_return_enum edges_result = zeroslam_get_map_edges(system, &edges);
        if (edges_result == zeroslam_return_success) {
            REQUIRE(edges.edges_length == 0);
        }
        else {
            REQUIRE(edges_result == zeroslam_return_failure_insufficient_data_length);
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
        }
        zeroslam_map_keyframes_struct keyframes{};
        const zeroslam_return_enum keyframes_result = zeroslam_get_map_keyframes(system, &keyframes);
        if (keyframes_result == zeroslam_return_success) {
            REQUIRE(keyframes.keyframes_length == 0);
        }
        else {
            REQUIRE(keyframes_result == zeroslam_return_failure_insufficient_data_length);
            std::vector<long long int> keyframe_buffer(static_cast<size_t>(keyframes.keyframes_length));
            keyframes.keyframes = keyframe_buffer.data();
            REQUIRE(zeroslam_get_map_keyframes(system, &keyframes) == zeroslam_return_success);
            for (size_t i = 0; i < keyframe_buffer.size(); ++i) {
                REQUIRE((i == 0) || (keyframe_buffer[i] > keyframe_buffer[i - 1]));
                zeroslam_pose_struct stored_pose{};
                REQUIRE(zeroslam_get_pose_at_timestamp(system, &stored_pose, keyframe_buffer[i]) == zeroslam_return_success);
            }
        }
        REQUIRE(zeroslam_get_map_lines(nullptr, &lines) == zeroslam_return_failure_invalid_system);
        REQUIRE(zeroslam_get_map_edges(system, nullptr) == zeroslam_return_failure_invalid_argument);

        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_success);
    }

    return EXIT_SUCCESS;
}
