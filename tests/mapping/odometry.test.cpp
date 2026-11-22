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

#include "mapping/odometry.hpp"

#include "image/image.hpp"
#include "mapping/rig.hpp"
#include "math/lie.hpp"
#include "optimisation/edges/odometry.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertices/pose.hpp"
#include "sensor/camera.hpp"
#include "sensor/odometry/ackermann.hpp"
#include "sensor/odometry/model.hpp"
#include "simulation/ground_vehicle.hpp"
#include "slam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
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
    constexpr static const double focal = 250.0;

    // The camera on the vehicle, looking forward along the odometry frame's x axis, pitched 5 degrees down, 1.5 m ahead of the
    // rear axle and 1.3 m up: the camera's pose in the odometry frame.
    math::se3<double> odometry_from_camera() {
        const math::so3<double> axes(math::matrix<double, 3, 3>({ { 0.0, 0.0, 1.0 }, { -1.0, 0.0, 0.0 }, { 0.0, -1.0, 0.0 } }));
        return math::se3<double>(math::so3<double>::exp({ { 0.0, 0.087, 0.0 } }) * axes, { { 1.5, 0.0, 1.3 } });
    }

    // The inside of a yard 60 m square with walls 8 m high, the vehicle's world (z up, the floor at z = 0), every surface
    // textured in hashed 50 cm cells, seen by a pinhole camera posed world to camera.
    void render_yard(const math::se3<double>& pose, image::image& frame) {
        const math::matrix<double, 3, 3> camera_to_world = math::transpose(pose.rotation().get_matrix());
        const math::matrix<double, 3, 1> centre = -(camera_to_world * pose.translation());
        const double lower[3] = { -30.0, -30.0, 0.0 };
        const double upper[3] = { 30.0, 30.0, 8.0 };
        unsigned char* const pixels = frame.get_data();
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                const math::matrix<double, 3, 1> ray_camera({ (static_cast<double>(column) + 0.5 - (0.5 * width)) / focal, (static_cast<double>(row) + 0.5 - (0.5 * height)) / focal, 1.0 });
                const math::matrix<double, 3, 1> ray = camera_to_world * ray_camera;
                double nearest = 1.0e30;
                int axis = 0;
                for (int a = 0; a < 3; ++a) {
                    if (std::abs(ray[static_cast<size_t>(a)]) < 1.0e-12) {
                        continue;
                    }
                    const double bound = (ray[static_cast<size_t>(a)] > 0.0) ? upper[a] : lower[a];
                    const double t = (bound - centre[static_cast<size_t>(a)]) / ray[static_cast<size_t>(a)];
                    if ((t > 0.0) && (t < nearest)) {
                        nearest = t;
                        axis = a;
                    }
                }
                const math::matrix<double, 3, 1> hit = centre + (ray * nearest);
                unsigned long long hash = static_cast<unsigned long long>(axis + 1) * 83492791ull;
                for (int a = 0; a < 3; ++a) {
                    if (a == axis) {
                        hash ^= (hit[static_cast<size_t>(a)] > 0.0) ? 0x9e3779b9ull : 0x7f4a7c15ull;
                        continue;
                    }
                    hash = (hash ^ static_cast<unsigned long long>(static_cast<long long>(std::floor(2.0 * hit[static_cast<size_t>(a)])))) * 0x100000001b3ull;
                }
                hash ^= hash >> 13;
                hash *= 0x5bd1e995ull;
                hash ^= hash >> 15;
                pixels[static_cast<size_t>((row * width) + column)] = static_cast<unsigned char>(40ull + (hash % 176ull));
            }
        }
    }

    sensor::odometry::model rear_axle() {
        return sensor::odometry::model(sensor::odometry::ackermann(sensor::odometry::ackermann::measured_wheel::rear_axle));
    }

    mapping::odometry::settings declared_settings(const double radius, const double offset, const bool free) {
        mapping::odometry::settings settings;
        settings.sensor_id = 2;
        settings.kinematics = rear_axle();
        const double values[4] = { radius, 2.7, 1.0, offset };
        settings.calibration = sensor::odometry::calibration(settings.kinematics, &values[0], 4);
        settings.calibration.free[sensor::odometry::ackermann::radius] = free;
        settings.calibration.free[sensor::odometry::ackermann::steering_offset] = free;
        settings.calibration.prior_sigma[sensor::odometry::ackermann::radius] = 0.05;
        settings.calibration.prior_sigma[sensor::odometry::ackermann::steering_offset] = 0.1;
        settings.extrinsic_deviations[5] = 0.01;
        return settings;
    }

    // The camera rig of one camera and the odometer, its extrinsic the camera from odometry transform.
    mapping::rig vehicle_rig(const bool refined) {
        mapping::rig rig;
        rig.add(1, mapping::rig::sensor_kind::camera, math::se3<double>::identity(), false);
        rig.add(2, mapping::rig::sensor_kind::odometry, odometry_from_camera().inverse(), refined);
        return rig;
    }

    // The odometer's own bookkeeping: frame times, motions kept and merged, carried past the newest sample, predicted through the
    // extrinsic, and the edges of an adjustment.
    void test_bookkeeping() {
        simulation::ground_vehicle_options options;
        options.duration = 6.0;
        options.add_noise = false;
        const simulation::ground_vehicle vehicle = simulation::ground_vehicle::generate(options);
        mapping::odometry odometry;
        REQUIRE(!odometry.is_enabled());
        mapping::odometry::settings wrong = declared_settings(0.3, 0.0, true);
        wrong.calibration.count = 3;
        REQUIRE(!odometry.configure(wrong));
        REQUIRE(odometry.configure(declared_settings(0.3, 0.0, true)));
        REQUIRE(odometry.is_enabled());
        REQUIRE(odometry.parameter_count() == 4);
        REQUIRE(odometry.calibration()[0] == 0.3);

        for (const sensor::odometry::sample& sample : vehicle.samples) {
            REQUIRE(odometry.add_sample(sample));
        }
        REQUIRE(!odometry.add_sample(vehicle.samples.front()));
        REQUIRE(odometry.sample_count() == vehicle.samples.size());
        const long long int times[4] = { options.start_time + 500000000ll, options.start_time + 2030000000ll, options.start_time + 3500000000ll, vehicle.end_time() + 50000000ll };
        for (int frame = 0; frame < 4; ++frame) {
            odometry.set_frame_time(frame, times[frame]);
        }
        long long int read = 0;
        REQUIRE(odometry.frame_time(1, read) && (read == times[1]));
        REQUIRE(!odometry.frame_time(9, read));

        sensor::odometry::preintegration first;
        sensor::odometry::preintegration second;
        sensor::odometry::preintegration whole;
        REQUIRE(odometry.motion(0, 1, first));
        REQUIRE(odometry.motion(1, 2, second));
        // The motion across a culled keyframe is merged from the two kept on either side of it, the samples kept with it.
        REQUIRE(odometry.motion(0, 2, whole));
        REQUIRE(whole.get_samples().size() == first.get_samples().size() + second.get_samples().size() - 1);
        sensor::odometry::preintegration direct(rear_axle(), sensor::odometry::noise(), odometry.calibration(), 4);
        REQUIRE(direct.integrate(vehicle.samples.data(), vehicle.samples.size(), times[0], times[2]));
        for (size_t r = 0; r < 3; ++r) {
            REQUIRE(std::abs(whole.get_delta()[r] - direct.get_delta()[r]) < 1e-9);
        }
        // Past the newest sample by less than the limit the motion is carried on, and not before the first.
        sensor::odometry::preintegration extrapolated;
        REQUIRE(odometry.motion(2, 3, extrapolated));
        REQUIRE(extrapolated.get_end() == times[3]);
        odometry.set_frame_time(4, vehicle.end_time() + 500000000ll);
        REQUIRE(!odometry.motion(2, 4, extrapolated));
        odometry.set_frame_time(5, options.start_time - 1);
        REQUIRE(!odometry.motion(5, 0, extrapolated));
        REQUIRE(!odometry.motion(1, 0, extrapolated));

        // Through the extrinsic the odometry predicts the camera's motion, and a two view initialisation's scale.
        mapping::rig rig = vehicle_rig(false);
        math::se3<double> moved;
        mapping::rig no_odometer;
        no_odometer.add(1, mapping::rig::sensor_kind::camera, math::se3<double>::identity(), false);
        REQUIRE(!odometry.camera_motion(0, 1, no_odometer, moved));
        REQUIRE(odometry.camera_motion(0, 1, rig, moved));
        const math::se3<double> world_from_camera_0 = vehicle.pose_at(times[0]) * odometry_from_camera();
        const math::se3<double> world_from_camera_1 = vehicle.pose_at(times[1]) * odometry_from_camera();
        const math::se3<double> truth = world_from_camera_1.inverse() * world_from_camera_0;
        REQUIRE(std::sqrt((moved.inverse() * truth).log().get_length_squared()) < 1e-3);
        math::matrix<double, 3, 1> unit = truth.translation() * (0.1 / std::sqrt(truth.translation().get_length_squared()));
        REQUIRE(odometry.scale_initialisation(0, 1, rig, unit));
        REQUIRE(std::abs(std::sqrt(unit.get_length_squared()) - std::sqrt(truth.translation().get_length_squared())) < 1e-3);

        // An adjustment's frames take an edge between each two consecutive ones the odometry reaches, all sharing the
        // calibration and extrinsic vertices, and the calibration they leave is read back, or restored.
        optimisation::factor_graph graph;
        std::unordered_map<int, optimisation::vertex*> frames;
        for (int frame = 0; frame < 3; ++frame) {
            optimisation::vertex node{ optimisation::vertices::pose() };
            double parameters[7] = {};
            optimisation::edges::odometry::write_pose((vehicle.pose_at(times[frame]) * odometry_from_camera()).inverse(), &parameters[0]);
            REQUIRE(node.set_parameters(&parameters[0], 7));
            node.set_fixed(frame == 0);
            frames[frame] = graph.add_vertex(static_cast<optimisation::vertex&&>(node));
        }
        const mapping::odometry::graph_vertices vertices = odometry.add_to_graph(graph, frames, rig);
        REQUIRE(vertices.edges == 2);
        REQUIRE((vertices.calibration != nullptr) && (vertices.extrinsic != nullptr));
        REQUIRE(vertices.extrinsic->is_fixed());
        REQUIRE(vertices.calibration->get_local_dimensions() == 2);
        REQUIRE(graph.get_connected_edges(vertices.calibration).size() == 3);
        REQUIRE(graph.solve(20) >= 0);
        odometry.read_from_graph(vertices, rig);
        // Exact samples and poses: the radius is drawn from 0.3 to the truth's.
        std::printf("Odometry bookkeeping: radius %.5f, offset %.5f after a three frame adjustment.\n", odometry.calibration()[0], odometry.calibration()[3]);
        std::fflush(stdout);
        REQUIRE(std::abs(odometry.calibration()[0] - 0.3) < 0.01);
        odometry.restore(vertices);
        REQUIRE(odometry.calibration()[0] == 0.3);
        REQUIRE(odometry.add_to_graph(graph, frames, no_odometer).edges == 0);
    }

    // The drive written as a scene in the directory form, into the directory ZEROSLAM_ODOMETRY_SCENE names (its sensor/image_01
    // directory made beforehand), for `zeroslam dataset collapse` and `zeroslam process --wheel wheel_01`: the frames, the
    // camera's calibration, the odometer with its parameters and extrinsic, and the ground truth, the camera's pose in the yard.
    void write_scene(const char* const directory, const simulation::ground_vehicle& vehicle, const std::vector<long long int>& times, const std::vector<math::se3<double>>& truth, const std::vector<image::image>& frames) {
        const std::string root(directory);
        std::FILE* const calibration = std::fopen((root + "/sensor/image_01.txt").c_str(), "wb");
        std::FILE* const trajectory = std::fopen((root + "/trajectory.txt").c_str(), "wb");
        std::FILE* const odometer = std::fopen((root + "/sensor/wheel_01.txt").c_str(), "wb");
        REQUIRE((calibration != nullptr) && (trajectory != nullptr) && (odometer != nullptr));
        for (size_t i = 0; i < times.size(); ++i) {
            char name[64];
            std::snprintf(name, sizeof(name), "/sensor/image_01/%020lld.pnm", times[i]);
            std::FILE* const image = std::fopen((root + name).c_str(), "wb");
            REQUIRE(image != nullptr);
            std::fprintf(image, "P5\n%d %d\n255\n", width, height);
            std::fwrite(frames[i].get_data(), 1, static_cast<size_t>(width * height), image);
            std::fclose(image);
            std::fprintf(calibration, "%lld.%09lld 0 0 0 0 0 0 1 plumb_bob %.17g %.17g %.17g %.17g\n", times[i] / 1000000000ll, times[i] % 1000000000ll, focal, focal, 0.5 * width, 0.5 * height);
            const math::se3<double> camera_to_world = truth[i].inverse();
            const math::matrix<double, 4, 1> quaternion = camera_to_world.rotation().get_quaternion();
            std::fprintf(trajectory, "%lld.%09lld %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", times[i] / 1000000000ll, times[i] % 1000000000ll, camera_to_world.translation()[0], camera_to_world.translation()[1], camera_to_world.translation()[2], quaternion[1], quaternion[2], quaternion[3], quaternion[0]);
        }
        const math::se3<double> camera_from_odometry = odometry_from_camera().inverse();
        const math::matrix<double, 4, 1> quaternion = camera_from_odometry.rotation().get_quaternion();
        std::fprintf(odometer, "# model=ackermann_rear_axle parameters=0.30,2.7,1.0,0.0 free_mask=9 parameter_deviations=0.05,0,0,0.1 extrinsic_deviations=0,0,0,0,0,0.01\n");
        for (const sensor::odometry::sample& sample : vehicle.samples) {
            std::fprintf(odometer, "%lld.%09lld %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", sample.timestamp / 1000000000ll, sample.timestamp % 1000000000ll, camera_from_odometry.translation()[0], camera_from_odometry.translation()[1], camera_from_odometry.translation()[2], quaternion[1], quaternion[2], quaternion[3], quaternion[0], sample.rotation[0], sample.rotation[1], sample.steering);
        }
        std::fclose(calibration);
        std::fclose(trajectory);
        std::fclose(odometer);
    }

    // A ground vehicle driving the yard through the whole system, a camera and the wheel odometer: the map is in metres, and the
    // wheel radius, declared 6% small, and the steering offset, declared zero, are recovered.
    void test_drive(const bool refined) {
        simulation::ground_vehicle_options options;
        options.duration = 20.0;
        options.speed = 1.5;
        options.speed_amplitude = 0.3;
        options.curvature = 0.03;
        options.curvature_amplitude = 0.12;
        options.parameters[sensor::odometry::ackermann::radius] = 0.32;
        options.parameters[sensor::odometry::ackermann::steering_offset] = 0.03;
        const simulation::ground_vehicle vehicle = simulation::ground_vehicle::generate(options);

        slam system;
        system.set_rig(vehicle_rig(refined));
        REQUIRE(system.reconstruction.wheel_odometry.configure(declared_settings(0.30, 0.0, true)));
        const double parameters[4] = { focal, focal, 0.5 * width, 0.5 * height };
        const sensor::model camera(sensor::camera::pinhole<double>(&parameters[0], 4));

        constexpr static const long long int frame_period = 66666667ll;
        std::vector<math::se3<double>> truth;
        std::vector<long long int> times;
        std::vector<image::image> frames;
        const char* const scene_directory = refined ? nullptr : std::getenv("ZEROSLAM_ODOMETRY_SCENE");
        size_t next_sample = 0;
        for (long long int time = options.start_time; time <= vehicle.end_time(); time += frame_period) {
            while ((next_sample < vehicle.samples.size()) && (vehicle.samples[next_sample].timestamp <= time)) {
                REQUIRE(system.reconstruction.wheel_odometry.add_sample(vehicle.samples[next_sample]));
                ++next_sample;
            }
            truth.push_back((vehicle.pose_at(time) * odometry_from_camera()).inverse());
            image::image frame(static_cast<size_t>(height), static_cast<size_t>(width));
            render_yard(truth.back(), frame);
            if (scene_directory != nullptr) {
                times.push_back(time);
                frames.push_back(frame);
            }
            system.reconstruction.wheel_odometry.set_frame_time(system.reconstruction.next_frame_id, time);
            system.process_frame(camera, frame);
        }
        while (next_sample < vehicle.samples.size()) {
            REQUIRE(system.reconstruction.wheel_odometry.add_sample(vehicle.samples[next_sample]));
            ++next_sample;
        }
        system.finalise();
        if (scene_directory != nullptr) {
            write_scene(scene_directory, vehicle, times, truth, frames);
        }

        REQUIRE(system.reconstruction.metric_scale);
        int first_id = -1;
        for (int i = 0; (i < static_cast<int>(truth.size())) && (first_id < 0); ++i) {
            first_id = (system.reconstruction.frames.count(i) != 0) ? i : -1;
        }
        REQUIRE(first_id >= 0);
        const auto centre_of = [](const math::matrix<double, 3, 3>& rotation, const math::matrix<double, 3, 1>& translation) {
            return math::matrix<double, 3, 1>(-(math::transpose(rotation) * translation));
        };
        const mapping::frame& first_frame = system.reconstruction.frames.at(first_id);
        const math::matrix<double, 3, 1> first_centre = centre_of(first_frame.rotation, first_frame.translation);
        const math::matrix<double, 3, 1> first_truth = centre_of(truth[static_cast<size_t>(first_id)].rotation().get_matrix(), truth[static_cast<size_t>(first_id)].translation());
        std::vector<double> ratios;
        for (const auto& [frame_id, frame] : system.reconstruction.frames) {
            const math::se3<double>& pose = truth[static_cast<size_t>(frame_id)];
            const double travelled = std::sqrt((centre_of(pose.rotation().get_matrix(), pose.translation()) - first_truth).get_length_squared());
            if (travelled >= 2.0) {
                ratios.push_back(std::sqrt((centre_of(frame.rotation, frame.translation) - first_centre).get_length_squared()) / travelled);
            }
        }
        REQUIRE(ratios.size() > 10);
        std::sort(ratios.begin(), ratios.end());
        const double scale = ratios[ratios.size() / 2];
        const double* const calibration = system.reconstruction.wheel_odometry.calibration();
        std::printf("Odometry drive%s: %zu frames posed, %zu keyframes, scale %.4f (from %.4f to %.4f), radius %.4f, offset %.4f.\n", refined ? " (extrinsic refined)" : "", system.reconstruction.frames.size(), system.keyframe_ids().size(), scale, ratios.front(), ratios.back(), calibration[0], calibration[3]);
        std::fflush(stdout);
        REQUIRE(system.state() == slam::tracking_state::tracking);
        REQUIRE(std::abs(scale - 1.0) < 0.03);
        REQUIRE(std::abs(calibration[0] - 0.32) < 0.008);
        REQUIRE(std::abs(calibration[3] - 0.03) < 0.01);
        REQUIRE(calibration[1] == 2.7);
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    test_bookkeeping();
    test_drive(false);
    test_drive(true);

    return EXIT_SUCCESS;
}
