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

#include "api_odometry.hpp"

#include "math/lie.hpp"
#include "simulation/ground_vehicle.hpp"

#include <zeroslam/zeroslam.h>

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
    constexpr static const double focal = 250.0;

    math::se3<double> odometry_from_camera() {
        const math::so3<double> axes(math::matrix<double, 3, 3>({ { 0.0, 0.0, 1.0 }, { -1.0, 0.0, 0.0 }, { 0.0, -1.0, 0.0 } }));
        return math::se3<double>(math::so3<double>::exp({ { 0.0, 0.087, 0.0 } }) * axes, { { 1.5, 0.0, 1.3 } });
    }

    // The inside of a yard 60 m square with walls 8 m high (z up), every surface textured in hashed 50 cm cells.
    void render_yard(const math::se3<double>& pose, std::vector<unsigned char>& pixels) {
        const math::matrix<double, 3, 3> camera_to_world = math::transpose(pose.rotation().get_matrix());
        const math::matrix<double, 3, 1> centre = -(camera_to_world * pose.translation());
        const double lower[3] = { -30.0, -30.0, 0.0 };
        const double upper[3] = { 30.0, 30.0, 8.0 };
        pixels.assign(static_cast<size_t>(width * height), 0);
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

    zeroslam_sensor_parameters_wheel_odometry_struct declared_parameters() {
        zeroslam_sensor_parameters_wheel_odometry_struct parameters = {};
        parameters.model = zeroslam_odometry_ackermann_rear_axle;
        parameters.free_mask = (1 << 0) | (1 << 3);
        parameters.parameters[0] = 0.30;
        parameters.parameters[1] = 2.7;
        parameters.parameters[2] = 1.0;
        parameters.parameters[3] = 0.0;
        parameters.parameter_deviations[0] = 0.05;
        parameters.parameter_deviations[3] = 0.1;
        const double noise[8] = { 0.01, 0.002, 0.005, 0.002, 0.01, 0.005, 0.001, 0.0005 };
        for (size_t i = 0; i < 8; ++i) {
            parameters.noise[i] = noise[i];
        }
        parameters.extrinsic_deviations[5] = 0.01;
        return parameters;
    }

    zeroslam_sensor_parameters_camera_struct camera_parameters() {
        zeroslam_sensor_parameters_camera_struct camera = {};
        camera.width = width;
        camera.height = height;
        camera.focal_x = focal;
        camera.focal_y = focal;
        camera.centre_x = 0.5 * width;
        camera.centre_y = 0.5 * height;
        return camera;
    }

    // The rig's declaration: what it takes and what it refuses.
    void test_declaration() {
        zeroslam_sensor_parameters_camera_struct camera = camera_parameters();
        zeroslam_sensor_parameters_wheel_odometry_struct odometer = declared_parameters();
        zeroslam_sensor_rig_struct rig[3] = {
            { zeroslam_sensor_camera, 1, static_cast<int>(sizeof(camera)), &camera },
            { zeroslam_sensor_wheel_odometry, 2, static_cast<int>(sizeof(odometer)), &odometer },
            { zeroslam_sensor_wheel_odometry, 3, static_cast<int>(sizeof(odometer)), &odometer }
        };
        zeroslam_system* system = nullptr;
        REQUIRE(zeroslam_create(&system) == zeroslam_return_success);
        zeroslam_odometry_calibration_struct calibration = {};
        REQUIRE(zeroslam_get_odometry_calibration(system, &calibration) == zeroslam_return_failure_invalid_rig_sensor);
        REQUIRE(zeroslam_get_odometry_calibration(system, nullptr) == zeroslam_return_failure_invalid_argument);
        // Two odometers, an odometer with the camera's id, and refused parameters.
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 3) == zeroslam_return_failure_invalid_rig_sensor);
        rig[1].sensor_id = 1;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
        rig[1].sensor_id = 2;
        odometer.parameters[0] = -0.3;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
        odometer = declared_parameters();
        odometer.model = 7;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
        odometer = declared_parameters();
        odometer.free_mask = 1 << 5;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
        odometer = declared_parameters();
        rig[1].parameters_length = 8;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
        rig[1].parameters_length = static_cast<int>(sizeof(odometer));
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[1], 1) == zeroslam_return_failure_invalid_rig_sensor);
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 2) == zeroslam_return_success);

        // Before any data the declared calibration and extrinsic, which the extrinsics set beside the camera's.
        const math::se3<double> camera_from_odometry = odometry_from_camera().inverse();
        zeroslam_sensor_extrinsics_struct extrinsics[2] = {};
        extrinsics[0].sensor_id = 1;
        extrinsics[0].pose[6] = 1.0;
        extrinsics[1].sensor_id = 2;
        extrinsics[1].refinement = zeroslam_extrinsic_refine;
        const math::matrix<double, 4, 1> quaternion = camera_from_odometry.rotation().get_quaternion();
        const double pose[7] = { camera_from_odometry.translation()[0], camera_from_odometry.translation()[1], camera_from_odometry.translation()[2], quaternion[1], quaternion[2], quaternion[3], quaternion[0] };
        for (size_t i = 0; i < 7; ++i) {
            extrinsics[1].pose[i] = pose[i];
        }
        extrinsics[1].pose[6] = 2.0;
        REQUIRE(zeroslam_set_sensor_extrinsics(system, &extrinsics[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
        extrinsics[1].pose[6] = pose[6];
        REQUIRE(zeroslam_set_sensor_extrinsics(system, &extrinsics[0], 2) == zeroslam_return_success);
        REQUIRE(zeroslam_set_sensor_extrinsics(system, &extrinsics[1], 1) == zeroslam_return_success);
        REQUIRE(zeroslam_get_odometry_calibration(system, &calibration) == zeroslam_return_success);
        REQUIRE((calibration.sensor_id == 2) && (calibration.parameters_length == 4) && (calibration.parameters[0] == 0.30) && (calibration.parameters[1] == 2.7));
        for (size_t i = 0; i < 7; ++i) {
            REQUIRE(std::abs(calibration.pose[i] - pose[i]) < 1e-12);
        }

        // Samples of the wrong size, out of order or not finite are refused, before any of the batch is taken.
        double values[3] = { 0.0, 0.0, 0.0 };
        zeroslam_sensor_data_struct samples[2] = { { 1000, 2, static_cast<int>(sizeof(values)), &values[0] }, { 1000, 2, static_cast<int>(sizeof(values)), &values[0] } };
        REQUIRE(zeroslam_set_sensor_data(system, &samples[0], 2) == zeroslam_return_failure_invalid_sensor_data);
        samples[1].timestamp = 2000;
        samples[1].measurement_length = 16;
        REQUIRE(zeroslam_set_sensor_data(system, &samples[0], 2) == zeroslam_return_failure_invalid_sensor_data);
        samples[1].measurement_length = static_cast<int>(sizeof(values));
        REQUIRE(zeroslam_set_sensor_data(system, &samples[0], 2) == zeroslam_return_success);
        REQUIRE(zeroslam_set_sensor_data(system, &samples[0], 1) == zeroslam_return_failure_invalid_sensor_data);
        values[2] = std::nan("");
        samples[0].timestamp = 3000;
        REQUIRE(zeroslam_set_sensor_data(system, &samples[0], 1) == zeroslam_return_failure_invalid_sensor_data);
        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_success);
    }

    // A ground vehicle through the api, its wheel samples interleaved with the camera's frames in time: the map is metric and
    // the calibration refined.
    void test_drive() {
        simulation::ground_vehicle_options options;
        options.duration = 14.0;
        options.speed = 1.5;
        options.speed_amplitude = 0.3;
        options.curvature = 0.03;
        options.curvature_amplitude = 0.12;
        options.parameters[0] = 0.32;
        options.parameters[3] = 0.03;
        const simulation::ground_vehicle vehicle = simulation::ground_vehicle::generate(options);

        zeroslam_sensor_parameters_camera_struct camera = camera_parameters();
        zeroslam_sensor_parameters_wheel_odometry_struct odometer = declared_parameters();
        const zeroslam_sensor_rig_struct rig[2] = {
            { zeroslam_sensor_wheel_odometry, 7, static_cast<int>(sizeof(odometer)), &odometer },
            { zeroslam_sensor_camera, 1, static_cast<int>(sizeof(camera)), &camera }
        };
        zeroslam_system* system = nullptr;
        REQUIRE(zeroslam_create(&system) == zeroslam_return_success);
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 2) == zeroslam_return_success);
        const math::se3<double> camera_from_odometry = odometry_from_camera().inverse();
        const math::matrix<double, 4, 1> quaternion = camera_from_odometry.rotation().get_quaternion();
        const zeroslam_sensor_extrinsics_struct extrinsic = { 7, zeroslam_extrinsic_fixed, { camera_from_odometry.translation()[0], camera_from_odometry.translation()[1], camera_from_odometry.translation()[2], quaternion[1], quaternion[2], quaternion[3], quaternion[0] } };
        REQUIRE(zeroslam_set_sensor_extrinsics(system, &extrinsic, 1) == zeroslam_return_success);

        constexpr static const long long int frame_period = 66666667ll;
        std::vector<long long int> times;
        std::vector<math::se3<double>> truth;
        size_t next_sample = 0;
        std::vector<unsigned char> pixels;
        std::vector<double> values;
        std::vector<zeroslam_sensor_data_struct> batch;
        for (long long int time = options.start_time; time <= vehicle.end_time(); time += frame_period) {
            // The samples up to the frame, then the frame, in one batch.
            batch.clear();
            values.clear();
            const size_t first_sample = next_sample;
            while ((next_sample < vehicle.samples.size()) && (vehicle.samples[next_sample].timestamp <= time)) {
                values.push_back(vehicle.samples[next_sample].rotation[0]);
                values.push_back(vehicle.samples[next_sample].rotation[1]);
                values.push_back(vehicle.samples[next_sample].steering);
                ++next_sample;
            }
            for (size_t i = first_sample; i < next_sample; ++i) {
                batch.push_back({ vehicle.samples[i].timestamp, 7, static_cast<int>(3 * sizeof(double)), &values[3 * (i - first_sample)] });
            }
            truth.push_back((vehicle.pose_at(time) * odometry_from_camera()).inverse());
            times.push_back(time);
            render_yard(truth.back(), pixels);
            batch.push_back({ time, 1, width * height, pixels.data() });
            REQUIRE(zeroslam_set_sensor_data(system, batch.data(), static_cast<int>(batch.size())) == zeroslam_return_success);
        }
        REQUIRE(zeroslam_finalise(system) == zeroslam_return_success);

        zeroslam_odometry_calibration_struct calibration = {};
        REQUIRE(zeroslam_get_odometry_calibration(system, &calibration) == zeroslam_return_success);
        // The poses the api reports are metric: the distance between the first and the last pose is the truth's.
        zeroslam_pose_struct first = {};
        zeroslam_pose_struct last = {};
        size_t first_index = 0;
        while ((first_index < times.size()) && (zeroslam_get_pose_at_timestamp(system, &first, times[first_index]) != zeroslam_return_success)) {
            ++first_index;
        }
        REQUIRE(first_index < times.size() / 4);
        REQUIRE(zeroslam_get_pose_at_timestamp(system, &last, times.back()) == zeroslam_return_success);
        const double estimated = std::sqrt(((last.pose[0] - first.pose[0]) * (last.pose[0] - first.pose[0])) + ((last.pose[1] - first.pose[1]) * (last.pose[1] - first.pose[1])) + ((last.pose[2] - first.pose[2]) * (last.pose[2] - first.pose[2])));
        const math::matrix<double, 3, 1> first_truth = truth[first_index].inverse().translation();
        const math::matrix<double, 3, 1> last_truth = truth.back().inverse().translation();
        const double travelled = std::sqrt((last_truth - first_truth).get_length_squared());
        std::printf("Odometry through the api: %.3f m between the first and last poses for %.3f m, radius %.4f, offset %.4f.\n", estimated, travelled, calibration.parameters[0], calibration.parameters[3]);
        std::fflush(stdout);
        REQUIRE(std::abs((estimated / travelled) - 1.0) < 0.03);
        REQUIRE(std::abs(calibration.parameters[0] - 0.32) < 0.01);
        REQUIRE(std::abs(calibration.parameters[3] - 0.03) < 0.015);
        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_success);
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    REQUIRE(api_odometry::declares({ zeroslam_sensor_wheel_odometry, 2, 0, nullptr }));
    REQUIRE(!api_odometry::declares({ zeroslam_sensor_camera, 1, 0, nullptr }));
    test_declaration();
    test_drive();

    return EXIT_SUCCESS;
}
