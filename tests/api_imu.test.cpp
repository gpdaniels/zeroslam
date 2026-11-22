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

#include "api_imu.hpp"

#include "math/lie.hpp"
#include "sensor/imu/measurement.hpp"
#include "zeroslam/zeroslam.h"

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

    // A wall of hashed cells 4 m ahead of a camera that moves sideways along x, seen by a pinhole camera of 250 px.
    void render_wall(std::vector<unsigned char>& pixels, const double camera_x) {
        pixels.resize(static_cast<size_t>(width * height));
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                const double x = camera_x + (4.0 * ((static_cast<double>(column) + 0.5 - (0.5 * width)) / 250.0));
                const double y = 4.0 * ((static_cast<double>(row) + 0.5 - (0.5 * height)) / 250.0);
                unsigned long long hash = (static_cast<unsigned long long>(static_cast<long long>(std::floor(10.0 * x))) * 0x9e3779b97f4a7c15ull) ^ (static_cast<unsigned long long>(static_cast<long long>(std::floor(10.0 * y))) * 0xbf58476d1ce4e5b9ull);
                hash ^= hash >> 29;
                hash *= 0x94d049bb133111ebull;
                hash ^= hash >> 32;
                pixels[static_cast<size_t>((row * width) + column)] = static_cast<unsigned char>(30ull + (hash % 190ull));
            }
        }
    }

    zeroslam_sensor_parameters_imu_struct euroc_imu() {
        zeroslam_sensor_parameters_imu_struct parameters{};
        parameters.gyroscope_noise_density = 1.6968e-04;
        parameters.accelerometer_noise_density = 2.0e-03;
        parameters.gyroscope_random_walk = 1.9393e-05;
        parameters.accelerometer_random_walk = 3.0e-03;
        parameters.rate = 200.0;
        return parameters;
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    zeroslam_sensor_parameters_camera_struct camera{};
    camera.width = width;
    camera.height = height;
    camera.focal_x = 250.0;
    camera.focal_y = 250.0;
    camera.centre_x = 0.5 * width;
    camera.centre_y = 0.5 * height;
    zeroslam_sensor_parameters_imu_struct imu = euroc_imu();

    // The rig entry of an imu: its own type and size, positive densities and rate, a known gravity and a finite clock offset.
    {
        zeroslam_sensor_rig_struct entry{};
        entry.type = zeroslam_sensor_imu;
        entry.sensor_id = 5;
        entry.parameters_length = static_cast<int>(sizeof(imu));
        entry.parameters_data = &imu;
        REQUIRE(api_imu::accepts(entry));
        entry.parameters_length = 48;
        REQUIRE(!api_imu::accepts(entry));
        entry.parameters_length = static_cast<int>(sizeof(imu));
        entry.sensor_id = 0;
        REQUIRE(!api_imu::accepts(entry));
        entry.sensor_id = 5;
        zeroslam_sensor_parameters_imu_struct bad = imu;
        bad.accelerometer_random_walk = 0.0;
        entry.parameters_data = &bad;
        REQUIRE(!api_imu::accepts(entry));
        bad = imu;
        bad.rate = std::nan("");
        REQUIRE(!api_imu::accepts(entry));
        bad = imu;
        bad.gravity = -1.0;
        REQUIRE(!api_imu::accepts(entry));
        entry.parameters_data = &imu;
        entry.type = zeroslam_sensor_camera;
        REQUIRE(!api_imu::accepts(entry));

        // Its extrinsic is taken out of a set of extrinsics, the cameras' left; its noise model and pose come back.
        entry.type = zeroslam_sensor_imu;
        imu.gravity = 9.80665;
        api_imu declared;
        REQUIRE(!declared.declared());
        declared.declare(&entry);
        REQUIRE(declared.declared());
        REQUIRE(declared.sensor_id() == 5);
        REQUIRE(declared.noise().gravity == 9.80665);
        REQUIRE(declared.noise().rate == 200.0);
        REQUIRE(!declared.refined());
        zeroslam_sensor_extrinsics_struct extrinsics[2] = {};
        extrinsics[0].sensor_id = 1;
        extrinsics[0].pose[6] = 1.0;
        extrinsics[1].sensor_id = 5;
        extrinsics[1].refinement = zeroslam_extrinsic_refine;
        extrinsics[1].pose[0] = 0.1;
        extrinsics[1].pose[5] = 1.0;
        extrinsics[1].pose[6] = 1.0;
        std::vector<zeroslam_sensor_extrinsics_struct> rest;
        REQUIRE(declared.take_extrinsic(&extrinsics[0], 2, rest));
        REQUIRE(rest.size() == 1);
        REQUIRE(rest[0].sensor_id == 1);
        REQUIRE(declared.refined());
        const math::se3<double> pose = declared.body_from_imu();
        REQUIRE(std::abs(pose.translation()[0] - 0.1) < 1e-15);
        REQUIRE(std::abs(pose.rotation().get_quaternion()[0] - std::sqrt(0.5)) < 1e-12);
        extrinsics[1].pose[5] = std::nan("");
        REQUIRE(!declared.take_extrinsic(&extrinsics[0], 2, rest));
        imu.gravity = 0.0;
        REQUIRE(declared.noise().gravity == 9.80665);
    }

    // Through the api: a camera and an imu declared together, the imu's extrinsic with the cameras', and its samples between the
    // frames. A rig with two imus, or an imu without a camera, is refused, as are samples that are malformed or out of order.
    {
        zeroslam_system* system = nullptr;
        REQUIRE(zeroslam_create(&system) == zeroslam_return_success);
        zeroslam_sensor_rig_struct rig[3] = {};
        rig[0].type = zeroslam_sensor_camera;
        rig[0].sensor_id = 1;
        rig[0].parameters_length = static_cast<int>(sizeof(camera));
        rig[0].parameters_data = &camera;
        rig[1].type = zeroslam_sensor_imu;
        rig[1].sensor_id = 2;
        rig[1].parameters_length = static_cast<int>(sizeof(imu));
        rig[1].parameters_data = &imu;
        rig[2] = rig[1];
        rig[2].sensor_id = 3;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[1], 1) == zeroslam_return_failure_invalid_rig_sensor);
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 3) == zeroslam_return_failure_invalid_rig_sensor);
        rig[1].sensor_id = 1;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 2) == zeroslam_return_failure_invalid_rig_sensor);
        rig[1].sensor_id = 2;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 2) == zeroslam_return_success);
        zeroslam_sensor_extrinsics_struct extrinsics[2] = {};
        extrinsics[0].sensor_id = 1;
        extrinsics[0].pose[6] = 1.0;
        extrinsics[1].sensor_id = 2;
        extrinsics[1].pose[0] = 0.1;
        extrinsics[1].pose[6] = 1.0;
        REQUIRE(zeroslam_set_sensor_extrinsics(system, &extrinsics[0], 2) == zeroslam_return_success);
        // The extrinsics come back the camera's first, then the imu's.
        int held_length = 0;
        REQUIRE(zeroslam_get_sensor_extrinsics(system, nullptr, &held_length) == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(held_length == 2);
        zeroslam_sensor_extrinsics_struct held[2] = {};
        REQUIRE(zeroslam_get_sensor_extrinsics(system, &held[0], &held_length) == zeroslam_return_success);
        REQUIRE((held[0].sensor_id == 1) && (held[1].sensor_id == 2));
        REQUIRE(held[1].pose[0] == 0.1);
        REQUIRE(held[1].refinement == zeroslam_extrinsic_fixed);

        zeroslam_sensor_measurement_imu_struct still{};
        still.linear_acceleration[1] = -9.81;
        zeroslam_sensor_data_struct sample{};
        sample.timestamp = 1000000000ll;
        sample.sensor_id = 2;
        sample.measurement_length = 40;
        sample.measurement_data = &still;
        REQUIRE(zeroslam_set_sensor_data(system, &sample, 1) == zeroslam_return_failure_invalid_sensor_data);
        sample.measurement_length = static_cast<int>(sizeof(still));
        REQUIRE(zeroslam_set_sensor_data(system, &sample, 1) == zeroslam_return_success);
        REQUIRE(zeroslam_set_sensor_data(system, &sample, 1) == zeroslam_return_failure_invalid_sensor_data);
        // Once data has come, the rig is fixed.
        REQUIRE(zeroslam_set_sensor_rig(system, &rig[0], 2) == zeroslam_return_failure_invalid_rig_sensor);

        // Frames of a camera moving sideways at a steady 0.5 m/s with the imu's samples between them, in batches mixing the two.
        std::vector<unsigned char> pixels;
        long long imu_time = sample.timestamp;
        size_t frames = 0;
        for (int frame = 1; frame <= 30; ++frame) {
            const long long frame_time = 1000000000ll + (50000000ll * frame);
            std::vector<zeroslam_sensor_data_struct> batch;
            std::vector<zeroslam_sensor_measurement_imu_struct> measurements;
            measurements.reserve(16);
            while (imu_time + 5000000ll <= frame_time) {
                imu_time += 5000000ll;
                measurements.push_back(still);
                zeroslam_sensor_data_struct entry{};
                entry.timestamp = imu_time;
                entry.sensor_id = 2;
                entry.measurement_length = static_cast<int>(sizeof(still));
                entry.measurement_data = &measurements.back();
                batch.push_back(entry);
            }
            render_wall(pixels, 0.025 * frame);
            zeroslam_sensor_data_struct image{};
            image.timestamp = frame_time;
            image.sensor_id = 1;
            image.measurement_length = width * height;
            image.measurement_data = pixels.data();
            batch.push_back(image);
            REQUIRE(zeroslam_set_sensor_data(system, batch.data(), static_cast<int>(batch.size())) == zeroslam_return_success);
            ++frames;
        }
        REQUIRE(frames == 30);
        // A frame that does not follow the last is refused whatever the imu's samples.
        zeroslam_sensor_data_struct late{};
        late.timestamp = 1000000000ll;
        late.sensor_id = 1;
        late.measurement_length = width * height;
        late.measurement_data = pixels.data();
        REQUIRE(zeroslam_set_sensor_data(system, &late, 1) == zeroslam_return_failure_invalid_sensor_data);
        REQUIRE(zeroslam_finalise(system) == zeroslam_return_success);
        zeroslam_pose_struct pose{};
        REQUIRE(zeroslam_get_pose(system, &pose) == zeroslam_return_success);
        REQUIRE(pose.timestamp > 1000000000ll);
        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_success);
    }

    return EXIT_SUCCESS;
}
