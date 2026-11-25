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

#include "simulation/imu.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "sensor/imu/measurement.hpp"

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

static double norm(const math::matrix<double, 3, 1>& vector) {
    return std::sqrt(vector.get_length_squared());
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const simulation::imu_trajectory trajectory = simulation::imu_trajectory::generate(simulation::imu_trajectory::options());

    // The derivatives against central differences of the position and the rotation.
    {
        const double step = 1e-5;
        for (const double time : { 0.0, 1.3, 4.7, 9.2 }) {
            const math::matrix<double, 3, 1> velocity = (1.0 / (2.0 * step)) * (trajectory.position(time + step) - trajectory.position(time - step));
            const math::matrix<double, 3, 1> acceleration = (1.0 / (2.0 * step)) * (trajectory.velocity(time + step) - trajectory.velocity(time - step));
            REQUIRE(norm(velocity - trajectory.velocity(time)) < 1e-8);
            REQUIRE(norm(acceleration - trajectory.acceleration(time)) < 1e-8);
            // R(t)^T R(t + h) = Exp(w h) to first order, in the imu's frame.
            const math::matrix<double, 3, 1> angular_velocity = (1.0 / (2.0 * step)) * (trajectory.rotation(time - step).inverse() * trajectory.rotation(time + step)).normalised().log();
            REQUIRE(norm(angular_velocity - trajectory.angular_velocity(time)) < 1e-8);
            // The specific force is the acceleration less gravity in the imu's frame.
            const math::matrix<double, 3, 1> world_force = trajectory.rotation(time) * trajectory.specific_force(time);
            REQUIRE(norm(world_force - (trajectory.acceleration(time) - trajectory.gravity())) < 1e-12);
            // The pose is the inverse of the imu's place in the world.
            const math::se3<double> pose = trajectory.world_to_imu(time);
            REQUIRE(norm(pose * trajectory.position(time)) < 1e-12);
        }
        REQUIRE(trajectory.gravity()[2] == -9.81);
        // The motion excites the imu: an acceleration of a metre per second squared or more, and some turning.
        double largest_acceleration = 0.0;
        double largest_turn = 0.0;
        for (int k = 0; k <= 100; ++k) {
            largest_acceleration = std::fmax(largest_acceleration, norm(trajectory.acceleration(0.1 * k)));
            largest_turn = std::fmax(largest_turn, norm(trajectory.angular_velocity(0.1 * k)));
        }
        REQUIRE(largest_acceleration > 1.0);
        REQUIRE(largest_turn > 0.2);
    }

    // The samples of an ideal imu are the trajectory's, at the rate from the start to the end, and the same seed gives the same
    // noisy samples.
    {
        sensor::imu::parameters parameters;
        simulation::imu_sensor::options options;
        options.start = 500000000ll;
        options.end = 1500000000ll;
        options.noise = false;
        options.random_walk = false;
        options.initial_bias.gyroscope = { { 0.1, 0.2, 0.3 } };
        const simulation::imu_sensor::recording ideal = simulation::imu_sensor::generate(trajectory, parameters, options);
        REQUIRE(ideal.samples.size() == 201);
        REQUIRE(ideal.biases.size() == 201);
        REQUIRE(ideal.samples.front().timestamp == options.start);
        REQUIRE(ideal.samples.back().timestamp == options.end);
        for (size_t k = 0; k < ideal.samples.size(); ++k) {
            const double time = 1e-9 * static_cast<double>(ideal.samples[k].timestamp);
            REQUIRE(norm(ideal.samples[k].angular_velocity - trajectory.angular_velocity(time) - options.initial_bias.gyroscope) < 1e-15);
            REQUIRE(norm(ideal.samples[k].linear_acceleration - trajectory.specific_force(time)) < 1e-15);
            REQUIRE(ideal.biases[k].gyroscope[2] == 0.3);
        }

        options.noise = true;
        options.random_walk = true;
        const simulation::imu_sensor::recording first = simulation::imu_sensor::generate(trajectory, parameters, options);
        const simulation::imu_sensor::recording second = simulation::imu_sensor::generate(trajectory, parameters, options);
        options.seed += 1;
        const simulation::imu_sensor::recording third = simulation::imu_sensor::generate(trajectory, parameters, options);
        REQUIRE(first.samples.size() == second.samples.size());
        bool same = true;
        bool different = false;
        for (size_t k = 0; k < first.samples.size(); ++k) {
            same = same && (first.samples[k].linear_acceleration == second.samples[k].linear_acceleration);
            different = different || (first.samples[k].linear_acceleration != third.samples[k].linear_acceleration);
        }
        REQUIRE(same);
        REQUIRE(different);
    }

    // The white noise has the standard deviation of the density times the square root of the rate, and the biases wander with
    // a variance that grows as the random walk density squared times the time.
    {
        sensor::imu::parameters parameters;
        parameters.rate = 100.0;
        simulation::imu_sensor::options options;
        options.end = 100000000000ll;
        options.random_walk = false;
        const simulation::imu_sensor::recording noisy = simulation::imu_sensor::generate(trajectory, parameters, options);
        REQUIRE(noisy.samples.size() == 10001);
        double gyroscope_sum = 0.0;
        double accelerometer_sum = 0.0;
        for (const sensor::imu::sample& sample : noisy.samples) {
            const double time = 1e-9 * static_cast<double>(sample.timestamp);
            gyroscope_sum += (sample.angular_velocity - trajectory.angular_velocity(time)).get_length_squared();
            accelerometer_sum += (sample.linear_acceleration - trajectory.specific_force(time)).get_length_squared();
        }
        const double count = 3.0 * static_cast<double>(noisy.samples.size());
        REQUIRE(std::abs((gyroscope_sum / count) / parameters.gyroscope_sample_variance() - 1.0) < 0.05);
        REQUIRE(std::abs((accelerometer_sum / count) / parameters.accelerometer_sample_variance() - 1.0) < 0.05);

        options.noise = false;
        options.random_walk = true;
        options.end = 10000000000ll;
        double gyroscope_walk = 0.0;
        double accelerometer_walk = 0.0;
        const int runs = 200;
        for (int run = 0; run < runs; ++run) {
            options.seed = 100 + static_cast<uint64_t>(run);
            const simulation::imu_sensor::recording walking = simulation::imu_sensor::generate(trajectory, parameters, options);
            gyroscope_walk += walking.biases.back().gyroscope.get_length_squared() / (3.0 * runs);
            accelerometer_walk += walking.biases.back().accelerometer.get_length_squared() / (3.0 * runs);
        }
        REQUIRE(std::abs(gyroscope_walk / parameters.gyroscope_walk_variance(10.0) - 1.0) < 0.15);
        REQUIRE(std::abs(accelerometer_walk / parameters.accelerometer_walk_variance(10.0) - 1.0) < 0.15);
    }

    // A gyroscope whose axes are scaled and misaligned measures the angular velocity that its matrix corrects to the true one
    // plus the bias, (I + M) * w_measured = w + b_g, and leaves the accelerometer alone; a zero matrix changes no sample.
    {
        sensor::imu::parameters parameters;
        simulation::imu_sensor::options options;
        options.noise = false;
        options.random_walk = false;
        options.initial_bias.gyroscope = { { 0.01, -0.02, 0.03 } };
        options.end = 2000000000ll;
        const simulation::imu_sensor::recording ideal = simulation::imu_sensor::generate(trajectory, parameters, options);
        const simulation::imu_sensor::recording unchanged = simulation::imu_sensor::generate(trajectory, parameters, options);
        options.intrinsics.gyroscope = { { { 0.011, -0.006, 0.004 }, { 0.007, -0.009, -0.005 }, { -0.003, 0.008, 0.012 } } };
        const simulation::imu_sensor::recording misaligned = simulation::imu_sensor::generate(trajectory, parameters, options);
        REQUIRE(misaligned.samples.size() == ideal.samples.size());
        double largest_change = 0.0;
        for (size_t k = 0; k < ideal.samples.size(); ++k) {
            REQUIRE(unchanged.samples[k].angular_velocity == ideal.samples[k].angular_velocity);
            REQUIRE(misaligned.samples[k].timestamp == ideal.samples[k].timestamp);
            REQUIRE(misaligned.samples[k].linear_acceleration == ideal.samples[k].linear_acceleration);
            const math::matrix<double, 3, 1> measured = misaligned.samples[k].angular_velocity;
            const math::matrix<double, 3, 1> corrected = measured + (options.intrinsics.gyroscope * measured);
            REQUIRE((corrected - ideal.samples[k].angular_velocity).get_length_squared() < 1e-28);
            largest_change = std::fmax(largest_change, std::sqrt((measured - ideal.samples[k].angular_velocity).get_length_squared()));
        }
        // A percent of an angular velocity of the order of a radian a second.
        REQUIRE(largest_change > 1e-3);
        REQUIRE(largest_change < 5e-2);
    }

    return 0;
}
