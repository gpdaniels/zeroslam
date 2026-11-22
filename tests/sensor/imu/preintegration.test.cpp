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

#include "sensor/imu/preintegration.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "sensor/imu/measurement.hpp"
#include "simulation/imu.hpp"

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

static double norm(const math::matrix<double, 3, 1>& vector) {
    return std::sqrt(vector.get_length_squared());
}

static double rotation_distance(const math::so3<double>& lhs, const math::so3<double>& rhs) {
    return norm((lhs.inverse() * rhs).normalised().log());
}

static double largest_difference(const math::matrix<double, 3, 3>& lhs, const math::matrix<double, 3, 3>& rhs) {
    double result = 0.0;
    for (size_t i = 0; i < 9; ++i) {
        result = std::fmax(result, std::abs(lhs.data()[i] - rhs.data()[i]));
    }
    return result;
}

static double largest_value(const math::matrix<double, 3, 3>& value) {
    return largest_difference(value, math::matrix<double, 3, 3>::zero());
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    sensor::imu::parameters parameters;

    // An empty preintegration is the identity, and steps that are not positive and finite are ignored.
    {
        sensor::imu::preintegration preintegration(parameters);
        REQUIRE(preintegration.get_delta_time() == 0.0);
        REQUIRE(preintegration.get_steps().empty());
        REQUIRE(rotation_distance(preintegration.get_delta_rotation(), math::so3<double>::identity()) == 0.0);
        REQUIRE(norm(preintegration.get_delta_velocity()) == 0.0);
        REQUIRE(norm(preintegration.get_delta_position()) == 0.0);
        preintegration.integrate({ { 1.0, 0.0, 0.0 } }, { { 0.0, 0.0, 9.81 } }, 0.0);
        preintegration.integrate({ { 1.0, 0.0, 0.0 } }, { { 0.0, 0.0, 9.81 } }, -0.1);
        preintegration.integrate({ { 1.0, 0.0, 0.0 } }, { { 0.0, 0.0, 9.81 } }, std::nan(""));
        REQUIRE(preintegration.get_steps().empty());
        REQUIRE(preintegration.get_delta_time() == 0.0);
    }

    // A constant angular velocity and a constant specific force against the closed form motion. The rotation is exact, and the
    // velocity, dv = T * J_l(w * T) * a, and the position, the integral of (T - s) * Exp(w * s) * a, are approached at first order
    // in the step length.
    {
        const math::matrix<double, 3, 1> angular_velocity = { { 0.3, -0.5, 0.8 } };
        const math::matrix<double, 3, 1> acceleration = { { 1.0, -2.0, 0.5 } };
        const double duration = 2.0;
        const math::so3<double> expected_rotation = math::so3<double>::exp(duration * angular_velocity);
        const math::matrix<double, 3, 1> expected_velocity = duration * (math::so3<double>::left_jacobian(duration * angular_velocity) * acceleration);
        math::matrix<double, 3, 1> expected_position = math::matrix<double, 3, 1>::zero();
        {
            // Simpson's rule on a fine grid.
            const int intervals = 20000;
            const double width = duration / intervals;
            for (int k = 0; k <= intervals; ++k) {
                const double s = k * width;
                const double weight = ((k == 0) || (k == intervals)) ? 1.0 : ((k % 2) == 1 ? 4.0 : 2.0);
                expected_position = expected_position + ((weight * width / 3.0) * (duration - s)) * (math::so3<double>::exp(s * angular_velocity) * acceleration);
            }
        }

        double previous_velocity_error = 0.0;
        double previous_position_error = 0.0;
        for (const int steps : { 1000, 10000 }) {
            sensor::imu::preintegration preintegration(parameters);
            for (int k = 0; k < steps; ++k) {
                preintegration.integrate(angular_velocity, acceleration, duration / steps);
            }
            REQUIRE(preintegration.get_steps().size() == static_cast<size_t>(steps));
            REQUIRE(std::abs(preintegration.get_delta_time() - duration) < 1e-9);
            REQUIRE(rotation_distance(preintegration.get_delta_rotation(), expected_rotation) < 1e-9);
            const double velocity_error = norm(preintegration.get_delta_velocity() - expected_velocity);
            const double position_error = norm(preintegration.get_delta_position() - expected_position);
            REQUIRE(velocity_error < 5e-3 * norm(acceleration) * duration * 1000.0 / steps);
            REQUIRE(position_error < 5e-3 * norm(acceleration) * duration * duration * 1000.0 / steps);
            if (previous_velocity_error > 0.0) {
                REQUIRE(velocity_error < previous_velocity_error / 8.0);
                REQUIRE(position_error < previous_position_error / 8.0);
            }
            previous_velocity_error = velocity_error;
            previous_position_error = position_error;
        }
    }

    // The samples of a simulated imu between two timestamps that fall between samples. The preintegrated prediction from the
    // true state at the start equals the navigation state integrated directly with the same steps, and stays near the truth.
    const simulation::imu_trajectory trajectory = simulation::imu_trajectory::generate(simulation::imu_trajectory::options());
    const math::matrix<double, 3, 1> gravity = trajectory.gravity();
    {
        simulation::imu_sensor::options sensor_options;
        sensor_options.noise = false;
        sensor_options.random_walk = false;
        sensor_options.initial_bias.gyroscope = { { 0.01, -0.02, 0.015 } };
        sensor_options.initial_bias.accelerometer = { { 0.1, 0.05, -0.08 } };
        const simulation::imu_sensor::recording recording = simulation::imu_sensor::generate(trajectory, parameters, sensor_options);
        REQUIRE(recording.samples.size() == 2001);

        const long long int start = 2002500000ll;
        const long long int end = 3501250000ll;
        sensor::imu::preintegration preintegration(parameters, sensor_options.initial_bias);
        REQUIRE(preintegration.integrate(recording.samples, start, end));
        REQUIRE(std::abs(preintegration.get_delta_time() - 1.49875) < 1e-12);
        REQUIRE(preintegration.get_steps().size() == 301);

        sensor::imu::preintegration::state initial;
        initial.rotation = trajectory.rotation(1e-9 * start);
        initial.position = trajectory.position(1e-9 * start);
        initial.velocity = trajectory.velocity(1e-9 * start);
        const sensor::imu::preintegration::state predicted = preintegration.predict(initial, sensor_options.initial_bias, gravity);

        sensor::imu::preintegration::state direct = initial;
        for (const sensor::imu::preintegration::step& step : preintegration.get_steps()) {
            // The specific force turned by the rotation at the middle of the step.
            const math::matrix<double, 3, 1> rotation_vector = step.duration * (step.angular_velocity - sensor_options.initial_bias.gyroscope);
            const math::so3<double> middle = (direct.rotation * math::so3<double>::exp(0.5 * rotation_vector)).normalised();
            const math::matrix<double, 3, 1> acceleration = (middle * (step.linear_acceleration - sensor_options.initial_bias.accelerometer)) + gravity;
            direct.position = direct.position + (step.duration * direct.velocity) + ((0.5 * step.duration * step.duration) * acceleration);
            direct.velocity = direct.velocity + (step.duration * acceleration);
            direct.rotation = (direct.rotation * math::so3<double>::exp(rotation_vector)).normalised();
        }
        REQUIRE(rotation_distance(predicted.rotation, direct.rotation) < 1e-12);
        REQUIRE(norm(predicted.velocity - direct.velocity) < 1e-10);
        REQUIRE(norm(predicted.position - direct.position) < 1e-10);

        const double time = 1e-9 * end;
        const double rotation_error = rotation_distance(predicted.rotation, trajectory.rotation(time));
        const double velocity_error = norm(predicted.velocity - trajectory.velocity(time));
        const double position_error = norm(predicted.position - trajectory.position(time));
        REQUIRE(rotation_error < 1e-5);
        REQUIRE(velocity_error < 2e-4);
        REQUIRE(position_error < 2e-4);

        // The samples must span the interval and be in time order.
        sensor::imu::preintegration rejected(parameters);
        REQUIRE(!rejected.integrate(recording.samples, -1, end));
        REQUIRE(!rejected.integrate(recording.samples, start, recording.samples.back().timestamp + 1));
        REQUIRE(!rejected.integrate(recording.samples, end, start));
        REQUIRE(!rejected.integrate(nullptr, 0, start, end));
        std::vector<sensor::imu::sample> unordered(recording.samples.begin(), recording.samples.begin() + 10);
        unordered[4].timestamp = unordered[6].timestamp;
        REQUIRE(!rejected.integrate(unordered, unordered.front().timestamp, unordered.back().timestamp));
        REQUIRE(rejected.get_steps().empty());
        REQUIRE(rejected.integrate(recording.samples, start, start));
        REQUIRE(rejected.get_steps().empty());

        // Repeated samples at one timestamp add nothing.
        std::vector<sensor::imu::sample> repeated(recording.samples.begin(), recording.samples.begin() + 10);
        repeated.insert(repeated.begin() + 5, repeated[5]);
        sensor::imu::preintegration with_repeat(parameters);
        sensor::imu::preintegration without_repeat(parameters);
        REQUIRE(with_repeat.integrate(repeated, repeated.front().timestamp, repeated.back().timestamp));
        REQUIRE(without_repeat.integrate(recording.samples.data(), 10, recording.samples.front().timestamp, recording.samples[9].timestamp));
        REQUIRE(with_repeat.get_steps().size() == without_repeat.get_steps().size());
        REQUIRE(norm(with_repeat.get_delta_position() - without_repeat.get_delta_position()) == 0.0);
    }

    // The covariance against the scatter of the deltas over many noisy recordings of one motion, with ten times the noise of the
    // EuRoC imu so that it dominates the rounding.
    {
        sensor::imu::parameters noisy = parameters;
        noisy.gyroscope_noise_density *= 10.0;
        noisy.accelerometer_noise_density *= 10.0;
        const long long int start = 1000000000ll;
        const long long int end = 2000000000ll;
        simulation::imu_sensor::options sensor_options;
        sensor_options.random_walk = false;
        sensor_options.noise = false;
        sensor_options.start = start;
        sensor_options.end = end;
        const simulation::imu_sensor::recording exact = simulation::imu_sensor::generate(trajectory, noisy, sensor_options);
        sensor::imu::preintegration reference(noisy);
        REQUIRE(reference.integrate(exact.samples, start, end));

        const int runs = 2000;
        sensor_options.noise = true;
        math::matrix<double, 9, 9> scatter = math::matrix<double, 9, 9>::zero();
        for (int run = 0; run < runs; ++run) {
            sensor_options.seed = 1000 + static_cast<uint64_t>(run);
            const simulation::imu_sensor::recording recording = simulation::imu_sensor::generate(trajectory, noisy, sensor_options);
            sensor::imu::preintegration preintegration(noisy);
            REQUIRE(preintegration.integrate(recording.samples, start, end));
            const math::matrix<double, 3, 1> rotation_error = (reference.get_delta_rotation().inverse() * preintegration.get_delta_rotation()).normalised().log();
            const math::matrix<double, 3, 1> velocity_error = preintegration.get_delta_velocity() - reference.get_delta_velocity();
            const math::matrix<double, 3, 1> position_error = preintegration.get_delta_position() - reference.get_delta_position();
            double error[9];
            for (size_t i = 0; i < 3; ++i) {
                error[i] = rotation_error[i];
                error[3 + i] = velocity_error[i];
                error[6 + i] = position_error[i];
            }
            for (size_t r = 0; r < 9; ++r) {
                for (size_t c = 0; c < 9; ++c) {
                    scatter[r][c] += error[r] * error[c] / runs;
                }
            }
        }
        const math::matrix<double, 9, 9>& covariance = reference.get_covariance();
        for (size_t r = 0; r < 9; ++r) {
            REQUIRE(covariance[r][r] > 0.0);
            // The variance of a sample variance of 2000 draws is about 3%, the averaging of neighbouring samples in each step
            // correlates the noise of the steps a little.
            REQUIRE(std::abs(scatter[r][r] / covariance[r][r] - 1.0) < 0.12);
            for (size_t c = 0; c < 9; ++c) {
                REQUIRE(std::abs(covariance[r][c] - covariance[c][r]) <= 1e-12 * std::abs(covariance[r][c]) + 1e-30);
                const double expected = covariance[r][c] / std::sqrt(covariance[r][r] * covariance[c][c]);
                const double measured = scatter[r][c] / std::sqrt(scatter[r][r] * scatter[c][c]);
                REQUIRE(std::abs(expected - measured) < 0.1);
            }
        }
    }

    // The bias jacobians against integrating again: by central differences of the integration, and the first order correction
    // of a finite change, whose error is of second order in the change.
    {
        simulation::imu_sensor::options sensor_options;
        sensor_options.random_walk = false;
        const simulation::imu_sensor::recording recording = simulation::imu_sensor::generate(trajectory, parameters, sensor_options);
        const long long int start = 4000000000ll;
        const long long int end = 5500000000ll;
        sensor::imu::bias linearisation;
        linearisation.gyroscope = { { 0.002, -0.001, 0.003 } };
        linearisation.accelerometer = { { -0.05, 0.02, 0.04 } };
        sensor::imu::preintegration preintegration(parameters, linearisation);
        REQUIRE(preintegration.integrate(recording.samples, start, end));

        const double step = 1e-6;
        for (size_t parameter = 0; parameter < 6; ++parameter) {
            sensor::imu::bias plus = linearisation;
            sensor::imu::bias minus = linearisation;
            if (parameter < 3) {
                plus.gyroscope[parameter] += step;
                minus.gyroscope[parameter] -= step;
            }
            else {
                plus.accelerometer[parameter - 3] += step;
                minus.accelerometer[parameter - 3] -= step;
            }
            sensor::imu::preintegration forward = preintegration;
            forward.reintegrate(plus);
            sensor::imu::preintegration backward = preintegration;
            backward.reintegrate(minus);
            const math::matrix<double, 3, 1> rotation_column = (1.0 / (2.0 * step)) * (backward.get_delta_rotation().inverse() * forward.get_delta_rotation()).normalised().log();
            const math::matrix<double, 3, 1> velocity_column = (1.0 / (2.0 * step)) * (forward.get_delta_velocity() - backward.get_delta_velocity());
            const math::matrix<double, 3, 1> position_column = (1.0 / (2.0 * step)) * (forward.get_delta_position() - backward.get_delta_position());
            for (size_t row = 0; row < 3; ++row) {
                if (parameter < 3) {
                    // The rotation difference is taken at the linearisation point, where its jacobian is the right one.
                    REQUIRE(std::abs(rotation_column[row] - preintegration.get_rotation_gyroscope_jacobian()[row][parameter]) < 1e-6);
                    REQUIRE(std::abs(velocity_column[row] - preintegration.get_velocity_gyroscope_jacobian()[row][parameter]) < 1e-6);
                    REQUIRE(std::abs(position_column[row] - preintegration.get_position_gyroscope_jacobian()[row][parameter]) < 1e-6);
                }
                else {
                    REQUIRE(std::abs(rotation_column[row]) < 1e-7);
                    REQUIRE(std::abs(velocity_column[row] - preintegration.get_velocity_accelerometer_jacobian()[row][parameter - 3]) < 1e-6);
                    REQUIRE(std::abs(position_column[row] - preintegration.get_position_accelerometer_jacobian()[row][parameter - 3]) < 1e-6);
                }
            }
        }

        double previous_errors[3] = { 0.0, 0.0, 0.0 };
        for (const double scale : { 1.0, 0.5 }) {
            sensor::imu::bias moved = linearisation;
            moved.gyroscope = moved.gyroscope + scale * math::matrix<double, 3, 1>({ { 0.02, -0.03, 0.01 } });
            moved.accelerometer = moved.accelerometer + scale * math::matrix<double, 3, 1>({ { 0.2, 0.1, -0.15 } });
            sensor::imu::preintegration reintegrated = preintegration;
            reintegrated.reintegrate(moved);
            REQUIRE(reintegrated.get_linearisation().gyroscope == moved.gyroscope);
            REQUIRE(reintegrated.get_steps().size() == preintegration.get_steps().size());
            const double errors[3] = {
                rotation_distance(preintegration.get_delta_rotation(moved), reintegrated.get_delta_rotation()),
                norm(preintegration.get_delta_velocity(moved) - reintegrated.get_delta_velocity()),
                norm(preintegration.get_delta_position(moved) - reintegrated.get_delta_position())
            };
            const double uncorrected[3] = {
                rotation_distance(preintegration.get_delta_rotation(), reintegrated.get_delta_rotation()),
                norm(preintegration.get_delta_velocity() - reintegrated.get_delta_velocity()),
                norm(preintegration.get_delta_position() - reintegrated.get_delta_position())
            };
            for (size_t i = 0; i < 3; ++i) {
                REQUIRE(errors[i] < 0.05 * uncorrected[i]);
                if (previous_errors[i] > 0.0) {
                    const double ratio = previous_errors[i] / errors[i];
                    REQUIRE((ratio > 3.0) && (ratio < 5.0));
                }
                previous_errors[i] = errors[i];
            }
        }

        sensor::imu::bias near = linearisation;
        near.gyroscope[0] += 0.5 * sensor::imu::preintegration::default_gyroscope_threshold;
        near.accelerometer[2] -= 0.5 * sensor::imu::preintegration::default_accelerometer_threshold;
        REQUIRE(!preintegration.requires_reintegration(linearisation));
        REQUIRE(!preintegration.requires_reintegration(near));
        sensor::imu::bias gyroscope_far = linearisation;
        gyroscope_far.gyroscope[1] += 1.5 * sensor::imu::preintegration::default_gyroscope_threshold;
        REQUIRE(preintegration.requires_reintegration(gyroscope_far));
        sensor::imu::bias accelerometer_far = linearisation;
        accelerometer_far.accelerometer[0] -= 1.5 * sensor::imu::preintegration::default_accelerometer_threshold;
        REQUIRE(preintegration.requires_reintegration(accelerometer_far));
        REQUIRE(!preintegration.requires_reintegration(accelerometer_far, 1.0, 1.0));
    }

    // Merging consecutive preintegrations gives what integrating the whole interval gives, in closed form with the same
    // linearisation point and by integrating again with another.
    {
        simulation::imu_sensor::options sensor_options;
        const simulation::imu_sensor::recording recording = simulation::imu_sensor::generate(trajectory, parameters, sensor_options);
        const long long int start = 1000000000ll;
        const long long int middle = 1800000000ll;
        const long long int end = 2600000000ll;
        sensor::imu::bias linearisation;
        linearisation.gyroscope = { { -0.003, 0.002, 0.001 } };
        linearisation.accelerometer = { { 0.04, -0.03, 0.02 } };
        sensor::imu::preintegration whole(parameters, linearisation);
        REQUIRE(whole.integrate(recording.samples, start, end));

        for (const bool same : { true, false }) {
            sensor::imu::preintegration first(parameters, linearisation);
            REQUIRE(first.integrate(recording.samples, start, middle));
            sensor::imu::bias other = linearisation;
            if (!same) {
                other.gyroscope[2] += 0.05;
                other.accelerometer[0] -= 0.3;
            }
            sensor::imu::preintegration second(parameters, other);
            REQUIRE(second.integrate(recording.samples, middle, end));
            first.merge(second);
            REQUIRE(first.get_steps().size() == whole.get_steps().size());
            REQUIRE(std::abs(first.get_delta_time() - whole.get_delta_time()) < 1e-12);
            REQUIRE(rotation_distance(first.get_delta_rotation(), whole.get_delta_rotation()) < 1e-12);
            REQUIRE(norm(first.get_delta_velocity() - whole.get_delta_velocity()) < 1e-12);
            REQUIRE(norm(first.get_delta_position() - whole.get_delta_position()) < 1e-12);
            REQUIRE(largest_difference(first.get_rotation_gyroscope_jacobian(), whole.get_rotation_gyroscope_jacobian()) < 1e-12 * largest_value(whole.get_rotation_gyroscope_jacobian()));
            REQUIRE(largest_difference(first.get_velocity_gyroscope_jacobian(), whole.get_velocity_gyroscope_jacobian()) < 1e-12 * largest_value(whole.get_velocity_gyroscope_jacobian()));
            REQUIRE(largest_difference(first.get_velocity_accelerometer_jacobian(), whole.get_velocity_accelerometer_jacobian()) < 1e-12 * largest_value(whole.get_velocity_accelerometer_jacobian()));
            REQUIRE(largest_difference(first.get_position_gyroscope_jacobian(), whole.get_position_gyroscope_jacobian()) < 1e-12 * largest_value(whole.get_position_gyroscope_jacobian()));
            REQUIRE(largest_difference(first.get_position_accelerometer_jacobian(), whole.get_position_accelerometer_jacobian()) < 1e-12 * largest_value(whole.get_position_accelerometer_jacobian()));
            for (size_t r = 0; r < 9; ++r) {
                for (size_t c = 0; c < 9; ++c) {
                    const double scale = std::sqrt(whole.get_covariance()[r][r] * whole.get_covariance()[c][c]);
                    REQUIRE(std::abs(first.get_covariance()[r][c] - whole.get_covariance()[r][c]) < 1e-9 * scale);
                }
            }
        }

        // Merging an empty preintegration changes nothing, and merging into an empty one copies.
        sensor::imu::preintegration copy = whole;
        copy.merge(sensor::imu::preintegration(parameters, linearisation));
        REQUIRE(copy.get_delta_time() == whole.get_delta_time());
        REQUIRE(norm(copy.get_delta_position() - whole.get_delta_position()) == 0.0);
        sensor::imu::preintegration empty(parameters, linearisation);
        empty.merge(whole);
        REQUIRE(std::abs(empty.get_delta_time() - whole.get_delta_time()) < 1e-15);
        REQUIRE(norm(empty.get_delta_position() - whole.get_delta_position()) < 1e-15);

        // A split between samples interpolates the sample at the split, so the halves differ from the whole by the linear
        // interpolation only.
        sensor::imu::preintegration first(parameters, linearisation);
        REQUIRE(first.integrate(recording.samples, start, middle + 1234567));
        sensor::imu::preintegration second(parameters, linearisation);
        REQUIRE(second.integrate(recording.samples, middle + 1234567, end));
        REQUIRE(std::abs(second.get_delta_time() - 1e-9 * static_cast<double>(end - middle - 1234567)) < 1e-15);
        first.merge(second);
        REQUIRE(first.get_steps().size() == whole.get_steps().size() + 1);
        REQUIRE(std::abs(first.get_delta_time() - whole.get_delta_time()) < 1e-12);
        REQUIRE(rotation_distance(first.get_delta_rotation(), whole.get_delta_rotation()) < 1e-6);
        REQUIRE(norm(first.get_delta_velocity() - whole.get_delta_velocity()) < 1e-4);
        REQUIRE(norm(first.get_delta_position() - whole.get_delta_position()) < 1e-5);

        // Resetting clears the steps and takes the new linearisation point.
        first.reset(sensor::imu::bias());
        REQUIRE(first.get_steps().empty());
        REQUIRE(first.get_delta_time() == 0.0);
        REQUIRE(first.get_linearisation().gyroscope[0] == 0.0);
        REQUIRE(first.get_parameters().gyroscope_noise_density == parameters.gyroscope_noise_density);
    }

    return 0;
}
