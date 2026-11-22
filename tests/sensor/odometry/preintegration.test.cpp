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

#include "sensor/odometry/preintegration.hpp"

#include "math/matrix.hpp"
#include "sensor/odometry/ackermann.hpp"
#include "sensor/odometry/differential_drive.hpp"
#include "sensor/odometry/model.hpp"
#include "sensor/odometry/sample.hpp"
#include "simulation/ground_vehicle.hpp"
#include "simulation/scene.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdint>
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

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-12) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

static const double pi = 3.14159265358979323846;

static sensor::odometry::model rear_axle() {
    return sensor::odometry::model(sensor::odometry::ackermann(sensor::odometry::ackermann::measured_wheel::rear_axle));
}

static sensor::odometry::noise quiet() {
    sensor::odometry::noise noise;
    noise.slip = 0.0;
    noise.steering = 0.0;
    noise.lateral = 0.0;
    noise.heading = 0.0;
    return noise;
}

// Samples at 100 Hz from the start time of a wheel turning at a constant rate and a constant steering reading.
static std::vector<sensor::odometry::sample> constant_samples(const size_t count, const double rotation_rate_left, const double rotation_rate_right, const double steering) {
    std::vector<sensor::odometry::sample> samples(count);
    for (size_t i = 0; i < count; ++i) {
        samples[i].timestamp = 1000000000ll + (static_cast<long long int>(i) * 10000000ll);
        samples[i].rotation[0] = rotation_rate_left * 0.01 * static_cast<double>(i);
        samples[i].rotation[1] = rotation_rate_right * 0.01 * static_cast<double>(i);
        samples[i].steering = steering;
    }
    return samples;
}

static double angle_difference(double lhs, double rhs) {
    double difference = std::fmod(lhs - rhs, 2.0 * pi);
    if (difference > pi) {
        difference -= 2.0 * pi;
    }
    if (difference < -pi) {
        difference += 2.0 * pi;
    }
    return difference;
}

// The motion between two times of the vehicle's true trajectory, the end's planar pose in the start's.
static math::matrix<double, 3, 1> true_motion(const simulation::ground_vehicle& vehicle, const long long int start, const long long int end) {
    const math::matrix<double, 3, 1> first = vehicle.planar_at(start);
    const math::matrix<double, 3, 1> second = vehicle.planar_at(end);
    const double cosine = std::cos(first[2]);
    const double sine = std::sin(first[2]);
    const double dx = second[0] - first[0];
    const double dy = second[1] - first[1];
    return { { (cosine * dx) + (sine * dy), (-sine * dx) + (cosine * dy), second[2] - first[2] } };
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // The arc's displacement and its Jacobian, on either side of where its series takes over.
    {
        const double headings[] = { 0.0, 1e-9, 1e-4, 0.0099999, 0.0100001, -0.3, 1.0, 3.0, -6.0 };
        for (const double heading : headings) {
            const double length = 1.7;
            double dx = 0.0;
            double dy = 0.0;
            double jacobian[4] = {};
            sensor::odometry::preintegration::arc(length, heading, dx, dy, &jacobian[0]);
            if (std::abs(heading) > 1e-6) {
                REQUIRE(is_value_approx(dx, length * std::sin(heading) / heading, 1e-14));
                REQUIRE(is_value_approx(dy, length * (1.0 - std::cos(heading)) / heading, 1e-9));
            }
            else {
                REQUIRE(is_value_approx(dx, length));
                REQUIRE(std::abs(dy - (0.5 * length * heading)) < 1e-15);
            }
            const double step = 1e-6;
            double dx_plus = 0.0;
            double dy_plus = 0.0;
            double dx_minus = 0.0;
            double dy_minus = 0.0;
            sensor::odometry::preintegration::arc(length + step, heading, dx_plus, dy_plus, nullptr);
            sensor::odometry::preintegration::arc(length - step, heading, dx_minus, dy_minus, nullptr);
            REQUIRE(is_value_approx(jacobian[0], (dx_plus - dx_minus) / (2.0 * step), 1e-8));
            REQUIRE(is_value_approx(jacobian[2], (dy_plus - dy_minus) / (2.0 * step), 1e-8));
            sensor::odometry::preintegration::arc(length, heading + step, dx_plus, dy_plus, nullptr);
            sensor::odometry::preintegration::arc(length, heading - step, dx_minus, dy_minus, nullptr);
            REQUIRE(is_value_approx(jacobian[1], (dx_plus - dx_minus) / (2.0 * step), 1e-8));
            REQUIRE(is_value_approx(jacobian[3], (dy_plus - dy_minus) / (2.0 * step), 1e-8));
        }
    }

    // Composition on SE2 and its Jacobians.
    {
        const math::matrix<double, 3, 1> first = { { 1.0, -2.0, 0.7 } };
        const math::matrix<double, 3, 1> second = { { 0.5, 0.25, -1.3 } };
        math::matrix<double, 3, 3> jacobian_first;
        math::matrix<double, 3, 3> jacobian_second;
        const math::matrix<double, 3, 1> composed = sensor::odometry::preintegration::compose(first, second, &jacobian_first, &jacobian_second);
        REQUIRE(is_value_approx(composed[0], 1.0 + (std::cos(0.7) * 0.5) - (std::sin(0.7) * 0.25)));
        REQUIRE(is_value_approx(composed[1], -2.0 + (std::sin(0.7) * 0.5) + (std::cos(0.7) * 0.25)));
        REQUIRE(is_value_approx(composed[2], 0.7 - 1.3));
        const double step = 1e-6;
        for (size_t i = 0; i < 3; ++i) {
            math::matrix<double, 3, 1> plus = first;
            math::matrix<double, 3, 1> minus = first;
            plus[i] += step;
            minus[i] -= step;
            const math::matrix<double, 3, 1> moved_plus = sensor::odometry::preintegration::compose(plus, second, nullptr, nullptr);
            const math::matrix<double, 3, 1> moved_minus = sensor::odometry::preintegration::compose(minus, second, nullptr, nullptr);
            for (size_t r = 0; r < 3; ++r) {
                REQUIRE(is_value_approx(jacobian_first[r][i], (moved_plus[r] - moved_minus[r]) / (2.0 * step), 1e-8));
            }
            plus = second;
            minus = second;
            plus[i] += step;
            minus[i] -= step;
            const math::matrix<double, 3, 1> second_plus = sensor::odometry::preintegration::compose(first, plus, nullptr, nullptr);
            const math::matrix<double, 3, 1> second_minus = sensor::odometry::preintegration::compose(first, minus, nullptr, nullptr);
            for (size_t r = 0; r < 3; ++r) {
                REQUIRE(is_value_approx(jacobian_second[r][i], (second_plus[r] - second_minus[r]) / (2.0 * step), 1e-8));
            }
        }
    }

    // A constant steering angle drives a circle: after rolling d on curvature k the vehicle is at (sin(d k) / k, (1 - cos(d k)) / k)
    // heading d k, more than one full turn included, the heading not wrapped.
    {
        const double parameters[4] = { 0.3, 2.7, 1.1, 0.02 };
        const double steering = 0.25;
        const double curvature = std::tan((1.1 * steering) + 0.02) / 2.7;
        const std::vector<sensor::odometry::sample> samples = constant_samples(4001, 20.0, 0.0, steering);
        sensor::odometry::preintegration motion(rear_axle(), quiet(), &parameters[0], 4);
        REQUIRE(motion.integrate(samples.data(), samples.size(), samples.front().timestamp, samples.back().timestamp));
        REQUIRE(motion.is_valid());
        const double distance = 0.3 * 20.0 * 40.0;
        const double heading = distance * curvature;
        REQUIRE(heading > 2.0 * pi);
        REQUIRE(is_value_approx(motion.get_delta()[0], std::sin(heading) / curvature, 1e-9));
        REQUIRE(is_value_approx(motion.get_delta()[1], (1.0 - std::cos(heading)) / curvature, 1e-9));
        REQUIRE(is_value_approx(motion.get_delta()[2], heading, 1e-11));
        REQUIRE(is_value_approx(motion.get_distance(), distance, 1e-11));
        REQUIRE(motion.get_samples().size() == samples.size());
        REQUIRE(motion.get_start() == samples.front().timestamp);
        REQUIRE(motion.get_end() == samples.back().timestamp);

        // The same circle from a differential drive whose wheels roll at the radii of the circle either side of its middle.
        const double drive_parameters[3] = { 0.1, 0.12, 0.6 };
        const double radius = 5.0;
        const double left_rate = 2.0 * (radius - 0.3) / (radius * 0.1);
        const double right_rate = 2.0 * (radius + 0.3) / (radius * 0.12);
        const std::vector<sensor::odometry::sample> drive_samples = constant_samples(1001, left_rate, right_rate, 0.0);
        sensor::odometry::preintegration drive(sensor::odometry::model(sensor::odometry::differential_drive{}), quiet(), &drive_parameters[0], 3);
        REQUIRE(drive.integrate(drive_samples.data(), drive_samples.size(), drive_samples.front().timestamp, drive_samples.back().timestamp));
        const double drive_heading = 20.0 / radius;
        REQUIRE(is_value_approx(drive.get_delta()[0], radius * std::sin(drive_heading), 1e-9));
        REQUIRE(is_value_approx(drive.get_delta()[1], radius * (1.0 - std::cos(drive_heading)), 1e-9));
        REQUIRE(is_value_approx(drive.get_delta()[2], drive_heading, 1e-11));
    }

    // Over a smooth drive, between times that fall between samples, the preintegrated motion is the true one to within the
    // error of the arcs between samples; the integration split at a sample and merged is the whole one.
    {
        simulation::ground_vehicle_options options;
        options.duration = 12.0;
        options.add_noise = false;
        options.parameters[sensor::odometry::ackermann::steering_scale] = 1.2;
        options.parameters[sensor::odometry::ackermann::steering_offset] = 0.03;
        const simulation::ground_vehicle vehicle = simulation::ground_vehicle::generate(options);
        REQUIRE(vehicle.samples.size() == 1201);
        const long long int start = options.start_time + 1234567890ll;
        const long long int middle = options.start_time + 6540000000ll;
        const long long int end = options.start_time + 11111111111ll;

        sensor::odometry::preintegration whole(options.kinematics, sensor::odometry::noise(), &options.parameters[0], 4);
        REQUIRE(whole.integrate(vehicle.samples.data(), vehicle.samples.size(), start, end));
        REQUIRE(whole.get_start() == start);
        REQUIRE(whole.get_end() == end);
        const math::matrix<double, 3, 1> truth = true_motion(vehicle, start, end);
        REQUIRE(std::abs(whole.get_delta()[0] - truth[0]) < 2e-4);
        REQUIRE(std::abs(whole.get_delta()[1] - truth[1]) < 2e-4);
        REQUIRE(std::abs(angle_difference(whole.get_delta()[2], truth[2])) < 1e-5);
        std::fprintf(stderr, "distance error %.3g, position error %.3g %.3g, heading error %.3g\n", whole.get_distance() - (vehicle.distance_at(end) - vehicle.distance_at(start)), whole.get_delta()[0] - truth[0], whole.get_delta()[1] - truth[1], angle_difference(whole.get_delta()[2], truth[2]));
        REQUIRE(std::abs(whole.get_distance() - (vehicle.distance_at(end) - vehicle.distance_at(start))) < 1e-5);

        sensor::odometry::preintegration first(options.kinematics, sensor::odometry::noise(), &options.parameters[0], 4);
        sensor::odometry::preintegration second(options.kinematics, sensor::odometry::noise(), &options.parameters[0], 4);
        REQUIRE(first.integrate(vehicle.samples.data(), vehicle.samples.size(), start, middle));
        REQUIRE(second.integrate(vehicle.samples.data(), vehicle.samples.size(), middle, end));
        REQUIRE(first.merge(second));
        REQUIRE(first.get_samples().size() == whole.get_samples().size());
        REQUIRE(first.get_end() == end);
        for (size_t r = 0; r < 3; ++r) {
            REQUIRE(is_value_approx(first.get_delta()[r], whole.get_delta()[r], 1e-9));
            for (size_t c = 0; c < 3; ++c) {
                REQUIRE(std::abs(first.get_covariance()[r][c] - whole.get_covariance()[r][c]) <= 1e-9 * std::abs(whole.get_covariance()[r][r]));
            }
            for (size_t c = 0; c < 4; ++c) {
                REQUIRE(is_value_approx(first.get_jacobian()[r][c], whole.get_jacobian()[r][c], 1e-8));
            }
        }
        REQUIRE(is_value_approx(first.get_distance(), whole.get_distance(), 1e-12));

        // A following motion integrated with other parameters is integrated again with these before it is appended.
        const double other[4] = { 0.31, 2.7, 1.2, 0.0 };
        sensor::odometry::preintegration third(options.kinematics, sensor::odometry::noise(), &options.parameters[0], 4);
        sensor::odometry::preintegration fourth(options.kinematics, sensor::odometry::noise(), &other[0], 4);
        REQUIRE(third.integrate(vehicle.samples.data(), vehicle.samples.size(), start, middle));
        REQUIRE(fourth.integrate(vehicle.samples.data(), vehicle.samples.size(), middle, end));
        REQUIRE(third.merge(fourth));
        for (size_t r = 0; r < 3; ++r) {
            REQUIRE(is_value_approx(third.get_delta()[r], whole.get_delta()[r], 1e-9));
        }

        // Merging into an empty preintegration takes the other, and motions that do not meet, or of other models, do not merge.
        sensor::odometry::preintegration empty;
        REQUIRE(empty.merge(second));
        REQUIRE(empty.get_start() == middle);
        sensor::odometry::preintegration disjoint = second;
        REQUIRE(!disjoint.merge(second));
        sensor::odometry::preintegration other_model(sensor::odometry::model(sensor::odometry::ackermann(sensor::odometry::ackermann::measured_wheel::front_wheel)), sensor::odometry::noise(), &options.parameters[0], 4);
        REQUIRE(other_model.integrate(vehicle.samples.data(), vehicle.samples.size(), start, middle));
        REQUIRE(!other_model.merge(second));
        REQUIRE(!other_model.merge(sensor::odometry::preintegration()));
    }

    // The Jacobian by the calibration against integrating again, and the first order prediction against integrating again.
    {
        simulation::ground_vehicle_options options;
        options.duration = 6.0;
        const simulation::ground_vehicle vehicle = simulation::ground_vehicle::generate(options);
        const long long int start = options.start_time + 500000000ll;
        const long long int end = options.start_time + 5500000000ll;
        const sensor::odometry::model models[2] = { rear_axle(), sensor::odometry::model(sensor::odometry::ackermann(sensor::odometry::ackermann::measured_wheel::front_wheel)) };
        for (const sensor::odometry::model& kinematics : models) {
            const double parameters[4] = { 0.3, 2.7, 1.0, 0.01 };
            sensor::odometry::preintegration motion(kinematics, sensor::odometry::noise(), &parameters[0], 4);
            REQUIRE(motion.integrate(vehicle.samples.data(), vehicle.samples.size(), start, end));
            for (size_t i = 0; i < 4; ++i) {
                const double step = 1e-6;
                double plus[4] = { parameters[0], parameters[1], parameters[2], parameters[3] };
                double minus[4] = { parameters[0], parameters[1], parameters[2], parameters[3] };
                plus[i] += step;
                minus[i] -= step;
                sensor::odometry::preintegration moved_plus = motion;
                sensor::odometry::preintegration moved_minus = motion;
                REQUIRE(moved_plus.reintegrate(&plus[0], 4));
                REQUIRE(moved_minus.reintegrate(&minus[0], 4));
                for (size_t r = 0; r < 3; ++r) {
                    const double numeric = (moved_plus.get_delta()[r] - moved_minus.get_delta()[r]) / (2.0 * step);
                    REQUIRE(std::abs(motion.get_jacobian()[r][i] - numeric) <= (1e-6 * (std::abs(numeric) + 1.0)));
                }
            }
            for (size_t r = 0; r < 3; ++r) {
                for (size_t i = 4; i < sensor::odometry::preintegration::maximum_parameters; ++i) {
                    REQUIRE(motion.get_jacobian()[r][i] == 0.0);
                }
            }

            // A small change is predicted to second order, a large one is not and asks to be integrated again.
            const double small[4] = { 0.30025, 2.7, 1.0, 0.0103 };
            REQUIRE(!motion.needs_reintegration(&small[0], 4));
            sensor::odometry::preintegration small_again = motion;
            REQUIRE(small_again.reintegrate(&small[0], 4));
            const math::matrix<double, 3, 1> small_prediction = motion.predict(&small[0], 4);
            const double large[4] = { 0.33, 2.7, 1.0, 0.05 };
            REQUIRE(motion.needs_reintegration(&large[0], 4));
            sensor::odometry::preintegration large_again = motion;
            REQUIRE(large_again.reintegrate(&large[0], 4));
            const math::matrix<double, 3, 1> large_prediction = motion.predict(&large[0], 4);
            double small_error = 0.0;
            double large_error = 0.0;
            for (size_t r = 0; r < 3; ++r) {
                small_error = std::max(small_error, std::abs(small_prediction[r] - small_again.get_delta()[r]));
                large_error = std::max(large_error, std::abs(large_prediction[r] - large_again.get_delta()[r]));
            }
            // The error of the prediction is of second order: half the change, a quarter of the error.
            const double half[4] = { 0.300125, 2.7, 1.0, 0.01015 };
            sensor::odometry::preintegration half_again = motion;
            REQUIRE(half_again.reintegrate(&half[0], 4));
            const math::matrix<double, 3, 1> half_prediction = motion.predict(&half[0], 4);
            double half_error = 0.0;
            for (size_t r = 0; r < 3; ++r) {
                half_error = std::max(half_error, std::abs(half_prediction[r] - half_again.get_delta()[r]));
            }
            REQUIRE(small_error < 5e-5);
            REQUIRE((small_error / half_error) > 3.5);
            REQUIRE((small_error / half_error) < 4.5);
            REQUIRE(large_error > 1000.0 * small_error);
            REQUIRE(small_again.get_linearisation()[0] == 0.30025);
            REQUIRE(motion.needs_reintegration(&small[0], 3));
            REQUIRE(!motion.needs_reintegration(&parameters[0], 4, 0.0));
        }
    }

    // The covariance against a Monte Carlo of the slip and the steering noise the simulated vehicle measures with.
    {
        simulation::ground_vehicle_options options;
        options.duration = 5.0;
        options.substeps = 4;
        options.noise.slip = 0.05;
        options.noise.steering = 0.01;
        options.curvature = 0.05;
        const long long int start = options.start_time + 250000000ll;
        const long long int end = options.start_time + 4750000000ll;
        sensor::odometry::noise noise = options.noise;
        noise.lateral = 0.0;
        noise.heading = 0.0;

        options.add_noise = false;
        const simulation::ground_vehicle exact = simulation::ground_vehicle::generate(options);
        sensor::odometry::preintegration predicted(options.kinematics, noise, &options.parameters[0], 4);
        REQUIRE(predicted.integrate(exact.samples.data(), exact.samples.size(), start, end));

        options.add_noise = true;
        const size_t runs = 3000;
        std::vector<math::matrix<double, 3, 1>> deltas;
        deltas.reserve(runs);
        math::matrix<double, 3, 1> mean = math::matrix<double, 3, 1>::zero();
        for (size_t run = 0; run < runs; ++run) {
            options.seed = 1000 + static_cast<uint64_t>(run);
            const simulation::ground_vehicle noisy = simulation::ground_vehicle::generate(options);
            sensor::odometry::preintegration motion(options.kinematics, noise, &options.parameters[0], 4);
            REQUIRE(motion.integrate(noisy.samples.data(), noisy.samples.size(), start, end));
            deltas.push_back(motion.get_delta());
            mean = mean + motion.get_delta();
        }
        mean = mean * (1.0 / static_cast<double>(runs));
        math::matrix<double, 3, 3> sampled = math::matrix<double, 3, 3>::zero();
        for (const math::matrix<double, 3, 1>& delta : deltas) {
            const math::matrix<double, 3, 1> difference = delta - mean;
            sampled = sampled + (difference * math::transpose(difference));
        }
        sampled = sampled * (1.0 / static_cast<double>(runs - 1));
        const math::matrix<double, 3, 3>& covariance = predicted.get_covariance();
        for (size_t r = 0; r < 3; ++r) {
            std::fprintf(stderr, "Monte Carlo variance %zu: predicted %.6g, sampled %.6g, mean error %.3g\n", r, covariance[r][r], sampled[r][r], mean[r] - predicted.get_delta()[r]);
            REQUIRE(sampled[r][r] > 0.0);
            REQUIRE(std::abs((covariance[r][r] / sampled[r][r]) - 1.0) < 0.1);
            // The mean is unbiased to within a few standard errors.
            REQUIRE(std::abs(mean[r] - predicted.get_delta()[r]) < 5.0 * std::sqrt(covariance[r][r] / static_cast<double>(runs)));
            for (size_t c = r + 1; c < 3; ++c) {
                const double predicted_correlation = covariance[r][c] / std::sqrt(covariance[r][r] * covariance[c][c]);
                const double sampled_correlation = sampled[r][c] / std::sqrt(sampled[r][r] * sampled[c][c]);
                std::fprintf(stderr, "Monte Carlo correlation %zu %zu: predicted %.4f, sampled %.4f\n", r, c, predicted_correlation, sampled_correlation);
                REQUIRE(std::abs(predicted_correlation - sampled_correlation) < 0.05);
            }
        }
    }

    // The covariance of the lateral slip and the heading noise against a Monte Carlo of the increments they perturb.
    {
        const double parameters[3] = { 0.1, 0.1, 0.5 };
        sensor::odometry::noise noise = quiet();
        noise.lateral = 0.03;
        noise.heading = 0.02;
        const std::vector<sensor::odometry::sample> samples = constant_samples(301, 10.0, 12.0, 0.0);
        const sensor::odometry::model kinematics(sensor::odometry::differential_drive{});
        sensor::odometry::preintegration predicted(kinematics, noise, &parameters[0], 3);
        REQUIRE(predicted.integrate(samples.data(), samples.size(), samples.front().timestamp, samples.back().timestamp));

        simulation::random_stream random(99);
        const size_t runs = 4000;
        std::vector<math::matrix<double, 3, 1>> deltas;
        deltas.reserve(runs);
        math::matrix<double, 3, 1> mean = math::matrix<double, 3, 1>::zero();
        for (size_t run = 0; run < runs; ++run) {
            math::matrix<double, 3, 1> pose = math::matrix<double, 3, 1>::zero();
            for (size_t k = 0; (k + 1) < samples.size(); ++k) {
                double inputs[2] = {};
                kinematics.inputs(samples[k], samples[k + 1], &inputs[0]);
                double length = 0.0;
                double heading = 0.0;
                REQUIRE(kinematics.motion(&parameters[0], &inputs[0], length, heading, nullptr, nullptr));
                double dx = 0.0;
                double dy = 0.0;
                sensor::odometry::preintegration::arc(length, heading, dx, dy, nullptr);
                dy += noise.lateral * std::sqrt(std::abs(length)) * random.gaussian();
                heading += noise.heading * std::sqrt(std::abs(length)) * random.gaussian();
                pose = sensor::odometry::preintegration::compose(pose, { { dx, dy, heading } }, nullptr, nullptr);
            }
            deltas.push_back(pose);
            mean = mean + pose;
        }
        mean = mean * (1.0 / static_cast<double>(runs));
        math::matrix<double, 3, 3> sampled = math::matrix<double, 3, 3>::zero();
        for (const math::matrix<double, 3, 1>& delta : deltas) {
            const math::matrix<double, 3, 1> difference = delta - mean;
            sampled = sampled + (difference * math::transpose(difference));
        }
        sampled = sampled * (1.0 / static_cast<double>(runs - 1));
        const math::matrix<double, 3, 3>& covariance = predicted.get_covariance();
        for (size_t r = 0; r < 3; ++r) {
            REQUIRE(std::abs((covariance[r][r] / sampled[r][r]) - 1.0) < 0.1);
            for (size_t c = r + 1; c < 3; ++c) {
                const double predicted_correlation = covariance[r][c] / std::sqrt(covariance[r][r] * covariance[c][c]);
                const double sampled_correlation = sampled[r][c] / std::sqrt(sampled[r][r] * sampled[c][c]);
                REQUIRE(std::abs(predicted_correlation - sampled_correlation) < 0.05);
            }
        }
    }

    // Samples that do not span the interval, are not in order, or a model or calibration that cannot integrate them, fail; an
    // empty interval is no motion.
    {
        const double parameters[4] = { 0.3, 2.7, 1.0, 0.0 };
        const std::vector<sensor::odometry::sample> samples = constant_samples(11, 10.0, 0.0, 0.1);
        const long long int first = samples.front().timestamp;
        const long long int last = samples.back().timestamp;
        sensor::odometry::preintegration motion(rear_axle(), sensor::odometry::noise(), &parameters[0], 4);
        REQUIRE(!motion.integrate(samples.data(), samples.size(), first - 1, last));
        REQUIRE(!motion.is_valid());
        REQUIRE(!motion.integrate(samples.data(), samples.size(), first, last + 1));
        REQUIRE(!motion.integrate(samples.data(), samples.size(), last, first));
        REQUIRE(!motion.integrate(samples.data(), 0, first, first));
        REQUIRE(!motion.integrate(nullptr, 3, first, first));
        std::vector<sensor::odometry::sample> unordered = samples;
        unordered[4].timestamp = unordered[3].timestamp;
        REQUIRE(!motion.integrate(unordered.data(), unordered.size(), first, last));

        REQUIRE(motion.integrate(samples.data(), samples.size(), last, last));
        REQUIRE(motion.is_valid());
        REQUIRE(motion.get_samples().size() == 1);
        REQUIRE((motion.get_delta()[0] == 0.0) && (motion.get_delta()[1] == 0.0) && (motion.get_delta()[2] == 0.0));
        REQUIRE(motion.get_covariance()[0][0] == 0.0);

        REQUIRE(motion.integrate(samples.data(), samples.size(), first + 5000000ll, first + 5000000ll));
        REQUIRE(motion.get_samples().size() == 1);
        REQUIRE(is_value_approx(motion.get_samples()[0].rotation[0], 0.05));

        sensor::odometry::preintegration invalid_model(sensor::odometry::model(), sensor::odometry::noise(), &parameters[0], 4);
        REQUIRE(!invalid_model.integrate(samples.data(), samples.size(), first, last));
        const double no_radius[4] = { 0.0, 2.7, 1.0, 0.0 };
        sensor::odometry::preintegration invalid_calibration(rear_axle(), sensor::odometry::noise(), &no_radius[0], 4);
        REQUIRE(!invalid_calibration.integrate(samples.data(), samples.size(), first, last));
        REQUIRE(!invalid_calibration.is_valid());

        sensor::odometry::preintegration valid(rear_axle(), sensor::odometry::noise(), &parameters[0], 4);
        REQUIRE(valid.integrate(samples.data(), samples.size(), first, last));
        REQUIRE(!valid.reintegrate(&parameters[0], 3));
        REQUIRE(!valid.reintegrate(nullptr, 4));
        REQUIRE(!valid.reintegrate(&no_radius[0], 4));
        REQUIRE(!valid.is_valid());
        REQUIRE(sensor::odometry::preintegration().get_start() == 0);
        REQUIRE(!sensor::odometry::preintegration().reintegrate(&parameters[0], 0));
    }

    return EXIT_SUCCESS;
}
