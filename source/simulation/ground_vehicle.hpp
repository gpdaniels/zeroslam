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

#pragma once
#ifndef ZEROSLAM_SIMULATION_GROUND_VEHICLE_HPP
#define ZEROSLAM_SIMULATION_GROUND_VEHICLE_HPP

#include "core/assert.hpp"
#include "math/lie.hpp"
#include "math/math.hpp"
#include "math/matrix.hpp"
#include "sensor/odometry/model.hpp"
#include "sensor/odometry/preintegration.hpp"
#include "sensor/odometry/sample.hpp"
#include "simulation/scene.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdint>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace simulation {
    struct ground_vehicle_options final {
        uint64_t seed = 7;
        // The first sample's time and the length of the drive (seconds), sampled at the rate (hertz), the truth integrated at that
        // many steps per sample.
        long long int start_time = 1000000000ll;
        double duration = 30.0;
        double sample_rate = 100.0;
        int substeps = 20;
        // The speed (metres per second) and the curvature (per metre) of the reference point, each a mean plus a sinusoid.
        double speed = 2.0;
        double speed_amplitude = 0.5;
        double speed_period = 11.0;
        double curvature = 0.02;
        double curvature_amplitude = 0.15;
        double curvature_period = 13.0;
        double curvature_phase = 0.0;
        // The vehicle and its true calibration, by which the wheel rotations and steering readings are made.
        sensor::odometry::model kinematics = sensor::odometry::model(sensor::odometry::ackermann(sensor::odometry::ackermann::measured_wheel::rear_axle));
        double parameters[sensor::odometry::model::maximum_parameters] = { 0.3, 2.7, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        // The noise of the measured samples: the wheel slip and the steering reading (the other fields are not simulated), and
        // the encoder resolution, zero for none.
        sensor::odometry::noise noise = sensor::odometry::noise();
        bool add_noise = true;
        double ticks_per_revolution = 0.0;
        // Bumps in the road under the reference point, a sinusoid of height along the distance travelled (metres), which pitch
        // the vehicle out of the plane the odometer assumes.
        double bump_amplitude = 0.0;
        double bump_wavelength = 5.0;
    };

    // A ground vehicle driving a smooth trajectory on a plane, the world's z = 0 with z up, and the wheel odometry samples it
    // gives: the true planar pose of the odometry frame at every integration step, the exact samples and the measured ones. The
    // trajectory is the arcs of the integration steps, each at the speed and curvature of its middle, so the truth is exact
    // between steps and the samples are those of the true wheel rotations and steering angles of the calibration.
    class ground_vehicle final {
    public:
        ground_vehicle_options options;
        long long int sample_period = 0;
        long long int step_period = 0;
        std::vector<sensor::odometry::sample> samples;
        std::vector<sensor::odometry::sample> samples_exact;
        // The planar pose (x, y, theta) of the odometry frame at every step, and the arc (length, heading change) after it.
        std::vector<math::matrix<double, 3, 1>> truth;
        std::vector<double> truth_distance;
        std::vector<math::matrix<double, 2, 1>> truth_arcs;

    public:
        static ground_vehicle generate(const ground_vehicle_options& generation_options) {
            ground_vehicle result;
            result.options = generation_options;
            result.sample_period = static_cast<long long int>(math::round(1.0e9 / generation_options.sample_rate));
            result.step_period = result.sample_period / static_cast<long long int>(generation_options.substeps);
            ASSERT((result.step_period * static_cast<long long int>(generation_options.substeps)) == result.sample_period, "The integration steps must divide the sample period.");
            const size_t sample_count = static_cast<size_t>(math::round(generation_options.duration * generation_options.sample_rate)) + 1;
            const size_t step_count = ((sample_count - 1) * static_cast<size_t>(generation_options.substeps)) + 1;
            random_stream random(generation_options.seed);

            const double* const parameters = &generation_options.parameters[0];
            const sensor::odometry::model::kind kind = generation_options.kinematics.get_kind();
            const bool front_wheel = (kind == sensor::odometry::model::kind::ackermann) && (generation_options.kinematics.get_ackermann().get_measured_wheel() == sensor::odometry::ackermann::measured_wheel::front_wheel);

            // The true wheel rotations accumulate over the steps, and the samples are taken from them at every sample time.
            double rotation[2] = { 0.0, 0.0 };
            double measured_rotation[2] = { 0.0, 0.0 };
            double sampled_rotation[2] = { 0.0, 0.0 };
            math::matrix<double, 3, 1> pose = math::matrix<double, 3, 1>::zero();
            double travelled = 0.0;
            result.truth.reserve(step_count);
            result.truth_distance.reserve(step_count);
            result.truth_arcs.reserve(step_count);
            for (size_t step = 0; step < step_count; ++step) {
                const long long int time = generation_options.start_time + (static_cast<long long int>(step) * result.step_period);
                result.truth.push_back(pose);
                result.truth_distance.push_back(travelled);
                if ((step % static_cast<size_t>(generation_options.substeps)) == 0) {
                    sensor::odometry::sample exact;
                    exact.timestamp = time;
                    exact.rotation[0] = rotation[0];
                    exact.rotation[1] = rotation[1];
                    exact.steering = result.steering_reading(time);
                    result.samples_exact.push_back(exact);

                    sensor::odometry::sample measured = exact;
                    if (generation_options.add_noise && !result.samples.empty()) {
                        // The slip of each wheel over the interval since the last sample, a random walk over the distance it rolled.
                        for (size_t wheel = 0; wheel < 2; ++wheel) {
                            const double change = rotation[wheel] - sampled_rotation[wheel];
                            const double radius = ground_vehicle::wheel_radius(generation_options, wheel);
                            const double deviation = (radius > 0.0) ? (generation_options.noise.slip * math::sqrt(math::abs(change) / radius)) : 0.0;
                            measured_rotation[wheel] += change + (deviation * random.gaussian());
                        }
                    }
                    else {
                        measured_rotation[0] = rotation[0];
                        measured_rotation[1] = rotation[1];
                    }
                    sampled_rotation[0] = rotation[0];
                    sampled_rotation[1] = rotation[1];
                    measured.rotation[0] = ground_vehicle::quantise(measured_rotation[0], generation_options.ticks_per_revolution);
                    measured.rotation[1] = ground_vehicle::quantise(measured_rotation[1], generation_options.ticks_per_revolution);
                    if (generation_options.add_noise) {
                        measured.steering += generation_options.noise.steering * random.gaussian();
                    }
                    result.samples.push_back(measured);
                }
                if ((step + 1) == step_count) {
                    result.truth_arcs.push_back({ { 0.0, 0.0 } });
                    break;
                }

                // The arc of the step at the speed and curvature of its middle.
                const double middle = (static_cast<double>(time - generation_options.start_time) + (0.5 * static_cast<double>(result.step_period))) * 1.0e-9;
                const double length = result.speed_at(middle) * static_cast<double>(result.step_period) * 1.0e-9;
                const double curvature = result.curvature_at(middle);
                const double heading = curvature * length;
                result.truth_arcs.push_back({ { length, heading } });
                double dx = 0.0;
                double dy = 0.0;
                sensor::odometry::preintegration::arc(length, heading, dx, dy, nullptr);
                pose = sensor::odometry::preintegration::compose(pose, { { dx, dy, heading } }, nullptr, nullptr);
                travelled += math::abs(length);

                if (kind == sensor::odometry::model::kind::ackermann) {
                    const double angle = math::atan2(curvature * parameters[sensor::odometry::ackermann::wheelbase], 1.0);
                    const double rolled = front_wheel ? (length / math::cos(angle)) : length;
                    rotation[0] += rolled / parameters[sensor::odometry::ackermann::radius];
                }
                else if (kind == sensor::odometry::model::kind::differential_drive) {
                    const double half_track = 0.5 * parameters[sensor::odometry::differential_drive::track_width];
                    rotation[0] += (length * (1.0 - (curvature * half_track))) / parameters[sensor::odometry::differential_drive::left_radius];
                    rotation[1] += (length * (1.0 + (curvature * half_track))) / parameters[sensor::odometry::differential_drive::right_radius];
                }
            }
            return result;
        }

        // The speed and curvature at a time in seconds from the start.
        double speed_at(const double seconds) const {
            return this->options.speed + (this->options.speed_amplitude * math::sin((2.0 * math::pi<double>() * seconds) / this->options.speed_period));
        }

        double curvature_at(const double seconds) const {
            return this->options.curvature + (this->options.curvature_amplitude * math::sin(((2.0 * math::pi<double>() * seconds) / this->options.curvature_period) + this->options.curvature_phase));
        }

        // The steering reading an Ackermann vehicle gives at a time, through its calibration, zero for another vehicle.
        double steering_reading(const long long int time) const {
            if (this->options.kinematics.get_kind() != sensor::odometry::model::kind::ackermann) {
                return 0.0;
            }
            const double seconds = static_cast<double>(time - this->options.start_time) * 1.0e-9;
            const double angle = math::atan2(this->curvature_at(seconds) * this->options.parameters[sensor::odometry::ackermann::wheelbase], 1.0);
            return (angle - this->options.parameters[sensor::odometry::ackermann::steering_offset]) / this->options.parameters[sensor::odometry::ackermann::steering_scale];
        }

        // The end time of the drive.
        long long int end_time() const {
            return this->options.start_time + (static_cast<long long int>(this->truth.size() - 1) * this->step_period);
        }

        // The true planar pose of the odometry frame at a time within the drive, along the arc of its step.
        math::matrix<double, 3, 1> planar_at(const long long int time) const {
            ASSERT(!this->truth.empty(), "The vehicle has no trajectory.");
            if (time <= this->options.start_time) {
                return this->truth.front();
            }
            const long long int offset = time - this->options.start_time;
            const size_t step = static_cast<size_t>(offset / this->step_period);
            if ((step + 1) >= this->truth.size()) {
                return this->truth.back();
            }
            const double fraction = static_cast<double>(offset - (static_cast<long long int>(step) * this->step_period)) / static_cast<double>(this->step_period);
            const double length = fraction * this->truth_arcs[step][0];
            const double heading = fraction * this->truth_arcs[step][1];
            double dx = 0.0;
            double dy = 0.0;
            sensor::odometry::preintegration::arc(length, heading, dx, dy, nullptr);
            return sensor::odometry::preintegration::compose(this->truth[step], { { dx, dy, heading } }, nullptr, nullptr);
        }

        // The distance travelled by the reference point up to a time.
        double distance_at(const long long int time) const {
            if (time <= this->options.start_time) {
                return 0.0;
            }
            const long long int offset = time - this->options.start_time;
            const size_t step = static_cast<size_t>(offset / this->step_period);
            if ((step + 1) >= this->truth.size()) {
                return this->truth_distance.back();
            }
            const double fraction = static_cast<double>(offset - (static_cast<long long int>(step) * this->step_period)) / static_cast<double>(this->step_period);
            return this->truth_distance[step] + (fraction * math::abs(this->truth_arcs[step][0]));
        }

        // The true pose of the odometry frame in the world (odometry to world) at a time, lifted off the plane by the bumps.
        math::se3<double> pose_at(const long long int time) const {
            const math::matrix<double, 3, 1> planar = this->planar_at(time);
            double height = 0.0;
            double pitch = 0.0;
            if (this->options.bump_amplitude != 0.0) {
                const double phase = (2.0 * math::pi<double>() * this->distance_at(time)) / this->options.bump_wavelength;
                height = this->options.bump_amplitude * math::sin(phase);
                const double slope = this->options.bump_amplitude * ((2.0 * math::pi<double>()) / this->options.bump_wavelength) * math::cos(phase);
                pitch = -math::atan2(slope, 1.0);
            }
            const math::so3<double> rotation = math::so3<double>::exp({ { 0.0, 0.0, planar[2] } }) * math::so3<double>::exp({ { 0.0, pitch, 0.0 } });
            return math::se3<double>(rotation, { { planar[0], planar[1], height } });
        }

    private:
        static double wheel_radius(const ground_vehicle_options& generation_options, const size_t wheel) {
            if (generation_options.kinematics.get_kind() == sensor::odometry::model::kind::ackermann) {
                return (wheel == 0) ? generation_options.parameters[sensor::odometry::ackermann::radius] : 0.0;
            }
            if (generation_options.kinematics.get_kind() == sensor::odometry::model::kind::differential_drive) {
                return generation_options.parameters[(wheel == 0) ? sensor::odometry::differential_drive::left_radius : sensor::odometry::differential_drive::right_radius];
            }
            return 0.0;
        }

        static double quantise(const double value, const double ticks_per_revolution) {
            if (!(ticks_per_revolution > 0.0)) {
                return value;
            }
            const double ticks = math::floor((value * ticks_per_revolution) / (2.0 * math::pi<double>()));
            return sensor::odometry::sample::rotation_from_ticks(static_cast<long long int>(ticks), ticks_per_revolution);
        }
    };
}

#endif // ZEROSLAM_SIMULATION_GROUND_VEHICLE_HPP
