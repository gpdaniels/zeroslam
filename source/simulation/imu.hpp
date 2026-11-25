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
#ifndef ZEROSLAM_SIMULATION_IMU_HPP
#define ZEROSLAM_SIMULATION_IMU_HPP

#include "core/assert.hpp"
#include "math/lie.hpp"
#include "math/math.hpp"
#include "math/matrix.hpp"
#include "sensor/imu/measurement.hpp"
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
    // A smooth trajectory of an imu in a world whose z axis is up, with gravity (0, 0, -g): each axis of the position and of the
    // rotation vector is a constant plus a sum of sinusoids, so the velocity, the acceleration and the angular velocity are exact.
    class imu_trajectory final {
    public:
        constexpr static const int harmonics = 3;

        class options final {
        public:
            uint64_t seed = 7;
            // The largest amplitude of a position harmonic, m, and the range of their frequencies, Hz.
            double position_amplitude = 1.0;
            double position_frequency_minimum = 0.05;
            double position_frequency_maximum = 0.4;
            // The largest amplitude of a rotation vector harmonic, rad, and the range of their frequencies, Hz.
            double rotation_amplitude = 0.4;
            double rotation_frequency_minimum = 0.05;
            double rotation_frequency_maximum = 0.3;
            double gravity = 9.81;
        };

        class harmonic final {
        public:
            double amplitude = 0.0;
            double angular_frequency = 0.0;
            double phase = 0.0;
        };

    private:
        double gravity_magnitude = 9.81;
        double position_offset[3] = { 0.0, 0.0, 0.0 };
        double rotation_offset[3] = { 0.0, 0.0, 0.0 };
        harmonic position_harmonics[3][harmonics] = {};
        harmonic rotation_harmonics[3][harmonics] = {};

    public:
        static imu_trajectory generate(const options& generation_options) {
            imu_trajectory result;
            result.gravity_magnitude = generation_options.gravity;
            random_stream random(generation_options.seed);
            const double two_pi = 2.0 * math::pi<double>();
            for (size_t axis = 0; axis < 3; ++axis) {
                result.position_offset[axis] = random.uniform(-2.0, 2.0);
                result.rotation_offset[axis] = random.uniform(-1.0, 1.0);
                for (size_t k = 0; k < static_cast<size_t>(harmonics); ++k) {
                    harmonic& position = result.position_harmonics[axis][k];
                    position.amplitude = random.uniform(0.3, 1.0) * generation_options.position_amplitude;
                    position.angular_frequency = two_pi * random.uniform(generation_options.position_frequency_minimum, generation_options.position_frequency_maximum);
                    position.phase = random.uniform(0.0, two_pi);
                    harmonic& rotation = result.rotation_harmonics[axis][k];
                    rotation.amplitude = random.uniform(0.3, 1.0) * generation_options.rotation_amplitude;
                    rotation.angular_frequency = two_pi * random.uniform(generation_options.rotation_frequency_minimum, generation_options.rotation_frequency_maximum);
                    rotation.phase = random.uniform(0.0, two_pi);
                }
            }
            return result;
        }

    public:
        math::matrix<double, 3, 1> gravity() const {
            return { { 0.0, 0.0, -this->gravity_magnitude } };
        }

        // The imu to world rotation and the position at a time in seconds.
        math::so3<double> rotation(const double time) const {
            return math::so3<double>::exp(this->rotation_vector(time, 0));
        }

        math::matrix<double, 3, 1> position(const double time) const {
            return this->evaluate(this->position_harmonics, this->position_offset, time, 0);
        }

        math::matrix<double, 3, 1> velocity(const double time) const {
            return this->evaluate(this->position_harmonics, this->position_offset, time, 1);
        }

        math::matrix<double, 3, 1> acceleration(const double time) const {
            return this->evaluate(this->position_harmonics, this->position_offset, time, 2);
        }

        // The angular velocity in the imu's frame: R(t) = Exp(theta(t)) turns at J_r(theta) * dtheta/dt.
        math::matrix<double, 3, 1> angular_velocity(const double time) const {
            const math::matrix<double, 3, 1> theta = this->rotation_vector(time, 0);
            return math::so3<double>::left_jacobian(-1.0 * theta) * this->rotation_vector(time, 1);
        }

        // The specific force in the imu's frame, what an ideal accelerometer measures.
        math::matrix<double, 3, 1> specific_force(const double time) const {
            return this->rotation(time).inverse() * (this->acceleration(time) - this->gravity());
        }

        // The pose of the imu as a world to imu transform, the convention of the pose vertices.
        math::se3<double> world_to_imu(const double time) const {
            return math::se3<double>(this->rotation(time), this->position(time)).inverse();
        }

    private:
        math::matrix<double, 3, 1> rotation_vector(const double time, const int derivative) const {
            return this->evaluate(this->rotation_harmonics, this->rotation_offset, time, derivative);
        }

        static math::matrix<double, 3, 1> evaluate(const harmonic (&terms)[3][harmonics], const double (&offset)[3], const double time, const int derivative) {
            math::matrix<double, 3, 1> result = math::matrix<double, 3, 1>::zero();
            for (size_t axis = 0; axis < 3; ++axis) {
                double value = (derivative == 0) ? offset[axis] : 0.0;
                for (size_t k = 0; k < static_cast<size_t>(harmonics); ++k) {
                    const harmonic& term = terms[axis][k];
                    const double angle = (term.angular_frequency * time) + term.phase;
                    if (derivative == 0) {
                        value += term.amplitude * math::sin(angle);
                    }
                    else if (derivative == 1) {
                        value += term.amplitude * term.angular_frequency * math::cos(angle);
                    }
                    else {
                        value -= term.amplitude * term.angular_frequency * term.angular_frequency * math::sin(angle);
                    }
                }
                result[axis] = value;
            }
            return result;
        }
    };

    // The samples of an imu that follows a trajectory, with the white noise and the wandering biases of a noise model.
    class imu_sensor final {
    public:
        class options final {
        public:
            uint64_t seed = 11;
            long long int start = 0;
            long long int end = 10000000000ll;
            bool noise = true;
            bool random_walk = true;
            sensor::imu::bias initial_bias = sensor::imu::bias();
            // The scale errors and the misalignment of the gyroscope's axes, as the matrix M that corrects them: the gyroscope
            // measures the w for which (I + M) * w is the true angular velocity plus the bias. Zero for an ideal gyroscope.
            sensor::imu::intrinsics intrinsics = sensor::imu::intrinsics();
        };

        class recording final {
        public:
            std::vector<sensor::imu::sample> samples;
            // The bias of each sample.
            std::vector<sensor::imu::bias> biases;
        };

    public:
        // Samples at the parameters' rate from the start to the end, the period rounded to whole nanoseconds. Each sample's white
        // noise has the density times the square root of the rate as its standard deviation, and the biases wander by the random
        // walk density times the square root of the period from sample to sample.
        static recording generate(const imu_trajectory& trajectory, const sensor::imu::parameters& parameters, const options& generation_options) {
            recording result;
            const long long int period = static_cast<long long int>(math::round(1.0e9 / parameters.rate));
            ASSERT(period > 0, "The sample period must be at least a nanosecond.");
            const double period_seconds = 1.0e-9 * static_cast<double>(period);
            const double gyroscope_sigma = math::sqrt(parameters.gyroscope_sample_variance());
            const double accelerometer_sigma = math::sqrt(parameters.accelerometer_sample_variance());
            const double gyroscope_walk_sigma = math::sqrt(parameters.gyroscope_walk_variance(period_seconds));
            const double accelerometer_walk_sigma = math::sqrt(parameters.accelerometer_walk_variance(period_seconds));
            random_stream random(generation_options.seed);
            sensor::imu::bias current = generation_options.initial_bias;
            const bool misaligned = generation_options.intrinsics.gyroscope != (math::matrix<double, 3, 3>::zero());
            math::matrix<double, 3, 3> measured_from_true = math::matrix<double, 3, 3>::identity();
            if (misaligned) {
                const bool inverted = math::invert(math::matrix<double, 3, 3>(math::matrix<double, 3, 3>::identity() + generation_options.intrinsics.gyroscope), measured_from_true);
                ASSERT(inverted, "The gyroscope's matrix must leave its axes independent.");
                static_cast<void>(inverted);
            }
            for (long long int timestamp = generation_options.start; timestamp <= generation_options.end; timestamp += period) {
                const double time = 1.0e-9 * static_cast<double>(timestamp);
                sensor::imu::sample measurement;
                measurement.timestamp = timestamp;
                measurement.angular_velocity = trajectory.angular_velocity(time) + current.gyroscope;
                if (misaligned) {
                    measurement.angular_velocity = measured_from_true * measurement.angular_velocity;
                }
                measurement.linear_acceleration = trajectory.specific_force(time) + current.accelerometer;
                if (generation_options.noise) {
                    for (size_t axis = 0; axis < 3; ++axis) {
                        measurement.angular_velocity[axis] += gyroscope_sigma * random.gaussian();
                        measurement.linear_acceleration[axis] += accelerometer_sigma * random.gaussian();
                    }
                }
                result.samples.push_back(measurement);
                result.biases.push_back(current);
                if (generation_options.random_walk) {
                    for (size_t axis = 0; axis < 3; ++axis) {
                        current.gyroscope[axis] += gyroscope_walk_sigma * random.gaussian();
                        current.accelerometer[axis] += accelerometer_walk_sigma * random.gaussian();
                    }
                }
            }
            return result;
        }
    };
}

#endif // ZEROSLAM_SIMULATION_IMU_HPP
