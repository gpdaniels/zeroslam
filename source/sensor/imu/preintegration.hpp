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
#ifndef ZEROSLAM_SENSOR_IMU_PREINTEGRATION_HPP
#define ZEROSLAM_SENSOR_IMU_PREINTEGRATION_HPP

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "sensor/imu/measurement.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor::imu {
    // The imu measurements between two instants preintegrated on the manifold, after Forster, Carlone, Dellaert and Scaramuzza,
    // "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry" (TRO 2017). From the instant i to the instant j the
    // deltas are those of the imu's own frame, free of the state at i and of gravity:
    //   R_j = R_i * dR,   v_j = v_i + g * dt + R_i * dv,   p_j = p_i + v_i * dt + g * dt^2 / 2 + R_i * dp,
    // with R, v and p the imu's orientation, velocity and position in a world where gravity is g. The measurements are corrected
    // by the bias they are integrated at, the linearisation point, and the deltas at a nearby bias b are corrected to first order:
    //   dR(b) = dR * Exp(J_R^g * (b_g - b_g0)),   dv(b) = dv + J_v^g * (b_g - b_g0) + J_v^a * (b_a - b_a0),   and dp(b) likewise.
    // The covariance is that of the errors (dphi, dv, dp) of the deltas, defined by dR~ = dR * Exp(dphi), dv~ = dv + dv_e and
    // dp~ = dp + dp_e, from the white noise of the measurements. The bias random walk is left to the edges between the biases.
    // The steps are kept, so the deltas can be integrated again at another bias and two consecutive preintegrations merged.
    //
    // The linearisation point also holds the gyroscope's matrix M (sensor::imu::intrinsics), after Van Meter and Paul, "SLAM Dunk:
    // A Recipe for Long-Horizon Egocentric SLAM in the Wild" (2026): each step turns by (I + M) * w_measured - b_g, and the deltas
    // carry their jacobians by the nine entries of M, row by row, through the recursion of the gyroscope bias's. The corrected
    // angular velocity changes by -I with the bias and by kron(I, w_measured^T) with M, so each step adds to the jacobians by M
    // what it takes from the bias's, times that. The deltas at a nearby M are corrected to first order as for the bias,
    //   dR(b, M) = dR * Exp(J_R^g * (b_g - b_g0) + J_R^M * vec(M - M0)),   dv(b, M) = dv(b) + J_v^M * vec(M - M0),
    // and dp(b, M) likewise. With M zero the measurements are integrated as they are.
    class preintegration final {
    public:
        // A measurement held over a step of the integration, its duration in seconds.
        class step final {
        public:
            math::matrix<double, 3, 1> angular_velocity = math::matrix<double, 3, 1>::zero();
            math::matrix<double, 3, 1> linear_acceleration = math::matrix<double, 3, 1>::zero();
            double duration = 0.0;
        };

        // The imu's orientation (imu to world), position and velocity in the world.
        class state final {
        public:
            math::so3<double> rotation = math::so3<double>::identity();
            math::matrix<double, 3, 1> position = math::matrix<double, 3, 1>::zero();
            math::matrix<double, 3, 1> velocity = math::matrix<double, 3, 1>::zero();
        };

        // The first order bias correction is good while the bias stays near the linearisation point: a change of the gyroscope bias
        // by 0.01 rad/s over a second of motion leaves an error of the order of 1e-4 rad in the corrected rotation. Beyond these
        // the deltas are integrated again.
        constexpr static const double default_gyroscope_threshold = 0.01;
        constexpr static const double default_accelerometer_threshold = 0.2;
        // And likewise while the gyroscope's matrix stays near the one integrated at: a change of 0.005 is the gyroscope bias's
        // threshold at an angular velocity of 2 rad/s.
        constexpr static const double default_gyroscope_matrix_threshold = 0.005;

    private:
        parameters noise;
        bias linearisation;
        intrinsics calibration;
        bool gyroscope_corrected;
        std::vector<step> steps;
        double delta_time;
        math::so3<double> delta_rotation;
        math::matrix<double, 3, 1> delta_velocity;
        math::matrix<double, 3, 1> delta_position;
        math::matrix<double, 9, 9> covariance;
        math::matrix<double, 3, 3> rotation_gyroscope;
        math::matrix<double, 3, 3> velocity_gyroscope;
        math::matrix<double, 3, 3> velocity_accelerometer;
        math::matrix<double, 3, 3> position_gyroscope;
        math::matrix<double, 3, 3> position_accelerometer;
        math::matrix<double, 3, 9> rotation_gyroscope_matrix;
        math::matrix<double, 3, 9> velocity_gyroscope_matrix;
        math::matrix<double, 3, 9> position_gyroscope_matrix;

    public:
        preintegration();
        explicit preintegration(const parameters& noise_parameters, const bias& linearisation_point = bias(), const intrinsics& linearisation_intrinsics = intrinsics());

    public:
        // Clears the deltas and the steps, and integrates from now on at the given bias, and at the given intrinsics or the ones
        // it has.
        void reset(const bias& linearisation_point);
        void reset(const bias& linearisation_point, const intrinsics& linearisation_intrinsics);

        // Integrates one measurement held over a duration in seconds, a duration that is not positive and finite is ignored.
        void integrate(const math::matrix<double, 3, 1>& angular_velocity, const math::matrix<double, 3, 1>& linear_acceleration, const double duration);

        // Integrates the samples between two timestamps (nanoseconds), each step between consecutive samples holding the mean of
        // its two ends, and the samples interpolated at the timestamps where those fall between samples. The samples must be in
        // time order and span the interval, otherwise nothing is integrated and false is returned.
        bool integrate(const sample* const samples, const size_t count, const long long int start, const long long int end);
        bool integrate(const std::vector<sample>& samples, const long long int start, const long long int end);

        // Integrates every step again at a new linearisation point, with the intrinsics it has or with new ones.
        void reintegrate(const bias& linearisation_point);
        void reintegrate(const bias& linearisation_point, const intrinsics& linearisation_intrinsics);

        // Whether a bias has moved far enough from the linearisation point that the first order correction no longer holds.
        bool requires_reintegration(const bias& estimate, const double gyroscope_threshold = default_gyroscope_threshold, const double accelerometer_threshold = default_accelerometer_threshold) const;

        // Whether the gyroscope's matrix has, by the root of the sum of the squares of its entries' changes.
        bool requires_reintegration(const intrinsics& estimate, const double gyroscope_matrix_threshold = default_gyroscope_matrix_threshold) const;

        // Appends the preintegration that follows this one, as when the keyframe between them is removed. With the same
        // linearisation point the deltas, the jacobians and the covariance compose in closed form; otherwise the following steps
        // are integrated at this linearisation point.
        void merge(const preintegration& next);

        // The state at j from the state at i, the bias and gravity in the world, with the intrinsics integrated at or with others.
        state predict(const state& start, const bias& estimate, const math::matrix<double, 3, 1>& gravity) const;
        state predict(const state& start, const bias& estimate, const intrinsics& calibration_estimate, const math::matrix<double, 3, 1>& gravity) const;

    public:
        const parameters& get_parameters() const;
        const bias& get_linearisation() const;
        const intrinsics& get_linearisation_intrinsics() const;
        const std::vector<step>& get_steps() const;
        double get_delta_time() const;
        const math::so3<double>& get_delta_rotation() const;
        const math::matrix<double, 3, 1>& get_delta_velocity() const;
        const math::matrix<double, 3, 1>& get_delta_position() const;
        const math::matrix<double, 9, 9>& get_covariance() const;
        const math::matrix<double, 3, 3>& get_rotation_gyroscope_jacobian() const;
        const math::matrix<double, 3, 3>& get_velocity_gyroscope_jacobian() const;
        const math::matrix<double, 3, 3>& get_velocity_accelerometer_jacobian() const;
        const math::matrix<double, 3, 3>& get_position_gyroscope_jacobian() const;
        const math::matrix<double, 3, 3>& get_position_accelerometer_jacobian() const;
        // The jacobians by the gyroscope's matrix, its entries row by row.
        const math::matrix<double, 3, 9>& get_rotation_gyroscope_matrix_jacobian() const;
        const math::matrix<double, 3, 9>& get_velocity_gyroscope_matrix_jacobian() const;
        const math::matrix<double, 3, 9>& get_position_gyroscope_matrix_jacobian() const;

        // The deltas corrected to first order for a bias.
        math::so3<double> get_delta_rotation(const bias& estimate) const;
        math::matrix<double, 3, 1> get_delta_velocity(const bias& estimate) const;
        math::matrix<double, 3, 1> get_delta_position(const bias& estimate) const;

        // The deltas corrected to first order for a bias and for the gyroscope's matrix.
        math::so3<double> get_delta_rotation(const bias& estimate, const intrinsics& calibration_estimate) const;
        math::matrix<double, 3, 1> get_delta_velocity(const bias& estimate, const intrinsics& calibration_estimate) const;
        math::matrix<double, 3, 1> get_delta_position(const bias& estimate, const intrinsics& calibration_estimate) const;

        // The change of the gyroscope's matrix from the one integrated at, its entries row by row as the jacobians take them.
        math::matrix<double, 9, 1> get_gyroscope_matrix_change(const intrinsics& calibration_estimate) const;

    private:
        void integrate_step(const step& measurement);
    };
}

#endif // ZEROSLAM_SENSOR_IMU_PREINTEGRATION_HPP
