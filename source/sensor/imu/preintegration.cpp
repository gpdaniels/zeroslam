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

#include "math/math.hpp"

namespace {
    math::matrix<double, 3, 3> skew(const math::matrix<double, 3, 1>& vector) {
        return { { { 0.0, -vector[2], vector[1] },
                   { vector[2], 0.0, -vector[0] },
                   { -vector[1], vector[0], 0.0 } } };
    }

    // The mean of a matrix and its transpose, which keeps a propagated covariance exactly symmetric.
    math::matrix<double, 9, 9> symmetric(const math::matrix<double, 9, 9>& value) {
        return 0.5 * (value + math::transpose(value));
    }

    // Sets the block of a 9x9 matrix at (row, column), in blocks of three.
    void set_part(math::matrix<double, 9, 9>& value, const size_t row, const size_t column, const math::matrix<double, 3, 3>& block) {
        math::set_block<double, 3, 3>(value, 3 * row, 3 * column, block);
    }
}

namespace sensor::imu {
    preintegration::preintegration()
        : preintegration(parameters(), bias()) {
    }

    preintegration::preintegration(const parameters& noise_parameters, const bias& linearisation_point)
        : noise(noise_parameters)
        , linearisation(linearisation_point)
        , steps()
        , delta_time(0.0)
        , delta_rotation(math::so3<double>::identity())
        , delta_velocity(math::matrix<double, 3, 1>::zero())
        , delta_position(math::matrix<double, 3, 1>::zero())
        , covariance(math::matrix<double, 9, 9>::zero())
        , rotation_gyroscope(math::matrix<double, 3, 3>::zero())
        , velocity_gyroscope(math::matrix<double, 3, 3>::zero())
        , velocity_accelerometer(math::matrix<double, 3, 3>::zero())
        , position_gyroscope(math::matrix<double, 3, 3>::zero())
        , position_accelerometer(math::matrix<double, 3, 3>::zero()) {
    }

    void preintegration::reset(const bias& linearisation_point) {
        *this = preintegration(this->noise, linearisation_point);
    }

    void preintegration::integrate(const math::matrix<double, 3, 1>& angular_velocity, const math::matrix<double, 3, 1>& linear_acceleration, const double duration) {
        if (!(duration > 0.0) || !math::isfinite(duration)) {
            return;
        }
        step measurement;
        measurement.angular_velocity = angular_velocity;
        measurement.linear_acceleration = linear_acceleration;
        measurement.duration = duration;
        this->steps.push_back(measurement);
        this->integrate_step(measurement);
    }

    bool preintegration::integrate(const sample* const samples, const size_t count, const long long int start, const long long int end) {
        if ((samples == nullptr) || (count == 0) || (end < start)) {
            return false;
        }
        if ((samples[0].timestamp > start) || (samples[count - 1].timestamp < end)) {
            return false;
        }
        for (size_t k = 1; k < count; ++k) {
            if (samples[k].timestamp < samples[k - 1].timestamp) {
                return false;
            }
        }
        for (size_t k = 0; (k + 1) < count; ++k) {
            const long long int first_time = samples[k].timestamp;
            const long long int second_time = samples[k + 1].timestamp;
            if ((second_time <= start) || (second_time == first_time)) {
                continue;
            }
            if (first_time >= end) {
                break;
            }
            const long long int step_start = (first_time < start) ? start : first_time;
            const long long int step_end = (second_time > end) ? end : second_time;
            const sample first = (step_start == first_time) ? samples[k] : sample::interpolate(samples[k], samples[k + 1], step_start);
            const sample second = (step_end == second_time) ? samples[k + 1] : sample::interpolate(samples[k], samples[k + 1], step_end);
            this->integrate(
                0.5 * (first.angular_velocity + second.angular_velocity),
                0.5 * (first.linear_acceleration + second.linear_acceleration),
                1.0e-9 * static_cast<double>(step_end - step_start)
            );
        }
        return true;
    }

    bool preintegration::integrate(const std::vector<sample>& samples, const long long int start, const long long int end) {
        return this->integrate(samples.data(), samples.size(), start, end);
    }

    void preintegration::reintegrate(const bias& linearisation_point) {
        const std::vector<step> measurements = static_cast<std::vector<step>&&>(this->steps);
        this->reset(linearisation_point);
        this->steps = measurements;
        for (const step& measurement : this->steps) {
            this->integrate_step(measurement);
        }
    }

    bool preintegration::requires_reintegration(const bias& estimate, const double gyroscope_threshold, const double accelerometer_threshold) const {
        const double gyroscope_change = (estimate.gyroscope - this->linearisation.gyroscope).get_length_squared();
        const double accelerometer_change = (estimate.accelerometer - this->linearisation.accelerometer).get_length_squared();
        return (gyroscope_change > (gyroscope_threshold * gyroscope_threshold)) || (accelerometer_change > (accelerometer_threshold * accelerometer_threshold));
    }

    void preintegration::merge(const preintegration& next) {
        if (next.steps.empty()) {
            return;
        }
        const bool same_linearisation = (next.linearisation.gyroscope == this->linearisation.gyroscope) && (next.linearisation.accelerometer == this->linearisation.accelerometer);
        if (!same_linearisation) {
            preintegration aligned = next;
            aligned.noise = this->noise;
            aligned.reintegrate(this->linearisation);
            this->merge(aligned);
            return;
        }

        // The errors of the merged deltas from those of the two parts, dphi_ik = dR_jk^T dphi_ij + dphi_jk,
        // dv_ik = dv_ij - dR_ij [dv_jk]x dphi_ij + dR_ij dv_jk and dp_ik = dp_ij + dt_jk dv_ij - dR_ij [dp_jk]x dphi_ij + dR_ij dp_jk,
        // and the bias jacobians compose the same way.
        const math::matrix<double, 3, 3> first_rotation = this->delta_rotation.get_matrix();
        const math::matrix<double, 3, 3> second_rotation_transpose = math::transpose(next.delta_rotation.get_matrix());
        const math::matrix<double, 3, 3> identity = math::matrix<double, 3, 3>::identity();
        const math::matrix<double, 3, 3> velocity_rotation = -1.0 * (first_rotation * skew(next.delta_velocity));
        const math::matrix<double, 3, 3> position_rotation = -1.0 * (first_rotation * skew(next.delta_position));
        const double second_time = next.delta_time;

        math::matrix<double, 9, 9> first_transition = math::matrix<double, 9, 9>::zero();
        set_part(first_transition, 0, 0, second_rotation_transpose);
        set_part(first_transition, 1, 0, velocity_rotation);
        set_part(first_transition, 1, 1, identity);
        set_part(first_transition, 2, 0, position_rotation);
        set_part(first_transition, 2, 1, second_time * identity);
        set_part(first_transition, 2, 2, identity);
        math::matrix<double, 9, 9> second_transition = math::matrix<double, 9, 9>::zero();
        set_part(second_transition, 0, 0, identity);
        set_part(second_transition, 1, 1, first_rotation);
        set_part(second_transition, 2, 2, first_rotation);
        this->covariance = symmetric((first_transition * this->covariance * math::transpose(first_transition)) + (second_transition * next.covariance * math::transpose(second_transition)));

        this->position_gyroscope = this->position_gyroscope + (second_time * this->velocity_gyroscope) + (position_rotation * this->rotation_gyroscope) + (first_rotation * next.position_gyroscope);
        this->position_accelerometer = this->position_accelerometer + (second_time * this->velocity_accelerometer) + (first_rotation * next.position_accelerometer);
        this->velocity_gyroscope = this->velocity_gyroscope + (velocity_rotation * this->rotation_gyroscope) + (first_rotation * next.velocity_gyroscope);
        this->velocity_accelerometer = this->velocity_accelerometer + (first_rotation * next.velocity_accelerometer);
        this->rotation_gyroscope = (second_rotation_transpose * this->rotation_gyroscope) + next.rotation_gyroscope;

        this->delta_position = this->delta_position + (second_time * this->delta_velocity) + (first_rotation * next.delta_position);
        this->delta_velocity = this->delta_velocity + (first_rotation * next.delta_velocity);
        this->delta_rotation = (this->delta_rotation * next.delta_rotation).normalised();
        this->delta_time += next.delta_time;
        this->steps.insert(this->steps.end(), next.steps.begin(), next.steps.end());
    }

    preintegration::state preintegration::predict(const state& start, const bias& estimate, const math::matrix<double, 3, 1>& gravity) const {
        const double duration = this->delta_time;
        const math::matrix<double, 3, 3> rotation = start.rotation.get_matrix();
        state result;
        result.rotation = (start.rotation * this->get_delta_rotation(estimate)).normalised();
        result.velocity = start.velocity + (duration * gravity) + (rotation * this->get_delta_velocity(estimate));
        result.position = start.position + (duration * start.velocity) + ((0.5 * duration * duration) * gravity) + (rotation * this->get_delta_position(estimate));
        return result;
    }

    const parameters& preintegration::get_parameters() const {
        return this->noise;
    }

    const bias& preintegration::get_linearisation() const {
        return this->linearisation;
    }

    const std::vector<preintegration::step>& preintegration::get_steps() const {
        return this->steps;
    }

    double preintegration::get_delta_time() const {
        return this->delta_time;
    }

    const math::so3<double>& preintegration::get_delta_rotation() const {
        return this->delta_rotation;
    }

    const math::matrix<double, 3, 1>& preintegration::get_delta_velocity() const {
        return this->delta_velocity;
    }

    const math::matrix<double, 3, 1>& preintegration::get_delta_position() const {
        return this->delta_position;
    }

    const math::matrix<double, 9, 9>& preintegration::get_covariance() const {
        return this->covariance;
    }

    const math::matrix<double, 3, 3>& preintegration::get_rotation_gyroscope_jacobian() const {
        return this->rotation_gyroscope;
    }

    const math::matrix<double, 3, 3>& preintegration::get_velocity_gyroscope_jacobian() const {
        return this->velocity_gyroscope;
    }

    const math::matrix<double, 3, 3>& preintegration::get_velocity_accelerometer_jacobian() const {
        return this->velocity_accelerometer;
    }

    const math::matrix<double, 3, 3>& preintegration::get_position_gyroscope_jacobian() const {
        return this->position_gyroscope;
    }

    const math::matrix<double, 3, 3>& preintegration::get_position_accelerometer_jacobian() const {
        return this->position_accelerometer;
    }

    math::so3<double> preintegration::get_delta_rotation(const bias& estimate) const {
        const math::matrix<double, 3, 1> gyroscope_change = estimate.gyroscope - this->linearisation.gyroscope;
        return (this->delta_rotation * math::so3<double>::exp(this->rotation_gyroscope * gyroscope_change)).normalised();
    }

    math::matrix<double, 3, 1> preintegration::get_delta_velocity(const bias& estimate) const {
        const math::matrix<double, 3, 1> gyroscope_change = estimate.gyroscope - this->linearisation.gyroscope;
        const math::matrix<double, 3, 1> accelerometer_change = estimate.accelerometer - this->linearisation.accelerometer;
        return this->delta_velocity + (this->velocity_gyroscope * gyroscope_change) + (this->velocity_accelerometer * accelerometer_change);
    }

    math::matrix<double, 3, 1> preintegration::get_delta_position(const bias& estimate) const {
        const math::matrix<double, 3, 1> gyroscope_change = estimate.gyroscope - this->linearisation.gyroscope;
        const math::matrix<double, 3, 1> accelerometer_change = estimate.accelerometer - this->linearisation.accelerometer;
        return this->delta_position + (this->position_gyroscope * gyroscope_change) + (this->position_accelerometer * accelerometer_change);
    }

    void preintegration::integrate_step(const step& measurement) {
        const double duration = measurement.duration;
        const double half_duration = 0.5 * duration;
        const double half_duration_squared = 0.5 * duration * duration;
        const math::matrix<double, 3, 1> acceleration = measurement.linear_acceleration - this->linearisation.accelerometer;
        const math::matrix<double, 3, 1> rotation_vector = duration * (measurement.angular_velocity - this->linearisation.gyroscope);
        const math::matrix<double, 3, 3> rotation = this->delta_rotation.get_matrix();
        const math::so3<double> step_rotation = math::so3<double>::exp(rotation_vector);
        const math::matrix<double, 3, 3> step_rotation_transpose = math::transpose(step_rotation.get_matrix());
        // The right jacobian of SO(3) is the left jacobian of the negated rotation vector.
        const math::matrix<double, 3, 3> step_right_jacobian = math::so3<double>::left_jacobian(-1.0 * rotation_vector);
        // The specific force of the step is turned into the frame at i by the rotation at the middle of the step, dR * Exp(w dt / 2).
        // The paper turns it by the rotation at the start, dR, which leaves an error of dR [w dt / 2]x a dt in every step: with
        // gravity in the specific force, 1.6 cm/s after 1.5 s of the simulated motion in the test at 200 Hz, against 0.006 cm/s from
        // the middle. The middle depends on the gyroscope too, which the jacobians below carry.
        const math::matrix<double, 3, 1> half_rotation_vector = 0.5 * rotation_vector;
        const math::matrix<double, 3, 3> half_rotation = math::so3<double>::exp(half_rotation_vector).get_matrix();
        const math::matrix<double, 3, 3> half_right_jacobian = math::so3<double>::left_jacobian(-1.0 * half_rotation_vector);
        const math::matrix<double, 3, 3> middle_rotation = rotation * half_rotation;
        const math::matrix<double, 3, 1> middle_acceleration = middle_rotation * acceleration;
        // d(R_mid a) = -R_mid [a]x (Exp(w dt / 2)^T dphi - J_r(w dt / 2) dt / 2 n_g) + R_mid n_a, and R_mid [a]x Exp(w dt / 2)^T is
        // dR [Exp(w dt / 2) a]x.
        const math::matrix<double, 3, 3> rotation_skew = rotation * skew(half_rotation * acceleration);
        const math::matrix<double, 3, 3> middle_skew = middle_rotation * skew(acceleration);
        const math::matrix<double, 3, 3> identity = math::matrix<double, 3, 3>::identity();

        // The errors propagate as x' = A x + B_g n_g + B_a n_a, where the white noise of a step of duration dt has the density
        // squared over dt as its variance (eq. 62 of the paper), so a step contributes the density squared times dt whatever its
        // length.
        math::matrix<double, 9, 9> transition = math::matrix<double, 9, 9>::identity();
        set_part(transition, 0, 0, step_rotation_transpose);
        set_part(transition, 1, 0, -duration * rotation_skew);
        set_part(transition, 2, 0, -half_duration_squared * rotation_skew);
        set_part(transition, 2, 1, duration * identity);
        const math::matrix<double, 3, 3> middle_gyroscope = (-half_duration * duration) * (middle_skew * half_right_jacobian);
        math::matrix<double, 9, 3> gyroscope_noise = math::matrix<double, 9, 3>::zero();
        math::set_block<double, 3, 3>(gyroscope_noise, 0, 0, duration * step_right_jacobian);
        math::set_block<double, 3, 3>(gyroscope_noise, 3, 0, middle_gyroscope);
        math::set_block<double, 3, 3>(gyroscope_noise, 6, 0, half_duration * middle_gyroscope);
        math::matrix<double, 9, 3> accelerometer_noise = math::matrix<double, 9, 3>::zero();
        math::set_block<double, 3, 3>(accelerometer_noise, 3, 0, duration * middle_rotation);
        math::set_block<double, 3, 3>(accelerometer_noise, 6, 0, half_duration_squared * middle_rotation);
        const double gyroscope_variance = this->noise.gyroscope_noise_density * this->noise.gyroscope_noise_density / duration;
        const double accelerometer_variance = this->noise.accelerometer_noise_density * this->noise.accelerometer_noise_density / duration;
        this->covariance = symmetric((transition * this->covariance * math::transpose(transition)) + (gyroscope_variance * (gyroscope_noise * math::transpose(gyroscope_noise))) + (accelerometer_variance * (accelerometer_noise * math::transpose(accelerometer_noise))));

        // The bias jacobians (appendix B of the paper, with the middle rotation), position first as it reads the velocity's before
        // they change.
        const math::matrix<double, 3, 3> middle_rotation_gyroscope = (math::transpose(half_rotation) * this->rotation_gyroscope) - (half_duration * half_right_jacobian);
        this->position_accelerometer = this->position_accelerometer + (duration * this->velocity_accelerometer) - (half_duration_squared * middle_rotation);
        this->position_gyroscope = this->position_gyroscope + (duration * this->velocity_gyroscope) - (half_duration_squared * (middle_skew * middle_rotation_gyroscope));
        this->velocity_accelerometer = this->velocity_accelerometer - (duration * middle_rotation);
        this->velocity_gyroscope = this->velocity_gyroscope - (duration * (middle_skew * middle_rotation_gyroscope));
        this->rotation_gyroscope = (step_rotation_transpose * this->rotation_gyroscope) - (duration * step_right_jacobian);

        // The deltas.
        this->delta_position = this->delta_position + (duration * this->delta_velocity) + (half_duration_squared * middle_acceleration);
        this->delta_velocity = this->delta_velocity + (duration * middle_acceleration);
        this->delta_rotation = (this->delta_rotation * step_rotation).normalised();
        this->delta_time += duration;
    }
}
