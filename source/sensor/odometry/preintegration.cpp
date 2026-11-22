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

#include "math/math.hpp"

namespace sensor::odometry {
    preintegration::preintegration()
        : kinematics()
        , noise_model()
        , parameter_count(0)
        , linearisation{}
        , samples()
        , delta(math::matrix<double, 3, 1>::zero())
        , covariance(math::matrix<double, 3, 3>::zero())
        , jacobian(math::matrix<double, 3, maximum_parameters>::zero())
        , distance(0.0)
        , valid(false) {
    }

    preintegration::preintegration(const model& kinematics_value, const noise& noise_value, const double* const parameters, const size_t parameters_length)
        : preintegration() {
        this->kinematics = kinematics_value;
        this->noise_model = noise_value;
        this->parameter_count = kinematics_value.parameter_count();
        for (size_t i = 0; (i < this->parameter_count) && (i < parameters_length) && (parameters != nullptr); ++i) {
            this->linearisation[i] = parameters[i];
        }
    }

    bool preintegration::integrate(const sample* const sample_values, const size_t sample_count, const long long int start, const long long int end) {
        this->valid = false;
        this->samples.clear();
        if (!this->kinematics.is_valid() || (sample_values == nullptr) || (sample_count == 0) || (start > end)) {
            return false;
        }
        if ((sample_values[0].timestamp > start) || (sample_values[sample_count - 1].timestamp < end)) {
            return false;
        }
        for (size_t i = 1; i < sample_count; ++i) {
            if (sample_values[i].timestamp <= sample_values[i - 1].timestamp) {
                return false;
            }
        }

        // The last sample at or before the start, then the samples strictly inside the interval, the ends interpolated.
        size_t first = 0;
        while (((first + 1) < sample_count) && (sample_values[first + 1].timestamp <= start)) {
            ++first;
        }
        if ((first + 1) < sample_count) {
            this->samples.push_back(sample::interpolate(sample_values[first], sample_values[first + 1], start));
        }
        else {
            this->samples.push_back(sample_values[first]);
        }
        this->samples.back().timestamp = start;
        if (end > start) {
            size_t next = first + 1;
            while ((next < sample_count) && (sample_values[next].timestamp < end)) {
                this->samples.push_back(sample_values[next]);
                ++next;
            }
            this->samples.push_back(sample::interpolate(sample_values[next - 1], sample_values[next], end));
            this->samples.back().timestamp = end;
        }
        return this->integrate_samples();
    }

    bool preintegration::reintegrate(const double* const parameters, const size_t parameters_length) {
        if ((parameters == nullptr) || (parameters_length != this->parameter_count) || this->samples.empty()) {
            return false;
        }
        for (size_t i = 0; i < this->parameter_count; ++i) {
            this->linearisation[i] = parameters[i];
        }
        return this->integrate_samples();
    }

    bool preintegration::merge(const preintegration& next) {
        if (!next.valid) {
            return false;
        }
        if (!this->valid) {
            *this = next;
            return true;
        }
        if ((this->kinematics != next.kinematics) || (this->parameter_count != next.parameter_count) || (this->get_end() != next.get_start())) {
            return false;
        }
        preintegration following = next;
        bool same_linearisation = true;
        for (size_t i = 0; i < this->parameter_count; ++i) {
            same_linearisation = same_linearisation && (this->linearisation[i] == next.linearisation[i]);
        }
        if (!same_linearisation) {
            following.noise_model = this->noise_model;
            if (!following.reintegrate(&this->linearisation[0], this->parameter_count)) {
                return false;
            }
        }
        math::matrix<double, 3, 3> jacobian_first;
        math::matrix<double, 3, 3> jacobian_second;
        this->delta = preintegration::compose(this->delta, following.delta, &jacobian_first, &jacobian_second);
        this->covariance = (jacobian_first * this->covariance * math::transpose(jacobian_first)) + (jacobian_second * following.covariance * math::transpose(jacobian_second));
        this->jacobian = (jacobian_first * this->jacobian) + (jacobian_second * following.jacobian);
        this->distance += following.distance;
        this->samples.insert(this->samples.end(), following.samples.begin() + 1, following.samples.end());
        return true;
    }

    bool preintegration::needs_reintegration(const double* const parameters, const size_t parameters_length, const double tolerance) const {
        if ((parameters == nullptr) || (parameters_length != this->parameter_count)) {
            return true;
        }
        for (size_t i = 0; i < this->parameter_count; ++i) {
            const double change = math::abs(parameters[i] - this->linearisation[i]);
            const double limit = this->kinematics.parameter_is_angle(i) ? tolerance : (tolerance * math::max(math::abs(this->linearisation[i]), 1.0e-9));
            if (!(change <= limit)) {
                return true;
            }
        }
        return false;
    }

    math::matrix<double, 3, 1> preintegration::predict(const double* const parameters, const size_t parameters_length) const {
        math::matrix<double, 3, 1> result = this->delta;
        if ((parameters == nullptr) || (parameters_length != this->parameter_count)) {
            return result;
        }
        for (size_t row = 0; row < 3; ++row) {
            for (size_t i = 0; i < this->parameter_count; ++i) {
                result[row] += this->jacobian[row][i] * (parameters[i] - this->linearisation[i]);
            }
        }
        return result;
    }

    bool preintegration::is_valid() const {
        return this->valid;
    }

    const model& preintegration::get_model() const {
        return this->kinematics;
    }

    const noise& preintegration::get_noise() const {
        return this->noise_model;
    }

    size_t preintegration::get_parameter_count() const {
        return this->parameter_count;
    }

    const double* preintegration::get_linearisation() const {
        return &this->linearisation[0];
    }

    const std::vector<sample>& preintegration::get_samples() const {
        return this->samples;
    }

    long long int preintegration::get_start() const {
        return this->samples.empty() ? 0 : this->samples.front().timestamp;
    }

    long long int preintegration::get_end() const {
        return this->samples.empty() ? 0 : this->samples.back().timestamp;
    }

    const math::matrix<double, 3, 1>& preintegration::get_delta() const {
        return this->delta;
    }

    const math::matrix<double, 3, 3>& preintegration::get_covariance() const {
        return this->covariance;
    }

    const math::matrix<double, 3, preintegration::maximum_parameters>& preintegration::get_jacobian() const {
        return this->jacobian;
    }

    double preintegration::get_distance() const {
        return this->distance;
    }

    void preintegration::arc(const double length, const double heading, double& dx, double& dy, double* const jacobian_arc) {
        // sin(h) / h, (1 - cos(h)) / h and their derivatives, by their series where the closed forms cancel.
        double sinc = 1.0;
        double cosc = 0.0;
        double sinc_derivative = 0.0;
        double cosc_derivative = 0.5;
        const double h = heading;
        const double h2 = h * h;
        if (math::abs(h) < 1.0e-2) {
            sinc = 1.0 - (h2 / 6.0) * (1.0 - (h2 / 20.0) * (1.0 - (h2 / 42.0)));
            cosc = (h / 2.0) * (1.0 - (h2 / 12.0) * (1.0 - (h2 / 30.0) * (1.0 - (h2 / 56.0))));
            sinc_derivative = -(h / 3.0) * (1.0 - (h2 / 10.0) * (1.0 - (h2 / 28.0) * (1.0 - (h2 / 54.0))));
            cosc_derivative = 0.5 - (h2 / 8.0) * (1.0 - (h2 / 18.0) * (1.0 - (h2 / 40.0)));
        }
        else {
            double sine = 0.0;
            double cosine = 1.0;
            math::sincos(h, sine, cosine);
            double half_sine = 0.0;
            double half_cosine = 1.0;
            math::sincos(0.5 * h, half_sine, half_cosine);
            const double one_minus_cosine = 2.0 * half_sine * half_sine;
            sinc = sine / h;
            cosc = one_minus_cosine / h;
            sinc_derivative = ((h * cosine) - sine) / h2;
            cosc_derivative = ((h * sine) - one_minus_cosine) / h2;
        }
        dx = length * sinc;
        dy = length * cosc;
        if (jacobian_arc != nullptr) {
            jacobian_arc[0] = sinc;
            jacobian_arc[1] = length * sinc_derivative;
            jacobian_arc[2] = cosc;
            jacobian_arc[3] = length * cosc_derivative;
        }
    }

    math::matrix<double, 3, 1> preintegration::compose(const math::matrix<double, 3, 1>& first, const math::matrix<double, 3, 1>& second, math::matrix<double, 3, 3>* const jacobian_first, math::matrix<double, 3, 3>* const jacobian_second) {
        double sine = 0.0;
        double cosine = 1.0;
        math::sincos(first[2], sine, cosine);
        const double rotated_x = (cosine * second[0]) - (sine * second[1]);
        const double rotated_y = (sine * second[0]) + (cosine * second[1]);
        if (jacobian_first != nullptr) {
            *jacobian_first = math::matrix<double, 3, 3>::identity();
            (*jacobian_first)[0][2] = -rotated_y;
            (*jacobian_first)[1][2] = rotated_x;
        }
        if (jacobian_second != nullptr) {
            *jacobian_second = math::matrix<double, 3, 3>::identity();
            (*jacobian_second)[0][0] = cosine;
            (*jacobian_second)[0][1] = -sine;
            (*jacobian_second)[1][0] = sine;
            (*jacobian_second)[1][1] = cosine;
        }
        return { { first[0] + rotated_x, first[1] + rotated_y, first[2] + second[2] } };
    }

    bool preintegration::integrate_samples() {
        this->valid = false;
        this->delta = math::matrix<double, 3, 1>::zero();
        this->covariance = math::matrix<double, 3, 3>::zero();
        this->jacobian = math::matrix<double, 3, maximum_parameters>::zero();
        this->distance = 0.0;
        if (this->samples.empty()) {
            return false;
        }
        const size_t inputs = this->kinematics.input_count();
        const size_t parameters = this->parameter_count;
        for (size_t k = 0; (k + 1) < this->samples.size(); ++k) {
            double values[model::maximum_inputs] = {};
            double variances[model::maximum_inputs] = {};
            double jacobian_inputs[2 * model::maximum_inputs] = {};
            double jacobian_parameters[2 * model::maximum_parameters] = {};
            double length = 0.0;
            double heading = 0.0;
            this->kinematics.inputs(this->samples[k], this->samples[k + 1], &values[0]);
            if (!this->kinematics.motion(&this->linearisation[0], &values[0], length, heading, &jacobian_inputs[0], &jacobian_parameters[0])) {
                return false;
            }
            this->kinematics.input_variances(&this->linearisation[0], &values[0], this->noise_model, &variances[0]);

            double dx = 0.0;
            double dy = 0.0;
            double jacobian_arc[4] = {};
            preintegration::arc(length, heading, dx, dy, &jacobian_arc[0]);

            // The increment (dx, dy, dheading) by (length, heading).
            math::matrix<double, 3, 2> jacobian_increment = math::matrix<double, 3, 2>::zero();
            jacobian_increment[0][0] = jacobian_arc[0];
            jacobian_increment[0][1] = jacobian_arc[1];
            jacobian_increment[1][0] = jacobian_arc[2];
            jacobian_increment[1][1] = jacobian_arc[3];
            jacobian_increment[2][1] = 1.0;

            // The covariance of (length, heading) from the inputs, and the increment's with the lateral slip and heading noise.
            math::matrix<double, 2, 2> motion_covariance = math::matrix<double, 2, 2>::zero();
            for (size_t r = 0; r < 2; ++r) {
                for (size_t c = 0; c < 2; ++c) {
                    double sum = 0.0;
                    for (size_t i = 0; i < inputs; ++i) {
                        sum += jacobian_inputs[(r * inputs) + i] * variances[i] * jacobian_inputs[(c * inputs) + i];
                    }
                    motion_covariance[r][c] = sum;
                }
            }
            math::matrix<double, 3, 3> increment_covariance = jacobian_increment * motion_covariance * math::transpose(jacobian_increment);
            const double travelled = math::abs(length);
            increment_covariance[1][1] += this->noise_model.lateral * this->noise_model.lateral * travelled;
            increment_covariance[2][2] += this->noise_model.heading * this->noise_model.heading * travelled;

            // The increment by the parameters.
            math::matrix<double, 3, maximum_parameters> increment_parameters = math::matrix<double, 3, maximum_parameters>::zero();
            for (size_t i = 0; i < parameters; ++i) {
                const double length_parameter = jacobian_parameters[i];
                const double heading_parameter = jacobian_parameters[parameters + i];
                increment_parameters[0][i] = (jacobian_increment[0][0] * length_parameter) + (jacobian_increment[0][1] * heading_parameter);
                increment_parameters[1][i] = (jacobian_increment[1][0] * length_parameter) + (jacobian_increment[1][1] * heading_parameter);
                increment_parameters[2][i] = heading_parameter;
            }

            math::matrix<double, 3, 3> jacobian_first;
            math::matrix<double, 3, 3> jacobian_second;
            this->delta = preintegration::compose(this->delta, { { dx, dy, heading } }, &jacobian_first, &jacobian_second);
            this->covariance = (jacobian_first * this->covariance * math::transpose(jacobian_first)) + (jacobian_second * increment_covariance * math::transpose(jacobian_second));
            this->jacobian = (jacobian_first * this->jacobian) + (jacobian_second * increment_parameters);
            this->distance += travelled;
        }
        this->valid = true;
        return true;
    }
}
