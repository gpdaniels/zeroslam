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
#ifndef ZEROSLAM_SENSOR_ODOMETRY_PREINTEGRATION_HPP
#define ZEROSLAM_SENSOR_ODOMETRY_PREINTEGRATION_HPP

#include "math/matrix.hpp"
#include "sensor/odometry/model.hpp"
#include "sensor/odometry/sample.hpp"

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

namespace sensor::odometry {
    // The planar motion of the odometry frame between two times, integrated from the wheel samples between them: the pose
    // (x, y, theta) of the frame at the end in the frame at the start, metres and radians, theta not wrapped so that a long
    // interval keeps its turns. The samples are taken linearly in time at the two ends, each increment between two samples is an
    // arc of constant curvature, and the increments compose on SE2.
    //
    // With the motion come its covariance, propagated from each increment's noise (the model's input noise, the lateral slip and
    // the heading noise), and its Jacobian by the calibration parameters, so that a factor built from it stays valid as the
    // calibration moves: to first order the motion at parameters p is delta + J (p - p0) for the parameters p0 it was integrated
    // with. When p moves further than that holds the samples, which the preintegration keeps, are integrated again.
    class preintegration final {
    public:
        constexpr static const size_t maximum_parameters = model::maximum_parameters;
        // The relative change of a length or a gain, or the change of an angle in radians, past which integrating again is needed.
        constexpr static const double default_tolerance = 1.0e-3;

    private:
        model kinematics;
        noise noise_model;
        size_t parameter_count;
        double linearisation[maximum_parameters];
        std::vector<sample> samples;
        math::matrix<double, 3, 1> delta;
        math::matrix<double, 3, 3> covariance;
        math::matrix<double, 3, maximum_parameters> jacobian;
        double distance;
        bool valid;

    public:
        preintegration();
        preintegration(const model& kinematics_value, const noise& noise_value, const double* const parameters, const size_t parameters_length);

    public:
        // Integrate the motion from start to end (nanoseconds, start <= end) from samples sorted by strictly increasing time that
        // span the interval, the first at or before the start and the last at or after the end. Fails, leaving the preintegration
        // invalid, on samples that do not, on a model that is not valid, or on parameters the model refuses.
        bool integrate(const sample* const sample_values, const size_t sample_count, const long long int start, const long long int end);

        // Integrate the kept samples again with other parameters, which become the linearisation point.
        bool reintegrate(const double* const parameters, const size_t parameters_length);

        // Append the motion that follows this one, from this one's end, as when the keyframe between them is culled. The result
        // is the motion of both, its covariance and Jacobian composed; a following motion integrated with other parameters is
        // integrated again with these first. An invalid preintegration takes the following one as it is.
        bool merge(const preintegration& next);

        // Whether the parameters have moved from the linearisation point further than the first order correction holds.
        bool needs_reintegration(const double* const parameters, const size_t parameters_length, const double tolerance = default_tolerance) const;

        // The motion at other parameters, to first order from the linearisation point.
        math::matrix<double, 3, 1> predict(const double* const parameters, const size_t parameters_length) const;

    public:
        bool is_valid() const;
        const model& get_model() const;
        const noise& get_noise() const;
        size_t get_parameter_count() const;
        const double* get_linearisation() const;
        const std::vector<sample>& get_samples() const;
        long long int get_start() const;
        long long int get_end() const;
        const math::matrix<double, 3, 1>& get_delta() const;
        const math::matrix<double, 3, 3>& get_covariance() const;
        // The Jacobian of the motion by the parameters, the columns past the model's parameter count zero.
        const math::matrix<double, 3, maximum_parameters>& get_jacobian() const;
        // The distance the reference point travelled, the sum of the arc lengths whatever their direction.
        double get_distance() const;

    public:
        // The displacement (dx, dy) along an arc of the given length and change of heading, with its Jacobian by the two (row
        // major 2 x 2) when not null.
        static void arc(const double length, const double heading, double& dx, double& dy, double* const jacobian_arc);

        // The pose that follows the pose first by the motion second, with the Jacobians of the result by first and second (row
        // major 3 x 3) when not null.
        static math::matrix<double, 3, 1> compose(const math::matrix<double, 3, 1>& first, const math::matrix<double, 3, 1>& second, math::matrix<double, 3, 3>* const jacobian_first, math::matrix<double, 3, 3>* const jacobian_second);

    private:
        bool integrate_samples();
    };
}

#endif // ZEROSLAM_SENSOR_ODOMETRY_PREINTEGRATION_HPP
