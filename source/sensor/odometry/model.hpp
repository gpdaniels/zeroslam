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
#ifndef ZEROSLAM_SENSOR_ODOMETRY_MODEL_HPP
#define ZEROSLAM_SENSOR_ODOMETRY_MODEL_HPP

#include "sensor/odometry/ackermann.hpp"
#include "sensor/odometry/differential_drive.hpp"
#include "sensor/odometry/sample.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor::odometry {
    // One of the vehicle kinematic models, chosen at run time: the preintegration, the factor graph and the interface hold this
    // rather than a model type, so that adding a model touches only this dispatch.
    class model final {
    public:
        enum class kind {
            none,
            ackermann,
            differential_drive
        };

        constexpr static const size_t maximum_parameters = 8;
        constexpr static const size_t maximum_inputs = 2;

    private:
        kind model_kind;
        odometry::ackermann ackermann_model;
        odometry::differential_drive differential_drive_model;

    public:
        model();
        explicit model(const odometry::ackermann& ackermann_value);
        explicit model(const odometry::differential_drive& differential_drive_value);

    public:
        bool is_valid() const;
        kind get_kind() const;
        const char* name() const;
        size_t parameter_count() const;
        size_t input_count() const;
        const char* parameter_name(const size_t index) const;
        bool parameter_is_angle(const size_t index) const;
        const odometry::ackermann& get_ackermann() const;

        bool operator==(const model& other) const;
        bool operator!=(const model& other) const;

        void inputs(const sample& from, const sample& to, double* const values) const;
        void input_variances(const double* const parameters, const double* const values, const noise& noise_model, double* const variances) const;
        bool motion(const double* const parameters, const double* const values, double& distance, double& heading, double* const jacobian_inputs, double* const jacobian_parameters) const;
    };

    // The calibration of a model: a value for each of its parameters, which of them are free to be estimated, and the standard
    // deviation of a prior on each (zero for none). A parameter that is not free is held at its value: with a monocular camera the
    // wheel radii, the wheelbase or the track width and the steering gain are not all observable together (a turn shows the ratio
    // of the distance rolled to the wheelbase or track width), so one length is measured and held, or given a tight prior.
    class calibration final {
    public:
        size_t count = 0;
        double values[model::maximum_parameters] = {};
        bool free[model::maximum_parameters] = {};
        double prior_sigma[model::maximum_parameters] = {};

    public:
        calibration();
        // The calibration of the model with the given values, every parameter held and none with a prior.
        calibration(const model& kinematics, const double* const parameter_values, const size_t parameter_count);

    public:
        // The free parameters as bits, bit i for parameter i.
        unsigned int free_mask() const;
        size_t free_count() const;
    };
}

#endif // ZEROSLAM_SENSOR_ODOMETRY_MODEL_HPP
