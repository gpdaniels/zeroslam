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

#include "sensor/odometry/model.hpp"

namespace sensor::odometry {
    static_assert(ackermann::parameter_count <= model::maximum_parameters, "The Ackermann parameters must fit a model.");
    static_assert(differential_drive::parameter_count <= model::maximum_parameters, "The differential drive parameters must fit a model.");
    static_assert(ackermann::input_count <= model::maximum_inputs, "The Ackermann inputs must fit a model.");
    static_assert(differential_drive::input_count <= model::maximum_inputs, "The differential drive inputs must fit a model.");

    model::model()
        : model_kind(kind::none)
        , ackermann_model()
        , differential_drive_model() {
    }

    model::model(const odometry::ackermann& ackermann_value)
        : model_kind(kind::ackermann)
        , ackermann_model(ackermann_value)
        , differential_drive_model() {
    }

    model::model(const odometry::differential_drive& differential_drive_value)
        : model_kind(kind::differential_drive)
        , ackermann_model()
        , differential_drive_model(differential_drive_value) {
    }

    bool model::is_valid() const {
        return this->model_kind != kind::none;
    }

    model::kind model::get_kind() const {
        return this->model_kind;
    }

    const char* model::name() const {
        switch (this->model_kind) {
            case kind::ackermann:
                return odometry::ackermann::name;
            case kind::differential_drive:
                return odometry::differential_drive::name;
            case kind::none:
                break;
        }
        return "";
    }

    size_t model::parameter_count() const {
        switch (this->model_kind) {
            case kind::ackermann:
                return odometry::ackermann::parameter_count;
            case kind::differential_drive:
                return odometry::differential_drive::parameter_count;
            case kind::none:
                break;
        }
        return 0;
    }

    size_t model::input_count() const {
        switch (this->model_kind) {
            case kind::ackermann:
                return odometry::ackermann::input_count;
            case kind::differential_drive:
                return odometry::differential_drive::input_count;
            case kind::none:
                break;
        }
        return 0;
    }

    const char* model::parameter_name(const size_t index) const {
        switch (this->model_kind) {
            case kind::ackermann:
                return odometry::ackermann::parameter_name(index);
            case kind::differential_drive:
                return odometry::differential_drive::parameter_name(index);
            case kind::none:
                break;
        }
        return "";
    }

    bool model::parameter_is_angle(const size_t index) const {
        switch (this->model_kind) {
            case kind::ackermann:
                return odometry::ackermann::parameter_is_angle(index);
            case kind::differential_drive:
                return odometry::differential_drive::parameter_is_angle(index);
            case kind::none:
                break;
        }
        return false;
    }

    const odometry::ackermann& model::get_ackermann() const {
        return this->ackermann_model;
    }

    bool model::operator==(const model& other) const {
        if (this->model_kind != other.model_kind) {
            return false;
        }
        return (this->model_kind != kind::ackermann) || (this->ackermann_model.get_measured_wheel() == other.ackermann_model.get_measured_wheel());
    }

    bool model::operator!=(const model& other) const {
        return !(*this == other);
    }

    void model::inputs(const sample& from, const sample& to, double* const values) const {
        switch (this->model_kind) {
            case kind::ackermann:
                this->ackermann_model.inputs(from, to, values);
                return;
            case kind::differential_drive:
                this->differential_drive_model.inputs(from, to, values);
                return;
            case kind::none:
                break;
        }
    }

    void model::input_variances(const double* const parameters, const double* const values, const noise& noise_model, double* const variances) const {
        switch (this->model_kind) {
            case kind::ackermann:
                this->ackermann_model.input_variances(parameters, values, noise_model, variances);
                return;
            case kind::differential_drive:
                this->differential_drive_model.input_variances(parameters, values, noise_model, variances);
                return;
            case kind::none:
                break;
        }
    }

    bool model::motion(const double* const parameters, const double* const values, double& distance, double& heading, double* const jacobian_inputs, double* const jacobian_parameters) const {
        switch (this->model_kind) {
            case kind::ackermann:
                return this->ackermann_model.motion(parameters, values, distance, heading, jacobian_inputs, jacobian_parameters);
            case kind::differential_drive:
                return this->differential_drive_model.motion(parameters, values, distance, heading, jacobian_inputs, jacobian_parameters);
            case kind::none:
                break;
        }
        return false;
    }

    calibration::calibration() {
    }

    calibration::calibration(const model& kinematics, const double* const parameter_values, const size_t parameter_count)
        : count(kinematics.parameter_count()) {
        for (size_t i = 0; (i < this->count) && (i < parameter_count) && (parameter_values != nullptr); ++i) {
            this->values[i] = parameter_values[i];
        }
    }

    unsigned int calibration::free_mask() const {
        unsigned int mask = 0;
        for (size_t i = 0; i < this->count; ++i) {
            if (this->free[i]) {
                mask |= (1u << i);
            }
        }
        return mask;
    }

    size_t calibration::free_count() const {
        size_t result = 0;
        for (size_t i = 0; i < this->count; ++i) {
            result += this->free[i] ? 1u : 0u;
        }
        return result;
    }
}
