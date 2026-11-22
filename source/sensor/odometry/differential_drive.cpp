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

#include "sensor/odometry/differential_drive.hpp"

#include "math/math.hpp"

namespace sensor::odometry {
    const char* differential_drive::parameter_name(const size_t index) {
        switch (index) {
            case differential_drive::left_radius:
                return "left_radius";
            case differential_drive::right_radius:
                return "right_radius";
            case differential_drive::track_width:
                return "track_width";
        }
        return "";
    }

    bool differential_drive::parameter_is_angle(const size_t index) {
        static_cast<void>(index);
        return false;
    }

    void differential_drive::inputs(const sample& from, const sample& to, double* const values) const {
        values[0] = to.rotation[0] - from.rotation[0];
        values[1] = to.rotation[1] - from.rotation[1];
    }

    void differential_drive::input_variances(const double* const parameters, const double* const values, const noise& model, double* const variances) const {
        for (size_t wheel = 0; wheel < 2; ++wheel) {
            const double radius = math::abs(parameters[wheel]);
            variances[wheel] = (radius > 0.0) ? ((model.slip * model.slip * math::abs(values[wheel])) / radius) : 0.0;
        }
    }

    bool differential_drive::motion(const double* const parameters, const double* const values, double& distance, double& heading, double* const jacobian_inputs, double* const jacobian_parameters) const {
        const double left_radius_value = parameters[differential_drive::left_radius];
        const double right_radius_value = parameters[differential_drive::right_radius];
        const double track = parameters[differential_drive::track_width];
        if (!(left_radius_value > 0.0) || !(right_radius_value > 0.0) || !(track > 0.0)) {
            return false;
        }
        const double left = left_radius_value * values[0];
        const double right = right_radius_value * values[1];
        distance = 0.5 * (left + right);
        heading = (right - left) / track;
        if (jacobian_inputs != nullptr) {
            jacobian_inputs[0] = 0.5 * left_radius_value;
            jacobian_inputs[1] = 0.5 * right_radius_value;
            jacobian_inputs[2] = -left_radius_value / track;
            jacobian_inputs[3] = right_radius_value / track;
        }
        if (jacobian_parameters != nullptr) {
            jacobian_parameters[0] = 0.5 * values[0];
            jacobian_parameters[1] = 0.5 * values[1];
            jacobian_parameters[2] = 0.0;
            jacobian_parameters[3] = -values[0] / track;
            jacobian_parameters[4] = values[1] / track;
            jacobian_parameters[5] = -heading / track;
        }
        return true;
    }
}
