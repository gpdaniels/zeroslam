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

#include "sensor/odometry/ackermann.hpp"

#include "math/math.hpp"

namespace sensor::odometry {
    ackermann::ackermann(const measured_wheel measured)
        : wheel(measured) {
    }

    ackermann::measured_wheel ackermann::get_measured_wheel() const {
        return this->wheel;
    }

    const char* ackermann::parameter_name(const size_t index) {
        switch (index) {
            case ackermann::radius:
                return "radius";
            case ackermann::wheelbase:
                return "wheelbase";
            case ackermann::steering_scale:
                return "steering_scale";
            case ackermann::steering_offset:
                return "steering_offset";
        }
        return "";
    }

    bool ackermann::parameter_is_angle(const size_t index) {
        return index == ackermann::steering_offset;
    }

    void ackermann::inputs(const sample& from, const sample& to, double* const values) const {
        values[0] = to.rotation[0] - from.rotation[0];
        values[1] = 0.5 * (from.steering + to.steering);
    }

    void ackermann::input_variances(const double* const parameters, const double* const values, const noise& model, double* const variances) const {
        // The slip's variance grows with the distance the wheel rolled, r |phi|, and is a variance of the rotation once divided by r^2.
        const double radius_value = math::abs(parameters[ackermann::radius]);
        variances[0] = (radius_value > 0.0) ? ((model.slip * model.slip * math::abs(values[0])) / radius_value) : 0.0;
        // A reading ends one increment and starts the next, so its noise enters two means by halves, which together carry it once.
        variances[1] = model.steering * model.steering;
    }

    bool ackermann::motion(const double* const parameters, const double* const values, double& distance, double& heading, double* const jacobian_inputs, double* const jacobian_parameters) const {
        const double radius_value = parameters[ackermann::radius];
        const double wheelbase_value = parameters[ackermann::wheelbase];
        const double scale = parameters[ackermann::steering_scale];
        const double offset = parameters[ackermann::steering_offset];
        const double rotation = values[0];
        const double reading = values[1];
        const double angle = (scale * reading) + offset;
        if (!(radius_value > 0.0) || !(wheelbase_value > 0.0) || !(math::abs(angle) <= ackermann::maximum_steering_angle)) {
            return false;
        }
        double sine = 0.0;
        double cosine = 1.0;
        math::sincos(angle, sine, cosine);
        const double rolled = radius_value * rotation;

        // The derivatives of distance and heading by the rolled distance and by the steering angle, then chained.
        double distance_rolled = 0.0;
        double distance_angle = 0.0;
        double heading_rolled = 0.0;
        double heading_angle = 0.0;
        if (this->wheel == measured_wheel::rear_axle) {
            const double tangent = sine / cosine;
            distance = rolled;
            heading = (rolled * tangent) / wheelbase_value;
            distance_rolled = 1.0;
            distance_angle = 0.0;
            heading_rolled = tangent / wheelbase_value;
            heading_angle = rolled / (wheelbase_value * cosine * cosine);
        }
        else {
            distance = rolled * cosine;
            heading = (rolled * sine) / wheelbase_value;
            distance_rolled = cosine;
            distance_angle = -rolled * sine;
            heading_rolled = sine / wheelbase_value;
            heading_angle = (rolled * cosine) / wheelbase_value;
        }

        if (jacobian_inputs != nullptr) {
            jacobian_inputs[0] = distance_rolled * radius_value;
            jacobian_inputs[1] = distance_angle * scale;
            jacobian_inputs[2] = heading_rolled * radius_value;
            jacobian_inputs[3] = heading_angle * scale;
        }
        if (jacobian_parameters != nullptr) {
            jacobian_parameters[0] = distance_rolled * rotation;
            jacobian_parameters[1] = 0.0;
            jacobian_parameters[2] = distance_angle * reading;
            jacobian_parameters[3] = distance_angle;
            jacobian_parameters[4] = heading_rolled * rotation;
            jacobian_parameters[5] = -heading / wheelbase_value;
            jacobian_parameters[6] = heading_angle * reading;
            jacobian_parameters[7] = heading_angle;
        }
        return true;
    }
}
