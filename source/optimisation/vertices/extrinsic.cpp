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

#include "optimisation/vertices/extrinsic.hpp"

namespace optimisation::vertices {
    void extrinsic::plus(double* const parameters, const double* const delta) const {
        const math::se3<double> current = extrinsic::from_parameters(parameters);
        const math::se3<double> update = math::se3<double>::exp({ { delta[0], delta[1], delta[2], delta[3], delta[4], delta[5] } });
        const math::se3<double> next = update * current;
        double next_parameters[extrinsic::parameter_count];
        extrinsic::to_parameters(math::se3<double>(next.rotation().normalised(), next.translation()), next_parameters);
        for (int i = 0; i < extrinsic::parameter_count; ++i) {
            parameters[i] = next_parameters[i];
        }
    }

    void extrinsic::to_parameters(const math::se3<double>& body_from_sensor, double (&parameters)[parameter_count]) {
        const math::matrix<double, 4, 1> quaternion = body_from_sensor.rotation().get_quaternion();
        parameters[0] = body_from_sensor.translation()[0];
        parameters[1] = body_from_sensor.translation()[1];
        parameters[2] = body_from_sensor.translation()[2];
        parameters[3] = quaternion[1];
        parameters[4] = quaternion[2];
        parameters[5] = quaternion[3];
        parameters[6] = quaternion[0];
    }

    math::se3<double> extrinsic::from_parameters(const double* const parameters) {
        return math::se3<double>(math::so3<double>(parameters[6], parameters[3], parameters[4], parameters[5]), { { parameters[0], parameters[1], parameters[2] } });
    }
}
