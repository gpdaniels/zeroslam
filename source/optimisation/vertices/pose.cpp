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

#include "optimisation/vertices/pose.hpp"

#include "math/lie.hpp"

namespace optimisation::vertices {
    void pose::plus(double* const parameters, const double* const delta) const {
        const math::so3<double> current_rotation(parameters[6], parameters[3], parameters[4], parameters[5]);
        const math::matrix<double, 3, 1> current_translation = { { parameters[0], parameters[1], parameters[2] } };
        const math::se3<double> current_pose(current_rotation, current_translation);
        const math::se3<double> update = math::se3<double>::exp({ { delta[0], delta[1], delta[2], delta[3], delta[4], delta[5] } });
        const math::se3<double> next_pose = update * current_pose;
        parameters[0] = next_pose.translation()[0];
        parameters[1] = next_pose.translation()[1];
        parameters[2] = next_pose.translation()[2];
        parameters[3] = next_pose.rotation().get_quaternion()[1];
        parameters[4] = next_pose.rotation().get_quaternion()[2];
        parameters[5] = next_pose.rotation().get_quaternion()[3];
        parameters[6] = next_pose.rotation().get_quaternion()[0];
    }
}
