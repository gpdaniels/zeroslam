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

#include "optimisation/vertices/similarity.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"

namespace optimisation::vertices {
    void similarity::plus(double* const parameters, const double* const delta) const {
        const math::so3<double> current_rotation(parameters[6], parameters[3], parameters[4], parameters[5]);
        const math::matrix<double, 3, 1> current_translation = { { parameters[0], parameters[1], parameters[2] } };
        const math::sim3<double> current_similarity(math::se3<double>(current_rotation, current_translation), parameters[7]);
        const math::sim3<double> update = math::sim3<double>::exp({ { delta[0], delta[1], delta[2], delta[3], delta[4], delta[5], delta[6] } });
        const math::sim3<double> next_similarity = update * current_similarity;
        parameters[0] = next_similarity.transformation().translation()[0];
        parameters[1] = next_similarity.transformation().translation()[1];
        parameters[2] = next_similarity.transformation().translation()[2];
        parameters[3] = next_similarity.transformation().rotation().get_quaternion()[1];
        parameters[4] = next_similarity.transformation().rotation().get_quaternion()[2];
        parameters[5] = next_similarity.transformation().rotation().get_quaternion()[3];
        parameters[6] = next_similarity.transformation().rotation().get_quaternion()[0];
        parameters[7] = next_similarity.scale();
    }
}
