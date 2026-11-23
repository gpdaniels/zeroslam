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

#include "optimisation/edges/bias_random_walk.hpp"

namespace optimisation::edges {
    math::matrix<double, 0, 0> bias_random_walk::information(const sensor::imu::parameters& noise, const double duration) {
        math::matrix<double, 0, 0> result = math::matrix<double, 0, 0>::zero(6, 6);
        const double gyroscope = 1.0 / noise.gyroscope_walk_variance(duration);
        const double accelerometer = 1.0 / noise.accelerometer_walk_variance(duration);
        for (size_t i = 0; i < 3; ++i) {
            result[i][i] = gyroscope;
            result[3 + i][3 + i] = accelerometer;
        }
        return result;
    }

    void bias_random_walk::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const double* const first = context.get_vertex(0)->get_parameters();
        const double* const second = context.get_vertex(1)->get_parameters();
        for (size_t i = 0; i < 6; ++i) {
            residual[i][0] = second[i] - first[i];
        }
    }

    void bias_random_walk::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        static_cast<void>(context);
        double first[36] = {};
        double second[36] = {};
        for (size_t i = 0; i < 6; ++i) {
            first[(i * 6) + i] = -1.0;
            second[(i * 6) + i] = 1.0;
        }
        edge::set_jacobian(jacobians[0], 6, 6, &first[0]);
        edge::set_jacobian(jacobians[1], 6, 6, &second[0]);
    }
}
