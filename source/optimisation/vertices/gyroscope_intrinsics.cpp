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

#include "optimisation/vertices/gyroscope_intrinsics.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace optimisation::vertices {
    void gyroscope_intrinsics::plus(double* const parameters, const double* const delta) const {
        for (int i = 0; i < gyroscope_intrinsics::local_count; ++i) {
            parameters[i] += delta[i];
        }
    }

    void gyroscope_intrinsics::to_parameters(const sensor::imu::intrinsics& intrinsics, double (&parameters)[parameter_count]) {
        for (size_t i = 0; i < static_cast<size_t>(gyroscope_intrinsics::parameter_count); ++i) {
            parameters[i] = intrinsics.gyroscope.data()[i];
        }
    }

    sensor::imu::intrinsics gyroscope_intrinsics::from_parameters(const double* const parameters) {
        sensor::imu::intrinsics intrinsics;
        for (size_t i = 0; i < static_cast<size_t>(gyroscope_intrinsics::parameter_count); ++i) {
            intrinsics.gyroscope.data()[i] = parameters[i];
        }
        return intrinsics;
    }
}
