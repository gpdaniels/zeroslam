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

#include "sensor/camera/pinhole.hpp"

#include "core/assert.hpp"

namespace sensor::camera {
    template <typename type>
    pinhole<type>::pinhole()
        : focal_lengths{ type(1), type(1) }
        , centre_points{ type(0.5), type(0.5) } {
    }

    template <typename type>
    pinhole<type>::pinhole(const type* const parameters, const size_t parameters_length)
        : pinhole() {
        const bool valid = this->set_parameters(parameters, parameters_length);
        ASSERT(valid, "The pinhole parameters must be four positive focal lengths and centre points.");
        static_cast<void>(valid);
    }

    template <typename type>
    size_t pinhole<type>::get_parameter_count() const {
        return pinhole::parameter_count;
    }

    template <typename type>
    bool pinhole<type>::set_parameters(const type* const parameters, const size_t parameters_length) {
        if ((parameters == nullptr) || (parameters_length != pinhole::parameter_count)) {
            return false;
        }
        if (!(parameters[0] > type(0)) || !(parameters[1] > type(0))) {
            return false;
        }
        this->focal_lengths[0] = parameters[0];
        this->focal_lengths[1] = parameters[1];
        this->centre_points[0] = parameters[2];
        this->centre_points[1] = parameters[3];
        return true;
    }

    template <typename type>
    bool pinhole<type>::get_parameters(type* const parameters, const size_t parameters_length) const {
        if ((parameters == nullptr) || (parameters_length != pinhole::parameter_count)) {
            return false;
        }
        parameters[0] = this->focal_lengths[0];
        parameters[1] = this->focal_lengths[1];
        parameters[2] = this->centre_points[0];
        parameters[3] = this->centre_points[1];
        return true;
    }

    template <typename type>
    bool pinhole<type>::project(
        const type* const point_xyz,
        type* const point_xy,
        type* const jacobian_projection,
        type* const jacobian_parameters
    ) const {
        if (point_xyz[2] < type(1.0e-12)) {
            return false;
        }

        const type inverse_z = type(1) / point_xyz[2];
        const type undistorted_xy[2] = {
            point_xyz[0] * inverse_z,
            point_xyz[1] * inverse_z
        };

        point_xy[0] = this->focal_lengths[0] * undistorted_xy[0] + this->centre_points[0];
        point_xy[1] = this->focal_lengths[1] * undistorted_xy[1] + this->centre_points[1];

        if (jacobian_projection != nullptr) {
            const type inverse_z_squared = inverse_z * inverse_z;
            jacobian_projection[0 * 3 + 0] = this->focal_lengths[0] * inverse_z;
            jacobian_projection[0 * 3 + 1] = type(0);
            jacobian_projection[0 * 3 + 2] = -this->focal_lengths[0] * point_xyz[0] * inverse_z_squared;
            jacobian_projection[1 * 3 + 0] = type(0);
            jacobian_projection[1 * 3 + 1] = this->focal_lengths[1] * inverse_z;
            jacobian_projection[1 * 3 + 2] = -this->focal_lengths[1] * point_xyz[1] * inverse_z_squared;
        }

        if (jacobian_parameters != nullptr) {
            jacobian_parameters[0 * pinhole::parameter_count + 0] = undistorted_xy[0];
            jacobian_parameters[0 * pinhole::parameter_count + 1] = type(0);
            jacobian_parameters[0 * pinhole::parameter_count + 2] = type(1);
            jacobian_parameters[0 * pinhole::parameter_count + 3] = type(0);
            jacobian_parameters[1 * pinhole::parameter_count + 0] = type(0);
            jacobian_parameters[1 * pinhole::parameter_count + 1] = undistorted_xy[1];
            jacobian_parameters[1 * pinhole::parameter_count + 2] = type(0);
            jacobian_parameters[1 * pinhole::parameter_count + 3] = type(1);
        }

        return true;
    }

    template <typename type>
    bool pinhole<type>::unproject(
        const type* const point_xy,
        type* const ray_xyz,
        type* const jacobian_unprojection
    ) const {
        const type undistorted_xy[2] = {
            (point_xy[0] - this->centre_points[0]) / this->focal_lengths[0],
            (point_xy[1] - this->centre_points[1]) / this->focal_lengths[1]
        };

        ray_xyz[0] = undistorted_xy[0];
        ray_xyz[1] = undistorted_xy[1];
        ray_xyz[2] = type(1);

        if (jacobian_unprojection != nullptr) {
            jacobian_unprojection[0 * 2 + 0] = type(1) / this->focal_lengths[0];
            jacobian_unprojection[0 * 2 + 1] = type(0);
            jacobian_unprojection[1 * 2 + 0] = type(0);
            jacobian_unprojection[1 * 2 + 1] = type(1) / this->focal_lengths[1];
            jacobian_unprojection[2 * 2 + 0] = type(0);
            jacobian_unprojection[2 * 2 + 1] = type(0);
        }

        return true;
    }

    template class pinhole<float>;
    template class pinhole<double>;
}
