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

#include "sensor/camera/unified.hpp"

#include "core/assert.hpp"
#include "math/math.hpp"

namespace sensor::camera {
    template <typename type>
    unified<type>::unified()
        : focal_lengths{ type(1), type(1) }
        , centre_points{ type(0.5), type(0.5) }
        , xi(type(0)) {
    }

    template <typename type>
    unified<type>::unified(const type* const parameters, const size_t parameters_length)
        : unified() {
        const bool valid = this->set_parameters(parameters, parameters_length);
        ASSERT(valid, "The unified parameters must be five finite values with positive focal lengths and xi >= 0.");
        static_cast<void>(valid);
    }

    template <typename type>
    size_t unified<type>::get_parameter_count() const {
        return unified::parameter_count;
    }

    template <typename type>
    bool unified<type>::set_parameters(const type* const parameters, const size_t parameters_length) {
        if ((parameters == nullptr) || (parameters_length != unified::parameter_count)) {
            return false;
        }
        for (size_t index = 0; index < unified::parameter_count; ++index) {
            if (!math::isfinite(parameters[index])) {
                return false;
            }
        }
        if (!(parameters[0] > type(0)) || !(parameters[1] > type(0)) || !(parameters[4] >= type(0))) {
            return false;
        }
        this->focal_lengths[0] = parameters[0];
        this->focal_lengths[1] = parameters[1];
        this->centre_points[0] = parameters[2];
        this->centre_points[1] = parameters[3];
        this->xi = parameters[4];
        return true;
    }

    template <typename type>
    bool unified<type>::get_parameters(type* const parameters, const size_t parameters_length) const {
        if ((parameters == nullptr) || (parameters_length != unified::parameter_count)) {
            return false;
        }
        parameters[0] = this->focal_lengths[0];
        parameters[1] = this->focal_lengths[1];
        parameters[2] = this->centre_points[0];
        parameters[3] = this->centre_points[1];
        parameters[4] = this->xi;
        return true;
    }

    template <typename type>
    bool unified<type>::get_pinhole(type* const pinhole_parameters, const size_t pinhole_parameters_length) const {
        if ((pinhole_parameters == nullptr) || (pinhole_parameters_length != 4)) {
            return false;
        }
        // At the centre d = z, so x_d = x / ((1 + xi) * z).
        pinhole_parameters[0] = this->focal_lengths[0] / (type(1) + this->xi);
        pinhole_parameters[1] = this->focal_lengths[1] / (type(1) + this->xi);
        pinhole_parameters[2] = this->centre_points[0];
        pinhole_parameters[3] = this->centre_points[1];
        return true;
    }

    template <typename type>
    bool unified<type>::project(
        const type* const point_xyz,
        type* const point_xy,
        type* const jacobian_projection,
        type* const jacobian_parameters
    ) const {
        const type x = point_xyz[0];
        const type y = point_xyz[1];
        const type z = point_xyz[2];
        const type d = math::sqrt((x * x) + (y * y) + (z * z));
        const type bound = (this->xi > type(1)) ? (type(1) / this->xi) : this->xi;
        const type denominator = z + (this->xi * d);
        if (!(z > -bound * d) || !(denominator > type(1.0e-12))) {
            return false;
        }
        const type inverse = type(1) / denominator;
        const type distorted_xy[2] = { x * inverse, y * inverse };
        point_xy[0] = (this->focal_lengths[0] * distorted_xy[0]) + this->centre_points[0];
        point_xy[1] = (this->focal_lengths[1] * distorted_xy[1]) + this->centre_points[1];

        if (jacobian_projection != nullptr) {
            const type inverse_squared = inverse * inverse;
            const type inverse_d = type(1) / d;
            const type slope[3] = { this->xi * x * inverse_d, this->xi * y * inverse_d, type(1) + (this->xi * z * inverse_d) };
            for (size_t row = 0; row < 2; ++row) {
                const type coordinate = point_xyz[row];
                for (size_t column = 0; column < 3; ++column) {
                    const type direct = (row == column) ? inverse : type(0);
                    jacobian_projection[row * 3 + column] = this->focal_lengths[row] * (direct - (coordinate * slope[column] * inverse_squared));
                }
            }
        }

        if (jacobian_parameters != nullptr) {
            constexpr static const size_t count = unified::parameter_count;
            const type inverse_squared = inverse * inverse;
            jacobian_parameters[0 * count + 0] = distorted_xy[0];
            jacobian_parameters[0 * count + 1] = type(0);
            jacobian_parameters[0 * count + 2] = type(1);
            jacobian_parameters[0 * count + 3] = type(0);
            jacobian_parameters[0 * count + 4] = -this->focal_lengths[0] * x * d * inverse_squared;
            jacobian_parameters[1 * count + 0] = type(0);
            jacobian_parameters[1 * count + 1] = distorted_xy[1];
            jacobian_parameters[1 * count + 2] = type(0);
            jacobian_parameters[1 * count + 3] = type(1);
            jacobian_parameters[1 * count + 4] = -this->focal_lengths[1] * y * d * inverse_squared;
        }

        return true;
    }

    template <typename type>
    bool unified<type>::unproject(
        const type* const point_xy,
        type* const ray_xyz,
        type* const jacobian_unprojection
    ) const {
        const type mx = (point_xy[0] - this->centre_points[0]) / this->focal_lengths[0];
        const type my = (point_xy[1] - this->centre_points[1]) / this->focal_lengths[1];
        const type r2 = (mx * mx) + (my * my);
        // With xi above one the image of the valid region is the disc 1 + (1 - xi^2) * r^2 > 0, its rim the fold.
        const type discriminant = type(1) + ((type(1) - (this->xi * this->xi)) * r2);
        if (!(discriminant > type(0))) {
            return false;
        }
        const type root = math::sqrt(discriminant);
        const type inverse_norm = type(1) / (type(1) + r2);
        // The point on the unit sphere: factor * (mx, my, 1) - (0, 0, xi).
        const type factor = (this->xi + root) * inverse_norm;
        ray_xyz[0] = factor * mx;
        ray_xyz[1] = factor * my;
        ray_xyz[2] = factor - this->xi;

        if (jacobian_unprojection != nullptr) {
            // d factor / d r^2, then d r^2 / d m = 2 m.
            const type slope = ((((type(1) - (this->xi * this->xi)) / (type(2) * root)) * (type(1) + r2)) - (this->xi + root)) * inverse_norm * inverse_norm;
            const type d_factor[2] = { type(2) * mx * slope, type(2) * my * slope };
            const type inverse_f[2] = { type(1) / this->focal_lengths[0], type(1) / this->focal_lengths[1] };
            for (size_t column = 0; column < 2; ++column) {
                jacobian_unprojection[0 * 2 + column] = (((column == 0) ? factor : type(0)) + (mx * d_factor[column])) * inverse_f[column];
                jacobian_unprojection[1 * 2 + column] = (((column == 1) ? factor : type(0)) + (my * d_factor[column])) * inverse_f[column];
                jacobian_unprojection[2 * 2 + column] = d_factor[column] * inverse_f[column];
            }
        }

        return true;
    }

    template class unified<float>;
    template class unified<double>;
}
