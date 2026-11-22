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

#include "sensor/camera/double_sphere.hpp"

#include "core/assert.hpp"
#include "math/math.hpp"

namespace sensor::camera {
    template <typename type>
    double_sphere<type>::double_sphere()
        : focal_lengths{ type(1), type(1) }
        , centre_points{ type(0.5), type(0.5) }
        , xi(type(0))
        , alpha(type(0.5))
        , valid_bound(type(1)) {
    }

    template <typename type>
    double_sphere<type>::double_sphere(const type* const parameters, const size_t parameters_length)
        : double_sphere() {
        const bool valid = this->set_parameters(parameters, parameters_length);
        ASSERT(valid, "The double sphere parameters must be six finite values with positive focal lengths, xi > -1 and alpha in [0, 1].");
        static_cast<void>(valid);
    }

    template <typename type>
    size_t double_sphere<type>::get_parameter_count() const {
        return double_sphere::parameter_count;
    }

    template <typename type>
    bool double_sphere<type>::set_parameters(const type* const parameters, const size_t parameters_length) {
        if ((parameters == nullptr) || (parameters_length != double_sphere::parameter_count)) {
            return false;
        }
        for (size_t index = 0; index < double_sphere::parameter_count; ++index) {
            if (!math::isfinite(parameters[index])) {
                return false;
            }
        }
        if (!(parameters[0] > type(0)) || !(parameters[1] > type(0)) || !(parameters[4] > type(-1)) || !(parameters[5] >= type(0)) || !(parameters[5] <= type(1))) {
            return false;
        }
        this->focal_lengths[0] = parameters[0];
        this->focal_lengths[1] = parameters[1];
        this->centre_points[0] = parameters[2];
        this->centre_points[1] = parameters[3];
        this->xi = parameters[4];
        this->alpha = parameters[5];
        const type w1 = (this->alpha > type(0.5)) ? ((type(1) - this->alpha) / this->alpha) : (this->alpha / (type(1) - this->alpha));
        this->valid_bound = (w1 + this->xi) / math::sqrt((type(2) * w1 * this->xi) + (this->xi * this->xi) + type(1));
        return true;
    }

    template <typename type>
    bool double_sphere<type>::get_parameters(type* const parameters, const size_t parameters_length) const {
        if ((parameters == nullptr) || (parameters_length != double_sphere::parameter_count)) {
            return false;
        }
        parameters[0] = this->focal_lengths[0];
        parameters[1] = this->focal_lengths[1];
        parameters[2] = this->centre_points[0];
        parameters[3] = this->centre_points[1];
        parameters[4] = this->xi;
        parameters[5] = this->alpha;
        return true;
    }

    template <typename type>
    bool double_sphere<type>::get_pinhole(type* const pinhole_parameters, const size_t pinhole_parameters_length) const {
        if ((pinhole_parameters == nullptr) || (pinhole_parameters_length != 4)) {
            return false;
        }
        // At the centre d1 = z and d2 = xi * d1 + z, so the denominator is (1 + xi) * z.
        pinhole_parameters[0] = this->focal_lengths[0] / (type(1) + this->xi);
        pinhole_parameters[1] = this->focal_lengths[1] / (type(1) + this->xi);
        pinhole_parameters[2] = this->centre_points[0];
        pinhole_parameters[3] = this->centre_points[1];
        return true;
    }

    template <typename type>
    bool double_sphere<type>::get_parameter_roles(parameter_role* const roles, const size_t roles_length) const {
        if ((roles == nullptr) || (roles_length != double_sphere::parameter_count)) {
            return false;
        }
        // xi and alpha, which together shape the projection.
        const parameter_role layout[double_sphere::parameter_count] = {
            parameter_role::focal_length,
            parameter_role::focal_length,
            parameter_role::principal_point,
            parameter_role::principal_point,
            parameter_role::distortion,
            parameter_role::distortion
        };
        for (size_t index = 0; index < double_sphere::parameter_count; ++index) {
            roles[index] = layout[index];
        }
        return true;
    }

    template <typename type>
    bool double_sphere<type>::project(
        const type* const point_xyz,
        type* const point_xy,
        type* const jacobian_projection,
        type* const jacobian_parameters
    ) const {
        const type x = point_xyz[0];
        const type y = point_xyz[1];
        const type z = point_xyz[2];
        const type r2 = (x * x) + (y * y);
        const type d1 = math::sqrt(r2 + (z * z));
        const type shifted_z = (this->xi * d1) + z;
        const type d2 = math::sqrt(r2 + (shifted_z * shifted_z));
        const type denominator = (this->alpha * d2) + ((type(1) - this->alpha) * shifted_z);
        if (!(z > -this->valid_bound * d1) || !(denominator > type(1.0e-12))) {
            return false;
        }
        const type inverse = type(1) / denominator;
        const type distorted_xy[2] = { x * inverse, y * inverse };
        point_xy[0] = (this->focal_lengths[0] * distorted_xy[0]) + this->centre_points[0];
        point_xy[1] = (this->focal_lengths[1] * distorted_xy[1]) + this->centre_points[1];

        if (jacobian_projection != nullptr) {
            const type inverse_squared = inverse * inverse;
            const type inverse_d1 = type(1) / d1;
            const type inverse_d2 = type(1) / d2;
            // d shifted_z / d point, then d d2 / d point, then d denominator / d point.
            const type shifted_slope[3] = { this->xi * x * inverse_d1, this->xi * y * inverse_d1, (this->xi * z * inverse_d1) + type(1) };
            const type planar[3] = { x, y, type(0) };
            type slope[3];
            for (size_t column = 0; column < 3; ++column) {
                const type d2_slope = (planar[column] + (shifted_z * shifted_slope[column])) * inverse_d2;
                slope[column] = (this->alpha * d2_slope) + ((type(1) - this->alpha) * shifted_slope[column]);
            }
            for (size_t row = 0; row < 2; ++row) {
                const type coordinate = point_xyz[row];
                for (size_t column = 0; column < 3; ++column) {
                    const type direct = (row == column) ? inverse : type(0);
                    jacobian_projection[row * 3 + column] = this->focal_lengths[row] * (direct - (coordinate * slope[column] * inverse_squared));
                }
            }
        }

        if (jacobian_parameters != nullptr) {
            constexpr static const size_t count = double_sphere::parameter_count;
            const type inverse_squared = inverse * inverse;
            const type d_xi = (this->alpha * shifted_z * d1 / d2) + ((type(1) - this->alpha) * d1);
            const type d_alpha = d2 - shifted_z;
            jacobian_parameters[0 * count + 0] = distorted_xy[0];
            jacobian_parameters[0 * count + 1] = type(0);
            jacobian_parameters[0 * count + 2] = type(1);
            jacobian_parameters[0 * count + 3] = type(0);
            jacobian_parameters[0 * count + 4] = -this->focal_lengths[0] * x * d_xi * inverse_squared;
            jacobian_parameters[0 * count + 5] = -this->focal_lengths[0] * x * d_alpha * inverse_squared;
            jacobian_parameters[1 * count + 0] = type(0);
            jacobian_parameters[1 * count + 1] = distorted_xy[1];
            jacobian_parameters[1 * count + 2] = type(0);
            jacobian_parameters[1 * count + 3] = type(1);
            jacobian_parameters[1 * count + 4] = -this->focal_lengths[1] * y * d_xi * inverse_squared;
            jacobian_parameters[1 * count + 5] = -this->focal_lengths[1] * y * d_alpha * inverse_squared;
        }

        return true;
    }

    template <typename type>
    bool double_sphere<type>::unproject(
        const type* const point_xy,
        type* const ray_xyz,
        type* const jacobian_unprojection
    ) const {
        const type mx = (point_xy[0] - this->centre_points[0]) / this->focal_lengths[0];
        const type my = (point_xy[1] - this->centre_points[1]) / this->focal_lengths[1];
        const type r2 = (mx * mx) + (my * my);
        // With alpha above a half the image of the valid region is the disc r^2 < 1 / (2 * alpha - 1).
        const type spread = (type(2) * this->alpha) - type(1);
        const type discriminant = type(1) - (spread * r2);
        if (!(discriminant > type(0))) {
            return false;
        }
        const type root = math::sqrt(discriminant);
        const type numerator = type(1) - (this->alpha * this->alpha * r2);
        const type denominator = (this->alpha * root) + (type(1) - this->alpha);
        const type mz = numerator / denominator;
        // The ray back through the first sphere: k * (mx, my, mz) - (0, 0, xi).
        const type sphere_discriminant = (mz * mz) + ((type(1) - (this->xi * this->xi)) * r2);
        if (!(sphere_discriminant > type(0))) {
            return false;
        }
        const type sphere_root = math::sqrt(sphere_discriminant);
        const type k_numerator = (mz * this->xi) + sphere_root;
        const type k_denominator = (mz * mz) + r2;
        const type k = k_numerator / k_denominator;
        ray_xyz[0] = k * mx;
        ray_xyz[1] = k * my;
        ray_xyz[2] = (k * mz) - this->xi;
        // A ray outside the valid region would not project back to this pixel.
        if (!(ray_xyz[2] > -this->valid_bound)) {
            return false;
        }

        if (jacobian_unprojection != nullptr) {
            const type mz_slope = ((-this->alpha * this->alpha * denominator) + (numerator * this->alpha * spread / (type(2) * root))) / (denominator * denominator);
            const type inverse_f[2] = { type(1) / this->focal_lengths[0], type(1) / this->focal_lengths[1] };
            const type m[2] = { mx, my };
            for (size_t column = 0; column < 2; ++column) {
                const type d_mz = type(2) * m[column] * mz_slope;
                const type d_root = ((mz * d_mz) + ((type(1) - (this->xi * this->xi)) * m[column])) / sphere_root;
                const type d_numerator = (this->xi * d_mz) + d_root;
                const type d_denominator = (type(2) * mz * d_mz) + (type(2) * m[column]);
                const type d_k = ((d_numerator * k_denominator) - (k_numerator * d_denominator)) / (k_denominator * k_denominator);
                jacobian_unprojection[0 * 2 + column] = (((column == 0) ? k : type(0)) + (mx * d_k)) * inverse_f[column];
                jacobian_unprojection[1 * 2 + column] = (((column == 1) ? k : type(0)) + (my * d_k)) * inverse_f[column];
                jacobian_unprojection[2 * 2 + column] = ((d_k * mz) + (k * d_mz)) * inverse_f[column];
            }
        }

        return true;
    }

    template class double_sphere<float>;
    template class double_sphere<double>;
}
