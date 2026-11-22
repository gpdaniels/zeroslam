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

#include "sensor/camera/extended_unified.hpp"

#include "core/assert.hpp"
#include "math/math.hpp"

namespace sensor::camera {
    template <typename type>
    extended_unified<type>::extended_unified()
        : focal_lengths{ type(1), type(1) }
        , centre_points{ type(0.5), type(0.5) }
        , alpha(type(0))
        , beta(type(1)) {
    }

    template <typename type>
    extended_unified<type>::extended_unified(const type* const parameters, const size_t parameters_length)
        : extended_unified() {
        const bool valid = this->set_parameters(parameters, parameters_length);
        ASSERT(valid, "The extended unified parameters must be six finite values with positive focal lengths, alpha in [0, 1] and beta > 0.");
        static_cast<void>(valid);
    }

    template <typename type>
    size_t extended_unified<type>::get_parameter_count() const {
        return extended_unified::parameter_count;
    }

    template <typename type>
    bool extended_unified<type>::set_parameters(const type* const parameters, const size_t parameters_length) {
        if ((parameters == nullptr) || (parameters_length != extended_unified::parameter_count)) {
            return false;
        }
        for (size_t index = 0; index < extended_unified::parameter_count; ++index) {
            if (!math::isfinite(parameters[index])) {
                return false;
            }
        }
        if (!(parameters[0] > type(0)) || !(parameters[1] > type(0)) || !(parameters[4] >= type(0)) || !(parameters[4] <= type(1)) || !(parameters[5] > type(0))) {
            return false;
        }
        this->focal_lengths[0] = parameters[0];
        this->focal_lengths[1] = parameters[1];
        this->centre_points[0] = parameters[2];
        this->centre_points[1] = parameters[3];
        this->alpha = parameters[4];
        this->beta = parameters[5];
        return true;
    }

    template <typename type>
    bool extended_unified<type>::get_parameters(type* const parameters, const size_t parameters_length) const {
        if ((parameters == nullptr) || (parameters_length != extended_unified::parameter_count)) {
            return false;
        }
        parameters[0] = this->focal_lengths[0];
        parameters[1] = this->focal_lengths[1];
        parameters[2] = this->centre_points[0];
        parameters[3] = this->centre_points[1];
        parameters[4] = this->alpha;
        parameters[5] = this->beta;
        return true;
    }

    template <typename type>
    bool extended_unified<type>::get_pinhole(type* const pinhole_parameters, const size_t pinhole_parameters_length) const {
        if ((pinhole_parameters == nullptr) || (pinhole_parameters_length != 4)) {
            return false;
        }
        // At the centre d = z, so the denominator is z.
        pinhole_parameters[0] = this->focal_lengths[0];
        pinhole_parameters[1] = this->focal_lengths[1];
        pinhole_parameters[2] = this->centre_points[0];
        pinhole_parameters[3] = this->centre_points[1];
        return true;
    }

    template <typename type>
    bool extended_unified<type>::project(
        const type* const point_xyz,
        type* const point_xy,
        type* const jacobian_projection,
        type* const jacobian_parameters
    ) const {
        const type x = point_xyz[0];
        const type y = point_xyz[1];
        const type z = point_xyz[2];
        const type r2 = (x * x) + (y * y);
        const type d = math::sqrt((this->beta * r2) + (z * z));
        const type bound = (this->alpha > type(0.5)) ? ((type(1) - this->alpha) / this->alpha) : (this->alpha / (type(1) - this->alpha));
        const type denominator = (this->alpha * d) + ((type(1) - this->alpha) * z);
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
            const type slope[3] = { this->alpha * this->beta * x * inverse_d, this->alpha * this->beta * y * inverse_d, (this->alpha * z * inverse_d) + (type(1) - this->alpha) };
            for (size_t row = 0; row < 2; ++row) {
                const type coordinate = point_xyz[row];
                for (size_t column = 0; column < 3; ++column) {
                    const type direct = (row == column) ? inverse : type(0);
                    jacobian_projection[row * 3 + column] = this->focal_lengths[row] * (direct - (coordinate * slope[column] * inverse_squared));
                }
            }
        }

        if (jacobian_parameters != nullptr) {
            constexpr static const size_t count = extended_unified::parameter_count;
            const type inverse_squared = inverse * inverse;
            const type d_alpha = d - z;
            const type d_beta = this->alpha * r2 / (type(2) * d);
            jacobian_parameters[0 * count + 0] = distorted_xy[0];
            jacobian_parameters[0 * count + 1] = type(0);
            jacobian_parameters[0 * count + 2] = type(1);
            jacobian_parameters[0 * count + 3] = type(0);
            jacobian_parameters[0 * count + 4] = -this->focal_lengths[0] * x * d_alpha * inverse_squared;
            jacobian_parameters[0 * count + 5] = -this->focal_lengths[0] * x * d_beta * inverse_squared;
            jacobian_parameters[1 * count + 0] = type(0);
            jacobian_parameters[1 * count + 1] = distorted_xy[1];
            jacobian_parameters[1 * count + 2] = type(0);
            jacobian_parameters[1 * count + 3] = type(1);
            jacobian_parameters[1 * count + 4] = -this->focal_lengths[1] * y * d_alpha * inverse_squared;
            jacobian_parameters[1 * count + 5] = -this->focal_lengths[1] * y * d_beta * inverse_squared;
        }

        return true;
    }

    template <typename type>
    bool extended_unified<type>::unproject(
        const type* const point_xy,
        type* const ray_xyz,
        type* const jacobian_unprojection
    ) const {
        const type mx = (point_xy[0] - this->centre_points[0]) / this->focal_lengths[0];
        const type my = (point_xy[1] - this->centre_points[1]) / this->focal_lengths[1];
        const type r2 = (mx * mx) + (my * my);
        // With alpha above a half the image of the valid region is the disc r^2 < 1 / (beta * (2 * alpha - 1)).
        const type spread = ((type(2) * this->alpha) - type(1)) * this->beta;
        const type discriminant = type(1) - (spread * r2);
        if (!(discriminant > type(0))) {
            return false;
        }
        const type root = math::sqrt(discriminant);
        const type numerator = type(1) - (this->beta * this->alpha * this->alpha * r2);
        const type denominator = (this->alpha * root) + (type(1) - this->alpha);
        const type mz = numerator / denominator;
        const type length = math::sqrt(r2 + (mz * mz));
        const type inverse_length = type(1) / length;
        ray_xyz[0] = mx * inverse_length;
        ray_xyz[1] = my * inverse_length;
        ray_xyz[2] = mz * inverse_length;

        if (jacobian_unprojection != nullptr) {
            // d mz / d r^2, then the unit vector's jacobian (I - n n^T) / |v| applied to dv / dm = (e_m, 2 m d mz / d r^2).
            const type slope = ((-this->beta * this->alpha * this->alpha * denominator) + (numerator * this->alpha * spread / (type(2) * root))) / (denominator * denominator);
            const type inverse_f[2] = { type(1) / this->focal_lengths[0], type(1) / this->focal_lengths[1] };
            const type m[2] = { mx, my };
            for (size_t column = 0; column < 2; ++column) {
                const type dv[3] = { (column == 0) ? type(1) : type(0), (column == 1) ? type(1) : type(0), type(2) * m[column] * slope };
                const type projection = (ray_xyz[0] * dv[0]) + (ray_xyz[1] * dv[1]) + (ray_xyz[2] * dv[2]);
                for (size_t row = 0; row < 3; ++row) {
                    jacobian_unprojection[row * 2 + column] = (dv[row] - (ray_xyz[row] * projection)) * inverse_length * inverse_f[column];
                }
            }
        }

        return true;
    }

    template class extended_unified<float>;
    template class extended_unified<double>;
}
