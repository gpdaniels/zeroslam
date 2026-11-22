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

#include "sensor/camera/field_of_view.hpp"

#include "core/assert.hpp"
#include "math/math.hpp"

namespace {
    // Below this the ratios below are taken from their series, which there are exact to the precision of the type and avoid
    // the cancellation of the closed forms: the eighth root of the machine epsilon.
    template <typename type>
    type series_limit() {
        constexpr static const type epsilon = math::epsilon<type>();
        return math::sqrt(math::sqrt(math::sqrt(epsilon)));
    }

    // atan(v) / v.
    template <typename type>
    type atan_ratio(const type v) {
        const type v2 = v * v;
        if (math::abs(v) < series_limit<type>()) {
            return type(1) + v2 * (-type(1) / type(3) + v2 * (type(1) / type(5) - v2 / type(7)));
        }
        return math::atan2(v, type(1)) / v;
    }

    // (d (atan(v) / v) / dv) / v.
    template <typename type>
    type atan_ratio_slope(const type v) {
        const type v2 = v * v;
        if (math::abs(v) < series_limit<type>()) {
            return -type(2) / type(3) + v2 * (type(4) / type(5) + v2 * (-type(6) / type(7) + v2 * type(8) / type(9)));
        }
        return ((type(1) / (type(1) + v2)) - atan_ratio(v)) / v2;
    }

    // tan(x) / x.
    template <typename type>
    type tan_ratio(const type x) {
        const type x2 = x * x;
        if (math::abs(x) < series_limit<type>()) {
            return type(1) + x2 * (type(1) / type(3) + x2 * (type(2) / type(15) + x2 * type(17) / type(315)));
        }
        return math::sin(x) / (math::cos(x) * x);
    }

    // (d (tan(x) / x) / dx) / x.
    template <typename type>
    type tan_ratio_slope(const type x) {
        const type x2 = x * x;
        if (math::abs(x) < series_limit<type>()) {
            return type(2) / type(3) + x2 * (type(8) / type(15) + x2 * (type(34) / type(105) + x2 * type(496) / type(2835)));
        }
        const type tangent = math::sin(x) / math::cos(x);
        return ((type(1) + (tangent * tangent)) - tan_ratio(x)) / x2;
    }
}

namespace sensor::camera {
    template <typename type>
    field_of_view<type>::field_of_view()
        : focal_lengths{ type(1), type(1) }
        , centre_points{ type(0.5), type(0.5) }
        , field(type(0)) {
    }

    template <typename type>
    field_of_view<type>::field_of_view(const type* const parameters, const size_t parameters_length)
        : field_of_view() {
        const bool valid = this->set_parameters(parameters, parameters_length);
        ASSERT(valid, "The field of view parameters must be five finite values with positive focal lengths and w in [0, pi).");
        static_cast<void>(valid);
    }

    template <typename type>
    size_t field_of_view<type>::get_parameter_count() const {
        return field_of_view::parameter_count;
    }

    template <typename type>
    bool field_of_view<type>::set_parameters(const type* const parameters, const size_t parameters_length) {
        if ((parameters == nullptr) || (parameters_length != field_of_view::parameter_count)) {
            return false;
        }
        for (size_t index = 0; index < field_of_view::parameter_count; ++index) {
            if (!math::isfinite(parameters[index])) {
                return false;
            }
        }
        if (!(parameters[0] > type(0)) || !(parameters[1] > type(0)) || !(parameters[4] >= type(0)) || !(parameters[4] < math::pi<type>())) {
            return false;
        }
        this->focal_lengths[0] = parameters[0];
        this->focal_lengths[1] = parameters[1];
        this->centre_points[0] = parameters[2];
        this->centre_points[1] = parameters[3];
        this->field = parameters[4];
        return true;
    }

    template <typename type>
    bool field_of_view<type>::get_parameters(type* const parameters, const size_t parameters_length) const {
        if ((parameters == nullptr) || (parameters_length != field_of_view::parameter_count)) {
            return false;
        }
        parameters[0] = this->focal_lengths[0];
        parameters[1] = this->focal_lengths[1];
        parameters[2] = this->centre_points[0];
        parameters[3] = this->centre_points[1];
        parameters[4] = this->field;
        return true;
    }

    template <typename type>
    bool field_of_view<type>::get_pinhole(type* const pinhole_parameters, const size_t pinhole_parameters_length) const {
        if ((pinhole_parameters == nullptr) || (pinhole_parameters_length != 4)) {
            return false;
        }
        // r_d = r_u * 2 * tan(w / 2) / w + O(r_u^3).
        const type gain = tan_ratio(type(0.5) * this->field);
        pinhole_parameters[0] = this->focal_lengths[0] * gain;
        pinhole_parameters[1] = this->focal_lengths[1] * gain;
        pinhole_parameters[2] = this->centre_points[0];
        pinhole_parameters[3] = this->centre_points[1];
        return true;
    }

    template <typename type>
    bool field_of_view<type>::project(
        const type* const point_xyz,
        type* const point_xy,
        type* const jacobian_projection,
        type* const jacobian_parameters
    ) const {
        if (point_xyz[2] < type(1.0e-12)) {
            return false;
        }
        const type inverse_z = type(1) / point_xyz[2];
        const type a = point_xyz[0] * inverse_z;
        const type b = point_xyz[1] * inverse_z;
        const type radius = math::sqrt((a * a) + (b * b));
        // With T = 2 * tan(w / 2): r_d / r_u = (T / w) * atan(T * r_u) / (T * r_u), both ratios bounded as w and r_u vanish.
        const type half = type(0.5) * this->field;
        const type gain = tan_ratio(half);
        const type two_tan = this->field * gain;
        const type v = two_tan * radius;
        const type factor = gain * atan_ratio(v);
        const type distorted_xy[2] = { factor * a, factor * b };

        point_xy[0] = (this->focal_lengths[0] * distorted_xy[0]) + this->centre_points[0];
        point_xy[1] = (this->focal_lengths[1] * distorted_xy[1]) + this->centre_points[1];

        if (jacobian_projection != nullptr) {
            // d factor / d r_u, over r_u.
            const type curvature = gain * two_tan * two_tan * atan_ratio_slope(v);
            const type j_aa = factor + (a * a * curvature);
            const type j_ab = a * b * curvature;
            const type j_bb = factor + (b * b * curvature);
            jacobian_projection[0 * 3 + 0] = this->focal_lengths[0] * j_aa * inverse_z;
            jacobian_projection[0 * 3 + 1] = this->focal_lengths[0] * j_ab * inverse_z;
            jacobian_projection[0 * 3 + 2] = -this->focal_lengths[0] * ((j_aa * a) + (j_ab * b)) * inverse_z;
            jacobian_projection[1 * 3 + 0] = this->focal_lengths[1] * j_ab * inverse_z;
            jacobian_projection[1 * 3 + 1] = this->focal_lengths[1] * j_bb * inverse_z;
            jacobian_projection[1 * 3 + 2] = -this->focal_lengths[1] * ((j_ab * a) + (j_bb * b)) * inverse_z;
        }

        if (jacobian_parameters != nullptr) {
            constexpr static const size_t count = field_of_view::parameter_count;
            // d (T / w) / dw = (d tan_ratio / dx at w / 2) / 2, dT / dw = 1 + T^2 / 4.
            const type gain_derivative = type(0.5) * tan_ratio_slope(half) * half;
            const type two_tan_derivative = type(1) + (type(0.25) * two_tan * two_tan);
            const type factor_derivative = (gain_derivative * atan_ratio(v)) + (gain * v * atan_ratio_slope(v) * two_tan_derivative * radius);
            jacobian_parameters[0 * count + 0] = distorted_xy[0];
            jacobian_parameters[0 * count + 1] = type(0);
            jacobian_parameters[0 * count + 2] = type(1);
            jacobian_parameters[0 * count + 3] = type(0);
            jacobian_parameters[0 * count + 4] = this->focal_lengths[0] * factor_derivative * a;
            jacobian_parameters[1 * count + 0] = type(0);
            jacobian_parameters[1 * count + 1] = distorted_xy[1];
            jacobian_parameters[1 * count + 2] = type(0);
            jacobian_parameters[1 * count + 3] = type(1);
            jacobian_parameters[1 * count + 4] = this->focal_lengths[1] * factor_derivative * b;
        }

        return true;
    }

    template <typename type>
    bool field_of_view<type>::unproject(
        const type* const point_xy,
        type* const ray_xyz,
        type* const jacobian_unprojection
    ) const {
        const type distorted_xy[2] = {
            (point_xy[0] - this->centre_points[0]) / this->focal_lengths[0],
            (point_xy[1] - this->centre_points[1]) / this->focal_lengths[1]
        };
        const type distorted_radius = math::sqrt((distorted_xy[0] * distorted_xy[0]) + (distorted_xy[1] * distorted_xy[1]));
        // r_u = tan(w * r_d) / T, which reaches infinity, ninety degrees from the axis, at w * r_d = pi / 2.
        const type angle = this->field * distorted_radius;
        if (!(angle < type(0.5) * math::pi<type>())) {
            return false;
        }
        const type inverse_gain = type(1) / tan_ratio(type(0.5) * this->field);
        const type g = tan_ratio(angle) * inverse_gain;
        ray_xyz[0] = g * distorted_xy[0];
        ray_xyz[1] = g * distorted_xy[1];
        ray_xyz[2] = type(1);

        if (jacobian_unprojection != nullptr) {
            // (d g / d r_d) / r_d.
            const type h = tan_ratio_slope(angle) * this->field * this->field * inverse_gain;
            const type inverse_fx = type(1) / this->focal_lengths[0];
            const type inverse_fy = type(1) / this->focal_lengths[1];
            jacobian_unprojection[0 * 2 + 0] = (g + (distorted_xy[0] * distorted_xy[0] * h)) * inverse_fx;
            jacobian_unprojection[0 * 2 + 1] = (distorted_xy[0] * distorted_xy[1] * h) * inverse_fy;
            jacobian_unprojection[1 * 2 + 0] = (distorted_xy[0] * distorted_xy[1] * h) * inverse_fx;
            jacobian_unprojection[1 * 2 + 1] = (g + (distorted_xy[1] * distorted_xy[1] * h)) * inverse_fy;
            jacobian_unprojection[2 * 2 + 0] = type(0);
            jacobian_unprojection[2 * 2 + 1] = type(0);
        }

        return true;
    }

    template class field_of_view<float>;
    template class field_of_view<double>;
}
