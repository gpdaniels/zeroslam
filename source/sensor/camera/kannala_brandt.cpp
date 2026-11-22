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

#include "sensor/camera/kannala_brandt.hpp"

#include "core/assert.hpp"
#include "math/math.hpp"
#include "math/polynomial.hpp"

namespace sensor::camera {
    template <typename type>
    kannala_brandt<type>::kannala_brandt()
        : focal_lengths{ type(1), type(1) }
        , centre_points{ type(0.5), type(0.5) }
        , radial_k{ type(0), type(0), type(0), type(0) }
        , maximum_angle(math::pi<type>())
        , maximum_distorted_radius(math::pi<type>()) {
    }

    template <typename type>
    kannala_brandt<type>::kannala_brandt(const type* const parameters, const size_t parameters_length)
        : kannala_brandt() {
        const bool valid = this->set_parameters(parameters, parameters_length);
        ASSERT(valid, "The Kannala-Brandt parameters must be eight finite values with positive focal lengths.");
        static_cast<void>(valid);
    }

    template <typename type>
    size_t kannala_brandt<type>::get_parameter_count() const {
        return kannala_brandt::parameter_count;
    }

    template <typename type>
    bool kannala_brandt<type>::set_parameters(const type* const parameters, const size_t parameters_length) {
        if ((parameters == nullptr) || (parameters_length != kannala_brandt::parameter_count)) {
            return false;
        }
        for (size_t index = 0; index < kannala_brandt::parameter_count; ++index) {
            if (!math::isfinite(parameters[index])) {
                return false;
            }
        }
        if (!(parameters[0] > type(0)) || !(parameters[1] > type(0))) {
            return false;
        }
        this->focal_lengths[0] = parameters[0];
        this->focal_lengths[1] = parameters[1];
        this->centre_points[0] = parameters[2];
        this->centre_points[1] = parameters[3];
        for (size_t index = 0; index < 4; ++index) {
            this->radial_k[index] = parameters[4 + index];
        }
        // d theta_d / d theta as a polynomial in theta^2, its first root is where the projection folds back.
        const double derivative[5] = {
            1.0,
            3.0 * static_cast<double>(this->radial_k[0]),
            5.0 * static_cast<double>(this->radial_k[1]),
            7.0 * static_cast<double>(this->radial_k[2]),
            9.0 * static_cast<double>(this->radial_k[3])
        };
        const double half_turn = math::pi<double>();
        const double fold_squared = math::polynomial::first_non_positive(&derivative[0], 4, half_turn * half_turn);
        this->maximum_angle = static_cast<type>(math::min(math::sqrt(fold_squared), half_turn));
        type derivative_at_maximum = type(0);
        this->distort_angle(this->maximum_angle, this->maximum_distorted_radius, derivative_at_maximum);
        return true;
    }

    template <typename type>
    bool kannala_brandt<type>::get_parameters(type* const parameters, const size_t parameters_length) const {
        if ((parameters == nullptr) || (parameters_length != kannala_brandt::parameter_count)) {
            return false;
        }
        parameters[0] = this->focal_lengths[0];
        parameters[1] = this->focal_lengths[1];
        parameters[2] = this->centre_points[0];
        parameters[3] = this->centre_points[1];
        for (size_t index = 0; index < 4; ++index) {
            parameters[4 + index] = this->radial_k[index];
        }
        return true;
    }

    template <typename type>
    bool kannala_brandt<type>::get_pinhole(type* const pinhole_parameters, const size_t pinhole_parameters_length) const {
        if ((pinhole_parameters == nullptr) || (pinhole_parameters_length != 4)) {
            return false;
        }
        // theta_d = theta + O(theta^3) and theta = r / z + O(r^3), so at the centre it is the pinhole of its focal lengths.
        pinhole_parameters[0] = this->focal_lengths[0];
        pinhole_parameters[1] = this->focal_lengths[1];
        pinhole_parameters[2] = this->centre_points[0];
        pinhole_parameters[3] = this->centre_points[1];
        return true;
    }

    template <typename type>
    type kannala_brandt<type>::get_maximum_angle() const {
        return this->maximum_angle;
    }

    template <typename type>
    void kannala_brandt<type>::distort_angle(const type theta, type& distorted, type& derivative) const {
        const type theta2 = theta * theta;
        const type polynomial = type(1) + theta2 * (this->radial_k[0] + theta2 * (this->radial_k[1] + theta2 * (this->radial_k[2] + theta2 * this->radial_k[3])));
        distorted = theta * polynomial;
        derivative = type(1) + theta2 * (type(3) * this->radial_k[0] + theta2 * (type(5) * this->radial_k[1] + theta2 * (type(7) * this->radial_k[2] + theta2 * type(9) * this->radial_k[3])));
    }

    template <typename type>
    bool kannala_brandt<type>::project(
        const type* const point_xyz,
        type* const point_xy,
        type* const jacobian_projection,
        type* const jacobian_parameters
    ) const {
        const type x = point_xyz[0];
        const type y = point_xyz[1];
        const type z = point_xyz[2];
        const type r2 = (x * x) + (y * y);
        const type r = math::sqrt(r2);
        const type theta = math::atan2(r, z);
        if (!(theta < this->maximum_angle) || !(r2 + (z * z) > type(0))) {
            return false;
        }
        type theta_d = type(0);
        type theta_d_derivative = type(0);
        this->distort_angle(theta, theta_d, theta_d_derivative);

        // scale = theta_d / r, theta_over_r = theta / r, and curvature = (d scale / d r) / r, by their series near the axis,
        // where r is a vanishing fraction of z (and z is positive, theta being below a half turn).
        constexpr static const type epsilon = math::epsilon<type>();
        const type series_limit = math::sqrt(math::sqrt(epsilon));
        type scale = type(0);
        type theta_over_r = type(0);
        type curvature = type(0);
        const type length_squared = r2 + (z * z);
        if (r < series_limit * z) {
            const type inverse_z = type(1) / z;
            const type ratio_squared = r2 * inverse_z * inverse_z;
            scale = inverse_z * (type(1) + ((this->radial_k[0] - (type(1) / type(3))) * ratio_squared));
            theta_over_r = inverse_z * (type(1) - (ratio_squared / type(3)));
            curvature = ((type(2) * this->radial_k[0]) - (type(2) / type(3))) * inverse_z * inverse_z * inverse_z;
        }
        else {
            scale = theta_d / r;
            theta_over_r = theta / r;
            curvature = ((theta_d_derivative * z / length_squared) - scale) / r2;
        }

        const type distorted_xy[2] = { scale * x, scale * y };
        point_xy[0] = (this->focal_lengths[0] * distorted_xy[0]) + this->centre_points[0];
        point_xy[1] = (this->focal_lengths[1] * distorted_xy[1]) + this->centre_points[1];

        if (jacobian_projection != nullptr) {
            const type along_z = -theta_d_derivative / length_squared;
            jacobian_projection[0 * 3 + 0] = this->focal_lengths[0] * (scale + (x * x * curvature));
            jacobian_projection[0 * 3 + 1] = this->focal_lengths[0] * (x * y * curvature);
            jacobian_projection[0 * 3 + 2] = this->focal_lengths[0] * (x * along_z);
            jacobian_projection[1 * 3 + 0] = this->focal_lengths[1] * (x * y * curvature);
            jacobian_projection[1 * 3 + 1] = this->focal_lengths[1] * (scale + (y * y * curvature));
            jacobian_projection[1 * 3 + 2] = this->focal_lengths[1] * (y * along_z);
        }

        if (jacobian_parameters != nullptr) {
            constexpr static const size_t count = kannala_brandt::parameter_count;
            jacobian_parameters[0 * count + 0] = distorted_xy[0];
            jacobian_parameters[0 * count + 1] = type(0);
            jacobian_parameters[0 * count + 2] = type(1);
            jacobian_parameters[0 * count + 3] = type(0);
            jacobian_parameters[1 * count + 0] = type(0);
            jacobian_parameters[1 * count + 1] = distorted_xy[1];
            jacobian_parameters[1 * count + 2] = type(0);
            jacobian_parameters[1 * count + 3] = type(1);
            // d theta_d / d k_i = theta^(2 i + 3), so d (theta_d / r) / d k_i = theta^(2 i + 2) * theta / r.
            const type theta2 = theta * theta;
            type power = theta2 * theta_over_r;
            for (size_t index = 0; index < 4; ++index) {
                jacobian_parameters[0 * count + 4 + index] = this->focal_lengths[0] * power * x;
                jacobian_parameters[1 * count + 4 + index] = this->focal_lengths[1] * power * y;
                power *= theta2;
            }
        }

        return true;
    }

    template <typename type>
    bool kannala_brandt<type>::unproject(
        const type* const point_xy,
        type* const ray_xyz,
        type* const jacobian_unprojection
    ) const {
        const type distorted_xy[2] = {
            (point_xy[0] - this->centre_points[0]) / this->focal_lengths[0],
            (point_xy[1] - this->centre_points[1]) / this->focal_lengths[1]
        };
        const type distorted_radius = math::sqrt((distorted_xy[0] * distorted_xy[0]) + (distorted_xy[1] * distorted_xy[1]));
        if (!(distorted_radius < this->maximum_distorted_radius)) {
            return false;
        }

        // theta_d grows monotonically up to the maximum angle, so Newton's steps are kept inside a shrinking bracket and
        // replaced by bisection when they would leave it.
        type lower = type(0);
        type upper = this->maximum_angle;
        type theta = math::min(distorted_radius, type(0.5) * (lower + upper));
        type theta_d = type(0);
        type theta_d_derivative = type(1);
        bool converged = false;
        constexpr static const type epsilon = math::epsilon<type>();
        const type tolerance = type(4) * epsilon * math::max(type(1), distorted_radius);
        for (size_t iteration = 0; iteration < kannala_brandt::maximum_unprojection_iterations; ++iteration) {
            this->distort_angle(theta, theta_d, theta_d_derivative);
            const type error = theta_d - distorted_radius;
            if (math::abs(error) <= tolerance) {
                converged = true;
                break;
            }
            if (error > type(0)) {
                upper = theta;
            }
            else {
                lower = theta;
            }
            type next = (theta_d_derivative > type(0)) ? (theta - (error / theta_d_derivative)) : lower;
            if (!(next > lower) || !(next < upper)) {
                next = type(0.5) * (lower + upper);
            }
            if (!(next != theta) || !((upper - lower) > tolerance)) {
                converged = true;
                break;
            }
            theta = next;
        }
        if (!converged) {
            return false;
        }
        this->distort_angle(theta, theta_d, theta_d_derivative);

        const type sine = math::sin(theta);
        const type cosine = math::cos(theta);
        // ray = (sin(theta) * m / |m|, cos(theta)), with g = sin(theta) / |m| and h = (d g / d |m|) / |m|.
        const type series_limit = math::sqrt(epsilon);
        type g = type(1);
        type h = type(0);
        if (distorted_radius > series_limit) {
            g = sine / distorted_radius;
            h = ((cosine * distorted_radius / theta_d_derivative) - sine) / (distorted_radius * distorted_radius * distorted_radius);
        }
        else {
            h = -((type(1) / type(3)) + (type(2) * this->radial_k[0]));
        }
        ray_xyz[0] = g * distorted_xy[0];
        ray_xyz[1] = g * distorted_xy[1];
        ray_xyz[2] = cosine;

        if (jacobian_unprojection != nullptr) {
            const type inverse_fx = type(1) / this->focal_lengths[0];
            const type inverse_fy = type(1) / this->focal_lengths[1];
            const type along_z = -g / theta_d_derivative;
            jacobian_unprojection[0 * 2 + 0] = (g + (distorted_xy[0] * distorted_xy[0] * h)) * inverse_fx;
            jacobian_unprojection[0 * 2 + 1] = (distorted_xy[0] * distorted_xy[1] * h) * inverse_fy;
            jacobian_unprojection[1 * 2 + 0] = (distorted_xy[0] * distorted_xy[1] * h) * inverse_fx;
            jacobian_unprojection[1 * 2 + 1] = (g + (distorted_xy[1] * distorted_xy[1] * h)) * inverse_fy;
            jacobian_unprojection[2 * 2 + 0] = along_z * distorted_xy[0] * inverse_fx;
            jacobian_unprojection[2 * 2 + 1] = along_z * distorted_xy[1] * inverse_fy;
        }

        return true;
    }

    template class kannala_brandt<float>;
    template class kannala_brandt<double>;
}
