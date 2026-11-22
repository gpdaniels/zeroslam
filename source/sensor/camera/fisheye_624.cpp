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

#include "sensor/camera/fisheye_624.hpp"

#include "core/assert.hpp"
#include "math/math.hpp"
#include "math/polynomial.hpp"

namespace sensor::camera {
    template <typename type>
    fisheye_624<type>::fisheye_624()
        : focal_lengths{ type(1), type(1) }
        , centre_points{ type(0.5), type(0.5) }
        , radial_k{ type(0), type(0), type(0), type(0), type(0), type(0) }
        , tangential_p{ type(0), type(0) }
        , prism_s{ type(0), type(0), type(0), type(0) }
        , maximum_angle(math::pi<type>())
        , maximum_distorted_radius(math::pi<type>()) {
    }

    template <typename type>
    fisheye_624<type>::fisheye_624(const type* const parameters, const size_t parameters_length)
        : fisheye_624() {
        const bool valid = this->set_parameters(parameters, parameters_length);
        ASSERT(valid, "The Fisheye624 parameters must be sixteen finite values with positive focal lengths.");
        static_cast<void>(valid);
    }

    template <typename type>
    size_t fisheye_624<type>::get_parameter_count() const {
        return fisheye_624::parameter_count;
    }

    template <typename type>
    bool fisheye_624<type>::set_parameters(const type* const parameters, const size_t parameters_length) {
        if ((parameters == nullptr) || (parameters_length != fisheye_624::parameter_count)) {
            return false;
        }
        for (size_t index = 0; index < fisheye_624::parameter_count; ++index) {
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
        for (size_t index = 0; index < 6; ++index) {
            this->radial_k[index] = parameters[4 + index];
        }
        this->tangential_p[0] = parameters[10];
        this->tangential_p[1] = parameters[11];
        for (size_t index = 0; index < 4; ++index) {
            this->prism_s[index] = parameters[12 + index];
        }
        // d theta_d / d theta as a polynomial in theta^2, its first root is where the projection folds back.
        double derivative[7] = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        for (size_t index = 0; index < 6; ++index) {
            derivative[index + 1] = static_cast<double>((2 * index) + 3) * static_cast<double>(this->radial_k[index]);
        }
        const double half_turn = math::pi<double>();
        const double fold_squared = math::polynomial::first_non_positive(&derivative[0], 6, half_turn * half_turn);
        this->maximum_angle = static_cast<type>(math::min(math::sqrt(fold_squared), half_turn));
        type derivative_at_maximum = type(0);
        this->distort_angle(this->maximum_angle, this->maximum_distorted_radius, derivative_at_maximum);
        return true;
    }

    template <typename type>
    bool fisheye_624<type>::get_parameters(type* const parameters, const size_t parameters_length) const {
        if ((parameters == nullptr) || (parameters_length != fisheye_624::parameter_count)) {
            return false;
        }
        parameters[0] = this->focal_lengths[0];
        parameters[1] = this->focal_lengths[1];
        parameters[2] = this->centre_points[0];
        parameters[3] = this->centre_points[1];
        for (size_t index = 0; index < 6; ++index) {
            parameters[4 + index] = this->radial_k[index];
        }
        parameters[10] = this->tangential_p[0];
        parameters[11] = this->tangential_p[1];
        for (size_t index = 0; index < 4; ++index) {
            parameters[12 + index] = this->prism_s[index];
        }
        return true;
    }

    template <typename type>
    bool fisheye_624<type>::get_pinhole(type* const pinhole_parameters, const size_t pinhole_parameters_length) const {
        if ((pinhole_parameters == nullptr) || (pinhole_parameters_length != 4)) {
            return false;
        }
        // The radial distortion is the identity to first order at the centre, the tangential and thin prism terms second order.
        pinhole_parameters[0] = this->focal_lengths[0];
        pinhole_parameters[1] = this->focal_lengths[1];
        pinhole_parameters[2] = this->centre_points[0];
        pinhole_parameters[3] = this->centre_points[1];
        return true;
    }

    template <typename type>
    type fisheye_624<type>::get_maximum_angle() const {
        return this->maximum_angle;
    }

    template <typename type>
    void fisheye_624<type>::distort_angle(const type theta, type& distorted, type& derivative) const {
        const type theta2 = theta * theta;
        type polynomial = this->radial_k[5];
        type slope = type(13) * this->radial_k[5];
        for (size_t index = 5; index > 0; --index) {
            polynomial = (polynomial * theta2) + this->radial_k[index - 1];
            slope = (slope * theta2) + (type((2 * (index - 1)) + 3) * this->radial_k[index - 1]);
        }
        distorted = theta * (type(1) + (theta2 * polynomial));
        derivative = type(1) + (theta2 * slope);
    }

    template <typename type>
    bool fisheye_624<type>::distort_plane(const type* const radial_xy, type* const distorted_xy, type* const jacobian) const {
        const type x = radial_xy[0];
        const type y = radial_xy[1];
        const type rho2 = (x * x) + (y * y);
        const type p0 = this->tangential_p[0];
        const type p1 = this->tangential_p[1];
        distorted_xy[0] = x + (p0 * (rho2 + (type(2) * x * x))) + (type(2) * p1 * x * y) + (this->prism_s[0] * rho2) + (this->prism_s[1] * rho2 * rho2);
        distorted_xy[1] = y + (p1 * (rho2 + (type(2) * y * y))) + (type(2) * p0 * x * y) + (this->prism_s[2] * rho2) + (this->prism_s[3] * rho2 * rho2);
        const type prism_x = this->prism_s[0] + (type(2) * this->prism_s[1] * rho2);
        const type prism_y = this->prism_s[2] + (type(2) * this->prism_s[3] * rho2);
        jacobian[0 * 2 + 0] = type(1) + (type(6) * p0 * x) + (type(2) * p1 * y) + (type(2) * x * prism_x);
        jacobian[0 * 2 + 1] = (type(2) * p0 * y) + (type(2) * p1 * x) + (type(2) * y * prism_x);
        jacobian[1 * 2 + 0] = (type(2) * p1 * x) + (type(2) * p0 * y) + (type(2) * x * prism_y);
        jacobian[1 * 2 + 1] = type(1) + (type(6) * p1 * y) + (type(2) * p0 * x) + (type(2) * y * prism_y);
        return ((jacobian[0] * jacobian[3]) - (jacobian[1] * jacobian[2])) > type(0);
    }

    template <typename type>
    bool fisheye_624<type>::project(
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

        // scale = theta_d / r, theta_over_r = theta / r, and curvature = (d scale / d r) / r, by their series near the axis.
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

        const type radial_xy[2] = { scale * x, scale * y };
        type distorted_xy[2];
        type plane[2 * 2];
        if (!this->distort_plane(&radial_xy[0], &distorted_xy[0], &plane[0])) {
            return false;
        }
        point_xy[0] = (this->focal_lengths[0] * distorted_xy[0]) + this->centre_points[0];
        point_xy[1] = (this->focal_lengths[1] * distorted_xy[1]) + this->centre_points[1];

        if (jacobian_projection != nullptr) {
            const type along_z = -theta_d_derivative / length_squared;
            const type radial[2 * 3] = {
                scale + (x * x * curvature),
                x * y * curvature,
                x * along_z,
                x * y * curvature,
                scale + (y * y * curvature),
                y * along_z
            };
            for (size_t row = 0; row < 2; ++row) {
                for (size_t column = 0; column < 3; ++column) {
                    jacobian_projection[row * 3 + column] = this->focal_lengths[row] * ((plane[row * 2 + 0] * radial[0 * 3 + column]) + (plane[row * 2 + 1] * radial[1 * 3 + column]));
                }
            }
        }

        if (jacobian_parameters != nullptr) {
            constexpr static const size_t count = fisheye_624::parameter_count;
            for (size_t index = 0; index < 2 * count; ++index) {
                jacobian_parameters[index] = type(0);
            }
            jacobian_parameters[0 * count + 0] = distorted_xy[0];
            jacobian_parameters[0 * count + 2] = type(1);
            jacobian_parameters[1 * count + 1] = distorted_xy[1];
            jacobian_parameters[1 * count + 3] = type(1);
            // d theta_d / d k_i = theta^(2 i + 3), so d (x_r, y_r) / d k_i = theta^(2 i + 2) * theta / r * (x, y).
            const type theta2 = theta * theta;
            type power = theta2 * theta_over_r;
            for (size_t index = 0; index < 6; ++index) {
                const type radial[2] = { power * x, power * y };
                for (size_t row = 0; row < 2; ++row) {
                    jacobian_parameters[row * count + 4 + index] = this->focal_lengths[row] * ((plane[row * 2 + 0] * radial[0]) + (plane[row * 2 + 1] * radial[1]));
                }
                power *= theta2;
            }
            const type rho2 = (radial_xy[0] * radial_xy[0]) + (radial_xy[1] * radial_xy[1]);
            const type cross = type(2) * radial_xy[0] * radial_xy[1];
            jacobian_parameters[0 * count + 10] = this->focal_lengths[0] * (rho2 + (type(2) * radial_xy[0] * radial_xy[0]));
            jacobian_parameters[0 * count + 11] = this->focal_lengths[0] * cross;
            jacobian_parameters[1 * count + 10] = this->focal_lengths[1] * cross;
            jacobian_parameters[1 * count + 11] = this->focal_lengths[1] * (rho2 + (type(2) * radial_xy[1] * radial_xy[1]));
            jacobian_parameters[0 * count + 12] = this->focal_lengths[0] * rho2;
            jacobian_parameters[0 * count + 13] = this->focal_lengths[0] * rho2 * rho2;
            jacobian_parameters[1 * count + 14] = this->focal_lengths[1] * rho2;
            jacobian_parameters[1 * count + 15] = this->focal_lengths[1] * rho2 * rho2;
        }

        return true;
    }

    template <typename type>
    bool fisheye_624<type>::unproject(
        const type* const point_xy,
        type* const ray_xyz,
        type* const jacobian_unprojection
    ) const {
        const type distorted_xy[2] = {
            (point_xy[0] - this->centre_points[0]) / this->focal_lengths[0],
            (point_xy[1] - this->centre_points[1]) / this->focal_lengths[1]
        };

        // Undo the tangential and thin prism distortion by Newton's method, every iterate held inside the radial fold.
        type radial_xy[2] = { distorted_xy[0], distorted_xy[1] };
        const type limit_squared = this->maximum_distorted_radius * this->maximum_distorted_radius;
        const type start_squared = (radial_xy[0] * radial_xy[0]) + (radial_xy[1] * radial_xy[1]);
        if (start_squared >= limit_squared) {
            const type shrink = type(0.5) * math::sqrt(limit_squared / start_squared);
            radial_xy[0] *= shrink;
            radial_xy[1] *= shrink;
        }
        constexpr static const type epsilon = math::epsilon<type>();
        const type plane_tolerance = type(16) * epsilon * epsilon * math::max(type(1), (distorted_xy[0] * distorted_xy[0]) + (distorted_xy[1] * distorted_xy[1]));
        type plane[2 * 2];
        bool plane_converged = false;
        for (size_t iteration = 0; iteration < fisheye_624::maximum_undistortion_iterations; ++iteration) {
            type redistorted_xy[2];
            if (!this->distort_plane(&radial_xy[0], &redistorted_xy[0], &plane[0])) {
                return false;
            }
            const type error[2] = { distorted_xy[0] - redistorted_xy[0], distorted_xy[1] - redistorted_xy[1] };
            const type error_squared = (error[0] * error[0]) + (error[1] * error[1]);
            if (error_squared <= plane_tolerance) {
                plane_converged = true;
                break;
            }
            const type determinant_inverse = type(1) / ((plane[0] * plane[3]) - (plane[1] * plane[2]));
            radial_xy[0] += ((plane[3] * error[0]) - (plane[1] * error[1])) * determinant_inverse;
            radial_xy[1] += ((plane[0] * error[1]) - (plane[2] * error[0])) * determinant_inverse;
            if (!(((radial_xy[0] * radial_xy[0]) + (radial_xy[1] * radial_xy[1])) < limit_squared)) {
                return false;
            }
        }
        if (!plane_converged) {
            // Newton's method stalls a few units of rounding short of the tolerance near the rim, accept a last check there.
            type redistorted_xy[2];
            if (!this->distort_plane(&radial_xy[0], &redistorted_xy[0], &plane[0])) {
                return false;
            }
            const type error[2] = { distorted_xy[0] - redistorted_xy[0], distorted_xy[1] - redistorted_xy[1] };
            if (!(((error[0] * error[0]) + (error[1] * error[1])) <= (type(1.0e6) * plane_tolerance))) {
                return false;
            }
        }
        type redistorted_xy[2];
        if (!this->distort_plane(&radial_xy[0], &redistorted_xy[0], &plane[0])) {
            return false;
        }

        // Undo the radial distortion: theta_d grows monotonically up to the maximum angle, so Newton's steps are kept inside
        // a shrinking bracket and replaced by bisection when they would leave it.
        const type distorted_radius = math::sqrt((radial_xy[0] * radial_xy[0]) + (radial_xy[1] * radial_xy[1]));
        if (!(distorted_radius < this->maximum_distorted_radius)) {
            return false;
        }
        type lower = type(0);
        type upper = this->maximum_angle;
        type theta = math::min(distorted_radius, type(0.5) * (lower + upper));
        type theta_d = type(0);
        type theta_d_derivative = type(1);
        bool converged = false;
        const type tolerance = type(4) * epsilon * math::max(type(1), distorted_radius);
        for (size_t iteration = 0; iteration < fisheye_624::maximum_unprojection_iterations; ++iteration) {
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
        ray_xyz[0] = g * radial_xy[0];
        ray_xyz[1] = g * radial_xy[1];
        ray_xyz[2] = cosine;

        if (jacobian_unprojection != nullptr) {
            // d ray / d (x_r, y_r), then through the inverse of the plane distortion's jacobian and the focal lengths.
            const type along_z = -g / theta_d_derivative;
            const type radial[3 * 2] = {
                g + (radial_xy[0] * radial_xy[0] * h),
                radial_xy[0] * radial_xy[1] * h,
                radial_xy[0] * radial_xy[1] * h,
                g + (radial_xy[1] * radial_xy[1] * h),
                along_z * radial_xy[0],
                along_z * radial_xy[1]
            };
            const type determinant_inverse = type(1) / ((plane[0] * plane[3]) - (plane[1] * plane[2]));
            const type inverse_plane[2 * 2] = {
                plane[3] * determinant_inverse,
                -plane[1] * determinant_inverse,
                -plane[2] * determinant_inverse,
                plane[0] * determinant_inverse
            };
            const type inverse_f[2] = { type(1) / this->focal_lengths[0], type(1) / this->focal_lengths[1] };
            for (size_t row = 0; row < 3; ++row) {
                for (size_t column = 0; column < 2; ++column) {
                    jacobian_unprojection[row * 2 + column] = ((radial[row * 2 + 0] * inverse_plane[0 * 2 + column]) + (radial[row * 2 + 1] * inverse_plane[1 * 2 + column])) * inverse_f[column];
                }
            }
        }

        return true;
    }

    template class fisheye_624<float>;
    template class fisheye_624<double>;
}
