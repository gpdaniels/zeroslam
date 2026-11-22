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

#pragma once
#ifndef ZEROSLAM_SENSOR_CAMERA_FISHEYE_624_HPP
#define ZEROSLAM_SENSOR_CAMERA_FISHEYE_624_HPP

#include "sensor/camera/parameter_role.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor::camera {
    // Project Aria's Fisheye624 (FisheyeRadTanThinPrism in projectaria_tools, RAD_TAN_THIN_PRISM_FISHEYE in COLMAP): the
    // Kannala-Brandt radial distortion to the twelfth power of the angle, then tangential and thin prism distortion of the
    // distorted point (x_r, y_r) = theta_d * (x, y) / r at radius rho:
    //   theta_d = theta * (1 + k0 * theta^2 + k1 * theta^4 + ... + k5 * theta^12)
    //   x_d = x_r + p0 * (rho^2 + 2 * x_r^2) + 2 * p1 * x_r * y_r + s0 * rho^2 + s1 * rho^4
    //   y_d = y_r + p1 * (rho^2 + 2 * y_r^2) + 2 * p0 * x_r * y_r + s2 * rho^2 + s3 * rho^4
    // and the pixel is (fx * x_d + cx, fy * y_d + cy). Parameters [fx fy cx cy k0 k1 k2 k3 k4 k5 p0 p1 s0 s1 s2 s3], COLMAP's
    // order; Aria's own calibrations have one focal length, fx = fy. Note p0 pairs with x as OpenCV's p2 does. The valid
    // region ends where theta_d stops growing with the angle, at most a half turn, and where the tangential and thin prism
    // distortion would fold the image (its jacobian determinant reaching zero).
    template <typename type>
    class fisheye_624 final {
    public:
        constexpr static const char* name = "fisheye_624";
        constexpr static const size_t parameter_count = 4 + 6 + 2 + 4;
        constexpr static const size_t maximum_undistortion_iterations = 20;
        constexpr static const size_t maximum_unprojection_iterations = 64;

    private:
        type focal_lengths[2];
        type centre_points[2];
        type radial_k[6];
        type tangential_p[2];
        type prism_s[4];
        // The angle from the optical axis past which theta_d folds back, at most a half turn, and theta_d there.
        type maximum_angle;
        type maximum_distorted_radius;

    public:
        fisheye_624();

        fisheye_624(const type* const parameters, const size_t parameters_length);

    public:
        size_t get_parameter_count() const;

        bool set_parameters(const type* const parameters, const size_t parameters_length);

        bool get_parameters(type* const parameters, const size_t parameters_length) const;

        bool get_pinhole(type* const pinhole_parameters, const size_t pinhole_parameters_length) const;

        bool get_parameter_roles(parameter_role* const roles, const size_t roles_length) const;

        type get_maximum_angle() const;

        bool project(
            const type* const point_xyz,
            type* const point_xy,
            type* const jacobian_projection = nullptr,
            type* const jacobian_parameters = nullptr
        ) const;

        bool unproject(
            const type* const point_xy,
            type* const ray_xyz,
            type* const jacobian_unprojection = nullptr
        ) const;

    private:
        // theta_d and its derivative with respect to theta.
        void distort_angle(const type theta, type& distorted, type& derivative) const;

        // The tangential and thin prism distortion of a radially distorted point, with its row major 2x2 jacobian, false
        // where the jacobian's determinant is not positive.
        bool distort_plane(const type* const radial_xy, type* const distorted_xy, type* const jacobian) const;
    };

    extern template class fisheye_624<float>;
    extern template class fisheye_624<double>;
}

#endif // ZEROSLAM_SENSOR_CAMERA_FISHEYE_624_HPP
