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
#ifndef ZEROSLAM_SENSOR_CAMERA_KANNALA_BRANDT_HPP
#define ZEROSLAM_SENSOR_CAMERA_KANNALA_BRANDT_HPP

#include "sensor/camera/parameter_role.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor::camera {
    // The Kannala-Brandt equidistant fisheye (Kannala and Brandt, PAMI 2006), as OpenCV's fisheye module and Kalibr's
    // pinhole-equi have it: a ray at theta from the optical axis lands at the distorted radius
    //   theta_d = theta * (1 + k1 * theta^2 + k2 * theta^4 + k3 * theta^6 + k4 * theta^8)
    // in the direction of its azimuth, and the pixel is (fx * theta_d * x / r + cx, fy * theta_d * y / r + cy) with
    // r = sqrt(x^2 + y^2). Parameters [fx fy cx cy k1 k2 k3 k4]; with the k all zero it is the equidistant projection.
    // The angle is taken from the whole ray, so the model reaches past ninety degrees. Its valid region ends where the
    // distorted radius stops growing with the angle, or at a half turn, whichever comes first.
    template <typename type>
    class kannala_brandt final {
    public:
        constexpr static const char* name = "kannala_brandt";
        constexpr static const size_t parameter_count = 4 + 4;
        constexpr static const size_t maximum_unprojection_iterations = 64;

    private:
        type focal_lengths[2];
        type centre_points[2];
        type radial_k[4];
        // The angle from the optical axis past which theta_d folds back, at most a half turn, and theta_d there.
        type maximum_angle;
        type maximum_distorted_radius;

    public:
        kannala_brandt();

        kannala_brandt(const type* const parameters, const size_t parameters_length);

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
    };

    extern template class kannala_brandt<float>;
    extern template class kannala_brandt<double>;
}

#endif // ZEROSLAM_SENSOR_CAMERA_KANNALA_BRANDT_HPP
