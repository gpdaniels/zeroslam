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
#ifndef ZEROSLAM_SENSOR_CAMERA_PINHOLE_RADIAL_TANGENTIAL_HPP
#define ZEROSLAM_SENSOR_CAMERA_PINHOLE_RADIAL_TANGENTIAL_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor::camera {
    template <typename type>
    class pinhole_radial_tangential final {
    public:
        constexpr static const char* name = "pinhole_radial_tangential";
        constexpr static const size_t parameter_count = 4 + 8;
        constexpr static const size_t maximum_undistortion_iterations = 20;

    private:
        type focal_lengths[2];
        type centre_points[2];
        type radial_k[6];
        type tangential_p[2];

    public:
        pinhole_radial_tangential();

        pinhole_radial_tangential(const type* const parameters, const size_t parameters_length);

    public:
        size_t get_parameter_count() const;

        bool set_parameters(const type* const parameters, const size_t parameters_length);

        bool get_parameters(type* const parameters, const size_t parameters_length) const;

        // Camera frame point to image coordinates normalised by the image width, jacobians row major 2 by 3 and 2 by parameter count.
        bool project(
            const type* const point_xyz,
            type* const point_xy,
            type* const jacobian_projection = nullptr,
            type* const jacobian_parameters = nullptr
        ) const;

        // Image coordinates normalised by the image width to the ray with unit depth, jacobian row major 3 by 2.
        bool unproject(
            const type* const point_xy,
            type* const ray_xyz,
            type* const jacobian_unprojection = nullptr
        ) const;

    private:
        bool distort(
            const type* const undistorted_xy,
            type* const distorted_xy,
            type* const jacobian_distortion = nullptr,
            type* const jacobian_coefficients = nullptr
        ) const;

        bool undistort(
            const type* const distorted_xy,
            type* const undistorted_xy,
            type* const jacobian_undistortion = nullptr
        ) const;
    };

    extern template class pinhole_radial_tangential<float>;
    extern template class pinhole_radial_tangential<double>;
}

#endif // ZEROSLAM_SENSOR_CAMERA_PINHOLE_RADIAL_TANGENTIAL_HPP
