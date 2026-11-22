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
#ifndef ZEROSLAM_SENSOR_CAMERA_UNIFIED_HPP
#define ZEROSLAM_SENSOR_CAMERA_UNIFIED_HPP

#include "sensor/camera/parameter_role.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor::camera {
    // The unified camera model (Geyer and Daniilidis, ECCV 2000; Barreto and Araujo; Mei and Rives, ICRA 2007, without the
    // distortion Mei adds): the point is put on the unit sphere and projected from a centre a distance xi behind the sphere's
    // centre onto the image plane,
    //   (x_d, y_d) = (x, y) / (z + xi * d),  d = |(x, y, z)|
    // and the pixel is (fx * x_d + cx, fy * y_d + cy). Parameters [fx fy cx cy xi], xi >= 0 (Kalibr's and OpenCV omnidir's
    // xi, with no distortion; xi = 0 is the pinhole, xi = 1 the stereographic projection with focal lengths fx / 2). The
    // valid region is z > -w * d with w = min(xi, 1 / xi), where the denominator stays positive and the projection does
    // not fold back: past ninety degrees from the axis for any xi above zero.
    template <typename type>
    class unified final {
    public:
        constexpr static const char* name = "unified";
        constexpr static const size_t parameter_count = 4 + 1;

    private:
        type focal_lengths[2];
        type centre_points[2];
        type xi;

    public:
        unified();

        unified(const type* const parameters, const size_t parameters_length);

    public:
        size_t get_parameter_count() const;

        bool set_parameters(const type* const parameters, const size_t parameters_length);

        bool get_parameters(type* const parameters, const size_t parameters_length) const;

        bool get_pinhole(type* const pinhole_parameters, const size_t pinhole_parameters_length) const;

        bool get_parameter_roles(parameter_role* const roles, const size_t roles_length) const;

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
    };

    extern template class unified<float>;
    extern template class unified<double>;
}

#endif // ZEROSLAM_SENSOR_CAMERA_UNIFIED_HPP
