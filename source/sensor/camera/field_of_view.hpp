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
#ifndef ZEROSLAM_SENSOR_CAMERA_FIELD_OF_VIEW_HPP
#define ZEROSLAM_SENSOR_CAMERA_FIELD_OF_VIEW_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor::camera {
    // The field of view (arctangent) model of Devernay and Faugeras ("Straight lines have to be straight", MVA 2001), the
    // ATANCamera of PTAM (Klein and Murray, ISMAR 2007). The image plane point (x / z, y / z) at radius r_u is moved along
    // its radius to
    //   r_d = atan(2 * r_u * tan(w / 2)) / w
    // and the pixel is (fx * x_d + cx, fy * y_d + cy). Parameters [fx fy cx cy w], w in [0, pi) radians, w = 0 being the
    // pinhole. The model is a pinhole one, so its valid region is the points in front of the camera (z > 0), and the pixels
    // with w * r_d below a quarter turn.
    // PTAM's calibration gives its parameters as fractions of the image size, [fx / width, fy / height, cx' / width,
    // cy' / height, w] with its principal point cx' + 0.5 in its own pixel frame; in the pixel-centre frame of this library
    // that is fx = width * p0, fy = height * p1, cx = width * p2, cy = height * p3.
    template <typename type>
    class field_of_view final {
    public:
        constexpr static const char* name = "field_of_view";
        constexpr static const size_t parameter_count = 4 + 1;

    private:
        type focal_lengths[2];
        type centre_points[2];
        type field;

    public:
        field_of_view();

        field_of_view(const type* const parameters, const size_t parameters_length);

    public:
        size_t get_parameter_count() const;

        bool set_parameters(const type* const parameters, const size_t parameters_length);

        bool get_parameters(type* const parameters, const size_t parameters_length) const;

        bool get_pinhole(type* const pinhole_parameters, const size_t pinhole_parameters_length) const;

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

    extern template class field_of_view<float>;
    extern template class field_of_view<double>;
}

#endif // ZEROSLAM_SENSOR_CAMERA_FIELD_OF_VIEW_HPP
