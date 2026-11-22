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
#ifndef ZEROSLAM_SENSOR_CAMERA_DOUBLE_SPHERE_HPP
#define ZEROSLAM_SENSOR_CAMERA_DOUBLE_SPHERE_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor::camera {
    // The double sphere camera model (Usenko, Demmel and Cremers, "The Double Sphere Camera Model", 3DV 2018): the point is
    // projected onto two unit spheres a distance xi apart and then onto the image plane from a centre at alpha / (1 - alpha)
    // in front of the second,
    //   (x_d, y_d) = (x, y) / (alpha * d2 + (1 - alpha) * (xi * d1 + z)),
    //   d1 = |(x, y, z)|,  d2 = |(x, y, xi * d1 + z)|
    // and the pixel is (fx * x_d + cx, fy * y_d + cy). Parameters [fx fy cx cy xi alpha], alpha in [0, 1], xi > -1. The
    // valid region is z > -w2 * d1 with w1 = alpha / (1 - alpha) for alpha <= 0.5 and (1 - alpha) / alpha above, and
    // w2 = (w1 + xi) / sqrt(2 * w1 * xi + xi^2 + 1).
    template <typename type>
    class double_sphere final {
    public:
        constexpr static const char* name = "double_sphere";
        constexpr static const size_t parameter_count = 4 + 2;

    private:
        type focal_lengths[2];
        type centre_points[2];
        type xi;
        type alpha;
        // The bound of the valid region, z > -w2 * d1, from xi and alpha.
        type valid_bound;

    public:
        double_sphere();

        double_sphere(const type* const parameters, const size_t parameters_length);

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

    extern template class double_sphere<float>;
    extern template class double_sphere<double>;
}

#endif // ZEROSLAM_SENSOR_CAMERA_DOUBLE_SPHERE_HPP
