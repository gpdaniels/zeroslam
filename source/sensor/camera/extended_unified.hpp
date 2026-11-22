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
#ifndef ZEROSLAM_SENSOR_CAMERA_EXTENDED_UNIFIED_HPP
#define ZEROSLAM_SENSOR_CAMERA_EXTENDED_UNIFIED_HPP

#include "sensor/camera/parameter_role.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor::camera {
    // The extended unified camera model (Khomenko, Bamberger and Stettner, ICRA 2016), as Usenko, Demmel and Cremers state
    // it ("The Double Sphere Camera Model", 3DV 2018):
    //   (x_d, y_d) = (x, y) / (alpha * d + (1 - alpha) * z),  d = sqrt(beta * (x^2 + y^2) + z^2)
    // and the pixel is (fx * x_d + cx, fy * y_d + cy). Parameters [fx fy cx cy alpha beta], alpha in [0, 1], beta > 0
    // (alpha = 0 is the pinhole, beta = 1 the unified model in its alpha form). The valid region is z > -w * d with
    // w = alpha / (1 - alpha) for alpha <= 0.5 and (1 - alpha) / alpha above.
    template <typename type>
    class extended_unified final {
    public:
        constexpr static const char* name = "extended_unified";
        constexpr static const size_t parameter_count = 4 + 2;

    private:
        type focal_lengths[2];
        type centre_points[2];
        type alpha;
        type beta;

    public:
        extended_unified();

        extended_unified(const type* const parameters, const size_t parameters_length);

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

    extern template class extended_unified<float>;
    extern template class extended_unified<double>;
}

#endif // ZEROSLAM_SENSOR_CAMERA_EXTENDED_UNIFIED_HPP
