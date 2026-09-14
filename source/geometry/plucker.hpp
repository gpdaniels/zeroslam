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
#ifndef ZEROSLAM_GEOMETRY_PLUCKER_HPP
#define ZEROSLAM_GEOMETRY_PLUCKER_HPP

#include "math/matrix.hpp"

namespace geometry {
    class plucker final {
    public:
        math::matrix<double, 3, 1> moment;
        math::matrix<double, 3, 1> direction;

    public:
        plucker();
        plucker(const math::matrix<double, 3, 1>& moment_value, const math::matrix<double, 3, 1>& direction_value);

    public:
        static math::matrix<double, 3, 1> cross(const math::matrix<double, 3, 1>& lhs, const math::matrix<double, 3, 1>& rhs);
        static double dot(const math::matrix<double, 3, 1>& lhs, const math::matrix<double, 3, 1>& rhs);

    public:
        static bool from_points(const math::matrix<double, 3, 1>& point_a, const math::matrix<double, 3, 1>& point_b, plucker& result);

        static bool from_planes(
            const math::matrix<double, 3, 1>& normal_1,
            const double offset_1,
            const math::matrix<double, 3, 1>& normal_2,
            const double offset_2,
            const double minimum_sine,
            plucker& result
        );

        plucker transformed(const math::matrix<double, 3, 3>& rotation, const math::matrix<double, 3, 1>& translation) const;

        math::matrix<double, 3, 1> closest_point_to_origin() const;

        double distance_to_origin() const;

        math::matrix<double, 3, 1> project_point(const math::matrix<double, 3, 1>& point) const;

        bool closest_point_to_ray(
            const math::matrix<double, 3, 1>& ray_origin,
            const math::matrix<double, 3, 1>& ray_direction,
            math::matrix<double, 3, 1>& result
        ) const;

    public:
        void to_orthonormal(math::matrix<double, 3, 3>& u, double& w1, double& w2, double& scale) const;
        static plucker from_orthonormal(const math::matrix<double, 3, 3>& u, const double w1, const double w2, const double scale);

        bool oplus(const double delta_theta_1, const double delta_theta_2, const double delta_theta_3, const double delta_phi);
    };
}

#endif // ZEROSLAM_GEOMETRY_PLUCKER_HPP
