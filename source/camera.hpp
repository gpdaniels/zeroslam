/*
Copyright (C) 2025 Geoffrey Daniels. https://gpdaniels.com/

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
#ifndef CAMERA_HPP
#define CAMERA_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace camera {
    class pinhole final {
    public:
        constexpr static const size_t parameter_count = 4;

    private:
        double focal_lengths[2];
        double centre_points[2];

    public:
        pinhole();
        pinhole(const double* const __restrict parameters, const size_t parameters_length);

    public:
        void set_parameters(const double* const __restrict parameters, const size_t parameters_length);
        void get_parameters(double* const __restrict parameters, const size_t parameters_length) const;

    public:
        bool project(
            const double* const __restrict point_xyz,
            double* const __restrict point_xy,
            double* const __restrict jacobian_projection = nullptr,
            double* const __restrict jacobian_parameters = nullptr
        ) const;

        bool unproject(
            const double* const __restrict point_xy,
            double* const __restrict ray_xyz,
            double* const __restrict jacobian_unprojection = nullptr
        ) const;
    };
}

#include "assert.hpp"

namespace camera {
    pinhole::pinhole()
        : focal_lengths{ 1.0, 1.0 }
        , centre_points{ 0.5, 0.5 } {
    }

    pinhole::pinhole(const double* const __restrict parameters, const size_t parameters_length) {
        ASSERT(parameters_length == pinhole::parameter_count, "Parameters length must match internal number of parameters.");
        this->set_parameters(parameters, parameters_length);
    }

    void pinhole::set_parameters(const double* const __restrict parameters, const size_t parameters_length) {
        ASSERT(parameters_length == pinhole::parameter_count, "Parameters length must match internal number of parameters.");
        static_cast<void>(parameters_length);
        this->focal_lengths[0] = parameters[0];
        this->focal_lengths[1] = parameters[1];
        this->centre_points[0] = parameters[2];
        this->centre_points[1] = parameters[3];
        ASSERT(this->focal_lengths[0] > 0.0, "Horiztonal focal length must be greater than zero.");
        ASSERT(this->focal_lengths[1] > 0.0, "Vertical focal length must be greater than zero.");
    }

    void pinhole::get_parameters(double* const __restrict parameters, const size_t parameters_length) const {
        ASSERT(parameters_length == pinhole::parameter_count, "Parameters length must match internal number of parameters.");
        static_cast<void>(parameters_length);
        parameters[0] = this->focal_lengths[0];
        parameters[1] = this->focal_lengths[1];
        parameters[2] = this->centre_points[0];
        parameters[3] = this->centre_points[1];
    }

    bool pinhole::project(
        const double* const __restrict point_xyz,
        double* const __restrict point_xy,
        double* const __restrict jacobian_projection,
        double* const __restrict jacobian_parameters
    ) const {
        // Cannot project points with zero depth, also return false for those behind the camera.
        if (point_xyz[2] < 1.0e-12) {
            return false;
        }

        // Project.
        const double inverse_z = 1.0 / point_xyz[2];
        const double undistorted_xy[2] = {
            point_xyz[0] * inverse_z,
            point_xyz[1] * inverse_z
        };

        // Scale and offset.
        point_xy[0] = this->focal_lengths[0] * undistorted_xy[0] + this->centre_points[0];
        point_xy[1] = this->focal_lengths[1] * undistorted_xy[1] + this->centre_points[1];

        // Compute the projection jacobian.
        if (jacobian_projection != nullptr) {
            // Python:
            // import sympy
            // fx, fy, cx, cy, X, Y, Z = sympy.symbols('fx, fy, cx, cy, X, Y, Z')
            // sympy.cse(sympy.Matrix([fx*X/Z+cx, fy*Y/Z+cy]).jacobian(sympy.Matrix([X, Y, Z])))
            const double inverse_z_squared = inverse_z * inverse_z;
            jacobian_projection[0 * 3 + 0] = this->focal_lengths[0] * inverse_z;
            jacobian_projection[0 * 3 + 1] = 0.0;
            jacobian_projection[0 * 3 + 2] = -this->focal_lengths[0] * point_xyz[0] * inverse_z_squared;
            jacobian_projection[1 * 3 + 0] = 0.0;
            jacobian_projection[1 * 3 + 1] = this->focal_lengths[1] * inverse_z;
            jacobian_projection[1 * 3 + 2] = -this->focal_lengths[1] * point_xyz[1] * inverse_z_squared;
        }

        // Compute the parameters jacobian.
        if (jacobian_parameters != nullptr) {
            // Python:
            // import sympy
            // fx, fy, cx, cy, X, Y, Z = sympy.symbols('fx, fy, cx, cy, X, Y, Z')
            // sympy.cse(sympy.Matrix([fx*X/Z+cx, fy*Y/Z+cy]).jacobian(sympy.Matrix([fx, fy, cx, cy])))
            jacobian_parameters[0 * pinhole::parameter_count + 0] = undistorted_xy[0];
            jacobian_parameters[0 * pinhole::parameter_count + 1] = 0.0;
            jacobian_parameters[0 * pinhole::parameter_count + 2] = 1.0;
            jacobian_parameters[0 * pinhole::parameter_count + 3] = 0.0;
            jacobian_parameters[1 * pinhole::parameter_count + 0] = 0.0;
            jacobian_parameters[1 * pinhole::parameter_count + 1] = undistorted_xy[1];
            jacobian_parameters[1 * pinhole::parameter_count + 2] = 0.0;
            jacobian_parameters[1 * pinhole::parameter_count + 3] = 1.0;
        }

        return true;
    }

    bool pinhole::unproject(
        const double* const __restrict point_xy,
        double* const __restrict ray_xyz,
        double* const __restrict jacobian_unprojection
    ) const {
        // Scale and offset.
        const double undistorted_xy[2] = {
            (point_xy[0] - this->centre_points[0]) / this->focal_lengths[0],
            (point_xy[1] - this->centre_points[1]) / this->focal_lengths[1]
        };

        // Unproject.
        ray_xyz[0] = undistorted_xy[0];
        ray_xyz[1] = undistorted_xy[1];
        ray_xyz[2] = 1.0;

        // Compute the unprojection jacobian.
        if (jacobian_unprojection != nullptr) {
            // Python:
            // import sympy
            // fx, fy, cx, cy, X, Y, Z = sympy.symbols('fx, fy, cx, cy, X, Y, Z')
            // sympy.cse(sympy.Matrix([(X-cx)/fx, (Y-cy)/fy, 1]).jacobian(sympy.Matrix([X, Y])))
            jacobian_unprojection[0 * 2 + 0] = 1.0 / this->focal_lengths[0];
            jacobian_unprojection[0 * 2 + 1] = 0.0;
            jacobian_unprojection[1 * 2 + 0] = 0.0;
            jacobian_unprojection[1 * 2 + 1] = 1.0 / this->focal_lengths[1];
            jacobian_unprojection[2 * 2 + 0] = 0.0;
            jacobian_unprojection[2 * 2 + 1] = 0.0;
        }

        return true;
    }
}

#endif // CAMERA_HPP
