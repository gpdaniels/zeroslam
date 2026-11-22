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
#ifndef ZEROSLAM_SENSOR_CAMERA_PINHOLE_OPENCV_HPP
#define ZEROSLAM_SENSOR_CAMERA_PINHOLE_OPENCV_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor::camera {
    // The whole of OpenCV's pinhole distortion (calib3d's projectPoints with CALIB_RATIONAL_MODEL, CALIB_THIN_PRISM_MODEL
    // and CALIB_TILTED_MODEL): on the image plane point (x, y) = (X / Z, Y / Z) at radius r,
    //   x' = x * (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2) + s1 r^2 + s2 r^4
    //   y' = y * (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6) + p1 (r^2 + 2 y^2) + 2 p2 x y + s3 r^2 + s4 r^4
    // then the tilt of the sensor, (x_d, y_d, 1) proportional to T(tau_x, tau_y) (x', y', 1) with OpenCV's
    // computeTiltProjectionMatrix, and the pixel is (fx * x_d + cx, fy * y_d + cy). Parameters
    // [fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4 tau_x tau_y]: OpenCV's distortion vector in its own order, the tilt
    // angles in radians. With the last six zero it is pinhole_radial_tangential, to rounding. The valid region is the points
    // in front of the camera inside the radius where the radial distortion folds back, as for pinhole_radial_tangential,
    // that the tilt keeps in front of the sensor.
    template <typename type>
    class pinhole_opencv final {
    public:
        constexpr static const char* name = "pinhole_opencv";
        constexpr static const size_t parameter_count = 4 + 14;
        constexpr static const size_t maximum_undistortion_iterations = 20;

    private:
        type focal_lengths[2];
        type centre_points[2];
        type radial_k[6];
        type tangential_p[2];
        type prism_s[4];
        type tilt[2];
        // T(tau_x, tau_y), row major.
        type tilt_matrix[9];
        // Past this squared radius the radial distortion folds back, so points there are rejected, infinite when it never folds.
        type maximum_radius_squared;

    public:
        pinhole_opencv();

        pinhole_opencv(const type* const parameters, const size_t parameters_length);

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

    private:
        // The radial, tangential and thin prism distortion, before the tilt, with its jacobians with respect to the point and
        // to [k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4].
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

    extern template class pinhole_opencv<float>;
    extern template class pinhole_opencv<double>;
}

#endif // ZEROSLAM_SENSOR_CAMERA_PINHOLE_OPENCV_HPP
