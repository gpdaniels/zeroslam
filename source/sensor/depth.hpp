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
#ifndef ZEROSLAM_SENSOR_DEPTH_HPP
#define ZEROSLAM_SENSOR_DEPTH_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor {
    // A depth image registered to a camera: the camera's own pixel grid, each pixel holding the depth in metres of the surface
    // that pixel sees, as the aligned depth of a structured light or stereo camera (TUM RGB-D, the OAK-D) comes. A pixel
    // without a measurement holds zero.
    //
    // What a depth is depends on the camera model only through its rays: an axial depth is the distance along the optical
    // axis, the z of the point in the camera frame, as structured light and rectified stereo cameras give it, and a radial
    // depth is the distance along the pixel's ray from the camera centre, as a time of flight camera or a renderer through a
    // wide angle model gives it. Either is turned into a point by scaling the ray the camera model unprojects the pixel to,
    // so a depth image composes with any camera model that unprojects, and an axial depth needs a ray in front of the camera.
    class depth final {
    public:
        enum class measure_kind {
            axial,
            radial
        };

        // The standard deviation of a depth measurement d, in metres, sigma(d) = constant + linear d + quadratic d^2. A
        // structured light or stereo camera measures a disparity with a roughly constant error, which is an error in depth
        // that grows with its square: Khoshelham and Elberink (Sensors 2012) give sigma = 1.425e-3 d^2 for the Kinect.
        struct noise_model final {
            double constant = 0.0;
            double linear = 0.0;
            double quadratic = 1.425e-3;
        };

        struct settings final {
            measure_kind measure = measure_kind::axial;
            // The range the sensor measures reliably; a depth outside it is taken as missing.
            double minimum = 0.0;
            double maximum = 1.0e30;
            noise_model noise;
        };

        // A depth differing from the depth at a pixel by more than this fraction of it, among the eight around the pixel,
        // puts the pixel on an occluding edge, where a feature can sit on either surface and the depth image's registration
        // to the camera is least certain: its depth is not taken. A slanted surface changes far less from pixel to pixel
        // (1% at a metre and 80 degrees from the view on TUM's 525 px focal length), an occluding edge far more.
        constexpr static const double discontinuity_fraction = 0.05;

        // The smallest standard deviation a measurement is given, so that a noise model of zeros cannot give infinite weight.
        constexpr static const double sigma_minimum = 1.0e-4;

    private:
        size_t rows;
        size_t columns;
        std::vector<float> metres;
        settings configuration;

    public:
        depth();

        // A depth image of the given size from metres in row major order, a zero, negative or non-finite value being missing.
        depth(const size_t image_rows, const size_t image_columns, const float* const image_metres, const settings& depth_settings);

        // The same from 16 bit unsigned values of scale metres each, zero being missing.
        depth(const size_t image_rows, const size_t image_columns, const unsigned short* const image_values, const double scale, const settings& depth_settings);

    public:
        bool empty() const;

        size_t get_rows() const;

        size_t get_cols() const;

        const settings& get_settings() const;

        // The stored metres of a pixel, zero when it has no measurement within the range.
        float at(const size_t row, const size_t column) const;

        // Frees the image, leaving an empty depth with the same settings.
        void release();

        // The depth at an image position in the pixel-centre frame, from the pixel it falls in, false when the position is
        // outside the image, the pixel has no measurement within the range or it lies on an occluding edge.
        bool sample(const double x, const double y, double& value) const;

        // The standard deviation of a measurement of the given depth.
        double sigma(const double value) const;

        // The standard deviation of the inverse of a measurement of the given depth, sigma(d) / d^2.
        double inverse_sigma(const double value) const;

        // The depth a camera frame point measures: its z for an axial depth, its distance from the centre for a radial one.
        static double measured(const measure_kind measure, const double (&point)[3]);

        // The camera frame point at the given depth along the ray the camera unprojects the image position to.
        template <typename camera_type>
        static bool back_project(const camera_type& camera, const measure_kind measure, const double (&pixel)[2], const double value, double (&point)[3]) {
            double ray[3] = { 0.0, 0.0, 0.0 };
            if (!(value > 0.0) || !camera.unproject(&pixel[0], &ray[0])) {
                return false;
            }
            const double along = depth::measured(measure, ray);
            if ((measure == measure_kind::axial) ? !(along > 1.0e-12) : !(along > 1.0e-300)) {
                return false;
            }
            for (size_t axis = 0; axis < 3; ++axis) {
                point[axis] = ray[axis] * (value / along);
            }
            return true;
        }
    };
}

#endif // ZEROSLAM_SENSOR_DEPTH_HPP
