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
#ifndef ZEROSLAM_TOOLS_ANDROID_CAMERA_MODEL_HPP
#define ZEROSLAM_TOOLS_ANDROID_CAMERA_MODEL_HPP

#include "zeroslam/zeroslam.h"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdlib>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mobile {
    // What a camera2 device reports of its geometry, in the pixel coordinates of its calibration: the pre-correction active
    // array with (0, 0) the top left corner of its top left pixel, so the centre of pixel i is at i + 0.5 as in the library.
    struct camera_characteristics final {
        // The part of the array a stream is cut from (left, top, width, height): the whole pre-correction active array when
        // distortion correction is off, which is how the app runs the camera.
        double region[4] = { 0.0, 0.0, 0.0, 0.0 };
        // The intrinsic calibration [f_x, f_y, c_x, c_y, s], when the device reports one.
        bool calibrated = false;
        double intrinsics[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        // The Brown-Conrady distortion [kappa_1, kappa_2, kappa_3] radial and [kappa_4, kappa_5] tangential, when reported.
        bool distorted = false;
        double distortion[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        // Without a calibration the focal length is estimated from the lens and the sensor: the focal length in millimetres,
        // the physical size of the full pixel array in millimetres, and its size in pixels.
        double focal_length_millimetres = 0.0;
        double physical_size_millimetres[2] = { 0.0, 0.0 };
        int pixel_array_size[2] = { 0, 0 };
    };

    struct stream_size final {
        int width = 0;
        int height = 0;
    };

    // The stream closest in width to the one asked for among those with the aspect ratio of the region, so that the stream
    // sees the whole region, or among all of them when none has it.
    inline bool choose_stream_size(const std::vector<stream_size>& sizes, const camera_characteristics& characteristics, const int preferred_width, stream_size& chosen) {
        const double region_aspect = (characteristics.region[3] > 0.0) ? (characteristics.region[2] / characteristics.region[3]) : 0.0;
        for (int pass = 0; pass < 2; ++pass) {
            bool found = false;
            for (const stream_size& size : sizes) {
                if ((size.width <= 0) || (size.height <= 0)) {
                    continue;
                }
                const double aspect = static_cast<double>(size.width) / static_cast<double>(size.height);
                if ((pass == 0) && (std::abs(aspect - region_aspect) > 0.01 * region_aspect)) {
                    continue;
                }
                const int distance = std::abs(size.width - preferred_width);
                const int chosen_distance = std::abs(chosen.width - preferred_width);
                if (!found || (distance < chosen_distance) || ((distance == chosen_distance) && (size.width < chosen.width))) {
                    chosen = size;
                    found = true;
                }
            }
            if (found) {
                return true;
            }
        }
        return false;
    }

    // The library's camera for a stream of the given size. The device crops the region to the stream's aspect ratio about
    // its centre and scales the crop to the stream, so the focal lengths scale with it, the principal point moves with the
    // crop, and the distortion, which acts on normalised coordinates, is unchanged.
    inline bool stream_camera(const camera_characteristics& characteristics, const int width, const int height, zeroslam_sensor_parameters_camera_struct& camera) {
        const double region_width = characteristics.region[2];
        const double region_height = characteristics.region[3];
        if ((width <= 0) || (height <= 0) || (region_width <= 0.0) || (region_height <= 0.0)) {
            return false;
        }
        double focal_x = characteristics.intrinsics[0];
        double focal_y = characteristics.intrinsics[1];
        double centre_x = characteristics.intrinsics[2];
        double centre_y = characteristics.intrinsics[3];
        if (!characteristics.calibrated || !(focal_x > 0.0) || !(focal_y > 0.0)) {
            if (!(characteristics.focal_length_millimetres > 0.0) || !(characteristics.physical_size_millimetres[0] > 0.0) || !(characteristics.physical_size_millimetres[1] > 0.0) || (characteristics.pixel_array_size[0] <= 0) || (characteristics.pixel_array_size[1] <= 0)) {
                return false;
            }
            focal_x = characteristics.focal_length_millimetres * static_cast<double>(characteristics.pixel_array_size[0]) / characteristics.physical_size_millimetres[0];
            focal_y = characteristics.focal_length_millimetres * static_cast<double>(characteristics.pixel_array_size[1]) / characteristics.physical_size_millimetres[1];
            centre_x = characteristics.region[0] + (0.5 * region_width);
            centre_y = characteristics.region[1] + (0.5 * region_height);
        }
        const double stream_aspect = static_cast<double>(width) / static_cast<double>(height);
        double crop_width = region_width;
        double crop_height = region_height;
        if (stream_aspect > (region_width / region_height)) {
            crop_height = region_width / stream_aspect;
        }
        else {
            crop_width = region_height * stream_aspect;
        }
        const double crop_left = characteristics.region[0] + (0.5 * (region_width - crop_width));
        const double crop_top = characteristics.region[1] + (0.5 * (region_height - crop_height));
        const double scale = static_cast<double>(width) / crop_width;
        camera = zeroslam_sensor_parameters_camera_struct{};
        camera.width = width;
        camera.height = height;
        camera.focal_x = focal_x * scale;
        camera.focal_y = focal_y * scale;
        camera.centre_x = (centre_x - crop_left) * scale;
        camera.centre_y = (centre_y - crop_top) * scale;
        if (characteristics.calibrated && characteristics.distorted) {
            // The library's order is [k1 k2 p1 p2 k3 k4 k5 k6], and the device's tangential terms are p1 and p2 in that order.
            camera.distortion[0] = characteristics.distortion[0];
            camera.distortion[1] = characteristics.distortion[1];
            camera.distortion[2] = characteristics.distortion[3];
            camera.distortion[3] = characteristics.distortion[4];
            camera.distortion[4] = characteristics.distortion[2];
        }
        return true;
    }
}

#endif // ZEROSLAM_TOOLS_ANDROID_CAMERA_MODEL_HPP
