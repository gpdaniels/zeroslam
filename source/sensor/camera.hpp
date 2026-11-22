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
#ifndef ZEROSLAM_SENSOR_CAMERA_HPP
#define ZEROSLAM_SENSOR_CAMERA_HPP

#include "math/math.hpp"
#include "sensor/camera/double_sphere.hpp"
#include "sensor/camera/extended_unified.hpp"
#include "sensor/camera/field_of_view.hpp"
#include "sensor/camera/fisheye_624.hpp"
#include "sensor/camera/kannala_brandt.hpp"
#include "sensor/camera/model.hpp"
#include "sensor/camera/pinhole.hpp"
#include "sensor/camera/pinhole_opencv.hpp"
#include "sensor/camera/pinhole_radial_tangential.hpp"
#include "sensor/camera/unified.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor {
    using pinhole = camera::pinhole<double>;

    // The camera the system holds: any of the models below behind the erased interface. Every model's parameters begin with
    // [fx fy cx cy], the pixel-centre frame's focal lengths and principal point, and continue with its own (see each model).
    using model = camera::model<double>;

    namespace camera {
        // The model called name (each model's own name) with the given parameters, false, and result left as it was, when
        // the name is not a model's or the parameters are not valid for it.
        inline bool make_model(const char* const name, const double* const parameters, const size_t parameters_length, sensor::model& result) {
            const auto make = [&](auto concrete) {
                if (!concrete.set_parameters(parameters, parameters_length)) {
                    return false;
                }
                result = sensor::model(concrete);
                return true;
            };
            const auto named = [name](const char* const candidate) {
                size_t index = 0;
                while ((name[index] != 0) && (name[index] == candidate[index])) {
                    ++index;
                }
                return name[index] == candidate[index];
            };
            if ((name == nullptr) || (parameters == nullptr)) {
                return false;
            }
            if (named(camera::pinhole<double>::name)) {
                return make(camera::pinhole<double>());
            }
            if (named(camera::pinhole_radial_tangential<double>::name)) {
                return make(camera::pinhole_radial_tangential<double>());
            }
            if (named(camera::pinhole_opencv<double>::name)) {
                return make(camera::pinhole_opencv<double>());
            }
            if (named(camera::kannala_brandt<double>::name)) {
                return make(camera::kannala_brandt<double>());
            }
            if (named(camera::fisheye_624<double>::name)) {
                return make(camera::fisheye_624<double>());
            }
            if (named(camera::field_of_view<double>::name)) {
                return make(camera::field_of_view<double>());
            }
            if (named(camera::unified<double>::name)) {
                return make(camera::unified<double>());
            }
            if (named(camera::extended_unified<double>::name)) {
                return make(camera::extended_unified<double>());
            }
            if (named(camera::double_sphere<double>::name)) {
                return make(camera::double_sphere<double>());
            }
            return false;
        }

        // The cosine of the largest angle from the optical axis at which the system takes a ray onto the image plane
        // (x / z, y / z), as the two-view and perspective solvers, the line measurements, the inverse depth anchors and the
        // epipolar matching do: 80 degrees. The image plane stretches a ray's angular noise by 1 / cos^2 of its angle, 33
        // times at 80 degrees, and has no point for a ray at or past 90. A pinhole camera's rays (z = 1) pass out to a radius
        // of 5.7 on the image plane, far past any real lens; a wide angle camera's rays further out are tracked,
        // triangulated and adjusted through its model, but left out of what works on the image plane.
        constexpr static const double image_plane_cosine_minimum = 0.17364817766693033;

        // Whether a ray, of any length, is within the image plane's angle of the optical axis.
        inline bool on_image_plane(const double* const ray) {
            return ray[2] > (camera::image_plane_cosine_minimum * math::sqrt((ray[0] * ray[0]) + (ray[1] * ray[1]) + (ray[2] * ray[2])));
        }

        // The pinhole [fx fy cx cy] of a camera (see model::get_pinhole), the unit pinhole at the origin for an empty model.
        inline void pinhole_of(const sensor::model& camera_model, double (&intrinsics)[4]) {
            if (!camera_model.get_pinhole(&intrinsics[0], 4)) {
                intrinsics[0] = 1.0;
                intrinsics[1] = 1.0;
                intrinsics[2] = 0.0;
                intrinsics[3] = 0.0;
            }
        }
    }
}

#endif // ZEROSLAM_SENSOR_CAMERA_HPP
