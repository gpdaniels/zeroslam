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
#ifndef ZEROSLAM_SENSOR_CAMERA_PARAMETER_ROLE_HPP
#define ZEROSLAM_SENSOR_CAMERA_PARAMETER_ROLE_HPP

namespace sensor::camera {
    // What each of a camera model's parameters is, which a model may report so that a calibration can choose the parameters
    // to refine without knowing the model's layout: the focal lengths, the principal point, the distortion that a sequence of
    // images pins down (the leading radial and the tangential terms), and the higher order distortion it barely does.
    enum class parameter_role : unsigned char {
        unknown,
        focal_length,
        principal_point,
        distortion,
        distortion_higher
    };
}

#endif // ZEROSLAM_SENSOR_CAMERA_PARAMETER_ROLE_HPP
