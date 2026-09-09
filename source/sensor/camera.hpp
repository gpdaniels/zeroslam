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

#include "sensor/camera/pinhole.hpp"
#include "sensor/camera/pinhole_radial_tangential.hpp"

namespace sensor {
    using pinhole = camera::pinhole<double>;
    using model = camera::pinhole_radial_tangential<double>;
}

#endif // ZEROSLAM_SENSOR_CAMERA_HPP
