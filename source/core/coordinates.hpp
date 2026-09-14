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
#ifndef ZEROSLAM_CORE_COORDINATES_HPP
#define ZEROSLAM_CORE_COORDINATES_HPP

namespace core {
    using pixel_index = int;
    using pixel_centre = float;

    constexpr static const float pixel_centre_offset = 0.5f;

    constexpr float to_pixel_centre(const float index_position) {
        return index_position + core::pixel_centre_offset;
    }

    constexpr float to_pixel_index_position(const float centre) {
        return centre - core::pixel_centre_offset;
    }

    constexpr pixel_index to_pixel_index(const float centre) {
        const int truncated = static_cast<int>(centre);
        return (static_cast<float>(truncated) > centre) ? (truncated - 1) : truncated;
    }
}

#endif // ZEROSLAM_CORE_COORDINATES_HPP
