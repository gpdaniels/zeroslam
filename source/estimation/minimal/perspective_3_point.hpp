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
#ifndef ZEROSLAM_ESTIMATION_MINIMAL_PERSPECTIVE_3_POINT_HPP
#define ZEROSLAM_ESTIMATION_MINIMAL_PERSPECTIVE_3_POINT_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::minimal {
    // Camera poses from three unit bearings and their world points, world to camera as row major rotations and translations; returns how many of the four candidates were written.
    template <typename type>
    class perspective_3_point final {
    public:
        static int solve(
            const type image_points[3][3],
            const type world_points[3][3],
            type rotations[4][9],
            type translations[4][3]
        );
    };

    extern template class perspective_3_point<float>;
    extern template class perspective_3_point<double>;
}

#endif // ZEROSLAM_ESTIMATION_MINIMAL_PERSPECTIVE_3_POINT_HPP
