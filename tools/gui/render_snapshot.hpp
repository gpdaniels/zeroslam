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
#ifndef ZEROSLAM_TOOLS_GUI_RENDER_SNAPSHOT_HPP
#define ZEROSLAM_TOOLS_GUI_RENDER_SNAPSHOT_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace gui {
    struct rotation3 {
        double rows[3][3];

        double* operator[](const int row) {
            return rows[row];
        }

        const double* operator[](const int row) const {
            return rows[row];
        }
    };

    struct render_snapshot {
        constexpr static const std::size_t camera_parameter_count = 12;

        struct frame {
            rotation3 rotation = { { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } } };
            double translation[3] = { 0.0, 0.0, 0.0 };
            double camera_parameters[camera_parameter_count] = {};
            int image_width = 0;
            int image_height = 0;
            long long timestamp_nanoseconds = -1;
        };

        struct landmark {
            double location[3];
            double colour[3];
        };

        struct line {
            double a[3];
            double b[3];
        };

        struct edge {
            int frame_a;
            int frame_b;
            int kind;
            int weight;
        };

        std::unordered_map<int, frame> frames;
        std::unordered_map<int, landmark> landmarks;
        std::unordered_map<int, line> lines;
        std::vector<edge> edges;
        std::unordered_set<int> keyframes;

        int processed_frame_count = 0;
        double last_process_seconds = 0.0;
        long long last_frame_timestamp_nanoseconds = -1;

        static void camera_centre(const frame& pose, double centre[3]) {
            for (int axis = 0; axis < 3; ++axis) {
                centre[axis] = -((pose.rotation[0][axis] * pose.translation[0]) + (pose.rotation[1][axis] * pose.translation[1]) + (pose.rotation[2][axis] * pose.translation[2]));
            }
        }
    };
}

#endif // ZEROSLAM_TOOLS_GUI_RENDER_SNAPSHOT_HPP
