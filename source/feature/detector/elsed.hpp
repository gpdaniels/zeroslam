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
#ifndef ZEROSLAM_FEATURE_DETECTOR_ELSED_HPP
#define ZEROSLAM_FEATURE_DETECTOR_ELSED_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace feature::detector {
    class elsed final {
    public:
        struct segment final {
            float x1;
            float y1;
            float x2;
            float y2;
            float length;
            float response;
            int support;
        };

        struct options final {
            int gradient_threshold = 30;
            int anchor_threshold = 8;
            int scan_interval = 2;
            int minimum_fit_pixels = 15;
            float establishment_rms_threshold = 0.5f;
            float point_to_line_threshold = 1.5f;
            int maximum_jump = 6;
            int maximum_outliers = 3;
            float minimum_length = 15.0f;
        };

    public:
        static size_t detect(
            const unsigned char* __restrict const data,
            const int width,
            const int height,
            const int stride,
            const options& detection_options,
            segment* __restrict const segment_buffer,
            const size_t segment_buffer_size
        );
    };
}

#endif // ZEROSLAM_FEATURE_DETECTOR_ELSED_HPP
