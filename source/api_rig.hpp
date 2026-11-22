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
#ifndef ZEROSLAM_API_RIG_HPP
#define ZEROSLAM_API_RIG_HPP

#include "api.hpp"
#include "mapping/rig.hpp"

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

// What the api holds of a rig of several cameras: each sensor's extrinsic as zeroslam_set_sensor_extrinsics gives it, and
// the frame that waits for the rest of its images. The cameras of a rig are synchronised: a frame is one image of every
// camera with the same timestamp, which may arrive in one batch or over several calls, and an image of a newer timestamp
// before the frame is whole is refused, as is a camera given twice in one frame.
class api_rig final {
public:
    constexpr static const size_t cameras_maximum = 16;

private:
    std::vector<zeroslam_sensor_extrinsics_struct> extrinsics;
    long long pending_timestamp = 0;
    size_t pending_count = 0;
    std::vector<unsigned char> pending_present;
    std::vector<std::vector<unsigned char>> pending_pixels;

public:
    static bool cameras_supported(const size_t count);

    // Takes the extrinsics of sensors among the rig's, each once, with a rotation and a known refinement; nothing on failure.
    bool set_extrinsics(const zeroslam_sensor_extrinsics_struct* const values, const int length, const std::vector<int>& sensor_ids);

    void clear_extrinsics();

    // The rig of the cameras, in their order, each at its extrinsic (the body's origin, held, when it was given none).
    mapping::rig make_rig(const std::vector<int>& sensor_ids) const;

    // Whether a batch of images, by timestamp and camera index, follows on from the frame that waits and from the data before
    // it (any_data, latest_timestamp), keeping the cameras synchronised.
    bool accepts(const long long* const timestamps, const size_t* const cameras, const size_t count, const size_t camera_count, const bool any_data, const long long latest_timestamp) const;

    // Takes an image of a frame, which accepts allowed; true when the frame is whole and its images are ready.
    bool take(const long long timestamp, const size_t camera, const unsigned char* const pixels, const size_t length, const size_t camera_count);

    long long timestamp() const;

    // The pixels of a camera's image in the frame that is whole; taking the next image begins a new frame.
    unsigned char* pixels(const size_t camera);
};

#endif // ZEROSLAM_API_RIG_HPP
