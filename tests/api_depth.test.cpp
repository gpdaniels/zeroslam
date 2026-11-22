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

#include "api_depth.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    zeroslam_sensor_parameters_depth_struct parameters{};
    parameters.camera_id = 1;
    parameters.encoding = zeroslam_depth_encoding_unsigned_16;
    parameters.measure = zeroslam_depth_measure_axial;
    parameters.registration = zeroslam_depth_registration_camera;
    parameters.scale = 0.001;
    parameters.minimum_depth = 0.1;
    parameters.maximum_depth = 8.0;
    parameters.noise_quadratic = 1.425e-3;
    parameters.pairing_tolerance = 10;

    {
        REQUIRE(api_depth::valid(parameters));
        zeroslam_sensor_parameters_depth_struct changed = parameters;
        changed.scale = -1.0;
        REQUIRE(!api_depth::valid(changed));
        changed = parameters;
        changed.encoding = zeroslam_depth_encoding_float_32;
        changed.scale = 0.0;
        REQUIRE(api_depth::valid(changed));
        changed = parameters;
        changed.minimum_depth = -0.5;
        REQUIRE(!api_depth::valid(changed));
        changed = parameters;
        changed.noise_quadratic = 0.0;
        REQUIRE(!api_depth::valid(changed));
        changed.noise_constant = 0.01;
        REQUIRE(api_depth::valid(changed));
        changed = parameters;
        changed.registration = 3;
        REQUIRE(!api_depth::valid(changed));

        zeroslam_sensor_rig_struct entry{};
        entry.type = zeroslam_sensor_depth;
        entry.sensor_id = 2;
        entry.parameters_length = static_cast<int>(sizeof(parameters));
        entry.parameters_data = &parameters;
        api_depth::sensor parsed{};
        REQUIRE(api_depth::parse(entry, parsed));
        REQUIRE((parsed.sensor_id == 2) && (parsed.parameters.camera_id == 1));
        entry.parameters_length = 4;
        REQUIRE(!api_depth::parse(entry, parsed));
        entry.parameters_length = static_cast<int>(sizeof(parameters));
        entry.type = zeroslam_sensor_camera;
        REQUIRE(!api_depth::parse(entry, parsed));
    }

    {
        api_depth depths;
        REQUIRE(depths.empty());
        depths.set_sensors({ api_depth::sensor{ 2, parameters } });
        REQUIRE(!depths.empty());
        REQUIRE(depths.find(2) != nullptr);
        REQUIRE(depths.find(1) == nullptr);
        REQUIRE(api_depth::frame_length(*depths.find(2), 4, 3) == 24);

        // Timestamps increase strictly among the depth frames.
        bool seen = false;
        long long int next = 0;
        depths.validation_start(seen, next);
        REQUIRE(!seen);
        REQUIRE(depths.follows(100, seen, next));
        REQUIRE(!depths.follows(100, seen, next));
        REQUIRE(depths.follows(101, seen, next));

        // Frames 2 ms apart; the camera frame at 105 takes the nearest within 10, the one at 104, the earlier one going with
        // it; one at 200 finds none near and lets the older frames go.
        const std::vector<unsigned short> millimetres(12, static_cast<unsigned short>(1500));
        for (const long long int timestamp : { 100ll, 102ll, 104ll, 108ll }) {
            depths.hold(*depths.find(2), timestamp, 4, 3, millimetres.data());
        }
        depths.validation_start(seen, next);
        REQUIRE(seen && (next == 108));
        const sensor::depth paired = depths.take(1, 105);
        REQUIRE(!paired.empty());
        double value = 0.0;
        REQUIRE(paired.sample(1.5, 1.5, value) && (value > 1.4999) && (value < 1.5001));
        // The frame at 108 is left for the next camera frame.
        REQUIRE(!depths.take(1, 110).empty());
        REQUIRE(depths.take(1, 111).empty());
        depths.hold(*depths.find(2), 150, 4, 3, millimetres.data());
        REQUIRE(depths.take(1, 200).empty());
        REQUIRE(depths.take(1, 151).empty());
        // A camera without a depth camera takes nothing.
        depths.hold(*depths.find(2), 300, 4, 3, millimetres.data());
        REQUIRE(depths.take(5, 300).empty());
        REQUIRE(!depths.take(1, 300).empty());

        // No more than the maximum unpaired frames are held, the oldest let go first.
        for (long long int timestamp = 400; timestamp < 400 + static_cast<long long int>(api_depth::pending_maximum) + 4; ++timestamp) {
            depths.hold(*depths.find(2), timestamp, 4, 3, millimetres.data());
        }
        REQUIRE(depths.take(1, 393).empty());
        REQUIRE(!depths.take(1, 400 + static_cast<long long int>(api_depth::pending_maximum) + 3).empty());
    }

    return EXIT_SUCCESS;
}
