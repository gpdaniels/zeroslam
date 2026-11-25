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

#include "api_rig.hpp"

#include "mapping/rig.hpp"
#include "math/lie.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
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

    REQUIRE(!api_rig::cameras_supported(0));
    REQUIRE(api_rig::cameras_supported(1));
    REQUIRE(api_rig::cameras_supported(api_rig::cameras_maximum));
    REQUIRE(!api_rig::cameras_supported(api_rig::cameras_maximum + 1));

    {
        // Extrinsics of the rig's sensors only, each once, with a rotation and a known refinement, and the rig they make: the
        // first camera the reference, each camera at its extrinsic, a camera given none at the body's origin.
        api_rig rig;
        const std::vector<int> ids = { 4, 7, 9 };
        zeroslam_sensor_extrinsics_struct extrinsics[2] = {};
        extrinsics[0].sensor_id = 4;
        extrinsics[0].pose[2] = -0.05;
        extrinsics[0].pose[6] = 2.0;
        extrinsics[1].sensor_id = 7;
        extrinsics[1].pose[0] = 0.11;
        extrinsics[1].pose[6] = 1.0;
        extrinsics[1].refinement = zeroslam_extrinsic_refine;
        extrinsics[1].sensor_id = 5;
        REQUIRE(!rig.set_extrinsics(&extrinsics[0], 2, ids));
        extrinsics[1].sensor_id = 4;
        REQUIRE(!rig.set_extrinsics(&extrinsics[0], 2, ids));
        extrinsics[1].sensor_id = 7;
        extrinsics[1].refinement = 3;
        REQUIRE(!rig.set_extrinsics(&extrinsics[0], 2, ids));
        extrinsics[1].refinement = zeroslam_extrinsic_refine;
        extrinsics[1].pose[6] = 0.0;
        REQUIRE(!rig.set_extrinsics(&extrinsics[0], 2, ids));
        extrinsics[1].pose[6] = 1.0;
        REQUIRE(rig.set_extrinsics(&extrinsics[0], 2, ids));
        const mapping::rig made = rig.make_rig(ids);
        REQUIRE(made.camera_count() == 3);
        REQUIRE(made.reference() == 0);
        REQUIRE(made.at(0).body_from_sensor.translation()[2] == -0.05);
        REQUIRE(std::abs(made.at(0).body_from_sensor.rotation().get_quaternion()[0] - 1.0) < 1e-12);
        REQUIRE(made.refines(1));
        REQUIRE(!made.refines(2));
        REQUIRE(std::abs(made.reference_from_sensor(1).translation()[0] - 0.11) < 1e-12);
        REQUIRE(std::abs(made.reference_from_sensor(1).translation()[2] - 0.05) < 1e-12);
        REQUIRE(made.metric());
        // A later call sets the sensors it names and leaves the others as they were; a refused one changes nothing.
        extrinsics[1].sensor_id = 9;
        extrinsics[1].pose[0] = -0.11;
        REQUIRE(rig.set_extrinsics(&extrinsics[1], 1, ids));
        extrinsics[1].refinement = 3;
        REQUIRE(!rig.set_extrinsics(&extrinsics[1], 1, ids));
        const mapping::rig remade = rig.make_rig(ids);
        REQUIRE(remade.at(0).body_from_sensor.translation()[2] == -0.05);
        REQUIRE(remade.refines(1) && remade.refines(2));
        REQUIRE(std::abs(remade.reference_from_sensor(1).translation()[0] - 0.11) < 1e-12);
        REQUIRE(std::abs(remade.reference_from_sensor(2).translation()[0] + 0.11) < 1e-12);
        rig.clear_extrinsics();
        REQUIRE(!rig.make_rig(ids).metric());
    }

    {
        // Frames of two cameras: each camera once per frame, a frame over several batches, and no newer image before a frame is
        // whole or at an earlier timestamp than the frame before.
        api_rig rig;
        const long long both_at_once[2] = { 100, 100 };
        const size_t cameras_both[2] = { 0, 1 };
        const size_t cameras_twice[2] = { 1, 1 };
        REQUIRE(rig.accepts(&both_at_once[0], &cameras_both[0], 2, 2, false, 0));
        REQUIRE(!rig.accepts(&both_at_once[0], &cameras_twice[0], 2, 2, false, 0));
        REQUIRE(!rig.accepts(&both_at_once[0], &cameras_both[0], 2, 3, true, 100));
        const long long split[2] = { 100, 200 };
        REQUIRE(!rig.accepts(&split[0], &cameras_both[0], 2, 2, false, 0));
        const size_t out_of_rig[1] = { 2 };
        REQUIRE(!rig.accepts(&split[0], &out_of_rig[0], 1, 2, false, 0));

        unsigned char left[4] = { 1, 2, 3, 4 };
        unsigned char right[4] = { 5, 6, 7, 8 };
        REQUIRE(!rig.take(100, 1, &right[0], 4, 2));
        // The frame waits for its first camera: the second again, or a newer frame, is refused.
        const long long again[1] = { 100 };
        const size_t second[1] = { 1 };
        const size_t first[1] = { 0 };
        REQUIRE(!rig.accepts(&again[0], &second[0], 1, 2, true, 0));
        const long long newer[1] = { 200 };
        REQUIRE(!rig.accepts(&newer[0], &first[0], 1, 2, true, 0));
        REQUIRE(rig.accepts(&again[0], &first[0], 1, 2, true, 0));
        REQUIRE(rig.take(100, 0, &left[0], 4, 2));
        REQUIRE(rig.timestamp() == 100);
        REQUIRE(rig.pixels(0)[3] == 4);
        REQUIRE(rig.pixels(1)[0] == 5);
        // The frame is whole: the next one comes after it.
        REQUIRE(!rig.accepts(&again[0], &first[0], 1, 2, true, 100));
        REQUIRE(rig.accepts(&newer[0], &second[0], 1, 2, true, 100));
    }

    return EXIT_SUCCESS;
}
