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

#include "mapping/calibration_drift.hpp"

#include "mapping/calibration.hpp"
#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "sensor/camera.hpp"
#include "sensor/camera/pinhole_radial_tangential.hpp"

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

static sensor::model make_camera(const double focal_x, const double focal_y) {
    const double parameters[sensor::camera::pinhole_radial_tangential<double>::parameter_count] = { focal_x, focal_y, 320.0, 240.0, -0.1, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    return sensor::model(sensor::camera::pinhole_radial_tangential<double>(&parameters[0], sensor::camera::pinhole_radial_tangential<double>::parameter_count));
}

static double parameter_of(const sensor::model& camera, const size_t index) {
    double parameters[sensor::camera::pinhole_radial_tangential<double>::parameter_count] = {};
    REQUIRE(camera.get_parameters(&parameters[0], sensor::camera::pinhole_radial_tangential<double>::parameter_count));
    return parameters[index];
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    constexpr static const long long second = 1000000000ll;
    const sensor::model first_given = make_camera(500.0, 510.0);
    const sensor::model second_given = make_camera(480.0, 480.0);
    const math::se3<double> declared(math::so3<double>::exp({ { 0.0, 0.01, 0.0 } }), { { 0.11, 0.0, 0.0 } });

    {
        // The paper's values: chunks of five seconds, and over five seconds a focal length walks 0.035 pixels in 240, an
        // extrinsic 0.001 radians and half a millimetre; the priors are 0.5 and 0.05 pixels in 240 and half a millimetre.
        const mapping::calibration_drift::settings defaults;
        REQUIRE(!defaults.enabled);
        REQUIRE(defaults.chunk_seconds == 5.0);
        REQUIRE(std::abs((defaults.focal_walk * 240.0) - 0.035) < 1.0e-15);
        REQUIRE(defaults.rotation_walk == 0.001);
        REQUIRE(defaults.translation_walk == 0.0005);
        REQUIRE(mapping::calibration_drift::walk_seconds == 5.0);
        REQUIRE(std::abs((mapping::calibration_drift::focal_prior_fraction * 240.0) - 0.5) < 1.0e-15);
        REQUIRE(std::abs((mapping::calibration_drift::aspect_prior_fraction * 240.0) - 0.05) < 1.0e-15);
        REQUIRE(mapping::calibration_drift::baseline_sigma == 0.0005);
    }

    {
        // The focal lengths of a camera are the parameters it says are, and none for a camera that says nothing.
        const std::vector<size_t> focal = mapping::calibration_drift::focal_parameters(first_given);
        REQUIRE((focal.size() == 2) && (focal[0] == 0) && (focal[1] == 1));
        REQUIRE(mapping::calibration_drift::focal_parameters(sensor::model()).empty());
    }

    {
        // Off, no frame is followed, and nothing is asked of an adjustment.
        mapping::calibration_drift drift;
        std::vector<sensor::model> cameras = { first_given };
        std::vector<math::se3<double>> extrinsics = { math::se3<double>::identity() };
        REQUIRE(drift.take_frame(0, cameras, extrinsics) == -1);
        REQUIRE(drift.chunks().empty());
        REQUIRE(drift.newest() == nullptr);
        REQUIRE(!drift.ready(100, 90.0));
        mapping::calibration_drift::request following;
        REQUIRE(!drift.make_request(mapping::calibration::level::off, false, {}, following));
    }

    mapping::calibration_drift drift;
    mapping::calibration_drift::settings settings;
    settings.enabled = true;
    drift.set(settings);
    REQUIRE(drift.get().enabled);
    REQUIRE(drift.chunk_nanoseconds() == 0);
    REQUIRE(!drift.ready(100, 90.0));

    {
        // The chunks count from the first frame followed, at whatever time it comes: a frame is in the chunk its time falls
        // in, and the first chunk is the calibration its frame came with.
        std::vector<sensor::model> cameras = { first_given, second_given };
        std::vector<math::se3<double>> extrinsics = { math::se3<double>::identity(), declared };
        REQUIRE(drift.take_frame(100 * second, cameras, extrinsics) == 0);
        REQUIRE(drift.chunk_nanoseconds() == 5 * second);
        REQUIRE(drift.chunk_start(0) == 100 * second);
        REQUIRE(drift.chunk_start(3) == 115 * second);
        REQUIRE(drift.take_frame((105 * second) - 1, cameras, extrinsics) == 0);
        REQUIRE(drift.chunks().size() == 1);
        REQUIRE(drift.take_frame(105 * second, cameras, extrinsics) == 1);
        // A chunk no frame is in is skipped.
        REQUIRE(drift.take_frame(121 * second, cameras, extrinsics) == 4);
        REQUIRE(drift.chunks().size() == 3);
        REQUIRE((drift.chunks()[0].index == 0) && (drift.chunks()[1].index == 1) && (drift.chunks()[2].index == 4));
        REQUIRE(drift.newest() == &drift.chunks()[2]);
        REQUIRE(drift.newest()->cameras.size() == 2);
        REQUIRE(drift.newest()->cameras[1].camera.is_equal(second_given));
        REQUIRE(drift.newest()->cameras[1].reference_from_camera.translation()[0] == 0.11);
        REQUIRE(drift.newest()->cameras[0].observations == 0);
        // The length of a chunk is the one the first frame was taken with.
        settings.chunk_seconds = 1.0;
        drift.set(settings);
        REQUIRE(drift.take_frame(124 * second, cameras, extrinsics) == 4);
        settings.chunk_seconds = 5.0;
        drift.set(settings);

        // Not followed: a frame before the first, one in a chunk before the newest that no frame was in, one with another
        // camera or another number of them, and any while the following is off. Its cameras are left as they came.
        REQUIRE(drift.take_frame(99 * second, cameras, extrinsics) == -1);
        REQUIRE(drift.take_frame(112 * second, cameras, extrinsics) == -1);
        REQUIRE(drift.take_frame(106 * second, cameras, extrinsics) == 1);
        std::vector<sensor::model> other = { first_given, make_camera(481.0, 480.0) };
        REQUIRE(drift.take_frame(125 * second, other, extrinsics) == -1);
        REQUIRE(parameter_of(other[1], 0) == 481.0);
        std::vector<sensor::model> one = { first_given };
        std::vector<math::se3<double>> one_extrinsic = { math::se3<double>::identity() };
        REQUIRE(drift.take_frame(125 * second, one, one_extrinsic) == -1);
        REQUIRE(drift.take_frame(125 * second, cameras, one_extrinsic) == -1);
        settings.enabled = false;
        drift.set(settings);
        REQUIRE(drift.take_frame(125 * second, cameras, extrinsics) == -1);
        settings.enabled = true;
        drift.set(settings);
        REQUIRE(drift.chunks().size() == 3);
    }

    {
        // An adjustment may follow the calibration once the keyframes have seen the scene turn, as a constant one waits.
        REQUIRE(!drift.ready(mapping::calibration::keyframes_minimum - 1, 90.0));
        REQUIRE(!drift.ready(100, 0.5 * mapping::calibration::view_spread_degrees_minimum));
        REQUIRE(drift.ready(mapping::calibration::keyframes_minimum, mapping::calibration::view_spread_degrees_minimum));
    }

    {
        // What is asked of an adjustment: the focal lengths of every camera, the extrinsic of every camera but the reference,
        // the walks as set, and a prior on the mean of each focal length that the level of the constant calibration widens
        // from the paper's to its own.
        mapping::calibration_drift::request following;
        REQUIRE(drift.make_request(mapping::calibration::level::off, false, { 0.0, 0.11 }, following));
        REQUIRE(!following.refined);
        // The request comes with the chunks as they are held.
        REQUIRE(following.chunks.size() == 3);
        REQUIRE((following.chunks[2].index == 4) && (following.chunks[2].cameras.size() == 2));
        REQUIRE(following.cameras.size() == 2);
        REQUIRE(following.cameras[0].follow_focal && !following.cameras[0].follow_extrinsic);
        REQUIRE(following.cameras[1].follow_focal && following.cameras[1].follow_extrinsic);
        REQUIRE(following.cameras[0].given.is_equal(first_given));
        REQUIRE(following.cameras[1].given.is_equal(second_given));
        REQUIRE(following.cameras[0].baseline == 0.0);
        REQUIRE(following.cameras[1].baseline == 0.11);
        REQUIRE(following.walk.chunk_seconds == 5.0);
        REQUIRE(following.walk.rotation_walk == 0.001);
        REQUIRE(following.focal_prior_fraction == mapping::calibration_drift::focal_prior_fraction);
        REQUIRE(drift.make_request(mapping::calibration::level::focal, false, { 0.0, 0.11 }, following));
        REQUIRE(following.focal_prior_fraction == mapping::calibration::focal_length_prior_fraction);
        REQUIRE(drift.make_request(mapping::calibration::level::intrinsics, false, { 0.0, 0.11 }, following));
        REQUIRE(following.focal_prior_fraction == mapping::calibration::focal_length_prior_fraction);

        // Held extrinsics leave the focal lengths, a focal length walk of zero the extrinsics, and both nothing; either of
        // the extrinsic's walks at zero holds the extrinsics.
        REQUIRE(drift.make_request(mapping::calibration::level::off, true, { 0.0, 0.11 }, following));
        REQUIRE(following.cameras[1].follow_focal && !following.cameras[1].follow_extrinsic);
        settings.focal_walk = 0.0;
        drift.set(settings);
        REQUIRE(drift.make_request(mapping::calibration::level::off, false, { 0.0, 0.11 }, following));
        REQUIRE(!following.cameras[0].follow_focal && !following.cameras[1].follow_focal && following.cameras[1].follow_extrinsic);
        REQUIRE(!drift.make_request(mapping::calibration::level::off, true, { 0.0, 0.11 }, following));
        settings.translation_walk = 0.0;
        drift.set(settings);
        REQUIRE(!drift.make_request(mapping::calibration::level::off, false, { 0.0, 0.11 }, following));
        settings = mapping::calibration_drift::settings();
        settings.enabled = true;
        drift.set(settings);
    }

    {
        // A camera and an extrinsic part of the way from one to another.
        const sensor::model between = mapping::calibration_drift::camera_between(make_camera(500.0, 510.0), make_camera(504.0, 518.0), 0.25);
        REQUIRE(std::abs(parameter_of(between, 0) - 501.0) < 1.0e-12);
        REQUIRE(std::abs(parameter_of(between, 1) - 512.0) < 1.0e-12);
        REQUIRE(parameter_of(between, 4) == -0.1);
        const math::se3<double> turned(declared.rotation() * math::so3<double>::exp({ { 0.02, 0.0, -0.04 } }), declared.translation() + math::matrix<double, 3, 1>({ 0.0, 0.004, 0.0 }));
        const math::se3<double> part = mapping::calibration_drift::extrinsic_between(declared, turned, 0.25);
        const math::matrix<double, 3, 1> turn = (declared.rotation().inverse() * part.rotation()).log();
        REQUIRE(std::abs(turn[0] - 0.005) < 1.0e-12);
        REQUIRE(std::abs(turn[1]) < 1.0e-12);
        REQUIRE(std::abs(turn[2] + 0.01) < 1.0e-12);
        REQUIRE(std::abs(part.translation()[1] - 0.001) < 1.0e-12);
        REQUIRE(std::abs(part.translation()[0] - 0.11) < 1.0e-12);
    }

    {
        // An adjustment's result: the chunks it gives take what it found, and a chunk it leaves out, the map holding no frame
        // of it, goes the way along between those around it. Chunks 0, 1 and 4 are held, and the result has 0 and 4.
        mapping::calibration_drift::request following;
        REQUIRE(drift.make_request(mapping::calibration::level::off, false, { 0.0, 0.11 }, following));
        drift.update(following);
        REQUIRE(drift.get_refinements() == 0);
        mapping::calibration_drift::chunk start;
        start.index = 0;
        start.cameras = { { make_camera(500.0, 510.0), math::se3<double>::identity(), 900 }, { make_camera(480.0, 480.0), declared, 800 } };
        mapping::calibration_drift::chunk end;
        end.index = 4;
        end.cameras = { { make_camera(504.0, 514.0), math::se3<double>::identity(), 700 }, { make_camera(482.0, 482.0), math::se3<double>(declared.rotation(), declared.translation() + math::matrix<double, 3, 1>({ 0.0, 0.004, 0.0 })), 600 } };
        following.chunks = { start, end };
        following.refined = true;
        drift.update(following);
        REQUIRE(drift.get_refinements() == 1);
        REQUIRE(drift.chunks().size() == 3);
        REQUIRE(parameter_of(drift.chunks()[0].cameras[0].camera, 0) == 500.0);
        REQUIRE(drift.chunks()[0].cameras[1].observations == 800);
        REQUIRE(std::abs(parameter_of(drift.chunks()[1].cameras[0].camera, 0) - 501.0) < 1.0e-12);
        REQUIRE(std::abs(parameter_of(drift.chunks()[1].cameras[1].camera, 1) - 480.5) < 1.0e-12);
        REQUIRE(std::abs(drift.chunks()[1].cameras[1].reference_from_camera.translation()[1] - 0.001) < 1.0e-12);
        REQUIRE(drift.chunks()[1].cameras[0].observations == 0);
        REQUIRE(parameter_of(drift.chunks()[2].cameras[0].camera, 0) == 504.0);
        REQUIRE(drift.chunks()[2].cameras[0].observations == 700);
        // Once followed, every later adjustment may follow.
        REQUIRE(drift.ready(1, 0.0));

        // A chunk that starts now starts from the newest, and its frames take it; the frames of an older chunk take theirs.
        std::vector<sensor::model> cameras = { first_given, second_given };
        std::vector<math::se3<double>> extrinsics = { math::se3<double>::identity(), declared };
        REQUIRE(drift.take_frame(131 * second, cameras, extrinsics) == 6);
        REQUIRE(parameter_of(cameras[0], 0) == 504.0);
        REQUIRE(parameter_of(cameras[1], 0) == 482.0);
        REQUIRE(extrinsics[1].translation()[1] == 0.004);
        REQUIRE(drift.newest()->cameras[0].observations == 0);
        cameras = { first_given, second_given };
        extrinsics = { math::se3<double>::identity(), declared };
        REQUIRE(drift.take_frame(107 * second, cameras, extrinsics) == 1);
        REQUIRE(std::abs(parameter_of(cameras[0], 0) - 501.0) < 1.0e-12);

        // A result that ends before the newest chunk leaves it with the result's last.
        end.cameras[0].camera = make_camera(506.0, 516.0);
        following.chunks = { start, end };
        drift.update(following);
        REQUIRE(drift.get_refinements() == 2);
        REQUIRE(parameter_of(drift.chunks()[3].cameras[0].camera, 0) == 506.0);
        REQUIRE(drift.chunks()[3].cameras[0].observations == 0);
        // A result that was undone changes nothing.
        following.refined = false;
        end.cameras[0].camera = make_camera(520.0, 530.0);
        following.chunks = { start, end };
        drift.update(following);
        REQUIRE(drift.get_refinements() == 2);
        REQUIRE(parameter_of(drift.chunks()[2].cameras[0].camera, 0) == 506.0);
    }

    return EXIT_SUCCESS;
}
