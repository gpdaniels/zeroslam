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

#include "api_calibration_drift.hpp"

#include "mapping/calibration.hpp"
#include "mapping/calibration_drift.hpp"
#include "mapping/rig.hpp"
#include "math/lie.hpp"
#include "sensor/camera.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static api_calibration_drift::parsed parse(const char* const key, const char* const value, mapping::calibration_drift::settings& settings) {
    return api_calibration_drift::parse(key, std::strlen(key), value, std::strlen(value), settings);
}

static std::string configuration_of(zeroslam_system* const system) {
    int length = 0;
    REQUIRE(zeroslam_get_configuration(system, nullptr, &length) == zeroslam_return_failure_insufficient_data_length);
    std::vector<char> buffer(static_cast<size_t>(length));
    int capacity = length;
    REQUIRE(zeroslam_get_configuration(system, buffer.data(), &capacity) == zeroslam_return_success);
    REQUIRE(capacity == length - 1);
    REQUIRE(std::strlen(buffer.data()) == static_cast<size_t>(capacity));
    return std::string(buffer.data());
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        // The keys: on and off, a chunk's length in seconds, and the three walks, each of which may be zero.
        mapping::calibration_drift::settings settings;
        REQUIRE(parse("calibration", "focal", settings) == api_calibration_drift::parsed::unknown);
        REQUIRE(parse("calibration_drift_", "on", settings) == api_calibration_drift::parsed::unknown);
        REQUIRE(parse("calibration_drift", "on", settings) == api_calibration_drift::parsed::taken);
        REQUIRE(settings.enabled);
        REQUIRE(parse("calibration_drift", "yes", settings) == api_calibration_drift::parsed::invalid);
        REQUIRE(settings.enabled);
        REQUIRE(parse("calibration_drift", "off", settings) == api_calibration_drift::parsed::taken);
        REQUIRE(!settings.enabled);
        REQUIRE(parse("calibration_drift_chunk", "2.5", settings) == api_calibration_drift::parsed::taken);
        REQUIRE(settings.chunk_seconds == 2.5);
        REQUIRE(parse("calibration_drift_chunk", "6e1", settings) == api_calibration_drift::parsed::taken);
        REQUIRE(settings.chunk_seconds == 60.0);
        // A chunk has a length, of a day at most.
        REQUIRE(parse("calibration_drift_chunk", "0", settings) == api_calibration_drift::parsed::invalid);
        REQUIRE(parse("calibration_drift_chunk", "100000", settings) == api_calibration_drift::parsed::invalid);
        REQUIRE(parse("calibration_drift_chunk", "-5", settings) == api_calibration_drift::parsed::invalid);
        REQUIRE(parse("calibration_drift_chunk", "", settings) == api_calibration_drift::parsed::invalid);
        REQUIRE(parse("calibration_drift_chunk", "5s", settings) == api_calibration_drift::parsed::invalid);
        REQUIRE(parse("calibration_drift_chunk", "1.5.2", settings) == api_calibration_drift::parsed::invalid);
        REQUIRE(parse("calibration_drift_chunk", "1e", settings) == api_calibration_drift::parsed::invalid);
        REQUIRE(settings.chunk_seconds == 60.0);
        REQUIRE(parse("calibration_drift_focal", "0.0003", settings) == api_calibration_drift::parsed::taken);
        REQUIRE(std::abs(settings.focal_walk - 0.0003) < 1.0e-18);
        REQUIRE(parse("calibration_drift_focal", "2.5e-4", settings) == api_calibration_drift::parsed::taken);
        REQUIRE(std::abs(settings.focal_walk - 0.00025) < 1.0e-18);
        REQUIRE(parse("calibration_drift_focal", "0", settings) == api_calibration_drift::parsed::taken);
        REQUIRE(settings.focal_walk == 0.0);
        REQUIRE(parse("calibration_drift_focal", "0.2", settings) == api_calibration_drift::parsed::invalid);
        REQUIRE(settings.focal_walk == 0.0);
        REQUIRE(parse("calibration_drift_rotation", "0.002", settings) == api_calibration_drift::parsed::taken);
        REQUIRE(std::abs(settings.rotation_walk - 0.002) < 1.0e-18);
        REQUIRE(parse("calibration_drift_rotation", "fast", settings) == api_calibration_drift::parsed::invalid);
        REQUIRE(parse("calibration_drift_translation", "0.001", settings) == api_calibration_drift::parsed::taken);
        REQUIRE(std::abs(settings.translation_walk - 0.001) < 1.0e-18);
        REQUIRE(parse("calibration_drift_translation", "1", settings) == api_calibration_drift::parsed::invalid);
    }

    {
        // The keys print as they parse: what is printed of the defaults, the paper's values, parses back to them.
        const mapping::calibration_drift::settings defaults;
        const int length = api_calibration_drift::print(nullptr, 0, defaults);
        REQUIRE(length > 0);
        std::vector<char> buffer(static_cast<size_t>(length) + 1);
        REQUIRE(api_calibration_drift::print(buffer.data(), buffer.size(), defaults) == length);
        const std::string printed(buffer.data());
        REQUIRE(printed == "calibration_drift=off\ncalibration_drift_chunk=5\ncalibration_drift_focal=0.000145833333\ncalibration_drift_rotation=0.001\ncalibration_drift_translation=0.0005\n");
        mapping::calibration_drift::settings parsed_back;
        parsed_back.enabled = true;
        parsed_back.chunk_seconds = 1.0;
        parsed_back.focal_walk = 1.0;
        parsed_back.rotation_walk = 1.0;
        parsed_back.translation_walk = 1.0;
        size_t start = 0;
        while (start < printed.size()) {
            const size_t end = printed.find('\n', start);
            const size_t equals = printed.find('=', start);
            REQUIRE((end != std::string::npos) && (equals != std::string::npos) && (equals < end));
            REQUIRE(api_calibration_drift::parse(printed.data() + start, equals - start, printed.data() + equals + 1, end - equals - 1, parsed_back) == api_calibration_drift::parsed::taken);
            start = end + 1;
        }
        REQUIRE(!parsed_back.enabled);
        REQUIRE(parsed_back.chunk_seconds == defaults.chunk_seconds);
        REQUIRE(std::abs((parsed_back.focal_walk / defaults.focal_walk) - 1.0) < 1.0e-8);
        REQUIRE(std::abs(parsed_back.rotation_walk - defaults.rotation_walk) < 1.0e-15);
        REQUIRE(std::abs(parsed_back.translation_walk - defaults.translation_walk) < 1.0e-15);
        // A short buffer takes what fits, as snprintf does.
        char few[8] = {};
        REQUIRE(api_calibration_drift::print(&few[0], sizeof(few), defaults) == length);
        REQUIRE(std::strcmp(&few[0], "calibra") == 0);
    }

    {
        // The chunks as entries: one per chunk and camera, by time and in the rig's order, each camera's focal lengths in the
        // chunk and its extrinsic on the rig's body, the reference camera's the rig's own.
        const double parameters[4] = { 240.0, 241.0, 320.0, 240.0 };
        const sensor::model given(sensor::camera::pinhole<double>(&parameters[0], 4));
        mapping::rig rig;
        const math::se3<double> body_from_first(math::so3<double>::identity(), { { 0.0, 0.0, 0.5 } });
        const math::se3<double> body_from_second(math::so3<double>::identity(), { { 0.1, 0.0, 0.5 } });
        rig.add(4, mapping::rig::sensor_kind::camera, body_from_first, false);
        rig.add(7, mapping::rig::sensor_kind::camera, body_from_second, false);
        mapping::calibration_drift drift;
        REQUIRE(api_calibration_drift::count(drift) == 0);
        mapping::calibration_drift::settings settings;
        settings.enabled = true;
        settings.chunk_seconds = 2.0;
        drift.set(settings);
        for (const long long timestamp : { 1000000000ll, 2500000000ll, 7000000000ll }) {
            std::vector<sensor::model> cameras = { given, given };
            std::vector<math::se3<double>> extrinsics = { math::se3<double>::identity(), rig.reference_from_sensor(1) };
            REQUIRE(drift.take_frame(timestamp, cameras, extrinsics) >= 0);
        }
        REQUIRE(drift.chunks().size() == 2);
        REQUIRE(api_calibration_drift::count(drift) == 4);
        // An adjustment's result for the last chunk, the second camera's focal length longer and its extrinsic moved.
        mapping::calibration_drift::request following;
        REQUIRE(drift.make_request(mapping::calibration::level::off, false, { 0.0, 0.1 }, following));
        following.refined = true;
        following.chunks = drift.chunks();
        const double longer[4] = { 240.5, 241.5, 320.0, 240.0 };
        following.chunks[1].cameras[1].camera = sensor::model(sensor::camera::pinhole<double>(&longer[0], 4));
        following.chunks[1].cameras[1].reference_from_camera = math::se3<double>(math::so3<double>::identity(), { { 0.1, 0.002, 0.0 } });
        following.chunks[1].cameras[1].observations = 321;
        drift.update(following);
        std::vector<zeroslam_calibration_chunk_struct> entries(4);
        api_calibration_drift::fill(drift, rig, { 4, 7 }, entries.data());
        REQUIRE((entries[0].sensor_id == 4) && (entries[1].sensor_id == 7) && (entries[2].sensor_id == 4) && (entries[3].sensor_id == 7));
        REQUIRE((entries[0].timestamp == 1000000000ll) && (entries[1].timestamp == 1000000000ll));
        // The frame at seven seconds is in the fourth chunk of two seconds from the first frame's time.
        REQUIRE((entries[2].timestamp == 7000000000ll) && (entries[3].timestamp == 7000000000ll));
        for (const zeroslam_calibration_chunk_struct& entry : entries) {
            REQUIRE(entry.duration == 2000000000ll);
            REQUIRE(entry.pose[6] == 1.0);
            REQUIRE(entry.pose[2] == 0.5);
        }
        REQUIRE((entries[0].focal_x == 240.0) && (entries[0].focal_y == 241.0));
        REQUIRE((entries[3].focal_x == 240.5) && (entries[3].focal_y == 241.5));
        REQUIRE(entries[0].pose[0] == 0.0);
        REQUIRE(std::abs(entries[1].pose[0] - 0.1) < 1.0e-15);
        REQUIRE(entries[1].pose[1] == 0.0);
        REQUIRE(std::abs(entries[3].pose[1] - 0.002) < 1.0e-15);
        REQUIRE((entries[0].observations == 0) && (entries[3].observations == 321));
        // A camera the ids do not reach has none.
        api_calibration_drift::fill(drift, rig, { 4 }, entries.data());
        REQUIRE((entries[0].sensor_id == 4) && (entries[1].sensor_id == 0));
    }

    {
        // Through the api: the keys are part of the configuration, and the chunks come back once frames with timestamps are
        // followed, here one camera's whose frames fall in two chunks of five seconds.
        zeroslam_system* system = nullptr;
        REQUIRE(zeroslam_create(&system) == zeroslam_return_success);
        std::string configuration = configuration_of(system);
        REQUIRE(configuration.find("\ncalibration_drift=off\ncalibration_drift_chunk=5\ncalibration_drift_focal=0.000145833333\ncalibration_drift_rotation=0.001\ncalibration_drift_translation=0.0005\n") != std::string::npos);
        // The whole configuration, as it is given, is taken back.
        REQUIRE(zeroslam_set_configuration(system, configuration.c_str(), static_cast<int>(configuration.size())) == zeroslam_return_success);
        REQUIRE(configuration_of(system) == configuration);
        const char* const on = "calibration_drift=on\ncalibration_drift_focal=0.0002\n";
        REQUIRE(zeroslam_set_configuration(system, on, static_cast<int>(std::strlen(on))) == zeroslam_return_success);
        configuration = configuration_of(system);
        REQUIRE(configuration.find("\ncalibration_drift=on\n") != std::string::npos);
        REQUIRE(configuration.find("\ncalibration_drift_focal=0.0002\n") != std::string::npos);
        const char* const bad = "calibration_drift=sometimes\n";
        REQUIRE(zeroslam_set_configuration(system, bad, static_cast<int>(std::strlen(bad))) == zeroslam_return_failure_invalid_configuration);
        const char* const bad_chunk = "calibration_drift_chunk=0\n";
        REQUIRE(zeroslam_set_configuration(system, bad_chunk, static_cast<int>(std::strlen(bad_chunk))) == zeroslam_return_failure_invalid_configuration);
        REQUIRE(configuration_of(system) == configuration);

        int length = 7;
        REQUIRE(zeroslam_get_calibration_drift(nullptr, nullptr, &length) == zeroslam_return_failure_invalid_system);
        REQUIRE(zeroslam_get_calibration_drift(system, nullptr, nullptr) == zeroslam_return_failure_invalid_argument);
        // Nothing is followed before a frame is.
        REQUIRE(zeroslam_get_calibration_drift(system, nullptr, &length) == zeroslam_return_success);
        REQUIRE(length == 0);

        zeroslam_sensor_parameters_camera_struct camera_parameters{};
        camera_parameters.width = 64;
        camera_parameters.height = 48;
        camera_parameters.focal_x = 50.0;
        camera_parameters.focal_y = 51.0;
        camera_parameters.centre_x = 32.0;
        camera_parameters.centre_y = 24.0;
        zeroslam_sensor_rig_struct rig{};
        rig.type = zeroslam_sensor_camera;
        rig.sensor_id = 3;
        rig.parameters_length = static_cast<int>(sizeof(camera_parameters));
        rig.parameters_data = &camera_parameters;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_success);
        std::vector<unsigned char> pixels(64 * 48, 0);
        for (size_t index = 0; index < pixels.size(); ++index) {
            pixels[index] = static_cast<unsigned char>((index * 2654435761u) >> 24);
        }
        for (const long long timestamp : { 10000000000ll, 13000000000ll, 16000000000ll }) {
            zeroslam_sensor_data_struct frame{};
            frame.timestamp = timestamp;
            frame.sensor_id = 3;
            frame.measurement_length = static_cast<int>(pixels.size());
            frame.measurement_data = pixels.data();
            REQUIRE(zeroslam_set_sensor_data(system, &frame, 1) == zeroslam_return_success);
        }
        REQUIRE(zeroslam_get_calibration_drift(system, nullptr, &length) == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(length == 2);
        zeroslam_calibration_chunk_struct chunks[2] = {};
        length = 1;
        REQUIRE(zeroslam_get_calibration_drift(system, &chunks[0], &length) == zeroslam_return_failure_insufficient_data_length);
        REQUIRE(length == 2);
        REQUIRE(zeroslam_get_calibration_drift(system, &chunks[0], &length) == zeroslam_return_success);
        REQUIRE(length == 2);
        REQUIRE((chunks[0].timestamp == 10000000000ll) && (chunks[1].timestamp == 15000000000ll));
        for (const zeroslam_calibration_chunk_struct& chunk : chunks) {
            REQUIRE(chunk.duration == 5000000000ll);
            REQUIRE(chunk.sensor_id == 3);
            REQUIRE(chunk.observations == 0);
            REQUIRE((chunk.focal_x == 50.0) && (chunk.focal_y == 51.0));
            REQUIRE((chunk.pose[0] == 0.0) && (chunk.pose[1] == 0.0) && (chunk.pose[2] == 0.0) && (chunk.pose[6] == 1.0));
        }
        // The camera as the system holds it now is the newest chunk's.
        zeroslam_sensor_parameters_camera_model_struct held{};
        REQUIRE(zeroslam_get_camera_calibration(system, 3, &held) == zeroslam_return_success);
        REQUIRE((held.parameters[0] == 50.0) && (held.parameters[1] == 51.0));
        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_success);
    }

    {
        // Off, which it is unless asked for, frames with timestamps are in no chunk.
        zeroslam_system* system = nullptr;
        REQUIRE(zeroslam_create(&system) == zeroslam_return_success);
        zeroslam_sensor_parameters_camera_struct camera_parameters{};
        camera_parameters.width = 64;
        camera_parameters.height = 48;
        camera_parameters.focal_x = 50.0;
        camera_parameters.focal_y = 50.0;
        camera_parameters.centre_x = 32.0;
        camera_parameters.centre_y = 24.0;
        zeroslam_sensor_rig_struct rig{};
        rig.type = zeroslam_sensor_camera;
        rig.sensor_id = 3;
        rig.parameters_length = static_cast<int>(sizeof(camera_parameters));
        rig.parameters_data = &camera_parameters;
        REQUIRE(zeroslam_set_sensor_rig(system, &rig, 1) == zeroslam_return_success);
        std::vector<unsigned char> pixels(64 * 48, 128);
        zeroslam_sensor_data_struct frame{};
        frame.timestamp = 10000000000ll;
        frame.sensor_id = 3;
        frame.measurement_length = static_cast<int>(pixels.size());
        frame.measurement_data = pixels.data();
        REQUIRE(zeroslam_set_sensor_data(system, &frame, 1) == zeroslam_return_success);
        int length = 7;
        REQUIRE(zeroslam_get_calibration_drift(system, nullptr, &length) == zeroslam_return_success);
        REQUIRE(length == 0);
        REQUIRE(zeroslam_destroy(&system) == zeroslam_return_success);
    }

    return EXIT_SUCCESS;
}
