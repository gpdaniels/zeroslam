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

#include "math/lie.hpp"
#include "math/math.hpp"
#include "math/matrix.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstring>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    bool token_equals(const char* const token, const size_t token_length, const char* const name) {
        return (token_length == std::strlen(name)) && (std::strncmp(token, name, token_length) == 0);
    }

    // A decimal that is not negative, with an optional exponent, as the keys are printed: digits, at most one point among
    // them, and e or E with a signed whole exponent.
    bool parse_decimal(const char* const token, const size_t token_length, double& value) {
        if ((token_length == 0) || (token_length > 24)) {
            return false;
        }
        double parsed = 0.0;
        double scale = 1.0;
        bool fraction = false;
        bool digits = false;
        for (size_t index = 0; index < token_length; ++index) {
            const char character = token[index];
            if ((character == '.') && !fraction) {
                fraction = true;
                continue;
            }
            if (((character == 'e') || (character == 'E')) && digits) {
                size_t exponent_index = index + 1;
                const bool negative = (exponent_index < token_length) && (token[exponent_index] == '-');
                if ((exponent_index < token_length) && ((token[exponent_index] == '-') || (token[exponent_index] == '+'))) {
                    ++exponent_index;
                }
                if (exponent_index == token_length) {
                    return false;
                }
                int exponent = 0;
                for (; exponent_index < token_length; ++exponent_index) {
                    if ((token[exponent_index] < '0') || (token[exponent_index] > '9')) {
                        return false;
                    }
                    exponent = math::min(99, (exponent * 10) + (token[exponent_index] - '0'));
                }
                for (int step = 0; step < exponent; ++step) {
                    parsed = negative ? (parsed * 0.1) : (parsed * 10.0);
                }
                break;
            }
            if ((character < '0') || (character > '9')) {
                return false;
            }
            digits = true;
            if (fraction) {
                scale *= 0.1;
                parsed += static_cast<double>(character - '0') * scale;
            }
            else {
                parsed = (parsed * 10.0) + static_cast<double>(character - '0');
            }
        }
        if (!digits || !math::isfinite(parsed)) {
            return false;
        }
        value = parsed;
        return true;
    }
}

api_calibration_drift::parsed api_calibration_drift::parse(const char* const key, const size_t key_length, const char* const value, const size_t value_length, mapping::calibration_drift::settings& settings) {
    if (token_equals(key, key_length, api_calibration_drift::key_enabled)) {
        if (token_equals(value, value_length, "on")) {
            settings.enabled = true;
            return parsed::taken;
        }
        if (token_equals(value, value_length, "off")) {
            settings.enabled = false;
            return parsed::taken;
        }
        return parsed::invalid;
    }
    const bool chunk = token_equals(key, key_length, api_calibration_drift::key_chunk);
    double* const walk = token_equals(key, key_length, api_calibration_drift::key_focal) ? &settings.focal_walk : (token_equals(key, key_length, api_calibration_drift::key_rotation) ? &settings.rotation_walk : (token_equals(key, key_length, api_calibration_drift::key_translation) ? &settings.translation_walk : nullptr));
    if (!chunk && (walk == nullptr)) {
        return parsed::unknown;
    }
    double decimal = 0.0;
    if (!parse_decimal(value, value_length, decimal)) {
        return parsed::invalid;
    }
    if (chunk) {
        if ((decimal < mapping::calibration_drift::chunk_seconds_minimum) || (decimal > mapping::calibration_drift::chunk_seconds_maximum)) {
            return parsed::invalid;
        }
        settings.chunk_seconds = decimal;
        return parsed::taken;
    }
    if (decimal > api_calibration_drift::walk_maximum) {
        return parsed::invalid;
    }
    *walk = decimal;
    return parsed::taken;
}

int api_calibration_drift::print(char* const buffer, const size_t capacity, const mapping::calibration_drift::settings& settings) {
    return std::snprintf(
        buffer,
        capacity,
        "%s=%s\n%s=%.9g\n%s=%.9g\n%s=%.9g\n%s=%.9g\n",
        api_calibration_drift::key_enabled,
        settings.enabled ? "on" : "off",
        api_calibration_drift::key_chunk,
        settings.chunk_seconds,
        api_calibration_drift::key_focal,
        settings.focal_walk,
        api_calibration_drift::key_rotation,
        settings.rotation_walk,
        api_calibration_drift::key_translation,
        settings.translation_walk
    );
}

int api_calibration_drift::count(const mapping::calibration_drift& drift) {
    size_t entries = 0;
    for (const mapping::calibration_drift::chunk& chunk : drift.chunks()) {
        entries += chunk.cameras.size();
    }
    return static_cast<int>(entries);
}

void api_calibration_drift::fill(const mapping::calibration_drift& drift, const mapping::rig& rig, const std::vector<int>& sensor_ids, zeroslam_calibration_chunk_struct* const chunks) {
    // The chunks hold each camera on the reference camera, which the rig has on the body.
    const math::se3<double> body_from_reference = (rig.reference() >= 0) ? rig.at(static_cast<size_t>(rig.reference())).body_from_sensor : math::se3<double>::identity();
    size_t entry = 0;
    for (const mapping::calibration_drift::chunk& chunk : drift.chunks()) {
        for (size_t camera = 0; camera < chunk.cameras.size(); ++camera) {
            const mapping::calibration_drift::camera_state& state = chunk.cameras[camera];
            zeroslam_calibration_chunk_struct filled;
            std::memset(&filled, 0, sizeof(filled));
            filled.timestamp = drift.chunk_start(chunk.index);
            filled.duration = drift.chunk_nanoseconds();
            filled.sensor_id = (camera < sensor_ids.size()) ? sensor_ids[camera] : 0;
            filled.observations = static_cast<int>(math::min(state.observations, static_cast<size_t>(2147483647)));
            // The parameters the camera says are its focal lengths, the horizontal first, which every model begins with.
            const std::vector<size_t> focal = mapping::calibration_drift::focal_parameters(state.camera);
            std::vector<double> parameters(state.camera.get_parameter_count(), 0.0);
            if (!focal.empty() && state.camera.get_parameters(parameters.data(), parameters.size())) {
                filled.focal_x = parameters[focal.front()];
                filled.focal_y = parameters[(focal.size() > 1) ? focal[1] : focal.front()];
            }
            const math::se3<double> body_from_camera = body_from_reference * state.reference_from_camera;
            const math::matrix<double, 4, 1> quaternion = body_from_camera.rotation().get_quaternion();
            filled.pose[0] = body_from_camera.translation()[0];
            filled.pose[1] = body_from_camera.translation()[1];
            filled.pose[2] = body_from_camera.translation()[2];
            filled.pose[3] = quaternion[1];
            filled.pose[4] = quaternion[2];
            filled.pose[5] = quaternion[3];
            filled.pose[6] = quaternion[0];
            std::memcpy(&chunks[entry], &filled, sizeof(filled));
            ++entry;
        }
    }
}
