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

#if !defined(ZEROSLAM_API_EXPORT)
#define ZEROSLAM_API_EXPORT
#endif

#include "api_depth.hpp"

#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstring>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

bool api_depth::valid(const zeroslam_sensor_parameters_depth_struct& parameters) {
    const bool encoding_valid = (parameters.encoding == zeroslam_depth_encoding_unsigned_16) || (parameters.encoding == zeroslam_depth_encoding_float_32);
    const bool measure_valid = (parameters.measure == zeroslam_depth_measure_axial) || (parameters.measure == zeroslam_depth_measure_radial);
    const bool scale_valid = (parameters.encoding != zeroslam_depth_encoding_unsigned_16) || (math::isfinite(parameters.scale) && (parameters.scale > 0.0));
    const bool range_valid = math::isfinite(parameters.minimum_depth) && (parameters.minimum_depth >= 0.0) && (parameters.maximum_depth > parameters.minimum_depth);
    const bool noise_finite = math::isfinite(parameters.noise_constant) && math::isfinite(parameters.noise_linear) && math::isfinite(parameters.noise_quadratic);
    const bool noise_valid = noise_finite && (parameters.noise_constant >= 0.0) && (parameters.noise_linear >= 0.0) && (parameters.noise_quadratic >= 0.0) && ((parameters.noise_constant + parameters.noise_linear + parameters.noise_quadratic) > 0.0);
    return encoding_valid && measure_valid && (parameters.registration == zeroslam_depth_registration_camera) && scale_valid && range_valid && noise_valid && (parameters.pairing_tolerance >= 0);
}

bool api_depth::parse(const zeroslam_sensor_rig_struct& entry, sensor& parsed) {
    if ((entry.type != zeroslam_sensor_depth) || (entry.sensor_id == 0) || (entry.parameters_length != static_cast<int>(sizeof(zeroslam_sensor_parameters_depth_struct))) || (entry.parameters_data == nullptr)) {
        return false;
    }
    parsed.sensor_id = entry.sensor_id;
    std::memcpy(&parsed.parameters, entry.parameters_data, sizeof(parsed.parameters));
    return api_depth::valid(parsed.parameters);
}

void api_depth::set_sensors(const std::vector<sensor>& rig_sensors) {
    this->sensors = rig_sensors;
    this->held.clear();
}

bool api_depth::empty() const {
    return this->sensors.empty();
}

const api_depth::sensor* api_depth::find(const int sensor_id) const {
    for (const sensor& candidate : this->sensors) {
        if (candidate.sensor_id == sensor_id) {
            return &candidate;
        }
    }
    return nullptr;
}

size_t api_depth::frame_length(const sensor& depth_sensor, const int width, const int height) {
    const size_t pixel_size = (depth_sensor.parameters.encoding == zeroslam_depth_encoding_unsigned_16) ? sizeof(unsigned short) : sizeof(float);
    return static_cast<size_t>(width) * static_cast<size_t>(height) * pixel_size;
}

void api_depth::validation_start(bool& seen, long long int& next) const {
    seen = this->any_frame;
    next = this->latest_timestamp;
}

bool api_depth::follows(const long long int timestamp, bool& seen, long long int& next) const {
    if (seen && !(timestamp > next)) {
        return false;
    }
    seen = true;
    next = timestamp;
    return true;
}

void api_depth::hold(const sensor& depth_sensor, const long long int timestamp, const int width, const int height, const void* const data) {
    ::sensor::depth::settings settings;
    settings.measure = (depth_sensor.parameters.measure == zeroslam_depth_measure_radial) ? ::sensor::depth::measure_kind::radial : ::sensor::depth::measure_kind::axial;
    settings.minimum = depth_sensor.parameters.minimum_depth;
    settings.maximum = depth_sensor.parameters.maximum_depth;
    settings.noise.constant = depth_sensor.parameters.noise_constant;
    settings.noise.linear = depth_sensor.parameters.noise_linear;
    settings.noise.quadratic = depth_sensor.parameters.noise_quadratic;
    const size_t rows = static_cast<size_t>(height);
    const size_t columns = static_cast<size_t>(width);
    if (depth_sensor.parameters.encoding == zeroslam_depth_encoding_unsigned_16) {
        std::vector<unsigned short> values(rows * columns, static_cast<unsigned short>(0));
        std::memcpy(values.data(), data, values.size() * sizeof(unsigned short));
        this->held.push_back({ timestamp, depth_sensor.parameters.camera_id, ::sensor::depth(rows, columns, values.data(), depth_sensor.parameters.scale, settings) });
    }
    else {
        std::vector<float> metres(rows * columns, 0.0f);
        std::memcpy(metres.data(), data, metres.size() * sizeof(float));
        this->held.push_back({ timestamp, depth_sensor.parameters.camera_id, ::sensor::depth(rows, columns, metres.data(), settings) });
    }
    this->latest_timestamp = timestamp;
    this->any_frame = true;
    if (this->held.size() > api_depth::pending_maximum) {
        this->held.erase(this->held.begin());
    }
}

::sensor::depth api_depth::take(const int camera_id, const long long int timestamp) {
    ::sensor::depth image;
    for (const sensor& depth_sensor : this->sensors) {
        if (depth_sensor.parameters.camera_id != camera_id) {
            continue;
        }
        size_t best = this->held.size();
        unsigned long long best_gap = 0;
        for (size_t index = 0; index < this->held.size(); ++index) {
            const pending& candidate = this->held[index];
            if (candidate.camera_id != camera_id) {
                continue;
            }
            const unsigned long long gap = (candidate.timestamp >= timestamp) ? (static_cast<unsigned long long>(candidate.timestamp) - static_cast<unsigned long long>(timestamp)) : (static_cast<unsigned long long>(timestamp) - static_cast<unsigned long long>(candidate.timestamp));
            if ((gap <= static_cast<unsigned long long>(depth_sensor.parameters.pairing_tolerance)) && ((best == this->held.size()) || (gap < best_gap))) {
                best = index;
                best_gap = gap;
            }
        }
        size_t used = 0;
        if (best < this->held.size()) {
            image = static_cast<::sensor::depth&&>(this->held[best].image);
            used = best + 1;
        }
        else {
            while ((used < this->held.size()) && (this->held[used].timestamp < timestamp)) {
                ++used;
            }
        }
        size_t kept = 0;
        for (size_t index = 0; index < this->held.size(); ++index) {
            if ((index < used) && (this->held[index].camera_id == camera_id)) {
                continue;
            }
            if (kept != index) {
                this->held[kept] = static_cast<pending&&>(this->held[index]);
            }
            ++kept;
        }
        this->held.erase(this->held.begin() + static_cast<std::ptrdiff_t>(kept), this->held.end());
    }
    return image;
}
