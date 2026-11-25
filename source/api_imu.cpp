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

#include "api_imu.hpp"

#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstring>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

bool api_imu::accepts(const zeroslam_sensor_rig_struct& entry) {
    if ((entry.type != zeroslam_sensor_imu) || (entry.sensor_id == 0) || (entry.parameters_data == nullptr) || (entry.parameters_length != static_cast<int>(sizeof(zeroslam_sensor_parameters_imu_struct)))) {
        return false;
    }
    zeroslam_sensor_parameters_imu_struct parameters;
    std::memcpy(&parameters, entry.parameters_data, sizeof(parameters));
    const double densities[4] = { parameters.gyroscope_noise_density, parameters.accelerometer_noise_density, parameters.gyroscope_random_walk, parameters.accelerometer_random_walk };
    for (const double density : densities) {
        if (!(density > 0.0) || !math::isfinite(density)) {
            return false;
        }
    }
    return (parameters.rate > 0.0) && math::isfinite(parameters.rate) && (parameters.gravity >= 0.0) && math::isfinite(parameters.gravity) && math::isfinite(parameters.time_offset);
}

void api_imu::declare(const zeroslam_sensor_rig_struct* const entry) {
    *this = api_imu();
    if (entry == nullptr) {
        return;
    }
    this->declared_ = true;
    this->sensor_id_ = entry->sensor_id;
    std::memcpy(&this->parameters_, entry->parameters_data, sizeof(this->parameters_));
}

bool api_imu::declared() const {
    return this->declared_;
}

int api_imu::sensor_id() const {
    return this->sensor_id_;
}

bool api_imu::take_extrinsic(const zeroslam_sensor_extrinsics_struct* const extrinsics, const int length, std::vector<zeroslam_sensor_extrinsics_struct>& rest) {
    rest.clear();
    bool found = false;
    zeroslam_sensor_extrinsics_struct taken{};
    for (int i = 0; i < length; ++i) {
        const zeroslam_sensor_extrinsics_struct& entry = extrinsics[i];
        if (!this->declared_ || (entry.sensor_id != this->sensor_id_)) {
            rest.push_back(entry);
            continue;
        }
        bool finite = true;
        for (const double value : entry.pose) {
            finite = finite && math::isfinite(value);
        }
        const double norm_squared = (entry.pose[3] * entry.pose[3]) + (entry.pose[4] * entry.pose[4]) + (entry.pose[5] * entry.pose[5]) + (entry.pose[6] * entry.pose[6]);
        if (found || !finite || !(norm_squared > 1.0e-12) || ((entry.refinement != zeroslam_extrinsic_fixed) && (entry.refinement != zeroslam_extrinsic_refine))) {
            return false;
        }
        found = true;
        taken = entry;
    }
    if (found) {
        this->extrinsic_ = taken;
        this->extrinsic_set_ = true;
    }
    return true;
}

void api_imu::fill_extrinsic(const mapping::rig& held, zeroslam_sensor_extrinsics_struct* const extrinsics, const int index) const {
    if (!this->declared_ || (extrinsics == nullptr)) {
        return;
    }
    size_t held_index = 0;
    const math::se3<double> body_from_imu = held.find(this->sensor_id_, held_index) ? held.at(held_index).body_from_sensor : this->body_from_imu();
    const math::matrix<double, 4, 1> quaternion = body_from_imu.rotation().get_quaternion();
    zeroslam_sensor_extrinsics_struct& value = extrinsics[index];
    value.sensor_id = this->sensor_id_;
    value.refinement = this->refined() ? zeroslam_extrinsic_refine : zeroslam_extrinsic_fixed;
    for (size_t axis = 0; axis < 3; ++axis) {
        value.pose[axis] = body_from_imu.translation()[axis];
    }
    value.pose[3] = quaternion[1];
    value.pose[4] = quaternion[2];
    value.pose[5] = quaternion[3];
    value.pose[6] = quaternion[0];
}

bool api_imu::fill_calibration(const mapping::inertial& held, zeroslam_imu_calibration_struct* const calibration) const {
    if (!this->declared_ || (calibration == nullptr)) {
        return false;
    }
    const mapping::gyroscope_intrinsics& gyroscope = held.gyroscope();
    calibration->sensor_id = this->sensor_id_;
    calibration->refinements = static_cast<int>(gyroscope.refinements());
    for (size_t i = 0; i < 9; ++i) {
        calibration->gyroscope[i] = gyroscope.estimate().gyroscope.data()[i];
    }
    return true;
}

sensor::imu::parameters api_imu::noise() const {
    sensor::imu::parameters noise;
    noise.gyroscope_noise_density = this->parameters_.gyroscope_noise_density;
    noise.accelerometer_noise_density = this->parameters_.accelerometer_noise_density;
    noise.gyroscope_random_walk = this->parameters_.gyroscope_random_walk;
    noise.accelerometer_random_walk = this->parameters_.accelerometer_random_walk;
    noise.rate = this->parameters_.rate;
    if (this->parameters_.gravity > 0.0) {
        noise.gravity = this->parameters_.gravity;
    }
    return noise;
}

math::se3<double> api_imu::body_from_imu() const {
    if (!this->extrinsic_set_) {
        return math::se3<double>::identity();
    }
    const double* const pose = this->extrinsic_.pose;
    const double norm = math::sqrt((pose[3] * pose[3]) + (pose[4] * pose[4]) + (pose[5] * pose[5]) + (pose[6] * pose[6]));
    return math::se3<double>(math::so3<double>(pose[6] / norm, pose[3] / norm, pose[4] / norm, pose[5] / norm), { { pose[0], pose[1], pose[2] } });
}

bool api_imu::refined() const {
    return this->extrinsic_set_ && (this->extrinsic_.refinement == zeroslam_extrinsic_refine);
}

bool api_imu::owns(const zeroslam_sensor_data_struct& entry) const {
    return this->declared_ && (entry.sensor_id == this->sensor_id_);
}

bool api_imu::accepts_samples(const zeroslam_sensor_data_struct* const data, const int length) const {
    long long latest = this->latest_timestamp_;
    bool any = this->any_sample_;
    for (int i = 0; i < length; ++i) {
        const zeroslam_sensor_data_struct& entry = data[i];
        if (!this->owns(entry)) {
            continue;
        }
        if ((entry.measurement_data == nullptr) || (entry.measurement_length != static_cast<int>(sizeof(zeroslam_sensor_measurement_imu_struct)))) {
            return false;
        }
        if (any && !(entry.timestamp > latest)) {
            return false;
        }
        zeroslam_sensor_measurement_imu_struct measurement;
        std::memcpy(&measurement, entry.measurement_data, sizeof(measurement));
        for (size_t axis = 0; axis < 3; ++axis) {
            if (!math::isfinite(measurement.angular_velocity[axis]) || !math::isfinite(measurement.linear_acceleration[axis])) {
                return false;
            }
        }
        latest = entry.timestamp;
        any = true;
    }
    return true;
}

sensor::imu::sample api_imu::take_sample(const zeroslam_sensor_data_struct& entry) {
    zeroslam_sensor_measurement_imu_struct measurement;
    std::memcpy(&measurement, entry.measurement_data, sizeof(measurement));
    sensor::imu::sample sample;
    sample.timestamp = entry.timestamp - static_cast<long long>(math::round(this->parameters_.time_offset * 1.0e9));
    sample.angular_velocity = { { measurement.angular_velocity[0], measurement.angular_velocity[1], measurement.angular_velocity[2] } };
    sample.linear_acceleration = { { measurement.linear_acceleration[0], measurement.linear_acceleration[1], measurement.linear_acceleration[2] } };
    this->latest_timestamp_ = entry.timestamp;
    this->any_sample_ = true;
    return sample;
}
