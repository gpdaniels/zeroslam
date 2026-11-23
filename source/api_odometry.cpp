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

#include "api_odometry.hpp"

#include "math/math.hpp"
#include "sensor/odometry/ackermann.hpp"
#include "sensor/odometry/differential_drive.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstring>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

bool api_odometry::declares(const zeroslam_sensor_rig_struct& entry) {
    return entry.type == zeroslam_sensor_wheel_odometry;
}

bool api_odometry::settings_of(const zeroslam_sensor_parameters_wheel_odometry_struct& values, const int id, mapping::odometry::settings& result) {
    mapping::odometry::settings settings;
    settings.sensor_id = id;
    switch (values.model) {
        case zeroslam_odometry_ackermann_rear_axle:
            settings.kinematics = sensor::odometry::model(sensor::odometry::ackermann(sensor::odometry::ackermann::measured_wheel::rear_axle));
            break;
        case zeroslam_odometry_ackermann_front_wheel:
            settings.kinematics = sensor::odometry::model(sensor::odometry::ackermann(sensor::odometry::ackermann::measured_wheel::front_wheel));
            break;
        case zeroslam_odometry_differential_drive:
            settings.kinematics = sensor::odometry::model(sensor::odometry::differential_drive{});
            break;
        default:
            return false;
    }
    const size_t count = settings.kinematics.parameter_count();
    if ((static_cast<unsigned int>(values.free_mask) >> count) != 0u) {
        return false;
    }
    settings.calibration = sensor::odometry::calibration(settings.kinematics, &values.parameters[0], count);
    for (size_t i = 0; i < count; ++i) {
        if (!math::isfinite(values.parameters[i]) || !math::isfinite(values.parameter_deviations[i]) || (values.parameter_deviations[i] < 0.0)) {
            return false;
        }
        settings.calibration.free[i] = ((static_cast<unsigned int>(values.free_mask) >> i) & 1u) != 0u;
        settings.calibration.prior_sigma[i] = values.parameter_deviations[i];
    }
    double* const noise[8] = { &settings.noise.slip, &settings.noise.steering, &settings.noise.lateral, &settings.noise.heading, &settings.noise.vertical, &settings.noise.tilt, &settings.noise.minimum_translation, &settings.noise.minimum_rotation };
    for (size_t i = 0; i < 8; ++i) {
        if (!math::isfinite(values.noise[i]) || (values.noise[i] < 0.0)) {
            return false;
        }
        *noise[i] = values.noise[i];
    }
    if (!(settings.noise.minimum_translation > 0.0) || !(settings.noise.minimum_rotation > 0.0)) {
        return false;
    }
    for (size_t i = 0; i < 6; ++i) {
        if (!math::isfinite(values.extrinsic_deviations[i]) || (values.extrinsic_deviations[i] < 0.0)) {
            return false;
        }
        settings.extrinsic_deviations[i] = values.extrinsic_deviations[i];
    }
    // The model must take the declared calibration: positive lengths, a steering offset within the steering limit.
    const double still[2] = { 0.0, 0.0 };
    double distance = 0.0;
    double heading = 0.0;
    if (!settings.kinematics.motion(&settings.calibration.values[0], &still[0], distance, heading, nullptr, nullptr)) {
        return false;
    }
    result = settings;
    return true;
}

bool api_odometry::take(const zeroslam_sensor_rig_struct& entry) {
    if (this->declared || (entry.parameters_data == nullptr) || (entry.parameters_length != static_cast<int>(sizeof(zeroslam_sensor_parameters_wheel_odometry_struct)))) {
        return false;
    }
    zeroslam_sensor_parameters_wheel_odometry_struct values;
    std::memcpy(&values, entry.parameters_data, sizeof(values));
    mapping::odometry::settings settings;
    if (!api_odometry::settings_of(values, entry.sensor_id, settings)) {
        return false;
    }
    this->declared = true;
    this->sensor_id = entry.sensor_id;
    this->parameters = values;
    return true;
}

bool api_odometry::is_declared() const {
    return this->declared;
}

int api_odometry::get_sensor_id() const {
    return this->sensor_id;
}

bool api_odometry::take_extrinsics(const zeroslam_sensor_extrinsics_struct* const values, const int length, std::vector<zeroslam_sensor_extrinsics_struct>& rest) {
    rest.clear();
    bool given = false;
    zeroslam_sensor_extrinsics_struct taken = {};
    for (int i = 0; i < length; ++i) {
        const zeroslam_sensor_extrinsics_struct& entry = values[i];
        if (!this->declared || (entry.sensor_id != this->sensor_id)) {
            rest.push_back(entry);
            continue;
        }
        double norm = 0.0;
        bool finite = true;
        for (size_t k = 0; k < 7; ++k) {
            finite = finite && math::isfinite(entry.pose[k]);
            norm += (k >= 3) ? (entry.pose[k] * entry.pose[k]) : 0.0;
        }
        if (given || !finite || (math::abs(norm - 1.0) > 1.0e-3) || ((entry.refinement != zeroslam_extrinsic_fixed) && (entry.refinement != zeroslam_extrinsic_refine))) {
            return false;
        }
        given = true;
        taken = entry;
    }
    if (given) {
        this->extrinsic_given = true;
        this->extrinsic = taken;
    }
    return true;
}

mapping::rig api_odometry::with_odometer(const mapping::rig& cameras) const {
    if (!this->declared) {
        return cameras;
    }
    mapping::rig result = cameras;
    math::se3<double> body_from_odometry = math::se3<double>::identity();
    bool refined = false;
    if (this->extrinsic_given) {
        const double* const pose = &this->extrinsic.pose[0];
        body_from_odometry = math::se3<double>(math::so3<double>(pose[6], pose[3], pose[4], pose[5]).normalised(), { { pose[0], pose[1], pose[2] } });
        refined = (this->extrinsic.refinement == zeroslam_extrinsic_refine);
    }
    result.add(this->sensor_id, mapping::rig::sensor_kind::odometry, body_from_odometry, refined);
    return result;
}

bool api_odometry::accepts(const zeroslam_sensor_data_struct* const data, const int length) const {
    bool any = this->any_sample;
    long long latest = this->latest_sample;
    for (int i = 0; i < length; ++i) {
        const zeroslam_sensor_data_struct& entry = data[i];
        if (entry.sensor_id != this->sensor_id) {
            continue;
        }
        if ((entry.measurement_data == nullptr) || (entry.measurement_length != static_cast<int>(api_odometry::sample_values * sizeof(double))) || (any && (entry.timestamp <= latest))) {
            return false;
        }
        const double* const values = static_cast<const double*>(entry.measurement_data);
        for (size_t k = 0; k < api_odometry::sample_values; ++k) {
            if (!math::isfinite(values[k])) {
                return false;
            }
        }
        any = true;
        latest = entry.timestamp;
    }
    return true;
}

bool api_odometry::is_dispatching() const {
    return this->dispatching;
}

void api_odometry::start(slam& system, const bool single_camera, const mapping::rig& cameras) {
    this->started = true;
    mapping::odometry::settings settings;
    if (!api_odometry::settings_of(this->parameters, this->sensor_id, settings)) {
        return;
    }
    static_cast<void>(system.reconstruction.wheel_odometry.configure(settings));
    // A rig of several cameras is set with the odometer when its first frame comes; one camera needs its rig set here.
    if (single_camera) {
        system.set_rig(this->with_odometer(cameras));
    }
}

void api_odometry::note_frame(slam& system, const long long timestamp) const {
    if (this->declared) {
        system.reconstruction.wheel_odometry.set_frame_time(system.reconstruction.next_frame_id, timestamp);
    }
}

bool api_odometry::fill_calibration(const mapping::map& map, zeroslam_odometry_calibration_struct* const calibration) const {
    if (!this->declared || (calibration == nullptr)) {
        return false;
    }
    zeroslam_odometry_calibration_struct result = {};
    result.sensor_id = this->sensor_id;
    mapping::odometry::settings settings;
    static_cast<void>(api_odometry::settings_of(this->parameters, this->sensor_id, settings));
    const size_t count = settings.kinematics.parameter_count();
    result.parameters_length = static_cast<int>(count);
    const double* const values = map.wheel_odometry.is_enabled() ? map.wheel_odometry.calibration() : &this->parameters.parameters[0];
    for (size_t i = 0; i < count; ++i) {
        result.parameters[i] = values[i];
    }
    math::se3<double> body_from_odometry = this->with_odometer(mapping::rig()).at(0).body_from_sensor;
    size_t index = 0;
    if (map.sensor_rig.find(this->sensor_id, index)) {
        body_from_odometry = map.sensor_rig.at(index).body_from_sensor;
    }
    const math::matrix<double, 4, 1> quaternion = body_from_odometry.rotation().get_quaternion();
    result.pose[0] = body_from_odometry.translation()[0];
    result.pose[1] = body_from_odometry.translation()[1];
    result.pose[2] = body_from_odometry.translation()[2];
    result.pose[3] = quaternion[1];
    result.pose[4] = quaternion[2];
    result.pose[5] = quaternion[3];
    result.pose[6] = quaternion[0];
    *calibration = result;
    return true;
}
