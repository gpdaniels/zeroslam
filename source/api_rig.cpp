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

#include "math/lie.hpp"
#include "math/math.hpp"

bool api_rig::cameras_supported(const size_t count) {
    return (count >= 1) && (count <= api_rig::cameras_maximum);
}

bool api_rig::set_extrinsics(const zeroslam_sensor_extrinsics_struct* const values, const int length, const std::vector<int>& sensor_ids) {
    std::vector<zeroslam_sensor_extrinsics_struct> accepted;
    for (int i = 0; i < length; ++i) {
        const zeroslam_sensor_extrinsics_struct& entry = values[i];
        bool known = false;
        for (const int sensor_id : sensor_ids) {
            known = known || (sensor_id == entry.sensor_id);
        }
        bool repeated = false;
        for (const zeroslam_sensor_extrinsics_struct& existing : accepted) {
            repeated = repeated || (existing.sensor_id == entry.sensor_id);
        }
        if (!known || repeated || ((entry.refinement != zeroslam_extrinsic_fixed) && (entry.refinement != zeroslam_extrinsic_refine))) {
            return false;
        }
        bool finite = true;
        for (const double value : entry.pose) {
            finite = finite && math::isfinite(value);
        }
        const double norm_squared = (entry.pose[3] * entry.pose[3]) + (entry.pose[4] * entry.pose[4]) + (entry.pose[5] * entry.pose[5]) + (entry.pose[6] * entry.pose[6]);
        if (!finite || !(norm_squared > 1.0e-12)) {
            return false;
        }
        accepted.push_back(entry);
    }
    this->extrinsics.swap(accepted);
    return true;
}

void api_rig::clear_extrinsics() {
    this->extrinsics.clear();
}

mapping::rig api_rig::make_rig(const std::vector<int>& sensor_ids) const {
    mapping::rig rig;
    for (const int sensor_id : sensor_ids) {
        math::se3<double> body_from_sensor = math::se3<double>::identity();
        bool refined = false;
        for (const zeroslam_sensor_extrinsics_struct& extrinsic : this->extrinsics) {
            if (extrinsic.sensor_id != sensor_id) {
                continue;
            }
            const double norm = math::sqrt((extrinsic.pose[3] * extrinsic.pose[3]) + (extrinsic.pose[4] * extrinsic.pose[4]) + (extrinsic.pose[5] * extrinsic.pose[5]) + (extrinsic.pose[6] * extrinsic.pose[6]));
            body_from_sensor = math::se3<double>(math::so3<double>(extrinsic.pose[6] / norm, extrinsic.pose[3] / norm, extrinsic.pose[4] / norm, extrinsic.pose[5] / norm), { { extrinsic.pose[0], extrinsic.pose[1], extrinsic.pose[2] } });
            refined = extrinsic.refinement == zeroslam_extrinsic_refine;
        }
        rig.add(sensor_id, mapping::rig::sensor_kind::camera, body_from_sensor, refined);
    }
    return rig;
}

void api_rig::fill_extrinsics(const mapping::rig& rig, zeroslam_sensor_extrinsics_struct* const values) {
    for (size_t camera = 0; camera < rig.camera_count(); ++camera) {
        const mapping::rig::sensor& sensor = rig.at(rig.camera(camera));
        const math::matrix<double, 4, 1> quaternion = sensor.body_from_sensor.rotation().get_quaternion();
        values[camera].sensor_id = sensor.id;
        values[camera].refinement = sensor.refined ? zeroslam_extrinsic_refine : zeroslam_extrinsic_fixed;
        for (size_t axis = 0; axis < 3; ++axis) {
            values[camera].pose[axis] = sensor.body_from_sensor.translation()[axis];
        }
        values[camera].pose[3] = quaternion[1];
        values[camera].pose[4] = quaternion[2];
        values[camera].pose[5] = quaternion[3];
        values[camera].pose[6] = quaternion[0];
    }
}

bool api_rig::accepts(const long long* const timestamps, const size_t* const cameras, const size_t count, const size_t camera_count, const bool any_data, const long long latest_timestamp) const {
    long long group_timestamp = (this->pending_count > 0) ? this->pending_timestamp : latest_timestamp;
    size_t group_count = this->pending_count;
    std::vector<unsigned char> group_present = this->pending_present;
    group_present.resize(camera_count, 0);
    bool any_group = any_data;
    for (size_t i = 0; i < count; ++i) {
        if (cameras[i] >= camera_count) {
            return false;
        }
        if ((group_count > 0) && (timestamps[i] == group_timestamp)) {
            if (group_present[cameras[i]] != 0) {
                return false;
            }
        }
        else {
            // A new frame, once the one before it is whole, and later than it.
            if ((group_count > 0) || (any_group && !(timestamps[i] > group_timestamp))) {
                return false;
            }
            group_timestamp = timestamps[i];
            group_present.assign(camera_count, 0);
        }
        group_present[cameras[i]] = 1;
        ++group_count;
        any_group = true;
        if (group_count == camera_count) {
            group_count = 0;
        }
    }
    return true;
}

bool api_rig::take(const long long timestamp, const size_t camera, const unsigned char* const pixels, const size_t length, const size_t camera_count) {
    if (this->pending_count == 0) {
        this->pending_present.assign(camera_count, 0);
        this->pending_pixels.resize(camera_count);
    }
    this->pending_timestamp = timestamp;
    this->pending_pixels[camera].assign(pixels, pixels + length);
    this->pending_present[camera] = 1;
    ++this->pending_count;
    if (this->pending_count < camera_count) {
        return false;
    }
    this->pending_count = 0;
    return true;
}

long long api_rig::timestamp() const {
    return this->pending_timestamp;
}

unsigned char* api_rig::pixels(const size_t camera) {
    return this->pending_pixels[camera].data();
}
