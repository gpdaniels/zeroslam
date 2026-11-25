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

#include "math/math.hpp"
#include "sensor/camera/parameter_role.hpp"

namespace mapping {
    void calibration_drift::set(const settings& value) {
        this->configuration = value;
    }

    const calibration_drift::settings& calibration_drift::get() const {
        return this->configuration;
    }

    long long calibration_drift::take_frame(const long long timestamp, std::vector<sensor::model>& cameras, std::vector<math::se3<double>>& extrinsics) {
        if (!this->configuration.enabled || cameras.empty() || (cameras.size() != extrinsics.size())) {
            return -1;
        }
        if (this->given_cameras.empty()) {
            const double seconds = math::max(calibration_drift::chunk_seconds_minimum, math::min(calibration_drift::chunk_seconds_maximum, this->configuration.chunk_seconds));
            this->given_cameras = cameras;
            this->origin = timestamp;
            this->chunk_duration = static_cast<long long>((seconds * 1.0e9) + 0.5);
        }
        if ((timestamp < this->origin) || (cameras.size() != this->given_cameras.size())) {
            return -1;
        }
        for (size_t camera = 0; camera < cameras.size(); ++camera) {
            if (!cameras[camera].is_equal(this->given_cameras[camera])) {
                return -1;
            }
        }
        const long long index = (timestamp - this->origin) / this->chunk_duration;
        if (this->held_chunks.empty()) {
            chunk first;
            first.index = index;
            for (size_t camera = 0; camera < cameras.size(); ++camera) {
                first.cameras.push_back(camera_state{ cameras[camera], extrinsics[camera], 0 });
            }
            this->held_chunks.push_back(first);
        }
        else if (this->held_chunks.back().index < index) {
            // A new chunk starts where the one before it is.
            chunk next = this->held_chunks.back();
            next.index = index;
            for (camera_state& state : next.cameras) {
                state.observations = 0;
            }
            this->held_chunks.push_back(next);
        }
        for (size_t held = this->held_chunks.size(); held > 0; --held) {
            const chunk& candidate = this->held_chunks[held - 1];
            if (candidate.index < index) {
                break;
            }
            if (candidate.index == index) {
                for (size_t camera = 0; camera < cameras.size(); ++camera) {
                    cameras[camera] = candidate.cameras[camera].camera;
                    extrinsics[camera] = candidate.cameras[camera].reference_from_camera;
                }
                return index;
            }
        }
        // A time before the newest chunk's whose own chunk no frame was in.
        return -1;
    }

    const std::vector<calibration_drift::chunk>& calibration_drift::chunks() const {
        return this->held_chunks;
    }

    const calibration_drift::chunk* calibration_drift::newest() const {
        return this->held_chunks.empty() ? nullptr : &this->held_chunks.back();
    }

    long long calibration_drift::chunk_start(const long long index) const {
        return this->origin + (index * this->chunk_duration);
    }

    long long calibration_drift::chunk_nanoseconds() const {
        return this->chunk_duration;
    }

    size_t calibration_drift::get_refinements() const {
        return this->refinement_count;
    }

    bool calibration_drift::ready(const size_t keyframes, const double view_spread_degrees) const {
        if (!this->configuration.enabled || this->held_chunks.empty()) {
            return false;
        }
        return (this->refinement_count > 0) || ((keyframes >= calibration::keyframes_minimum) && (view_spread_degrees >= calibration::view_spread_degrees_minimum));
    }

    bool calibration_drift::make_request(const calibration::level level, const bool extrinsics_held, const std::vector<double>& baselines, request& following) const {
        following = request();
        if (!this->configuration.enabled || this->held_chunks.empty()) {
            return false;
        }
        following.walk = this->configuration;
        // The length the chunks were cut at, whatever the setting has been changed to since.
        following.walk.chunk_seconds = static_cast<double>(this->chunk_duration) * 1.0e-9;
        following.focal_prior_fraction = (level == calibration::level::off) ? calibration_drift::focal_prior_fraction : calibration::focal_length_prior_fraction;
        const bool follow_extrinsics = !extrinsics_held && (this->configuration.rotation_walk > 0.0) && (this->configuration.translation_walk > 0.0);
        bool any = false;
        for (size_t index = 0; index < this->given_cameras.size(); ++index) {
            request::camera camera;
            camera.given = this->given_cameras[index];
            camera.follow_focal = (this->configuration.focal_walk > 0.0) && !calibration_drift::focal_parameters(camera.given).empty();
            // The reference camera is the gauge the others are posed on.
            camera.follow_extrinsic = (index > 0) && follow_extrinsics;
            camera.baseline = (index < baselines.size()) ? baselines[index] : 0.0;
            any = any || camera.follow_focal || camera.follow_extrinsic;
            following.cameras.push_back(camera);
        }
        following.chunks = this->held_chunks;
        return any;
    }

    void calibration_drift::update(const request& following) {
        if (!following.refined || following.chunks.empty()) {
            return;
        }
        size_t next = 0;
        for (chunk& held : this->held_chunks) {
            while ((next < following.chunks.size()) && (following.chunks[next].index < held.index)) {
                ++next;
            }
            const chunk* const after = (next < following.chunks.size()) ? &following.chunks[next] : nullptr;
            const chunk* const before = (next > 0) ? &following.chunks[next - 1] : nullptr;
            if ((after != nullptr) && (after->index == held.index)) {
                if (after->cameras.size() == held.cameras.size()) {
                    held.cameras = after->cameras;
                }
                continue;
            }
            const chunk* const nearest = (before != nullptr) ? before : after;
            if (nearest->cameras.size() != held.cameras.size()) {
                continue;
            }
            for (size_t camera = 0; camera < held.cameras.size(); ++camera) {
                if ((before != nullptr) && (after != nullptr) && (after->cameras.size() == held.cameras.size())) {
                    const double fraction = static_cast<double>(held.index - before->index) / static_cast<double>(after->index - before->index);
                    held.cameras[camera].camera = calibration_drift::camera_between(before->cameras[camera].camera, after->cameras[camera].camera, fraction);
                    held.cameras[camera].reference_from_camera = calibration_drift::extrinsic_between(before->cameras[camera].reference_from_camera, after->cameras[camera].reference_from_camera, fraction);
                }
                else {
                    held.cameras[camera].camera = nearest->cameras[camera].camera;
                    held.cameras[camera].reference_from_camera = nearest->cameras[camera].reference_from_camera;
                }
                held.cameras[camera].observations = 0;
            }
        }
        ++this->refinement_count;
    }

    std::vector<size_t> calibration_drift::focal_parameters(const sensor::model& camera) {
        std::vector<size_t> focal;
        const size_t count = camera.get_parameter_count();
        std::vector<sensor::camera::parameter_role> roles(count, sensor::camera::parameter_role::unknown);
        if ((count == 0) || !camera.get_parameter_roles(roles.data(), count)) {
            return focal;
        }
        for (size_t index = 0; index < count; ++index) {
            if (roles[index] == sensor::camera::parameter_role::focal_length) {
                focal.push_back(index);
            }
        }
        return focal;
    }

    sensor::model calibration_drift::camera_between(const sensor::model& before, const sensor::model& after, const double fraction) {
        const size_t count = before.get_parameter_count();
        std::vector<double> from(count, 0.0);
        std::vector<double> to(count, 0.0);
        sensor::model between = before;
        if ((count == 0) || (after.get_parameter_count() != count) || !before.get_parameters(from.data(), count) || !after.get_parameters(to.data(), count)) {
            return between;
        }
        for (size_t index = 0; index < count; ++index) {
            from[index] += fraction * (to[index] - from[index]);
        }
        if (!between.set_parameters(from.data(), count)) {
            return before;
        }
        return between;
    }

    math::se3<double> calibration_drift::extrinsic_between(const math::se3<double>& before, const math::se3<double>& after, const double fraction) {
        const math::matrix<double, 3, 1> turn = (before.rotation().inverse() * after.rotation()).log();
        return math::se3<double>(before.rotation() * math::so3<double>::exp(turn * fraction), before.translation() + ((after.translation() - before.translation()) * fraction));
    }
}
