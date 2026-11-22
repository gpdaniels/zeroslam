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

#include "mapping/calibration.hpp"

#include "math/math.hpp"
#include "sensor/camera/parameter_role.hpp"

namespace {
    template <typename camera_type>
    std::vector<double> parameters_of(const camera_type& camera) {
        std::vector<double> parameters(camera.get_parameter_count(), 0.0);
        if (!parameters.empty() && !camera.get_parameters(parameters.data(), parameters.size())) {
            parameters.clear();
        }
        return parameters;
    }
}

namespace mapping {
    void calibration::set_level(const level value) {
        this->refined_level = value;
    }

    calibration::level calibration::get_level() const {
        return this->refined_level;
    }

    const sensor::model& calibration::camera_for(const sensor::model& given) {
        if (!this->given_valid || (parameters_of(given) != parameters_of(this->given_camera))) {
            this->given_camera = given;
            this->current_camera = given;
            this->given_valid = true;
            this->refinement_count = 0;
            this->started = false;
        }
        return (this->refined_level == level::off) ? given : this->current_camera;
    }

    const sensor::model& calibration::get_camera() const {
        return this->current_camera;
    }

    const sensor::model& calibration::get_given_camera() const {
        return this->given_camera;
    }

    size_t calibration::get_refinements() const {
        return this->refinement_count;
    }

    bool calibration::has_started() const {
        return this->started;
    }

    bool calibration::ready(const size_t keyframes, const double view_spread_degrees) const {
        if (this->refined_level == level::off) {
            return false;
        }
        return this->started || ((keyframes >= calibration::keyframes_minimum) && (view_spread_degrees >= calibration::view_spread_degrees_minimum));
    }

    double calibration::view_spread_degrees(const std::vector<math::matrix<double, 3, 3>>& rotations) {
        if (rotations.size() < 2) {
            return 0.0;
        }
        // A world to camera rotation's last row is the camera's optical axis in the world.
        math::matrix<double, 3, 1> mean = math::matrix<double, 3, 1>::zero();
        for (const math::matrix<double, 3, 3>& rotation : rotations) {
            for (size_t axis = 0; axis < 3; ++axis) {
                mean[axis] += rotation[2][axis];
            }
        }
        const double length = math::sqrt(mean.get_length_squared());
        if (!(length > 1.0e-12)) {
            return 180.0;
        }
        double smallest_cosine = 1.0;
        for (const math::matrix<double, 3, 3>& rotation : rotations) {
            const double cosine = ((rotation[2][0] * mean[0]) + (rotation[2][1] * mean[1]) + (rotation[2][2] * mean[2])) / length;
            smallest_cosine = math::min(smallest_cosine, cosine);
        }
        return math::acos(math::max(-1.0, math::min(1.0, smallest_cosine))) * 180.0 / math::pi<double>();
    }

    bool calibration::free_parameters(const sensor::camera::model<double>& camera, std::vector<int>& free) const {
        const size_t count = camera.get_parameter_count();
        free.assign(count, -1);
        if ((this->refined_level == level::off) || (count == 0)) {
            return false;
        }
        std::vector<sensor::camera::parameter_role> roles(count, sensor::camera::parameter_role::unknown);
        const bool known = camera.get_parameter_roles(roles.data(), count);
        if (!known && (this->refined_level != level::all)) {
            return false;
        }
        int next = 0;
        int focal_local = -1;
        for (size_t index = 0; index < count; ++index) {
            const sensor::camera::parameter_role role = roles[index];
            bool chosen = (this->refined_level == level::all) || (role == sensor::camera::parameter_role::focal_length);
            chosen = chosen || ((this->refined_level != level::focal) && (role == sensor::camera::parameter_role::principal_point));
            chosen = chosen || ((this->refined_level == level::intrinsics) && (role == sensor::camera::parameter_role::distortion));
            if (!chosen) {
                continue;
            }
            // Below all, the focal lengths share one local dimension.
            if ((role == sensor::camera::parameter_role::focal_length) && (this->refined_level != level::all)) {
                if (focal_local < 0) {
                    focal_local = next++;
                }
                free[index] = focal_local;
                continue;
            }
            free[index] = next++;
        }
        return next > 0;
    }

    bool calibration::make_refinement(refinement& request) const {
        request.refined = false;
        request.parameters.clear();
        request.camera = sensor::camera::model<double>(this->current_camera);
        if (!this->free_parameters(request.camera, request.free_parameters)) {
            return false;
        }
        const size_t count = request.free_parameters.size();
        const sensor::camera::model<double> given(this->given_camera);
        request.prior_values = parameters_of(given);
        if (request.prior_values.size() != count) {
            return false;
        }
        std::vector<sensor::camera::parameter_role> roles(count, sensor::camera::parameter_role::unknown);
        if (!given.get_parameter_roles(roles.data(), count)) {
            roles.assign(count, sensor::camera::parameter_role::unknown);
        }
        double focal_sum = 0.0;
        size_t focal_count = 0;
        for (size_t index = 0; index < count; ++index) {
            if (roles[index] == sensor::camera::parameter_role::focal_length) {
                focal_sum += math::abs(request.prior_values[index]);
                ++focal_count;
            }
        }
        const double focal_mean = (focal_count > 0) ? (focal_sum / static_cast<double>(focal_count)) : 0.0;
        request.prior_sigmas.assign(count, 0.0);
        for (size_t index = 0; index < count; ++index) {
            switch (roles[index]) {
                case sensor::camera::parameter_role::focal_length:
                    request.prior_sigmas[index] = calibration::focal_length_prior_fraction * math::abs(request.prior_values[index]);
                    break;
                case sensor::camera::parameter_role::principal_point:
                    request.prior_sigmas[index] = calibration::principal_point_prior_fraction * focal_mean;
                    break;
                case sensor::camera::parameter_role::distortion:
                    request.prior_sigmas[index] = calibration::distortion_prior_sigma;
                    break;
                case sensor::camera::parameter_role::distortion_higher:
                    request.prior_sigmas[index] = calibration::distortion_higher_prior_sigma;
                    break;
                case sensor::camera::parameter_role::unknown:
                    break;
            }
        }
        return true;
    }

    void calibration::update(const refinement& request) {
        if (!request.refined || (request.parameters.size() != this->current_camera.get_parameter_count())) {
            return;
        }
        if (this->current_camera.set_parameters(request.parameters.data(), request.parameters.size())) {
            ++this->refinement_count;
            this->started = true;
        }
    }
}
