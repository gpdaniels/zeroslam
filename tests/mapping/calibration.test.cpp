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

#include "math/lie.hpp"
#include "sensor/camera.hpp"
#include "sensor/camera/model.hpp"
#include "sensor/camera/pinhole.hpp"
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

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-9) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

static std::vector<double> parameters_of(const sensor::model& camera) {
    std::vector<double> parameters(sensor::camera::pinhole_radial_tangential<double>::parameter_count, 0.0);
    REQUIRE(camera.get_parameters(parameters.data(), parameters.size()));
    return parameters;
}

// A camera model that does not say what its parameters are.
class anonymous_model final {
public:
    constexpr static const char* name = "anonymous";

    size_t get_parameter_count() const {
        return 2;
    }

    bool set_parameters(const double* const parameters, const size_t parameters_length) {
        return (parameters != nullptr) && (parameters_length == 2);
    }

    bool get_parameters(double* const parameters, const size_t parameters_length) const {
        if ((parameters == nullptr) || (parameters_length != 2)) {
            return false;
        }
        parameters[0] = 1.0;
        parameters[1] = 2.0;
        return true;
    }

    bool get_pinhole(double* const pinhole, const size_t pinhole_length) const {
        if ((pinhole == nullptr) || (pinhole_length != 4)) {
            return false;
        }
        pinhole[0] = 1.0;
        pinhole[1] = 1.0;
        pinhole[2] = 0.0;
        pinhole[3] = 0.0;
        return true;
    }

    bool project(const double* const point_xyz, double* const point_xy, double* const jacobian_projection, double* const jacobian_parameters) const {
        static_cast<void>(jacobian_projection);
        static_cast<void>(jacobian_parameters);
        point_xy[0] = point_xyz[0] / point_xyz[2];
        point_xy[1] = point_xyz[1] / point_xyz[2];
        return true;
    }

    bool unproject(const double* const point_xy, double* const ray_xyz, double* const jacobian_unprojection) const {
        static_cast<void>(jacobian_unprojection);
        ray_xyz[0] = point_xy[0];
        ray_xyz[1] = point_xy[1];
        ray_xyz[2] = 1.0;
        return true;
    }
};

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const double given_parameters[sensor::camera::pinhole_radial_tangential<double>::parameter_count] = { 500.0, 504.0, 320.0, 240.0, -0.1, 0.02, 0.001, -0.001, 0.003, 0.0, 0.0, 0.0 };
    const sensor::model given(sensor::camera::pinhole_radial_tangential<double>(&given_parameters[0], sensor::camera::pinhole_radial_tangential<double>::parameter_count));

    {
        // Off, each frame takes the camera it is given, and nothing is refined.
        mapping::calibration calibration;
        REQUIRE(calibration.get_level() == mapping::calibration::level::off);
        REQUIRE(parameters_of(calibration.camera_for(given)) == parameters_of(given));
        REQUIRE(parameters_of(calibration.get_camera()) == parameters_of(given));
        REQUIRE(parameters_of(calibration.get_given_camera()) == parameters_of(given));
        mapping::calibration::refinement request;
        REQUIRE(!calibration.make_refinement(request));
        REQUIRE(!calibration.ready(100, 90.0));
        REQUIRE(!calibration.has_started());
    }

    {
        // The parameters each level frees, by what the model says they are, the focal lengths moving together below all.
        mapping::calibration calibration;
        const sensor::camera::model<double> camera(given);
        std::vector<int> free;
        const int expected[5][12] = {
            { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
            { 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
            { 0, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1 },
            { 0, 0, 1, 2, 3, 4, 5, 6, -1, -1, -1, -1 },
            { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }
        };
        for (int level = 0; level < 5; ++level) {
            calibration.set_level(static_cast<mapping::calibration::level>(level));
            REQUIRE(calibration.free_parameters(camera, free) == (level > 0));
            REQUIRE(free.size() == 12);
            for (size_t index = 0; index < 12; ++index) {
                REQUIRE(free[index] == expected[level][index]);
            }
        }
        // A model that does not say what its parameters are can only be refined whole.
        const sensor::camera::model<double> anonymous{ anonymous_model() };
        calibration.set_level(mapping::calibration::level::focal_centre);
        REQUIRE(!calibration.free_parameters(anonymous, free));
        calibration.set_level(mapping::calibration::level::all);
        REQUIRE(calibration.free_parameters(anonymous, free));
        REQUIRE((free.size() == 2) && (free[0] == 0) && (free[1] == 1));
    }

    {
        // A request names the current camera with a prior on the given calibration for each free parameter.
        mapping::calibration calibration;
        calibration.set_level(mapping::calibration::level::intrinsics);
        REQUIRE(parameters_of(calibration.camera_for(given)) == parameters_of(given));
        mapping::calibration::refinement request;
        REQUIRE(calibration.make_refinement(request));
        REQUIRE(!request.refined);
        REQUIRE(request.prior_values.size() == 12);
        REQUIRE(request.prior_sigmas.size() == 12);
        for (size_t index = 0; index < 12; ++index) {
            REQUIRE(request.prior_values[index] == given_parameters[index]);
        }
        REQUIRE(is_value_approx(request.prior_sigmas[0], mapping::calibration::focal_length_prior_fraction * 500.0));
        REQUIRE(is_value_approx(request.prior_sigmas[1], mapping::calibration::focal_length_prior_fraction * 504.0));
        REQUIRE(is_value_approx(request.prior_sigmas[2], mapping::calibration::principal_point_prior_fraction * 502.0));
        REQUIRE(is_value_approx(request.prior_sigmas[3], mapping::calibration::principal_point_prior_fraction * 502.0));
        REQUIRE(request.prior_sigmas[4] == mapping::calibration::distortion_prior_sigma);
        REQUIRE(request.prior_sigmas[8] == mapping::calibration::distortion_higher_prior_sigma);
        double requested[12] = {};
        REQUIRE(request.camera.get_parameters(&requested[0], 12));
        REQUIRE(requested[0] == 500.0);

        // The refined camera is what the frames take from then on, and what the next request starts from, its prior still
        // on the given calibration.
        request.refined = true;
        request.parameters = { 525.0, 528.0, 318.0, 242.0, -0.12, 0.025, 0.0, 0.0, 0.003, 0.0, 0.0, 0.0 };
        calibration.update(request);
        REQUIRE(calibration.get_refinements() == 1);
        REQUIRE(calibration.has_started());
        REQUIRE(parameters_of(calibration.get_camera()) == request.parameters);
        REQUIRE(parameters_of(calibration.camera_for(given)) == request.parameters);
        REQUIRE(parameters_of(calibration.get_given_camera()) == parameters_of(given));
        mapping::calibration::refinement next;
        REQUIRE(calibration.make_refinement(next));
        REQUIRE(next.camera.get_parameters(&requested[0], 12));
        REQUIRE(requested[0] == 525.0);
        REQUIRE(next.prior_values[0] == 500.0);

        // A refinement that was not kept changes nothing, nor does one of the wrong size.
        mapping::calibration::refinement dropped = next;
        dropped.refined = false;
        dropped.parameters = std::vector<double>(12, 1.0);
        calibration.update(dropped);
        dropped.refined = true;
        dropped.parameters = std::vector<double>(4, 1.0);
        calibration.update(dropped);
        REQUIRE(calibration.get_refinements() == 1);
        REQUIRE(calibration.get_camera().get_parameters(&requested[0], 12));
        REQUIRE(requested[0] == 525.0);

        // Another camera given is a new calibration, which starts again from it.
        const double other_parameters[sensor::camera::pinhole_radial_tangential<double>::parameter_count] = { 400.0, 400.0, 300.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        const sensor::model other(sensor::camera::pinhole_radial_tangential<double>(&other_parameters[0], sensor::camera::pinhole_radial_tangential<double>::parameter_count));
        REQUIRE(parameters_of(calibration.camera_for(other)) == parameters_of(other));
        REQUIRE(calibration.get_refinements() == 0);
        REQUIRE(!calibration.has_started());
        REQUIRE(parameters_of(calibration.get_given_camera()) == parameters_of(other));
    }

    {
        // The refinement waits for enough keyframes spanning enough of a turn, and once started carries on.
        mapping::calibration calibration;
        calibration.set_level(mapping::calibration::level::focal);
        static_cast<void>(calibration.camera_for(given));
        REQUIRE(!calibration.ready(mapping::calibration::keyframes_minimum - 1, 90.0));
        REQUIRE(!calibration.ready(1000, 0.5 * mapping::calibration::view_spread_degrees_minimum));
        REQUIRE(calibration.ready(mapping::calibration::keyframes_minimum, mapping::calibration::view_spread_degrees_minimum));
        mapping::calibration::refinement request;
        REQUIRE(calibration.make_refinement(request));
        request.refined = true;
        request.parameters.assign(&given_parameters[0], &given_parameters[0] + 12);
        request.parameters[0] = 510.0;
        calibration.update(request);
        REQUIRE(calibration.ready(2, 0.0));
    }

    {
        // The spread of the views is the largest angle of an optical axis from their mean.
        std::vector<math::matrix<double, 3, 3>> rotations;
        REQUIRE(mapping::calibration::view_spread_degrees(rotations) == 0.0);
        rotations.push_back(math::matrix<double, 3, 3>::identity());
        REQUIRE(mapping::calibration::view_spread_degrees(rotations) == 0.0);
        rotations.push_back(math::matrix<double, 3, 3>::identity());
        REQUIRE(is_value_approx(mapping::calibration::view_spread_degrees(rotations), 0.0, 1e-6));
        const double radians = 20.0 * 3.14159265358979323846 / 180.0;
        rotations[0] = math::so3<double>::exp({ { 0.0, radians, 0.0 } }).get_matrix();
        rotations[1] = math::matrix<double, 3, 3>::identity();
        REQUIRE(is_value_approx(mapping::calibration::view_spread_degrees(rotations), 10.0, 1e-9));
        // A turn about the optical axis does not move it.
        rotations[0] = math::so3<double>::exp({ { 0.0, 0.0, 1.0 } }).get_matrix();
        REQUIRE(is_value_approx(mapping::calibration::view_spread_degrees(rotations), 0.0, 1e-6));
    }

    {
        // Any model whose parameters say what they are: a pinhole, refined in focal length only.
        const double pinhole_parameters[4] = { 300.0, 310.0, 160.0, 120.0 };
        const sensor::camera::model<double> pinhole{ sensor::camera::pinhole<double>(&pinhole_parameters[0], 4) };
        mapping::calibration calibration;
        calibration.set_level(mapping::calibration::level::focal);
        std::vector<int> free;
        REQUIRE(calibration.free_parameters(pinhole, free));
        REQUIRE(free.size() == 4);
        REQUIRE((free[0] == 0) && (free[1] == 0) && (free[2] == -1) && (free[3] == -1));
    }

    return EXIT_SUCCESS;
}
