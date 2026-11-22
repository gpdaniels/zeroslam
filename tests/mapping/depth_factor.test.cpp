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

#include "mapping/depth_factor.hpp"

#include "mapping/map.hpp"
#include "math/lie.hpp"
#include "sensor/camera.hpp"
#include "sensor/depth.hpp"
#include "simulation/scene.hpp"

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

namespace {
    constexpr static const size_t rows = 48;
    constexpr static const size_t columns = 64;

    // A frame of a 64x48 pinhole camera looking at a wall at the given axial depth, posed at the given pose.
    mapping::frame wall_frame(const math::se3<double>& pose, const float wall_depth) {
        mapping::frame frame;
        frame.id = 0;
        frame.rotation = pose.rotation().get_matrix();
        frame.translation = pose.translation();
        const double parameters[sensor::camera::pinhole_radial_tangential<double>::parameter_count] = { 50.0, 50.0, 32.0, 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        frame.camera = sensor::model(sensor::camera::pinhole_radial_tangential<double>(&parameters[0], sensor::camera::pinhole_radial_tangential<double>::parameter_count));
        const std::vector<float> metres(rows * columns, wall_depth);
        sensor::depth::settings settings;
        settings.noise.quadratic = 1.0e-3;
        frame.depth = sensor::depth(rows, columns, metres.data(), settings);
        return frame;
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const math::se3<double> pose(math::so3<double>::exp({ { 0.05, -0.1, 0.02 } }), { { 0.3, -0.2, 0.1 } });
    const mapping::frame frame = wall_frame(pose, 2.0f);

    {
        // The sample is the depth image's, zero outside it.
        REQUIRE(mapping::depth_factor::sample(frame, 10.5, 20.5) == 2.0);
        REQUIRE(mapping::depth_factor::sample(frame, -3.0, 20.5) == 0.0);
        const mapping::frame bare;
        REQUIRE(mapping::depth_factor::sample(bare, 10.5, 20.5) == 0.0);
    }

    {
        // A depth puts a landmark on the pixel's ray at that depth, where the frame projects it back and predicts the depth.
        math::matrix<double, 3, 1> world;
        double axial = 0.0;
        REQUIRE(mapping::depth_factor::location(frame, 10.5, 20.5, 2.0, world, axial));
        REQUIRE(std::abs(axial - 2.0) < 1.0e-12);
        mapping::point landmark(0, world, math::matrix<double, 3, 1>::zero());
        math::matrix<double, 2, 1> pixel;
        REQUIRE(mapping::map::project_landmark(frame, landmark, pixel));
        REQUIRE(std::abs(pixel[0] - 10.5) < 1.0e-9);
        REQUIRE(std::abs(pixel[1] - 20.5) < 1.0e-9);
        double predicted = 0.0;
        REQUIRE(mapping::depth_factor::predicted(frame, landmark, predicted));
        REQUIRE(std::abs(predicted - 2.0) < 1.0e-9);
        REQUIRE(mapping::depth_factor::agrees(frame, landmark, 2.0));
        // Off by many standard deviations (sigma 4 mm at 2 m) it does not agree.
        REQUIRE(!mapping::depth_factor::agrees(frame, landmark, 2.05));
        // Anchored as an inverse depth it predicts the same.
        mapping::point anchored = landmark;
        REQUIRE(anchored.anchor(frame.rotation, frame.translation));
        REQUIRE(mapping::depth_factor::predicted(frame, anchored, predicted));
        REQUIRE(std::abs(predicted - 2.0) < 1.0e-9);

        // The factor's residual is zero at the measured depth, for either parameterisation.
        for (const mapping::point* const held : { &landmark, &anchored }) {
            optimisation::edge factor = mapping::depth_factor::make_edge(frame, *held, 2.0);
            optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
            const double pose_parameters[7] = { pose.translation()[0], pose.translation()[1], pose.translation()[2], pose.rotation().get_quaternion()[1], pose.rotation().get_quaternion()[2], pose.rotation().get_quaternion()[3], pose.rotation().get_quaternion()[0] };
            REQUIRE(pose_vertex.set_parameters(&pose_parameters[0], 7));
            optimisation::vertex point_vertex{ optimisation::vertices::point() };
            REQUIRE(point_vertex.set_parameters(held->inverse_depth ? held->inverse_parameters.data() : held->location.data(), 3));
            factor.add_vertex(&pose_vertex);
            factor.add_vertex(&point_vertex);
            factor.compute_residual();
            REQUIRE(std::abs(factor.get_residual()[0][0]) < 1.0e-9);
            // Weighted by the inverse depth's deviation, sigma(d) / d^2.
            const double sigma = 1.0e-3 * 2.0 * 2.0 / (2.0 * 2.0);
            REQUIRE(std::abs(factor.get_information()[0][0] - (1.0 / (sigma * sigma))) < 1.0e-6 * factor.get_information()[0][0]);
        }

        // The depth adds information along the ray, where one view has none: with the reprojection's information, which is
        // blind along the ray, the sum is invertible.
        math::matrix<double, 3, 3> information = math::matrix<double, 3, 3>::zero();
        mapping::depth_factor::add_information(frame, landmark, 2.0, information);
        const math::matrix<double, 3, 1> ray = world + (math::transpose(frame.rotation) * frame.translation);
        const math::matrix<double, 3, 1> along = ray * (1.0 / std::sqrt(ray.get_length_squared()));
        double along_information = 0.0;
        for (size_t a = 0; a < 3; ++a) {
            for (size_t b = 0; b < 3; ++b) {
                along_information += along[a] * information[a][b] * along[b];
            }
        }
        REQUIRE(along_information > 1.0e3);
    }

    {
        // A landmark seen once with its depth stays in the map after the outlier removal; a depth that disagrees goes, and a
        // landmark seen once is then held by nothing and goes with it. In a map without metric scale nothing samples a depth.
        mapping::map reconstruction;
        reconstruction.metric_scale = true;
        reconstruction.add_frame(frame);
        math::matrix<double, 3, 1> world;
        double axial = 0.0;
        REQUIRE(mapping::depth_factor::location(frame, 30.5, 22.5, 2.0, world, axial));
        const mapping::point landmark(7, world, math::matrix<double, 3, 1>::zero());
        reconstruction.add_landmark(landmark);
        reconstruction.add_observation(0, landmark, 30.5, 22.5);
        REQUIRE(reconstruction.observations.at(7).front().depth == 2.0);
        REQUIRE(reconstruction.remove_outlier_observations({ 7 }, { 0 }) == 0);
        REQUIRE(reconstruction.landmarks.count(7) == 1);
        REQUIRE(reconstruction.observations.at(7).front().depth == 2.0);
        reconstruction.observations.at(7).front().depth = 2.2;
        REQUIRE(reconstruction.remove_outlier_observations({ 7 }, { 0 }) == 0);
        REQUIRE(reconstruction.landmarks.count(7) == 0);

        mapping::map monocular;
        monocular.add_frame(frame);
        monocular.add_landmark(landmark);
        monocular.add_observation(0, landmark, 30.5, 22.5);
        REQUIRE(monocular.observations.at(7).front().depth == 0.0);
        // A landmark seen once without a depth is not held by anything, and goes.
        monocular.remove_outlier_observations({ 7 }, { 0 });
        REQUIRE(monocular.landmarks.count(7) == 0);
    }

    {
        // Two frames of the simulated orbit, its points seen with their exact depths: the adjustment of a metric map holds
        // no baseline, and the depths alone give the scale, so landmarks started at half their depth and the second frame
        // started at half its baseline come back to the truth in metres.
        simulation::options options;
        options.frame_count = 2;
        options.orbit_arc_radians = 0.2;
        options.point_count = 120;
        const simulation::scene scene = simulation::scene::generate(options);
        mapping::map reconstruction;
        reconstruction.metric_scale = true;
        const double parameters[sensor::camera::pinhole_radial_tangential<double>::parameter_count] = { options.focal_x, options.focal_y, options.centre_x, options.centre_y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        sensor::depth::settings settings;
        settings.noise.quadratic = 1.0e-3;
        for (int index = 0; index < options.frame_count; ++index) {
            mapping::frame posed;
            posed.id = index;
            posed.camera = sensor::model(sensor::camera::pinhole_radial_tangential<double>(&parameters[0], sensor::camera::pinhole_radial_tangential<double>::parameter_count));
            const math::se3<double> relative = scene.poses[static_cast<size_t>(index)] * scene.poses[0].inverse();
            const math::se3<double> started = (index == 0) ? relative : math::se3<double>(relative.rotation(), relative.translation() * 0.5);
            posed.rotation = started.rotation().get_matrix();
            posed.translation = started.translation();
            // An empty depth image with the settings: the observations below carry their exact depths.
            posed.depth = sensor::depth(1, 1, std::vector<float>{ 0.0f }.data(), settings);
            reconstruction.add_frame(posed);
        }
        reconstruction.gauge_frame_id = 0;
        for (int point_index = 0; point_index < options.point_count; ++point_index) {
            const math::matrix<double, 3, 1> truth = scene.poses[0] * scene.points[static_cast<size_t>(point_index)];
            if (!(truth[2] > 0.5)) {
                continue;
            }
            mapping::point landmark(point_index, truth * 0.5, math::matrix<double, 3, 1>::zero());
            size_t seen = 0;
            for (const simulation::observation& observed : scene.observations_exact) {
                seen += (observed.point_index == point_index) ? 1u : 0u;
            }
            if (seen != 2) {
                continue;
            }
            reconstruction.add_landmark(landmark);
            for (const simulation::observation& observed : scene.observations_exact) {
                if (observed.point_index != point_index) {
                    continue;
                }
                reconstruction.add_observation(observed.frame_index, landmark, observed.x, observed.y);
                reconstruction.observations.at(point_index).back().depth = (scene.poses[static_cast<size_t>(observed.frame_index)] * scene.points[static_cast<size_t>(point_index)])[2];
            }
        }
        REQUIRE(reconstruction.landmarks.size() > 50);
        reconstruction.optimise(0, false, 50, true);
        const math::se3<double> relative = scene.poses[1] * scene.poses[0].inverse();
        const mapping::frame& second = reconstruction.frames.at(1);
        const double baseline = std::sqrt(relative.translation().get_length_squared());
        const double solved = std::sqrt(second.translation.get_length_squared());
        std::fprintf(stdout, "Metric adjustment: baseline %.6f m solved as %.6f m.\n", baseline, solved);
        REQUIRE(std::abs(solved - baseline) < 1.0e-4 * baseline);
        size_t checked = 0;
        for (const auto& [landmark_id, landmark] : reconstruction.landmarks) {
            const math::matrix<double, 3, 1> truth = scene.poses[0] * scene.points[static_cast<size_t>(landmark_id)];
            REQUIRE(std::sqrt((landmark.location - truth).get_length_squared()) < 1.0e-4 * std::sqrt(truth.get_length_squared()));
            ++checked;
        }
        REQUIRE(checked > 50);
    }

    return EXIT_SUCCESS;
}
