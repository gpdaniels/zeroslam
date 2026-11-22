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

#include "optimisation/edges/odometry.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/calibration_prior.hpp"
#include "optimisation/edges/extrinsic_prior.hpp"
#include "optimisation/edges/reprojection.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/calibration.hpp"
#include "optimisation/vertices/extrinsic.hpp"
#include "optimisation/vertices/point.hpp"
#include "optimisation/vertices/pose.hpp"
#include "sensor/camera.hpp"
#include "sensor/odometry/ackermann.hpp"
#include "sensor/odometry/differential_drive.hpp"
#include "sensor/odometry/model.hpp"
#include "sensor/odometry/preintegration.hpp"
#include "simulation/ground_vehicle.hpp"
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

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-9) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

static optimisation::vertex make_pose(const math::se3<double>& pose, const bool fixed) {
    optimisation::vertex vertex{ optimisation::vertices::pose() };
    double parameters[7] = {};
    optimisation::edges::odometry::write_pose(pose, &parameters[0]);
    REQUIRE(vertex.set_parameters(&parameters[0], 7));
    vertex.set_fixed(fixed);
    return vertex;
}

// The rig's extrinsic vertex of the odometry frame, which holds it as sensor to body, from the body to odometry transform.
static optimisation::vertex make_extrinsic(const math::se3<double>& odometry_from_body, const bool fixed) {
    optimisation::vertex vertex{ optimisation::vertices::extrinsic() };
    double parameters[7] = {};
    optimisation::vertices::extrinsic::to_parameters(odometry_from_body.inverse(), parameters);
    REQUIRE(vertex.set_parameters(&parameters[0], 7));
    vertex.set_fixed(fixed);
    return vertex;
}

static optimisation::edge make_odometry(const sensor::odometry::preintegration& motion, optimisation::vertex* first, optimisation::vertex* second, optimisation::vertex* extrinsic, optimisation::vertex* calibration) {
    optimisation::edge edge{ optimisation::edges::odometry() };
    edge.set_observation(optimisation::edges::odometry::observation(motion));
    edge.set_information(optimisation::edges::odometry::information(motion));
    REQUIRE(edge.add_vertex(first));
    REQUIRE(edge.add_vertex(second));
    REQUIRE(edge.add_vertex(extrinsic));
    REQUIRE(edge.add_vertex(calibration));
    return edge;
}

// The camera, the body, looking forward along the odometry frame's x axis, pitched down and turned a little, 1.6 m forward of the
// rear axle, 0.15 m to its left and 1.3 m up: the body to odometry transform.
static math::se3<double> camera_extrinsic() {
    const math::so3<double> camera_axes(math::matrix<double, 3, 3>({ { 0.0, 0.0, 1.0 }, { -1.0, 0.0, 0.0 }, { 0.0, -1.0, 0.0 } }));
    return math::se3<double>(math::so3<double>::exp({ { 0.0, 0.08, 0.05 } }) * camera_axes, { { 1.6, 0.15, 1.3 } });
}

// The analytic Jacobians of an edge against central differences through each vertex's own update.
static void check_jacobians(optimisation::edge& edge, const double tolerance) {
    edge.compute_residual();
    edge.compute_jacobians();
    const std::vector<math::matrix<double, 0, 0>> analytic = edge.get_jacobians();
    const double step = 1e-6;
    for (size_t v = 0; v < edge.num_vertices(); ++v) {
        optimisation::vertex* const vertex = edge.get_vertex(static_cast<int>(v));
        const size_t parameters = static_cast<size_t>(vertex->get_parameter_dimensions());
        const size_t local = static_cast<size_t>(vertex->get_local_dimensions());
        REQUIRE(analytic[v].rows() == 6);
        REQUIRE(analytic[v].cols() == local);
        double backup[optimisation::vertex::maximum_parameters] = {};
        REQUIRE(vertex->get_parameters(&backup[0], parameters));
        for (size_t d = 0; d < local; ++d) {
            double delta[optimisation::vertex::maximum_parameters] = {};
            delta[d] = step;
            vertex->plus(&delta[0]);
            edge.compute_residual();
            const math::matrix<double, 0, 0> plus = edge.get_residual();
            REQUIRE(vertex->set_parameters(&backup[0], parameters));
            delta[d] = -step;
            vertex->plus(&delta[0]);
            edge.compute_residual();
            const math::matrix<double, 0, 0> minus = edge.get_residual();
            REQUIRE(vertex->set_parameters(&backup[0], parameters));
            for (size_t r = 0; r < 6; ++r) {
                const double numeric = (plus[r][0] - minus[r][0]) / (2.0 * step);
                REQUIRE(std::abs(analytic[v][r][d] - numeric) < tolerance * (1.0 + std::abs(numeric)));
            }
        }
    }
    edge.compute_residual();
}

// A drive seen by a camera on the vehicle: landmarks beside the road, keyframes along it, and the wheel odometry between them.
class drive final {
public:
    simulation::ground_vehicle vehicle;
    math::se3<double> extrinsic;
    sensor::pinhole camera;
    std::vector<long long int> times;
    std::vector<math::se3<double>> poses;
    std::vector<math::matrix<double, 3, 1>> points;
    std::vector<simulation::observation> observations;
};

static drive make_drive(const simulation::ground_vehicle_options& options, const double keyframe_period, const uint64_t seed) {
    drive result;
    result.vehicle = simulation::ground_vehicle::generate(options);
    result.extrinsic = camera_extrinsic();
    const double camera_parameters[4] = { 500.0, 500.0, 320.0, 240.0 };
    result.camera = sensor::pinhole(&camera_parameters[0], 4);
    simulation::random_stream random(seed);

    const long long int period = static_cast<long long int>(keyframe_period * 1.0e9);
    for (long long int time = options.start_time; time <= result.vehicle.end_time(); time += period) {
        result.times.push_back(time);
        result.poses.push_back((result.vehicle.pose_at(time) * result.extrinsic).inverse());
    }

    // Landmarks on both sides of the road ahead of each keyframe, from 4 m to 12 m out and from the ground to 4 m up.
    for (size_t k = 0; k < result.times.size(); ++k) {
        for (int i = 0; i < 10; ++i) {
            const long long int ahead = result.times[k] + static_cast<long long int>(random.uniform(1.0, 5.0) * 1.0e9);
            const math::se3<double> place = result.vehicle.pose_at(std::min(ahead, result.vehicle.end_time()));
            const double side = (random.uniform() < 0.5) ? -1.0 : 1.0;
            const math::matrix<double, 3, 1> offset = { { random.uniform(-2.0, 2.0), side * random.uniform(4.0, 12.0), random.uniform(0.0, 4.0) } };
            result.points.push_back(place * offset);
        }
    }

    // Each landmark in every keyframe that sees it, with half a pixel of noise.
    for (size_t k = 0; k < result.poses.size(); ++k) {
        for (size_t i = 0; i < result.points.size(); ++i) {
            const math::matrix<double, 3, 1> in_camera = result.poses[k] * result.points[i];
            if ((in_camera[2] < 1.0) || (in_camera[2] > 40.0)) {
                continue;
            }
            math::matrix<double, 2, 1> pixel;
            if (!result.camera.project(in_camera.data(), pixel.data()) || (pixel[0] < 0.5) || (pixel[0] > 639.5) || (pixel[1] < 0.5) || (pixel[1] > 479.5)) {
                continue;
            }
            simulation::observation observation;
            observation.frame_index = static_cast<int>(k);
            observation.point_index = static_cast<int>(i);
            observation.x = pixel[0] + (0.5 * random.gaussian());
            observation.y = pixel[1] + (0.5 * random.gaussian());
            observation.outlier = false;
            result.observations.push_back(observation);
        }
    }
    return result;
}

class adjustment_result final {
public:
    double calibration[sensor::odometry::model::maximum_parameters] = {};
    double scale = 0.0;
    double position_error = 0.0;
    double extrinsic_rotation_error = 0.0;
    double extrinsic_translation_error[3] = {};
    int reintegrations = 0;
    bool used_square_root = false;
};

// Bundle adjust a drive from a monocular start, its keyframes and landmarks shrunk about the first keyframe to the given scale and
// jittered, with the wheel odometry between the keyframes integrated from the measured samples with a wrong calibration, the
// extrinsic perturbed but for its unobservable height, which a prior holds. The odometry is integrated again whenever the
// calibration moves past what its first order correction holds, as the pipeline will.
static adjustment_result adjust(const drive& scene, const double start_scale, const double* const start_calibration, const unsigned int free_mask, const double* const calibration_deviations, const bool free_extrinsic, const optimisation::factor_graph::strategy strategy, const optimisation::factor_graph::precision precision) {
    const sensor::odometry::model& kinematics = scene.vehicle.options.kinematics;
    const size_t parameter_count = kinematics.parameter_count();
    simulation::random_stream random(5);
    optimisation::factor_graph graph;
    graph.set_strategy(strategy);
    graph.set_precision(precision);

    // The keyframes, the first fixed at the truth, the rest shrunk towards it and jittered by half a degree and a centimetre.
    const math::matrix<double, 3, 1> origin = scene.poses[0].inverse().translation();
    std::vector<optimisation::vertex*> keyframes;
    for (size_t k = 0; k < scene.poses.size(); ++k) {
        const math::se3<double> camera_to_world = scene.poses[k].inverse();
        math::se3<double> start(camera_to_world.rotation(), origin + (start_scale * (camera_to_world.translation() - origin)));
        if (k > 0) {
            const math::matrix<double, 3, 1> turn = { { 0.009 * random.gaussian(), 0.009 * random.gaussian(), 0.009 * random.gaussian() } };
            const math::matrix<double, 3, 1> shift = { { 0.01 * random.gaussian(), 0.01 * random.gaussian(), 0.01 * random.gaussian() } };
            start = math::se3<double>(math::so3<double>::exp(turn) * start.rotation(), start.translation() + shift);
        }
        keyframes.push_back(graph.add_vertex(make_pose(start.inverse(), k == 0)));
    }

    // The landmarks, shrunk the same way, marginalised.
    std::vector<optimisation::vertex*> landmarks(scene.points.size(), nullptr);
    std::vector<int> seen(scene.points.size(), 0);
    for (const simulation::observation& observation : scene.observations) {
        ++seen[static_cast<size_t>(observation.point_index)];
    }
    for (size_t i = 0; i < scene.points.size(); ++i) {
        if (seen[i] < 2) {
            continue;
        }
        const math::matrix<double, 3, 1> position = origin + (start_scale * (scene.points[i] - origin));
        optimisation::vertex vertex{ optimisation::vertices::point() };
        const double parameters[3] = { position[0] + (0.02 * random.gaussian()), position[1] + (0.02 * random.gaussian()), position[2] + (0.02 * random.gaussian()) };
        REQUIRE(vertex.set_parameters(&parameters[0], 3));
        vertex.set_marginalised(true);
        landmarks[i] = graph.add_vertex(static_cast<optimisation::vertex&&>(vertex));
    }
    for (const simulation::observation& observation : scene.observations) {
        optimisation::vertex* const landmark = landmarks[static_cast<size_t>(observation.point_index)];
        if (landmark == nullptr) {
            continue;
        }
        optimisation::edge edge{ optimisation::edges::reprojection(scene.camera) };
        const double pixel[2] = { observation.x, observation.y };
        edge.set_observation(math::matrix<double, 0, 0>(2, 1, &pixel[0]));
        edge.set_information(math::matrix<double, 0, 0>::identity(2, 2) * 4.0);
        REQUIRE(edge.add_vertex(keyframes[static_cast<size_t>(observation.frame_index)]));
        REQUIRE(edge.add_vertex(landmark));
        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(edge)) != nullptr);
    }

    // The extrinsic, turned by a few degrees and shifted 20 cm in the plane, with a tight prior on its height and loose ones on the
    // rest about where it starts.
    const math::se3<double> extrinsic_start(math::so3<double>::exp({ { 0.03, -0.02, 0.04 } }) * scene.extrinsic.rotation(), scene.extrinsic.translation() + math::matrix<double, 3, 1>({ 0.2, -0.15, 0.0 }));
    optimisation::vertex* const extrinsic = graph.add_vertex(make_extrinsic(free_extrinsic ? extrinsic_start : scene.extrinsic, !free_extrinsic));
    if (free_extrinsic) {
        optimisation::edge prior{ optimisation::edges::extrinsic_prior() };
        double parameters[7] = {};
        optimisation::vertices::extrinsic::to_parameters(extrinsic_start.inverse(), parameters);
        prior.set_observation(optimisation::edges::extrinsic_prior::observation(&parameters[0]));
        const double deviations[6] = { 0.2, 0.2, 0.2, 1.0, 1.0, 0.005 };
        prior.set_information(optimisation::edges::extrinsic_prior::information(deviations));
        REQUIRE(prior.add_vertex(extrinsic));
        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(prior)) != nullptr);
    }

    // The calibration and its prior about where it starts.
    optimisation::vertex* const calibration = graph.add_vertex(optimisation::vertices::calibration::make(start_calibration, parameter_count, free_mask));
    {
        optimisation::edge prior{ optimisation::edges::calibration_prior() };
        prior.set_observation(optimisation::edges::calibration_prior::observation(start_calibration, parameter_count));
        prior.set_information(optimisation::edges::calibration_prior::information(calibration_deviations, parameter_count));
        REQUIRE(prior.add_vertex(calibration));
        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(prior)) != nullptr);
    }

    // The odometry between consecutive keyframes from the measured samples.
    std::vector<sensor::odometry::preintegration> motions;
    std::vector<optimisation::edge*> odometry_edges;
    for (size_t k = 0; (k + 1) < scene.times.size(); ++k) {
        sensor::odometry::preintegration motion(kinematics, scene.vehicle.options.noise, start_calibration, parameter_count);
        REQUIRE(motion.integrate(scene.vehicle.samples.data(), scene.vehicle.samples.size(), scene.times[k], scene.times[k + 1]));
        motions.push_back(motion);
        odometry_edges.push_back(graph.add_edge(make_odometry(motion, keyframes[k], keyframes[k + 1], extrinsic, calibration)));
        REQUIRE(odometry_edges.back() != nullptr);
    }

    adjustment_result result;
    for (int round = 0; round < 6; ++round) {
        REQUIRE(graph.solve(50, true) > 0);
        result.used_square_root = graph.get_diagnostics().used_square_root;
        const double* const values = calibration->get_parameters();
        bool moved = false;
        for (size_t k = 0; k < motions.size(); ++k) {
            if (motions[k].needs_reintegration(values, parameter_count)) {
                REQUIRE(motions[k].reintegrate(values, parameter_count));
                odometry_edges[k]->set_observation(optimisation::edges::odometry::observation(motions[k]));
                odometry_edges[k]->set_information(optimisation::edges::odometry::information(motions[k]));
                moved = true;
            }
        }
        if (!moved) {
            break;
        }
        ++result.reintegrations;
    }

    for (size_t i = 0; i < parameter_count; ++i) {
        result.calibration[i] = calibration->get_parameters()[i];
    }
    // The scale of the estimate is the ratio of the distances of the keyframes from the first, and the position error the
    // largest distance of a keyframe from the truth.
    double estimated_length = 0.0;
    double true_length = 0.0;
    for (size_t k = 0; k < keyframes.size(); ++k) {
        const math::matrix<double, 3, 1> estimate = optimisation::edges::odometry::read_pose(keyframes[k]->get_parameters()).inverse().translation();
        const math::matrix<double, 3, 1> truth = scene.poses[k].inverse().translation();
        estimated_length += std::sqrt((estimate - origin).get_length_squared());
        true_length += std::sqrt((truth - origin).get_length_squared());
        result.position_error = std::max(result.position_error, std::sqrt((estimate - truth).get_length_squared()));
    }
    result.scale = estimated_length / true_length;
    const math::se3<double> extrinsic_estimate = optimisation::vertices::extrinsic::from_parameters(extrinsic->get_parameters()).inverse();
    result.extrinsic_rotation_error = std::sqrt((extrinsic_estimate.rotation() * scene.extrinsic.rotation().inverse()).log().get_length_squared());
    for (size_t i = 0; i < 3; ++i) {
        result.extrinsic_translation_error[i] = extrinsic_estimate.translation()[i] - scene.extrinsic.translation()[i];
    }
    return result;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::odometry) <= optimisation::edge::maximum_size, "The odometry edge must fit.");

    const math::se3<double> extrinsic = camera_extrinsic();

    // The helpers: the adjoint moves a perturbation across a transform, a lifted planar motion is a turn about z and a shift in
    // the plane, and the predicted pose is the one the motion leads to.
    {
        const math::se3<double> pose(math::so3<double>::exp({ { 0.3, -0.2, 0.9 } }), { { 1.0, -2.0, 0.5 } });
        const math::matrix<double, 6, 1> xi = { { 0.01, -0.02, 0.03, 0.1, 0.2, -0.3 } };
        const math::se3<double> left = pose * math::se3<double>::exp(xi) * pose.inverse();
        const math::se3<double> right = math::se3<double>::exp(optimisation::edges::odometry::adjoint(pose) * xi);
        REQUIRE((left.inverse() * right).log().get_length_squared() < 1e-24);
        const math::matrix<double, 3, 1> vector = { { 1.0, 2.0, 3.0 } };
        const math::matrix<double, 3, 1> other = { { -0.5, 0.25, 2.0 } };
        const math::matrix<double, 3, 1> cross = optimisation::edges::odometry::skew(vector) * other;
        REQUIRE(is_value_approx(cross[0], (2.0 * 2.0) - (3.0 * 0.25)));
        REQUIRE(is_value_approx(cross[1], (3.0 * -0.5) - (1.0 * 2.0)));
        REQUIRE(is_value_approx(cross[2], (1.0 * 0.25) - (2.0 * -0.5)));

        const math::matrix<double, 3, 1> planar = { { 2.0, 0.5, 0.3 } };
        const math::se3<double> lifted = optimisation::edges::odometry::lift(planar);
        REQUIRE(is_value_approx(lifted.translation()[0], 2.0) && is_value_approx(lifted.translation()[1], 0.5) && (lifted.translation()[2] == 0.0));
        REQUIRE(is_value_approx(lifted.rotation().log()[2], 0.3));
        const math::se3<double> first = (math::se3<double>(math::so3<double>::exp({ { 0.0, 0.0, 0.4 } }), { { 3.0, -2.0, 0.0 } }) * extrinsic).inverse();
        const math::se3<double> second = optimisation::edges::odometry::predict(first, extrinsic.inverse(), planar);
        const math::se3<double> motion = optimisation::edges::odometry::relative(first, second, extrinsic.inverse());
        REQUIRE((motion.inverse() * lifted).log().get_length_squared() < 1e-24);

        double parameters[7] = {};
        optimisation::edges::odometry::write_pose(pose, &parameters[0]);
        const math::se3<double> read = optimisation::edges::odometry::read_pose(&parameters[0]);
        REQUIRE((read.inverse() * pose).log().get_length_squared() < 1e-24);
    }

    // The observation carries the motion, its linearisation point and Jacobian, the information the inverse covariance with the
    // out of plane rows, and the measurement follows the calibration to first order.
    {
        simulation::ground_vehicle_options options;
        options.duration = 4.0;
        const simulation::ground_vehicle vehicle = simulation::ground_vehicle::generate(options);
        const double parameters[4] = { 0.3, 2.7, 1.0, 0.01 };
        sensor::odometry::preintegration motion(options.kinematics, options.noise, &parameters[0], 4);
        REQUIRE(motion.integrate(vehicle.samples.data(), vehicle.samples.size(), options.start_time, vehicle.end_time()));
        const math::matrix<double, 0, 0> observation = optimisation::edges::odometry::observation(motion);
        REQUIRE(observation.rows() == 3 + (4 * 4));
        REQUIRE(observation[2][0] == motion.get_delta()[2]);
        REQUIRE(observation[3][0] == 0.3);
        REQUIRE(observation[7 + 4 + 3][0] == motion.get_jacobian()[1][3]);

        const math::matrix<double, 0, 0> information = optimisation::edges::odometry::information(motion);
        REQUIRE((information.rows() == 6) && (information.cols() == 6));
        for (size_t r = 0; r < 6; ++r) {
            REQUIRE(information[r][r] > 0.0);
            for (size_t c = 0; c < 6; ++c) {
                REQUIRE(is_value_approx(information[r][c], information[c][r], 1e-12));
            }
        }
        REQUIRE((information[0][1] == 0.0) && (information[0][2] == 0.0) && (information[5][3] == 0.0));
        const double tilt = (options.noise.tilt * options.noise.tilt * motion.get_distance()) + (options.noise.minimum_rotation * options.noise.minimum_rotation);
        REQUIRE(is_value_approx(information[0][0], 1.0 / tilt));
        // The planar block times the covariance with its floors is the identity.
        const size_t order[3] = { 2, 0, 1 };
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                double sum = 0.0;
                for (size_t k = 0; k < 3; ++k) {
                    double covariance = motion.get_covariance()[order[k]][order[c]];
                    if (k == c) {
                        covariance += (k == 0) ? (options.noise.minimum_rotation * options.noise.minimum_rotation) : (options.noise.minimum_translation * options.noise.minimum_translation);
                    }
                    sum += information[2 + r][2 + k] * covariance;
                }
                REQUIRE(std::abs(sum - ((r == c) ? 1.0 : 0.0)) < 1e-9);
            }
        }

        optimisation::vertex first = make_pose(math::se3<double>::identity(), false);
        optimisation::vertex second = make_pose(math::se3<double>::identity(), false);
        optimisation::vertex extrinsic_vertex = make_extrinsic(extrinsic, false);
        const double moved[4] = { 0.301, 2.7, 1.0, 0.012 };
        optimisation::vertex calibration = optimisation::vertices::calibration::make(&moved[0], 4, 0xFu);
        optimisation::edge edge = make_odometry(motion, &first, &second, &extrinsic_vertex, &calibration);
        const math::matrix<double, 3, 1> measured = optimisation::edges::odometry::measurement(edge);
        const math::matrix<double, 3, 1> predicted = motion.predict(&moved[0], 4);
        for (size_t r = 0; r < 3; ++r) {
            REQUIRE(is_value_approx(measured[r], predicted[r], 1e-14));
        }
    }

    // Body poses that the extrinsic carries along the measured motion have no residual; poses off the plane, the calibration
    // away from the linearisation point and a partial mask, and the analytic Jacobians match central differences.
    {
        simulation::ground_vehicle_options options;
        options.duration = 3.0;
        options.curvature = 0.1;
        const simulation::ground_vehicle vehicle = simulation::ground_vehicle::generate(options);
        const double parameters[4] = { 0.3, 2.7, 1.0, 0.01 };
        sensor::odometry::preintegration motion(options.kinematics, options.noise, &parameters[0], 4);
        REQUIRE(motion.integrate(vehicle.samples.data(), vehicle.samples.size(), options.start_time, vehicle.end_time()));

        const math::se3<double> first_pose = (math::se3<double>(math::so3<double>::exp({ { 0.0, 0.0, 0.4 } }), { { 3.0, -2.0, 0.0 } }) * extrinsic).inverse();
        const math::se3<double> second_pose = optimisation::edges::odometry::predict(first_pose, extrinsic.inverse(), motion.get_delta());
        optimisation::vertex first = make_pose(first_pose, false);
        optimisation::vertex second = make_pose(second_pose, false);
        optimisation::vertex extrinsic_vertex = make_extrinsic(extrinsic, false);
        optimisation::vertex calibration = optimisation::vertices::calibration::make(&parameters[0], 4, 0xFu);
        optimisation::edge edge = make_odometry(motion, &first, &second, &extrinsic_vertex, &calibration);
        edge.compute_residual();
        for (size_t r = 0; r < 6; ++r) {
            REQUIRE(std::abs(edge.get_residual()[r][0]) < 1e-12);
        }
        check_jacobians(edge, 1e-6);

        // Off the plane and off the measurement.
        const math::se3<double> off = math::se3<double>(math::so3<double>::exp({ { 0.05, -0.04, 0.1 } }), { { 0.3, -0.2, 0.15 } }) * second_pose;
        REQUIRE(second.set_parameters(make_pose(off, false).get_parameters(), 7));
        const math::se3<double> turned(math::so3<double>::exp({ { 0.1, 0.05, -0.2 } }) * extrinsic.rotation(), extrinsic.translation() + math::matrix<double, 3, 1>({ 0.1, 0.2, -0.3 }));
        REQUIRE(extrinsic_vertex.set_parameters(make_extrinsic(turned, false).get_parameters(), 7));
        const double moved[4] = { 0.31, 2.65, 1.05, 0.03 };
        optimisation::vertex partial = optimisation::vertices::calibration::make(&moved[0], 4, 0xBu);
        optimisation::edge off_edge = make_odometry(motion, &first, &second, &extrinsic_vertex, &partial);
        off_edge.compute_residual();
        REQUIRE(off_edge.chi2() > 1.0);
        check_jacobians(off_edge, 1e-6);
        REQUIRE(off_edge.get_jacobians()[3].cols() == 3);

        // A held calibration vertex gives a Jacobian of zeros the graph does not read.
        optimisation::vertex held = optimisation::vertices::calibration::make(&moved[0], 4, 0u);
        optimisation::edge held_edge = make_odometry(motion, &first, &second, &extrinsic_vertex, &held);
        held_edge.compute_jacobians();
        REQUIRE(held_edge.get_jacobians()[3].cols() == 1);
        for (size_t r = 0; r < 6; ++r) {
            REQUIRE(held_edge.get_jacobians()[3][r][0] == 0.0);
        }

        // A differential drive's edge too.
        simulation::ground_vehicle_options drive_options;
        drive_options.duration = 3.0;
        drive_options.kinematics = sensor::odometry::model(sensor::odometry::differential_drive{});
        drive_options.parameters[0] = 0.1;
        drive_options.parameters[1] = 0.11;
        drive_options.parameters[2] = 0.5;
        const simulation::ground_vehicle robot = simulation::ground_vehicle::generate(drive_options);
        sensor::odometry::preintegration drive_motion(drive_options.kinematics, drive_options.noise, &drive_options.parameters[0], 3);
        REQUIRE(drive_motion.integrate(robot.samples.data(), robot.samples.size(), drive_options.start_time, robot.end_time()));
        const double drive_moved[3] = { 0.102, 0.108, 0.52 };
        optimisation::vertex drive_calibration = optimisation::vertices::calibration::make(&drive_moved[0], 3, 0x7u);
        optimisation::edge drive_edge = make_odometry(drive_motion, &first, &second, &extrinsic_vertex, &drive_calibration);
        check_jacobians(drive_edge, 1e-6);
    }

    // A drive bundle adjusted from a monocular start at half the scale, a wheel radius 6% small and no steering offset, with the
    // extrinsic perturbed: the odometry brings the map to metric scale and recovers the radius, the steering offset and the
    // extrinsic, its height held by its prior. The wheelbase and the steering gain are held: a turn shows only the ratio of the
    // distance rolled to the wheelbase, and for small angles the gain only with the radius.
    {
        simulation::ground_vehicle_options options;
        options.duration = 40.0;
        options.parameters[sensor::odometry::ackermann::radius] = 0.32;
        options.parameters[sensor::odometry::ackermann::steering_offset] = 0.03;
        const drive scene = make_drive(options, 0.5, 11);
        std::fprintf(stderr, "Ackermann drive: %zu keyframes, %zu landmarks, %zu observations, %.1f m\n", scene.poses.size(), scene.points.size(), scene.observations.size(), scene.vehicle.truth_distance.back());
        const double start[4] = { 0.30, 2.7, 1.0, 0.0 };
        const double deviations[4] = { 0.05, 0.0, 0.0, 0.1 };
        const unsigned int mask = (1u << sensor::odometry::ackermann::radius) | (1u << sensor::odometry::ackermann::steering_offset);

        const adjustment_result result = adjust(scene, 0.5, &start[0], mask, &deviations[0], true, optimisation::factor_graph::strategy::dense_schur, optimisation::factor_graph::precision::double_precision);
        std::fprintf(stderr, "Dense: radius %.5f, offset %.5f, scale %.5f, position error %.4f m, extrinsic %.4f rad (%.4f, %.4f, %.4f) m, %d reintegrations\n", result.calibration[0], result.calibration[3], result.scale, result.position_error, result.extrinsic_rotation_error, result.extrinsic_translation_error[0], result.extrinsic_translation_error[1], result.extrinsic_translation_error[2], result.reintegrations);
        REQUIRE(!result.used_square_root);
        REQUIRE(std::abs(result.calibration[0] - 0.32) < 0.002);
        REQUIRE(result.calibration[1] == 2.7);
        REQUIRE(result.calibration[2] == 1.0);
        REQUIRE(std::abs(result.calibration[3] - 0.03) < 0.003);
        REQUIRE(std::abs(result.scale - 1.0) < 0.01);
        REQUIRE(result.position_error < 0.5);
        REQUIRE(result.extrinsic_rotation_error < 0.01);
        REQUIRE(std::abs(result.extrinsic_translation_error[0]) < 0.05);
        REQUIRE(std::abs(result.extrinsic_translation_error[1]) < 0.05);
        REQUIRE(std::abs(result.extrinsic_translation_error[2]) < 0.01);
        REQUIRE(result.reintegrations > 0);

        // With the extrinsic known the same holds.
        const adjustment_result known = adjust(scene, 0.5, &start[0], mask, &deviations[0], false, optimisation::factor_graph::strategy::dense_schur, optimisation::factor_graph::precision::double_precision);
        REQUIRE(std::abs(known.calibration[0] - 0.32) < 0.002);
        REQUIRE(std::abs(known.calibration[3] - 0.03) < 0.003);
        REQUIRE(std::abs(known.scale - 1.0) < 0.01);
        REQUIRE(known.extrinsic_rotation_error == 0.0);
    }

    // The square root solver, in double and in single precision, takes the calibration as a Hessian block beside the landmark
    // blocks and comes to the dense solver's estimate, on a shorter drive as its conjugate gradients are slow on this one.
    {
        simulation::ground_vehicle_options options;
        options.duration = 16.0;
        options.parameters[sensor::odometry::ackermann::radius] = 0.32;
        options.parameters[sensor::odometry::ackermann::steering_offset] = 0.03;
        const drive scene = make_drive(options, 0.5, 13);
        const double start[4] = { 0.30, 2.7, 1.0, 0.0 };
        const double deviations[4] = { 0.05, 0.0, 0.0, 0.1 };
        const unsigned int mask = (1u << sensor::odometry::ackermann::radius) | (1u << sensor::odometry::ackermann::steering_offset);
        const adjustment_result dense = adjust(scene, 0.5, &start[0], mask, &deviations[0], true, optimisation::factor_graph::strategy::dense_schur, optimisation::factor_graph::precision::double_precision);
        const optimisation::factor_graph::precision precisions[2] = { optimisation::factor_graph::precision::double_precision, optimisation::factor_graph::precision::single_precision };
        for (const optimisation::factor_graph::precision precision : precisions) {
            const adjustment_result result = adjust(scene, 0.5, &start[0], mask, &deviations[0], true, optimisation::factor_graph::strategy::square_root, precision);
            std::fprintf(stderr, "Square root (%s): radius %.6f (dense %.6f), offset %.6f (dense %.6f), scale %.6f (dense %.6f)\n", (precision == optimisation::factor_graph::precision::single_precision) ? "single" : "double", result.calibration[0], dense.calibration[0], result.calibration[3], dense.calibration[3], result.scale, dense.scale);
            REQUIRE(result.used_square_root);
            REQUIRE(std::abs(result.calibration[0] - dense.calibration[0]) < 1e-4);
            REQUIRE(std::abs(result.calibration[3] - dense.calibration[3]) < 1e-4);
            REQUIRE(std::abs(result.scale - dense.scale) < 1e-4);
            REQUIRE(std::abs(result.extrinsic_rotation_error - dense.extrinsic_rotation_error) < 1e-4);
        }
    }

    // A differential drive, its two wheel radii recovered with the track width held. The slip of either wheel turns it, so its
    // heading is far noisier than a car's for the same slip and a turn shows its radii less well; this robot slips less.
    {
        simulation::ground_vehicle_options options;
        options.duration = 60.0;
        options.noise.slip = 0.002;
        options.speed = 1.0;
        options.speed_amplitude = 0.3;
        options.curvature = 0.05;
        options.curvature_amplitude = 0.4;
        options.kinematics = sensor::odometry::model(sensor::odometry::differential_drive{});
        options.parameters[0] = 0.1;
        options.parameters[1] = 0.104;
        options.parameters[2] = 0.5;
        options.parameters[3] = 0.0;
        const drive scene = make_drive(options, 0.5, 12);
        const double start[3] = { 0.095, 0.095, 0.5 };
        const double deviations[3] = { 0.02, 0.02, 0.0 };
        const adjustment_result result = adjust(scene, 0.6, &start[0], 0x3u, &deviations[0], true, optimisation::factor_graph::strategy::dense_schur, optimisation::factor_graph::precision::double_precision);
        std::fprintf(stderr, "Differential drive: radii %.5f %.5f, scale %.5f, position error %.4f m, extrinsic %.4f rad (%.4f, %.4f, %.4f) m\n", result.calibration[0], result.calibration[1], result.scale, result.position_error, result.extrinsic_rotation_error, result.extrinsic_translation_error[0], result.extrinsic_translation_error[1], result.extrinsic_translation_error[2]);
        REQUIRE(std::abs(result.calibration[0] - 0.1) < 0.0015);
        REQUIRE(std::abs(result.calibration[1] - 0.104) < 0.0015);
        REQUIRE(result.calibration[2] == 0.5);
        REQUIRE(std::abs(result.scale - 1.0) < 0.015);
        REQUIRE(result.extrinsic_rotation_error < 0.01);
        REQUIRE(std::abs(result.extrinsic_translation_error[0]) < 0.05);
        REQUIRE(std::abs(result.extrinsic_translation_error[1]) < 0.05);
    }

    return EXIT_SUCCESS;
}
