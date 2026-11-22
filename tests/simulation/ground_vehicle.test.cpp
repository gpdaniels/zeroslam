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

#include "simulation/ground_vehicle.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "sensor/odometry/ackermann.hpp"
#include "sensor/odometry/differential_drive.hpp"
#include "sensor/odometry/model.hpp"
#include "sensor/odometry/preintegration.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-12) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

// The exact samples, integrated with the true calibration over the whole drive, come back to the true trajectory.
static void check_consistent(const simulation::ground_vehicle& vehicle, const double position_tolerance, const double heading_tolerance) {
    sensor::odometry::noise noise;
    sensor::odometry::preintegration motion(vehicle.options.kinematics, noise, &vehicle.options.parameters[0], vehicle.options.kinematics.parameter_count());
    REQUIRE(motion.integrate(vehicle.samples_exact.data(), vehicle.samples_exact.size(), vehicle.options.start_time, vehicle.end_time()));
    const math::matrix<double, 3, 1>& truth = vehicle.truth.back();
    REQUIRE(std::abs(motion.get_delta()[0] - truth[0]) < position_tolerance);
    REQUIRE(std::abs(motion.get_delta()[1] - truth[1]) < position_tolerance);
    REQUIRE(std::abs(motion.get_delta()[2] - truth[2]) < heading_tolerance);
    REQUIRE(std::abs(motion.get_distance() - vehicle.truth_distance.back()) < 1e-6 * vehicle.truth_distance.back());
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // The drive is sampled at the rate, the truth at every step, and starts at rest at the origin.
    {
        simulation::ground_vehicle_options options;
        options.duration = 20.0;
        const simulation::ground_vehicle vehicle = simulation::ground_vehicle::generate(options);
        REQUIRE(vehicle.sample_period == 10000000ll);
        REQUIRE(vehicle.step_period == 500000ll);
        REQUIRE(vehicle.samples.size() == 2001);
        REQUIRE(vehicle.samples_exact.size() == 2001);
        REQUIRE(vehicle.truth.size() == (2000 * 20) + 1);
        REQUIRE(vehicle.truth_distance.size() == vehicle.truth.size());
        REQUIRE(vehicle.truth_arcs.size() == vehicle.truth.size());
        REQUIRE(vehicle.samples.front().timestamp == options.start_time);
        REQUIRE(vehicle.samples.back().timestamp == options.start_time + 20000000000ll);
        REQUIRE(vehicle.end_time() == vehicle.samples.back().timestamp);
        REQUIRE((vehicle.truth[0][0] == 0.0) && (vehicle.truth[0][1] == 0.0) && (vehicle.truth[0][2] == 0.0));
        for (size_t i = 1; i < vehicle.samples.size(); ++i) {
            REQUIRE(vehicle.samples[i].timestamp - vehicle.samples[i - 1].timestamp == vehicle.sample_period);
        }

        // The distance is the speed integrated, and the planar pose between steps lies on the arc of the step.
        REQUIRE(std::abs(vehicle.truth_distance.back() - (2.0 * 20.0) - ((0.5 * 11.0 / (2.0 * 3.14159265358979323846)) * (1.0 - std::cos(2.0 * 3.14159265358979323846 * 20.0 / 11.0)))) < 1e-6);
        const math::matrix<double, 3, 1> at_step = vehicle.planar_at(options.start_time + (1000 * vehicle.step_period));
        REQUIRE((at_step[0] == vehicle.truth[1000][0]) && (at_step[1] == vehicle.truth[1000][1]) && (at_step[2] == vehicle.truth[1000][2]));
        const math::matrix<double, 3, 1> between = vehicle.planar_at(options.start_time + (1000 * vehicle.step_period) + (vehicle.step_period / 2));
        REQUIRE(is_value_approx(between[2], vehicle.truth[1000][2] + (0.5 * vehicle.truth_arcs[1000][1])));
        REQUIRE(is_value_approx(vehicle.distance_at(options.start_time + (1000 * vehicle.step_period) + (vehicle.step_period / 2)), vehicle.truth_distance[1000] + (0.5 * vehicle.truth_arcs[1000][0])));
        REQUIRE(vehicle.planar_at(0)[0] == 0.0);
        REQUIRE(vehicle.planar_at(vehicle.end_time() + 1)[0] == vehicle.truth.back()[0]);
        REQUIRE(vehicle.distance_at(0) == 0.0);
        REQUIRE(vehicle.distance_at(vehicle.end_time() + 1) == vehicle.truth_distance.back());

        // The steering reading is the steering angle of the curvature through the calibration.
        const long long int time = options.start_time + 3000000000ll;
        const double curvature = vehicle.curvature_at(3.0);
        REQUIRE(is_value_approx(std::tan(vehicle.steering_reading(time)) / 2.7, curvature));
        REQUIRE(vehicle.samples_exact[300].steering == vehicle.steering_reading(time));
        REQUIRE(vehicle.samples[300].steering != vehicle.samples_exact[300].steering);

        // The measured wheel drifts from the exact one by the slip, a random walk of a few millimetres over metres.
        const double drift = (vehicle.samples.back().rotation[0] - vehicle.samples_exact.back().rotation[0]) * 0.3;
        REQUIRE(drift != 0.0);
        REQUIRE(std::abs(drift) < 5.0 * 0.01 * std::sqrt(vehicle.truth_distance.back()));

        // The planar pose lifts to the plane z = 0.
        const math::se3<double> pose = vehicle.pose_at(time);
        REQUIRE(pose.translation()[2] == 0.0);
        REQUIRE(is_value_approx(pose.translation()[0], vehicle.planar_at(time)[0]));

        check_consistent(vehicle, 2e-4, 2e-5);
    }

    // The steering readings carry the gain and offset of the calibration, and front wheel and differential drive vehicles roll
    // their wheels by their own kinematics.
    {
        simulation::ground_vehicle_options options;
        options.duration = 10.0;
        options.add_noise = false;
        options.parameters[sensor::odometry::ackermann::steering_scale] = 0.8;
        options.parameters[sensor::odometry::ackermann::steering_offset] = -0.05;
        const simulation::ground_vehicle rear = simulation::ground_vehicle::generate(options);
        REQUIRE(rear.samples[100].steering == rear.samples_exact[100].steering);
        REQUIRE(rear.samples[100].rotation[0] == rear.samples_exact[100].rotation[0]);
        REQUIRE(is_value_approx((0.8 * rear.samples[100].steering) - 0.05, std::atan(rear.curvature_at(1.0) * 2.7)));
        check_consistent(rear, 2e-4, 2e-5);

        options.kinematics = sensor::odometry::model(sensor::odometry::ackermann(sensor::odometry::ackermann::measured_wheel::front_wheel));
        const simulation::ground_vehicle front = simulation::ground_vehicle::generate(options);
        REQUIRE(front.samples.back().rotation[0] > rear.samples.back().rotation[0]);
        check_consistent(front, 2e-4, 2e-5);

        options.kinematics = sensor::odometry::model(sensor::odometry::differential_drive{});
        options.parameters[0] = 0.1;
        options.parameters[1] = 0.11;
        options.parameters[2] = 0.5;
        options.curvature_amplitude = 0.5;
        const simulation::ground_vehicle drive = simulation::ground_vehicle::generate(options);
        REQUIRE(drive.samples[100].steering == 0.0);
        REQUIRE(drive.samples.back().rotation[1] != 0.0);
        check_consistent(drive, 2e-4, 2e-5);
    }

    // Encoder ticks quantise the measured rotations, and bumps lift and pitch the vehicle off the plane.
    {
        simulation::ground_vehicle_options options;
        options.duration = 4.0;
        options.add_noise = false;
        options.ticks_per_revolution = 100.0;
        options.bump_amplitude = 0.05;
        options.bump_wavelength = 2.0;
        const simulation::ground_vehicle vehicle = simulation::ground_vehicle::generate(options);
        const double tick = 2.0 * 3.14159265358979323846 / 100.0;
        for (size_t i = 0; i < vehicle.samples.size(); i += 37) {
            const double ticks = vehicle.samples[i].rotation[0] / tick;
            REQUIRE(std::abs(ticks - std::round(ticks)) < 1e-9);
            REQUIRE(vehicle.samples[i].rotation[0] <= vehicle.samples_exact[i].rotation[0]);
            REQUIRE(vehicle.samples_exact[i].rotation[0] - vehicle.samples[i].rotation[0] < tick);
        }
        // A quarter of a wavelength in, the road is at the crest, level, and half way up the slope before it it climbs.
        double crest = 0.0;
        bool found = false;
        for (size_t step = 0; (step < vehicle.truth.size()) && !found; ++step) {
            if (vehicle.truth_distance[step] >= 0.5) {
                crest = static_cast<double>(step);
                found = true;
            }
        }
        REQUIRE(found);
        const long long int crest_time = options.start_time + (static_cast<long long int>(crest) * vehicle.step_period);
        const math::se3<double> top = vehicle.pose_at(crest_time);
        REQUIRE(std::abs(top.translation()[2] - 0.05) < 1e-4);
        const math::matrix<double, 3, 1> forward = top.rotation() * math::matrix<double, 3, 1>({ 1.0, 0.0, 0.0 });
        REQUIRE(std::abs(forward[2]) < 1e-3);
        const math::se3<double> start = vehicle.pose_at(options.start_time);
        const math::matrix<double, 3, 1> climbing = start.rotation() * math::matrix<double, 3, 1>({ 1.0, 0.0, 0.0 });
        REQUIRE(climbing[2] > 0.1);
        REQUIRE(std::abs(start.translation()[2]) < 1e-12);
    }

    // The same seed makes the same drive.
    {
        simulation::ground_vehicle_options options;
        options.duration = 2.0;
        const simulation::ground_vehicle first = simulation::ground_vehicle::generate(options);
        const simulation::ground_vehicle second = simulation::ground_vehicle::generate(options);
        options.seed = 8;
        const simulation::ground_vehicle third = simulation::ground_vehicle::generate(options);
        REQUIRE(first.samples.back().rotation[0] == second.samples.back().rotation[0]);
        REQUIRE(first.samples.back().steering == second.samples.back().steering);
        REQUIRE(first.samples.back().rotation[0] != third.samples.back().rotation[0]);
    }

    return EXIT_SUCCESS;
}
