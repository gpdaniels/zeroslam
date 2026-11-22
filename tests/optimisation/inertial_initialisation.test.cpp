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

#include "optimisation/inertial_initialisation.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "sensor/imu/measurement.hpp"
#include "sensor/imu/preintegration.hpp"
#include "simulation/imu.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static double norm(const math::matrix<double, 3, 1>& vector) {
    return std::sqrt(vector.get_length_squared());
}

static double angle_between(const math::matrix<double, 3, 1>& lhs, const math::matrix<double, 3, 1>& rhs) {
    const double cosine = (lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]) / (norm(lhs) * norm(rhs));
    return std::acos(std::fmax(-1.0, std::fmin(1.0, cosine)));
}

// A window of keyframes as a monocular map would hold them: the body poses in a world turned from the gravity aligned one and in
// units of 1 / scale metres, with the imu preintegrated between them.
class window final {
public:
    std::vector<math::se3<double>> poses;
    std::vector<sensor::imu::preintegration> preintegrations;
    std::vector<math::matrix<double, 3, 1>> velocities;
};

static window make_window(const simulation::imu_trajectory& trajectory, const simulation::imu_sensor::recording& recording, const sensor::imu::parameters& parameters, const math::se3<double>& body_from_imu, const math::so3<double>& map_from_gravity, const double scale, const double first_time, const double period, const int count) {
    window result;
    for (int k = 0; k < count; ++k) {
        const double time = first_time + period * k;
        const math::se3<double> world_from_body = trajectory.world_to_imu(time).inverse() * body_from_imu.inverse();
        const math::se3<double> map_from_body(map_from_gravity * world_from_body.rotation(), (1.0 / scale) * (map_from_gravity * world_from_body.translation()));
        result.poses.push_back(map_from_body.inverse());
        result.velocities.push_back(map_from_gravity * trajectory.velocity(time));
        if (k > 0) {
            sensor::imu::preintegration preintegration(parameters);
            REQUIRE(preintegration.integrate(recording.samples, std::llround(1e9 * (time - period)), std::llround(1e9 * time)));
            result.preintegrations.push_back(preintegration);
        }
    }
    return result;
}

// Reads the whitespace separated rows of numbers of a text file, skipping comment lines.
static std::vector<std::vector<double>> read_rows(const std::string& path, const size_t columns) {
    std::vector<std::vector<double>> rows;
    std::FILE* const file = std::fopen(path.c_str(), "r");
    if (file == nullptr) {
        return rows;
    }
    char line[4096];
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        if (line[0] == '#') {
            continue;
        }
        std::vector<double> row;
        char* cursor = line;
        char* end = nullptr;
        for (size_t i = 0; i < columns; ++i) {
            const double value = std::strtod(cursor, &end);
            if (end == cursor) {
                break;
            }
            row.push_back(value);
            cursor = end;
        }
        if (row.size() == columns) {
            rows.push_back(row);
        }
    }
    std::fclose(file);
    return rows;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // The rotation from down.
    {
        const math::matrix<double, 3, 1> down = { { 0.0, 0.0, -1.0 } };
        for (const math::matrix<double, 3, 1>& direction : { math::matrix<double, 3, 1>({ { 0.0, 0.0, -2.0 } }), math::matrix<double, 3, 1>({ { 0.0, 0.0, 3.0 } }), math::matrix<double, 3, 1>({ { 1.0, -2.0, 0.5 } }), math::matrix<double, 3, 1>({ { 1e-14, 0.0, 1.0 } }) }) {
            const math::so3<double> rotation = optimisation::inertial_initialisation::rotation_from_down(direction);
            REQUIRE(rotation.is_unit());
            const math::matrix<double, 3, 1> turned = rotation * down;
            REQUIRE(norm(turned - (1.0 / norm(direction)) * direction) < 1e-12);
        }
        const math::so3<double> none = optimisation::inertial_initialisation::rotation_from_down(math::matrix<double, 3, 1>::zero());
        REQUIRE(none == math::so3<double>::identity());
    }

    sensor::imu::parameters parameters;
    const simulation::imu_trajectory trajectory = simulation::imu_trajectory::generate(simulation::imu_trajectory::options());
    const math::se3<double> body_from_imu(math::so3<double>::exp({ { 1.2, -0.3, 0.4 } }), { { 0.06, -0.02, 0.08 } });
    const math::so3<double> map_from_gravity = math::so3<double>::exp({ { -0.5, 0.9, 2.5 } });
    const double true_scale = 2.5;
    sensor::imu::bias true_bias;
    true_bias.gyroscope = { { 0.02, -0.01, 0.03 } };
    true_bias.accelerometer = { { 0.08, -0.12, 0.05 } };
    simulation::imu_sensor::options sensor_options;
    sensor_options.initial_bias = true_bias;
    const simulation::imu_sensor::recording recording = simulation::imu_sensor::generate(trajectory, parameters, sensor_options);
    const math::matrix<double, 3, 1> true_gravity = map_from_gravity * math::matrix<double, 3, 1>({ { 0.0, 0.0, -parameters.gravity } });

    // Too few keyframes, or preintegrations that do not match them, give nothing.
    {
        const window short_window = make_window(trajectory, recording, parameters, body_from_imu, map_from_gravity, true_scale, 1.0, 0.25, 3);
        const optimisation::inertial_initialisation::result failed = optimisation::inertial_initialisation::estimate(short_window.poses, short_window.preintegrations, optimisation::inertial_initialisation::options());
        REQUIRE(!failed.success);
        REQUIRE(!failed.observable);
        window mismatched = make_window(trajectory, recording, parameters, body_from_imu, map_from_gravity, true_scale, 1.0, 0.25, 6);
        mismatched.preintegrations.pop_back();
        REQUIRE(!optimisation::inertial_initialisation::estimate(mismatched.poses, mismatched.preintegrations, optimisation::inertial_initialisation::options()).success);
    }

    // Thirteen keyframes over three seconds of the simulated motion. With the defaults, the accelerometer bias held at zero, the
    // scale and gravity come out a little biased by it, and with the accelerometer bias free they come out right with it.
    {
        const window keyframes = make_window(trajectory, recording, parameters, body_from_imu, map_from_gravity, true_scale, 1.0, 0.25, 13);
        optimisation::inertial_initialisation::options options;
        options.body_from_imu = body_from_imu;
        const optimisation::inertial_initialisation::result held = optimisation::inertial_initialisation::estimate(keyframes.poses, keyframes.preintegrations, options);
        REQUIRE(held.success);
        REQUIRE(held.observable);
        REQUIRE(std::abs(held.scale / true_scale - 1.0) < 0.05);
        REQUIRE(std::abs(held.linear_scale / true_scale - 1.0) < 0.1);
        REQUIRE(angle_between(held.gravity, true_gravity) < 0.03);
        REQUIRE(std::abs(norm(held.gravity) - parameters.gravity) < 1e-9);
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(std::abs(held.bias.gyroscope[i] - true_bias.gyroscope[i]) < 2e-3);
            REQUIRE(std::abs(held.bias.accelerometer[i]) < 1e-4);
        }
        REQUIRE(held.velocities.size() == keyframes.poses.size());
        REQUIRE(held.preintegrations.size() == keyframes.preintegrations.size());
        for (const sensor::imu::preintegration& preintegration : held.preintegrations) {
            REQUIRE(preintegration.get_linearisation().gyroscope == held.bias.gyroscope);
        }

        options.accelerometer_bias_deviation = 1.0;
        const optimisation::inertial_initialisation::result free = optimisation::inertial_initialisation::estimate(keyframes.poses, keyframes.preintegrations, options);
        REQUIRE(free.success);
        REQUIRE(free.observable);
        REQUIRE(std::abs(free.scale / true_scale - 1.0) < 0.01);
        REQUIRE(angle_between(free.gravity, true_gravity) < 0.005);
        double largest_velocity_error = 0.0;
        for (size_t k = 0; k < keyframes.velocities.size(); ++k) {
            largest_velocity_error = std::fmax(largest_velocity_error, norm(free.velocities[k] - keyframes.velocities[k]));
        }
        REQUIRE(largest_velocity_error < 0.02);
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(std::abs(free.bias.gyroscope[i] - true_bias.gyroscope[i]) < 2e-3);
            REQUIRE(std::abs(free.bias.accelerometer[i] - true_bias.accelerometer[i]) < 0.05);
        }
        // The estimate is the same from run to run.
        const optimisation::inertial_initialisation::result again = optimisation::inertial_initialisation::estimate(keyframes.poses, keyframes.preintegrations, options);
        REQUIRE(again.scale == free.scale);
        REQUIRE(again.gravity == free.gravity);
        REQUIRE(again.scale_deviation == free.scale_deviation);
    }

    // A metric map, a stereo rig's, keeps its scale: only gravity, the velocities and the biases are estimated.
    {
        const window keyframes = make_window(trajectory, recording, parameters, body_from_imu, map_from_gravity, 1.0, 1.0, 0.25, 13);
        optimisation::inertial_initialisation::options options;
        options.body_from_imu = body_from_imu;
        options.fixed_scale = true;
        const optimisation::inertial_initialisation::result metric = optimisation::inertial_initialisation::estimate(keyframes.poses, keyframes.preintegrations, options);
        REQUIRE(metric.success);
        REQUIRE(metric.observable);
        REQUIRE(metric.scale == 1.0);
        REQUIRE(metric.scale_deviation == 0.0);
        REQUIRE(angle_between(metric.gravity, true_gravity) < 0.01);
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(std::abs(metric.bias.gyroscope[i] - true_bias.gyroscope[i]) < 2e-3);
        }
    }

    // A platform that barely moves holds neither the scale nor gravity's direction: the estimate is not observable.
    {
        simulation::imu_trajectory::options still_options;
        still_options.position_amplitude = 0.002;
        still_options.rotation_amplitude = 0.002;
        const simulation::imu_trajectory still = simulation::imu_trajectory::generate(still_options);
        const simulation::imu_sensor::recording still_recording = simulation::imu_sensor::generate(still, parameters, sensor_options);
        const window keyframes = make_window(still, still_recording, parameters, body_from_imu, map_from_gravity, true_scale, 1.0, 0.25, 13);
        optimisation::inertial_initialisation::options options;
        options.body_from_imu = body_from_imu;
        const optimisation::inertial_initialisation::result estimate = optimisation::inertial_initialisation::estimate(keyframes.poses, keyframes.preintegrations, options);
        REQUIRE(!estimate.observable);
        REQUIRE(estimate.scale_deviation > options.maximum_scale_deviation);
    }

    // EuRoC MAV, when ZEROSLAM_EUROC_SCENE names a scene expanded by `zeroslam dataset expand`: windows of ten keyframes four
    // per second along the ground truth, the ground truth turned and shrunk as a monocular map would be, and the imu's samples.
    const char* const scene = std::getenv("ZEROSLAM_EUROC_SCENE");
    if (scene != nullptr) {
        const std::string directory(scene);
        const std::vector<std::vector<double>> imu_rows = read_rows(directory + "/sensor/imu_01.txt", 14);
        const std::vector<std::vector<double>> truth_rows = read_rows(directory + "/trajectory.txt", 8);
        REQUIRE(!imu_rows.empty());
        REQUIRE(!truth_rows.empty());
        std::vector<sensor::imu::sample> samples;
        for (const std::vector<double>& row : imu_rows) {
            sensor::imu::sample sample;
            sample.timestamp = std::llround(row[0] * 1e9);
            sample.angular_velocity = { { row[8], row[9], row[10] } };
            sample.linear_acceleration = { { row[11], row[12], row[13] } };
            samples.push_back(sample);
        }
        // The imu's place on the body, the primary camera, from the first sample's extrinsic.
        const std::vector<double>& first = imu_rows.front();
        const math::se3<double> euroc_body_from_imu(math::so3<double>(first[7], first[4], first[5], first[6]).normalised(), { { first[1], first[2], first[3] } });
        std::vector<long long int> truth_times;
        std::vector<math::se3<double>> truth_poses;
        for (const std::vector<double>& row : truth_rows) {
            truth_times.push_back(std::llround(row[0] * 1e9));
            truth_poses.push_back(math::se3<double>(math::so3<double>(row[7], row[4], row[5], row[6]).normalised(), { { row[1], row[2], row[3] } }));
        }
        const double euroc_scale = 3.7;
        const math::matrix<double, 3, 1> euroc_gravity = map_from_gravity * math::matrix<double, 3, 1>({ { 0.0, 0.0, -parameters.gravity } });

        int windows = 0;
        int observable = 0;
        int good = 0;
        size_t index = 0;
        const long long int step = 250000000ll;
        while (index < truth_times.size()) {
            // The keyframes: the ground truth poses nearest to every quarter second from this one.
            std::vector<size_t> chosen;
            long long int target = truth_times[index];
            size_t cursor = index;
            while ((chosen.size() < 10) && (cursor < truth_times.size())) {
                while ((cursor + 1 < truth_times.size()) && (truth_times[cursor + 1] <= target)) {
                    ++cursor;
                }
                chosen.push_back(cursor);
                target += step;
                ++cursor;
            }
            if (chosen.size() < 10) {
                break;
            }
            std::vector<math::se3<double>> poses;
            std::vector<sensor::imu::preintegration> preintegrations;
            bool covered = true;
            for (size_t k = 0; k < chosen.size(); ++k) {
                const math::se3<double>& world_from_body = truth_poses[chosen[k]];
                const math::se3<double> map_from_body(map_from_gravity * world_from_body.rotation(), (1.0 / euroc_scale) * (map_from_gravity * world_from_body.translation()));
                poses.push_back(map_from_body.inverse());
                if (k > 0) {
                    sensor::imu::preintegration preintegration(parameters);
                    covered = covered && preintegration.integrate(samples, truth_times[chosen[k - 1]], truth_times[chosen[k]]);
                    preintegrations.push_back(preintegration);
                }
            }
            if (covered) {
                optimisation::inertial_initialisation::options options;
                options.body_from_imu = euroc_body_from_imu;
                const optimisation::inertial_initialisation::result estimate = optimisation::inertial_initialisation::estimate(poses, preintegrations, options);
                const double scale_error = estimate.scale / euroc_scale - 1.0;
                const double gravity_error = angle_between(estimate.gravity, euroc_gravity);
                std::fprintf(stderr, "euroc %7.2f s: scale %+.3f (deviation %.3f, linear %+.3f) gravity %.2f deg (deviation %.2f) gyroscope bias %+.4f %+.4f %+.4f observable %d\n", 1e-9 * static_cast<double>(truth_times[chosen[0]] - truth_times[0]), scale_error, estimate.scale_deviation, (estimate.linear_scale > 0.0) ? (estimate.linear_scale / euroc_scale - 1.0) : 0.0, gravity_error * 180.0 / 3.14159265358979323846, estimate.gravity_deviation * 180.0 / 3.14159265358979323846, estimate.bias.gyroscope[0], estimate.bias.gyroscope[1], estimate.bias.gyroscope[2], estimate.observable ? 1 : 0);
                ++windows;
                if (estimate.observable) {
                    ++observable;
                    good += ((std::abs(scale_error) < 0.1) && (gravity_error < 0.05)) ? 1 : 0;
                }
            }
            // The next window starts five seconds on.
            const long long int next = truth_times[index] + 5000000000ll;
            while ((index < truth_times.size()) && (truth_times[index] < next)) {
                ++index;
            }
        }
        std::fprintf(stderr, "euroc: %d windows, %d observable, %d of those within 10%% of the scale and 0.05 rad of gravity\n", windows, observable, good);
        REQUIRE(windows > 0);
        REQUIRE(observable > 0);
        REQUIRE(good * 10 >= observable * 9);
    }

    return EXIT_SUCCESS;
}
