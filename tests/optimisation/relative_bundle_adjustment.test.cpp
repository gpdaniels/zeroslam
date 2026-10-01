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

#include "optimisation/relative_bundle_adjustment.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edges/reprojection.hpp"
#include "sensor/camera.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdint>
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
    using problem_type = optimisation::relative_bundle_adjustment;

    class generator final {
    private:
        std::uint64_t state = 0x9E3779B97F4A7C15ull;

    public:
        double uniform(const double low, const double high) {
            this->state ^= this->state << 13;
            this->state ^= this->state >> 7;
            this->state ^= this->state << 17;
            return low + ((high - low) * static_cast<double>(this->state >> 11) / static_cast<double>(1ull << 53));
        }
    };

    double distance(const math::sim3<double>& lhs, const math::sim3<double>& rhs) {
        const math::matrix<double, 7, 1> difference = (lhs.inverse() * rhs).log();
        return std::sqrt(difference.get_length_squared());
    }

    // The world to camera transform of a camera at centre looking at the origin, with its y axis towards world +y.
    math::se3<double> look_at_origin(const math::matrix<double, 3, 1>& centre) {
        const double length = std::sqrt(centre.get_length_squared());
        const math::matrix<double, 3, 1> forward = { { -centre[0] / length, -centre[1] / length, -centre[2] / length } };
        const math::matrix<double, 3, 1> down = { { 0.0, 1.0, 0.0 } };
        math::matrix<double, 3, 1> right = { { (down[1] * forward[2]) - (down[2] * forward[1]), (down[2] * forward[0]) - (down[0] * forward[2]), (down[0] * forward[1]) - (down[1] * forward[0]) } };
        const double right_length = std::sqrt(right.get_length_squared());
        right = { { right[0] / right_length, right[1] / right_length, right[2] / right_length } };
        const math::matrix<double, 3, 1> up = { { (forward[1] * right[2]) - (forward[2] * right[1]), (forward[2] * right[0]) - (forward[0] * right[2]), (forward[0] * right[1]) - (forward[1] * right[0]) } };
        const math::matrix<double, 3, 3> rotation_cw = { { { right[0], right[1], right[2] }, { up[0], up[1], up[2] }, { forward[0], forward[1], forward[2] } } };
        const math::so3<double> rotation(rotation_cw);
        const math::matrix<double, 3, 1> rotated = rotation * centre;
        return { rotation, { { -rotated[0], -rotated[1], -rotated[2] } } };
    }

    class edge_link final {
    public:
        int parent;
        int child;
    };

    // The steps from the observing frame to the base frame along a shortest path of the graph.
    std::vector<problem_type::step> path_between(const std::vector<edge_link>& links, const int frames, const int observing, const int base) {
        std::vector<int> previous_transform(static_cast<size_t>(frames), -1);
        std::vector<int> previous_frame(static_cast<size_t>(frames), -1);
        std::vector<char> visited(static_cast<size_t>(frames), 0);
        std::vector<int> queue = { base };
        visited[static_cast<size_t>(base)] = 1;
        for (size_t head = 0; head < queue.size(); ++head) {
            const int current = queue[head];
            for (size_t index = 0; index < links.size(); ++index) {
                const int other = (links[index].parent == current) ? links[index].child : ((links[index].child == current) ? links[index].parent : -1);
                if ((other < 0) || (visited[static_cast<size_t>(other)] != 0)) {
                    continue;
                }
                visited[static_cast<size_t>(other)] = 1;
                previous_transform[static_cast<size_t>(other)] = static_cast<int>(index);
                previous_frame[static_cast<size_t>(other)] = current;
                queue.push_back(other);
            }
        }
        // Walking back from the observing frame towards the base, each step takes the next frame's coordinates into this one.
        std::vector<problem_type::step> path;
        for (int current = observing; current != base; current = previous_frame[static_cast<size_t>(current)]) {
            const int transform = previous_transform[static_cast<size_t>(current)];
            problem_type::step link;
            link.transform = transform;
            link.inverse = (links[static_cast<size_t>(transform)].child == current);
            path.push_back(link);
        }
        return path;
    }

    class scene final {
    public:
        std::vector<math::se3<double>> poses;
        std::vector<edge_link> links;
        std::vector<math::sim3<double>> truth_transforms;
        std::vector<math::matrix<double, 3, 1>> truth_landmarks;
        problem_type problem;
    };

    // Six frames on an arc around a cloud of points, chained by rigid transforms 0-4 and closed by the similarity 5.
    scene make_scene() {
        scene result;
        const double parameters[sensor::model::parameter_count] = { 450.0, 455.0, 320.0, 240.0, -0.05, 0.01, 0.001, -0.0005, 0.0, 0.0, 0.0, 0.0 };
        result.problem.cameras.push_back(sensor::model(&parameters[0], sensor::model::parameter_count));
        const int frames = 6;
        for (int index = 0; index < frames; ++index) {
            const double angle = -0.6 + (0.24 * static_cast<double>(index));
            result.poses.push_back(look_at_origin({ { 5.0 * std::sin(angle), 0.4 * static_cast<double>(index % 2) - 0.2, -5.0 * std::cos(angle) } }));
        }
        for (int index = 0; index + 1 < frames; ++index) {
            result.links.push_back({ index, index + 1 });
        }
        result.links.push_back({ 0, frames - 1 });
        for (size_t index = 0; index < result.links.size(); ++index) {
            const math::se3<double> relative = result.poses[static_cast<size_t>(result.links[index].parent)] * result.poses[static_cast<size_t>(result.links[index].child)].inverse();
            problem_type::transform transform;
            transform.estimate = math::sim3<double>(relative, 1.0);
            transform.similarity = (index + 1 == result.links.size());
            transform.active = true;
            result.problem.transforms.push_back(transform);
            result.truth_transforms.push_back(transform.estimate);
        }
        generator random;
        for (int index = 0; index < 60; ++index) {
            const math::matrix<double, 3, 1> world = { { random.uniform(-1.5, 1.5), random.uniform(-1.0, 1.0), random.uniform(-1.5, 1.5) } };
            const int base = index % frames;
            const math::matrix<double, 3, 1> in_base = result.poses[static_cast<size_t>(base)] * world;
            problem_type::landmark landmark;
            landmark.parameters[0] = in_base[0] / in_base[2];
            landmark.parameters[1] = in_base[1] / in_base[2];
            landmark.parameters[2] = 1.0 / in_base[2];
            landmark.active = true;
            result.problem.landmarks.push_back(landmark);
            result.truth_landmarks.push_back({ { landmark.parameters[0], landmark.parameters[1], landmark.parameters[2] } });
            for (int observing = 0; observing < frames; ++observing) {
                const math::matrix<double, 3, 1> in_frame = result.poses[static_cast<size_t>(observing)] * world;
                double pixel[2];
                if (!result.problem.cameras[0].project(in_frame.data(), &pixel[0]) || (pixel[0] < 0.0) || (pixel[0] > 640.0) || (pixel[1] < 0.0) || (pixel[1] > 480.0)) {
                    continue;
                }
                problem_type::observation measured;
                measured.landmark = index;
                measured.camera = 0;
                measured.pixel[0] = pixel[0];
                measured.pixel[1] = pixel[1];
                measured.sigma = 1.0;
                measured.path = path_between(result.links, frames, observing, base);
                result.problem.observations.push_back(measured);
            }
        }
        return result;
    }

    void perturb(scene& setup, const bool include_first) {
        generator random;
        for (size_t index = 0; index < setup.problem.transforms.size(); ++index) {
            problem_type::transform& transform = setup.problem.transforms[index];
            if (!include_first && (index == 0)) {
                continue;
            }
            const double scale = transform.similarity ? random.uniform(-0.05, 0.05) : 0.0;
            const math::matrix<double, 7, 1> delta = { { random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02), random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), scale } };
            transform.estimate = transform.estimate * math::sim3<double>::exp(delta);
        }
        for (problem_type::landmark& landmark : setup.problem.landmarks) {
            landmark.parameters[0] += random.uniform(-0.01, 0.01);
            landmark.parameters[1] += random.uniform(-0.01, 0.01);
            landmark.parameters[2] *= random.uniform(0.9, 1.1);
        }
    }

    bool recovered(const scene& setup, const double tolerance) {
        for (size_t index = 0; index < setup.problem.transforms.size(); ++index) {
            if (!(distance(setup.truth_transforms[index], setup.problem.transforms[index].estimate) < tolerance)) {
                std::fprintf(stderr, "transform %zu is %g from the truth\n", index, distance(setup.truth_transforms[index], setup.problem.transforms[index].estimate));
                return false;
            }
        }
        for (size_t index = 0; index < setup.problem.landmarks.size(); ++index) {
            for (size_t row = 0; row < 3; ++row) {
                if (!(std::fabs(setup.truth_landmarks[index][row] - setup.problem.landmarks[index].parameters[row]) < tolerance)) {
                    std::fprintf(stderr, "landmark %zu is %g from the truth\n", index, std::fabs(setup.truth_landmarks[index][row] - setup.problem.landmarks[index].parameters[row]));
                    return false;
                }
            }
        }
        return true;
    }

    // Compares the analytic jacobians with central differences of the prediction.
    void check_jacobians(const problem_type& problem, const problem_type::observation& measured) {
        double residual[2];
        std::vector<math::matrix<double, 2, 7>> step_jacobians;
        math::matrix<double, 2, 3> landmark_jacobian;
        problem.linearise(measured, residual, step_jacobians, landmark_jacobian);
        REQUIRE(step_jacobians.size() == measured.path.size());
        double predicted[2];
        REQUIRE(problem.predict(measured, predicted));
        REQUIRE(std::fabs(residual[0] - (measured.pixel[0] - predicted[0])) < 1e-9);
        REQUIRE(std::fabs(residual[1] - (measured.pixel[1] - predicted[1])) < 1e-9);
        const double epsilon = 1e-6;
        for (size_t a = 0; a < measured.path.size(); ++a) {
            const size_t transform_index = static_cast<size_t>(measured.path[a].transform);
            math::matrix<double, 2, 7> expected = math::matrix<double, 2, 7>::zero();
            for (size_t b = 0; b < measured.path.size(); ++b) {
                if (static_cast<size_t>(measured.path[b].transform) == transform_index) {
                    expected = expected + step_jacobians[b];
                }
            }
            for (size_t parameter = 0; parameter < 7; ++parameter) {
                math::matrix<double, 7, 1> delta = math::matrix<double, 7, 1>::zero();
                delta[parameter] = epsilon;
                problem_type plus = problem;
                plus.transforms[transform_index].estimate = problem.transforms[transform_index].estimate * math::sim3<double>::exp(delta);
                delta[parameter] = -epsilon;
                problem_type minus = problem;
                minus.transforms[transform_index].estimate = problem.transforms[transform_index].estimate * math::sim3<double>::exp(delta);
                double pixel_plus[2];
                double pixel_minus[2];
                REQUIRE(plus.predict(measured, pixel_plus));
                REQUIRE(minus.predict(measured, pixel_minus));
                for (size_t row = 0; row < 2; ++row) {
                    const double numeric = (pixel_plus[row] - pixel_minus[row]) / (2.0 * epsilon);
                    REQUIRE(std::fabs(numeric - expected[row][parameter]) < 1e-4 * (1.0 + std::fabs(numeric)));
                }
            }
        }
        for (size_t parameter = 0; parameter < 3; ++parameter) {
            problem_type plus = problem;
            plus.landmarks[static_cast<size_t>(measured.landmark)].parameters[parameter] += epsilon;
            problem_type minus = problem;
            minus.landmarks[static_cast<size_t>(measured.landmark)].parameters[parameter] -= epsilon;
            double pixel_plus[2];
            double pixel_minus[2];
            REQUIRE(plus.predict(measured, pixel_plus));
            REQUIRE(minus.predict(measured, pixel_minus));
            for (size_t row = 0; row < 2; ++row) {
                const double numeric = (pixel_plus[row] - pixel_minus[row]) / (2.0 * epsilon);
                REQUIRE(std::fabs(numeric - landmark_jacobian[row][parameter]) < 1e-4 * (1.0 + std::fabs(numeric)));
            }
        }
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // The truth predicts every observation exactly, through forward and inverse steps and the loop.
    {
        const scene setup = make_scene();
        REQUIRE(setup.problem.observations.size() > 200);
        bool forward = false;
        bool inverse = false;
        bool loop = false;
        for (const problem_type::observation& measured : setup.problem.observations) {
            double pixel[2];
            REQUIRE(setup.problem.predict(measured, pixel));
            REQUIRE(std::fabs(pixel[0] - measured.pixel[0]) < 1e-8);
            REQUIRE(std::fabs(pixel[1] - measured.pixel[1]) < 1e-8);
            for (const problem_type::step& link : measured.path) {
                forward = forward || !link.inverse;
                inverse = inverse || link.inverse;
                loop = loop || (link.transform == 5);
            }
        }
        REQUIRE(forward && inverse && loop);
        REQUIRE(setup.problem.cost() < 1e-12);
    }

    // An empty path observes the landmark in its own frame.
    {
        const scene setup = make_scene();
        problem_type::observation measured;
        measured.landmark = 0;
        measured.camera = 0;
        const math::matrix<double, 3, 1> point = setup.problem.point_in_frame(measured);
        REQUIRE(std::fabs(point[0] - setup.problem.landmarks[0].parameters[0]) < 1e-15);
        REQUIRE(std::fabs(point[1] - setup.problem.landmarks[0].parameters[1]) < 1e-15);
        REQUIRE(std::fabs(point[2] - 1.0) < 1e-15);
    }

    // The jacobians match numeric differences along every kind of path, away from the truth and with a scale on the loop.
    {
        scene setup = make_scene();
        perturb(setup, true);
        int checked = 0;
        for (const problem_type::observation& measured : setup.problem.observations) {
            if (measured.path.size() >= 2) {
                check_jacobians(setup.problem, measured);
                ++checked;
            }
        }
        REQUIRE(checked > 50);
        for (const problem_type::observation& measured : setup.problem.observations) {
            if (measured.path.empty()) {
                check_jacobians(setup.problem, measured);
                break;
            }
        }
    }

    // With one transform held for the scale, the rest of the graph and the landmarks are recovered.
    {
        scene setup = make_scene();
        setup.problem.transforms[0].active = false;
        perturb(setup, false);
        const double initial = setup.problem.cost();
        REQUIRE(initial > 1.0);
        const problem_type::summary summary = setup.problem.solve(50);
        REQUIRE(summary.active_transforms == 5);
        REQUIRE(summary.active_landmarks == 60);
        REQUIRE(std::fabs(summary.initial_cost - initial) < 1e-9 * initial);
        REQUIRE(summary.final_cost < 1e-12);
        REQUIRE(std::fabs(summary.final_cost - setup.problem.cost()) < 1e-12);
        REQUIRE(summary.accepted > 0);
        REQUIRE(recovered(setup, 1e-6));
    }

    // With every transform free, a length prior holds the scale instead.
    {
        scene setup = make_scene();
        perturb(setup, true);
        problem_type::length_prior prior;
        prior.transform = 0;
        prior.length = std::sqrt(setup.truth_transforms[0].transformation().translation().get_length_squared());
        prior.information = 1e4;
        setup.problem.priors.push_back(prior);
        const problem_type::summary summary = setup.problem.solve(50);
        REQUIRE(summary.active_transforms == 6);
        REQUIRE(summary.final_cost < 1e-12);
        REQUIRE(recovered(setup, 1e-6));
    }

    // Inactive transforms and landmarks are held exactly.
    {
        scene setup = make_scene();
        setup.problem.transforms[0].active = false;
        perturb(setup, false);
        setup.problem.transforms[2].active = false;
        setup.problem.landmarks[7].active = false;
        const math::sim3<double> held_transform = setup.problem.transforms[2].estimate;
        const double held_landmark[3] = { setup.problem.landmarks[7].parameters[0], setup.problem.landmarks[7].parameters[1], setup.problem.landmarks[7].parameters[2] };
        const problem_type::summary summary = setup.problem.solve(20);
        REQUIRE(summary.active_transforms == 4);
        REQUIRE(summary.active_landmarks == 59);
        REQUIRE(summary.final_cost < summary.initial_cost);
        REQUIRE(setup.problem.transforms[2].estimate == held_transform);
        REQUIRE(setup.problem.transforms[0].estimate == setup.truth_transforms[0]);
        for (size_t row = 0; row < 3; ++row) {
            REQUIRE(setup.problem.landmarks[7].parameters[row] == held_landmark[row]);
        }
    }

    // A gross outlier is down weighted rather than pulling the solution.
    {
        scene setup = make_scene();
        setup.problem.transforms[0].active = false;
        perturb(setup, false);
        setup.problem.observations[11].pixel[0] += 60.0;
        const problem_type::summary summary = setup.problem.solve(50);
        REQUIRE(summary.final_cost < summary.initial_cost);
        REQUIRE(recovered(setup, 1e-2));
        double pixel[2];
        REQUIRE(setup.problem.predict(setup.problem.observations[11], pixel));
        REQUIRE(std::fabs(setup.problem.observations[11].pixel[0] - pixel[0]) > 50.0);
    }

    // A loop seen only through points at infinity holds no information on its translation and scale, and the rest of the
    // problem still converges as fast: the reduced system is scaled to a unit diagonal before it is factorised.
    {
        scene setup = make_scene();
        problem_type::transform& loop = setup.problem.transforms[5];
        std::vector<problem_type::observation> kept;
        for (const problem_type::observation& measured : setup.problem.observations) {
            bool through_loop = false;
            for (const problem_type::step& link : measured.path) {
                through_loop = through_loop || (link.transform == 5);
            }
            if (!through_loop) {
                kept.push_back(measured);
            }
        }
        // Points at infinity from frame 0, observed by frame 5 across the loop.
        const int first_far = static_cast<int>(setup.problem.landmarks.size());
        generator random;
        for (int index = 0; index < 30; ++index) {
            const math::matrix<double, 3, 1> direction = { { random.uniform(-0.4, 0.4), random.uniform(-0.3, 0.3), 1.0 } };
            const math::matrix<double, 3, 1> in_first = setup.poses[0].rotation() * (setup.poses[0].inverse().rotation() * direction);
            problem_type::landmark distant;
            distant.parameters[0] = in_first[0] / in_first[2];
            distant.parameters[1] = in_first[1] / in_first[2];
            distant.parameters[2] = 0.0;
            distant.active = false;
            setup.problem.landmarks.push_back(distant);
            const math::matrix<double, 3, 1> in_last = setup.poses[5].rotation() * (setup.poses[0].inverse().rotation() * in_first);
            double pixel[2];
            if (!(in_last[2] > 0.0) || !setup.problem.cameras[0].project(in_last.data(), &pixel[0])) {
                continue;
            }
            problem_type::observation measured;
            measured.landmark = first_far + index;
            measured.camera = 0;
            measured.pixel[0] = pixel[0];
            measured.pixel[1] = pixel[1];
            problem_type::step across;
            across.transform = 5;
            across.inverse = true;
            measured.path.push_back(across);
            kept.push_back(measured);
        }
        setup.problem.observations = kept;
        setup.problem.transforms[0].active = false;
        std::vector<problem_type::landmark> far_points(setup.problem.landmarks.begin() + first_far, setup.problem.landmarks.end());
        perturb(setup, false);
        // The points at infinity are held exactly, so they alone fix the loop's rotation.
        for (size_t index = 0; index < far_points.size(); ++index) {
            setup.problem.landmarks[static_cast<size_t>(first_far) + index] = far_points[index];
        }
        const math::matrix<double, 3, 1> held_translation = loop.estimate.transformation().translation();
        const double held_scale = loop.estimate.scale();
        const problem_type::summary summary = setup.problem.solve(50);
        REQUIRE(summary.accepted > 0);
        REQUIRE(summary.final_cost < 1e-12);
        for (size_t index = 0; index < 5; ++index) {
            REQUIRE(distance(setup.truth_transforms[index], setup.problem.transforms[index].estimate) < 1e-6);
        }
        // The loop's rotation is recovered, its translation and scale stay where nothing moves them.
        const math::matrix<double, 3, 1> rotation_error = (setup.truth_transforms[5].transformation().rotation().inverse() * loop.estimate.transformation().rotation()).log();
        REQUIRE(std::sqrt(rotation_error.get_length_squared()) < 1e-6);
        REQUIRE(std::sqrt((loop.estimate.transformation().translation() - held_translation).get_length_squared()) < 1e-6);
        REQUIRE(std::fabs(loop.estimate.scale() - held_scale) < 1e-6);
    }

    // A point behind its camera does not project, and is penalised by its depth as the reprojection edges do.
    {
        scene setup = make_scene();
        bool checked = false;
        for (const problem_type::observation& measured : setup.problem.observations) {
            if (measured.path.empty()) {
                continue;
            }
            // The depth is linear in the inverse depth through the chain's translation, so it can be put behind the camera.
            problem_type::landmark& landmark = setup.problem.landmarks[static_cast<size_t>(measured.landmark)];
            const double held = landmark.parameters[2];
            landmark.parameters[2] = 0.0;
            const double at_infinity = setup.problem.point_in_frame(measured)[2];
            landmark.parameters[2] = 1.0;
            const double slope = setup.problem.point_in_frame(measured)[2] - at_infinity;
            if (!(std::fabs(slope) > 0.1) || !(at_infinity > 0.0)) {
                landmark.parameters[2] = held;
                continue;
            }
            landmark.parameters[2] = -2.0 * at_infinity / slope;
            const math::matrix<double, 3, 1> point = setup.problem.point_in_frame(measured);
            REQUIRE(point[2] < 0.0);
            double pixel[2];
            REQUIRE(!setup.problem.predict(measured, pixel));
            double residual[2];
            std::vector<math::matrix<double, 2, 7>> step_jacobians;
            math::matrix<double, 2, 3> landmark_jacobian;
            setup.problem.linearise(measured, residual, step_jacobians, landmark_jacobian);
            const double length = std::sqrt(point.get_length_squared());
            const double penalty = optimisation::edges::reprojection::behind_camera_residual(point[2] / length);
            REQUIRE(penalty > 0.0);
            REQUIRE(std::fabs(residual[0] - penalty) < 1e-12);
            REQUIRE(std::fabs(residual[1] - penalty) < 1e-12);
            // The jacobian is of the prediction, so of minus the penalty.
            for (size_t parameter = 0; parameter < 3; ++parameter) {
                const double value = landmark.parameters[parameter];
                double shifted[2];
                landmark.parameters[parameter] = value + 1e-6;
                setup.problem.linearise(measured, shifted, step_jacobians, landmark_jacobian);
                const double plus = shifted[0];
                landmark.parameters[parameter] = value - 1e-6;
                setup.problem.linearise(measured, shifted, step_jacobians, landmark_jacobian);
                const double minus = shifted[0];
                landmark.parameters[parameter] = value;
                setup.problem.linearise(measured, residual, step_jacobians, landmark_jacobian);
                REQUIRE(std::fabs(-((plus - minus) / 2e-6) - landmark_jacobian[0][parameter]) < 1e-6);
            }
            checked = true;
            break;
        }
        REQUIRE(checked);
    }

    return EXIT_SUCCESS;
}
