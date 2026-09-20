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

#pragma once
#ifndef ZEROSLAM_SIMULATION_SCENE_HPP
#define ZEROSLAM_SIMULATION_SCENE_HPP

#include "core/assert.hpp"
#include "math/lie.hpp"
#include "math/math.hpp"
#include "math/matrix.hpp"
#include "sensor/camera.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdint>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace simulation {
    class random_stream final {
    private:
        uint64_t state;

    public:
        explicit random_stream(const uint64_t seed)
            : state(seed) {
        }

        uint64_t next() {
            this->state += 0x9E3779B97F4A7C15ull;
            uint64_t z = this->state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return z ^ (z >> 31);
        }

        double uniform() {
            return static_cast<double>(this->next() >> 11) * (1.0 / 9007199254740992.0);
        }

        double uniform(const double minimum, const double maximum) {
            return minimum + (maximum - minimum) * this->uniform();
        }

        double gaussian() {
            double u1 = this->uniform();
            if (u1 < 1.0e-300) {
                u1 = 1.0e-300;
            }
            const double u2 = this->uniform();
            return math::sqrt(-2.0 * math::log(u1)) * math::cos(2.0 * 3.14159265358979323846 * u2);
        }
    };

    struct observation final {
        int frame_index;
        int point_index;
        double x;
        double y;
        bool outlier;
    };

    struct options final {
        uint64_t seed = 42;
        int point_count = 200;
        double point_extent = 2.0;
        int frame_count = 10;
        double orbit_radius = 6.0;
        double orbit_height = 1.0;
        double orbit_arc_radians = 0.8;
        double focal_x = 525.0;
        double focal_y = 525.0;
        double centre_x = 320.0;
        double centre_y = 240.0;
        double image_width = 640.0;
        double image_height = 480.0;
        double noise_sigma = 0.0;
        double outlier_fraction = 0.0;
    };

    class scene final {
    public:
        options generation_options;
        sensor::pinhole camera;
        std::vector<math::matrix<double, 3, 1>> points;
        std::vector<math::se3<double>> poses;
        std::vector<observation> observations;
        std::vector<observation> observations_exact;

    public:
        static scene generate(const options& generation_options) {
            scene result;
            result.generation_options = generation_options;
            const double camera_parameters[4] = {
                generation_options.focal_x,
                generation_options.focal_y,
                generation_options.centre_x,
                generation_options.centre_y
            };
            result.camera = sensor::pinhole(&camera_parameters[0], 4);
            random_stream random(generation_options.seed);

            result.points.reserve(static_cast<size_t>(generation_options.point_count));
            for (int i = 0; i < generation_options.point_count; ++i) {
                const double x = random.uniform(-generation_options.point_extent, generation_options.point_extent);
                const double y = random.uniform(-generation_options.point_extent, generation_options.point_extent);
                const double z = random.uniform(-generation_options.point_extent, generation_options.point_extent);
                result.points.push_back({ { x, y, z } });
            }

            result.poses.reserve(static_cast<size_t>(generation_options.frame_count));
            for (int k = 0; k < generation_options.frame_count; ++k) {
                const double angle = (generation_options.frame_count > 1)
                                         ? (generation_options.orbit_arc_radians * static_cast<double>(k) / static_cast<double>(generation_options.frame_count - 1))
                                         : 0.0;
                const math::matrix<double, 3, 1> centre = { { generation_options.orbit_radius * math::cos(angle),
                                                              generation_options.orbit_height,
                                                              generation_options.orbit_radius * math::sin(angle) } };
                result.poses.push_back(scene::look_at(centre));
            }

            for (int k = 0; k < generation_options.frame_count; ++k) {
                const math::se3<double>& pose = result.poses[static_cast<size_t>(k)];
                for (int i = 0; i < generation_options.point_count; ++i) {
                    const math::matrix<double, 3, 1> in_camera = pose * result.points[static_cast<size_t>(i)];
                    math::matrix<double, 2, 1> pixel;
                    if (!result.camera.project(in_camera.data(), pixel.data())) {
                        continue;
                    }
                    if ((pixel[0] < 0.5) || (pixel[0] > generation_options.image_width - 0.5) || (pixel[1] < 0.5) || (pixel[1] > generation_options.image_height - 0.5)) {
                        continue;
                    }
                    observation exact;
                    exact.frame_index = k;
                    exact.point_index = i;
                    exact.x = pixel[0];
                    exact.y = pixel[1];
                    exact.outlier = false;
                    result.observations_exact.push_back(exact);

                    observation corrupted = exact;
                    if (generation_options.noise_sigma > 0.0) {
                        corrupted.x += generation_options.noise_sigma * random.gaussian();
                        corrupted.y += generation_options.noise_sigma * random.gaussian();
                    }
                    if (generation_options.outlier_fraction > 0.0) {
                        const double draw = random.uniform();
                        if (draw < generation_options.outlier_fraction) {
                            corrupted.x = static_cast<double>(static_cast<long long>(random.uniform(0.0, generation_options.image_width))) + 0.5;
                            corrupted.y = static_cast<double>(static_cast<long long>(random.uniform(0.0, generation_options.image_height))) + 0.5;
                            corrupted.outlier = true;
                        }
                        else {
                            static_cast<void>(random.uniform());
                            static_cast<void>(random.uniform());
                        }
                    }
                    result.observations.push_back(corrupted);
                }
            }
            return result;
        }

    private:
        static math::se3<double> look_at(const math::matrix<double, 3, 1>& centre) {
            math::matrix<double, 3, 1> world_down = { { 0.0, 1.0, 0.0 } };
            math::matrix<double, 3, 1> forward = -1.0 * centre;
            const double forward_length = math::sqrt(forward.get_length_squared());
            ASSERT(forward_length > 1.0e-12, "The camera cannot look at the origin from the origin.");
            forward = forward * (1.0 / forward_length);
            if (math::abs(forward[0]) + math::abs(forward[2]) < 1.0e-9) {
                world_down = { { 0.0, 0.0, 1.0 } };
            }
            math::matrix<double, 3, 1> right = { { world_down[1] * forward[2] - world_down[2] * forward[1],
                                                   world_down[2] * forward[0] - world_down[0] * forward[2],
                                                   world_down[0] * forward[1] - world_down[1] * forward[0] } };
            const double right_length = math::sqrt(right.get_length_squared());
            right = right * (1.0 / right_length);
            const math::matrix<double, 3, 1> down = { { forward[1] * right[2] - forward[2] * right[1],
                                                        forward[2] * right[0] - forward[0] * right[2],
                                                        forward[0] * right[1] - forward[1] * right[0] } };
            math::matrix<double, 3, 3> rotation;
            for (size_t column = 0; column < 3; ++column) {
                rotation[0][column] = right[column];
                rotation[1][column] = down[column];
                rotation[2][column] = forward[column];
            }
            const math::matrix<double, 3, 1> translation = -1.0 * (rotation * centre);
            return math::se3<double>(rotation, translation);
        }
    };
}

#endif // ZEROSLAM_SIMULATION_SCENE_HPP
