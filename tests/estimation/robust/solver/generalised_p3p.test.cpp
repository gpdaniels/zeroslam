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

#include "estimation/robust/solver/generalised_p3p.hpp"

#include "core/random_pcg.hpp"
#include "math/lie.hpp"

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
    using solver = estimation::robust::solver::generalised_p3p<double>;

    // Points in front of a camera, seen by it, with a share of them moved to wrong places.
    void observe(const math::se3<double>& reference_from_world, const math::se3<double>& camera_from_reference, const size_t camera, const size_t count, const size_t outliers, core::random_pcg& random, std::vector<solver::correspondence>& correspondences) {
        const math::se3<double> world_from_camera = (camera_from_reference * reference_from_world).inverse();
        for (size_t i = 0; i < count; ++i) {
            const math::matrix<double, 3, 1> seen({ random.get_random(-2.0, 2.0), random.get_random(-1.5, 1.5), random.get_random(3.0, 8.0) });
            const math::matrix<double, 3, 1> world = world_from_camera * seen;
            solver::correspondence entry;
            entry.bearing[0] = seen[0] / seen[2];
            entry.bearing[1] = seen[1] / seen[2];
            if (i < outliers) {
                entry.bearing[0] += random.get_random(0.1, 0.3);
                entry.bearing[1] -= random.get_random(0.1, 0.3);
            }
            entry.point[0] = world[0];
            entry.point[1] = world[1];
            entry.point[2] = world[2];
            entry.camera = camera;
            correspondences.push_back(entry);
        }
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const math::se3<double> reference_from_world(math::so3<double>::exp({ { 0.2, -0.4, 0.1 } }), { { 0.5, -0.2, 1.5 } });
    // A rig of three cameras: the reference, one looking backwards half a metre away, and one turned sideways.
    const math::se3<double> camera_from_reference[3] = {
        math::se3<double>::identity(),
        math::se3<double>(math::so3<double>::exp({ { 0.0, 3.14159265358979, 0.0 } }), { { 0.0, 0.0, -0.5 } }).inverse(),
        math::se3<double>(math::so3<double>::exp({ { 0.0, 1.5707963267949, 0.0 } }), { { 0.2, 0.0, 0.0 } }).inverse()
    };
    const float threshold = 1.0e-5f;

    {
        // The reference camera sees too few points to propose a pose and the third none: the second poses the rig, and
        // every point that agrees with it, in whichever camera, is an inlier.
        core::random_pcg random(7);
        std::vector<solver::correspondence> correspondences;
        observe(reference_from_world, camera_from_reference[0], 0, 4, 0, random, correspondences);
        observe(reference_from_world, camera_from_reference[1], 1, 40, 10, random, correspondences);
        std::vector<size_t> inliers(correspondences.size());
        size_t inliers_size = 0;
        solver::model model;
        REQUIRE(solver::solve(correspondences.data(), correspondences.size(), &camera_from_reference[0], 3, inliers.data(), inliers_size, model, threshold));
        REQUIRE(model.camera == 1);
        REQUIRE(inliers_size == 34);
        REQUIRE(inliers[0] == 0);
        const math::matrix<double, 3, 3> expected = reference_from_world.rotation().get_matrix();
        for (size_t row = 0; row < 3; ++row) {
            REQUIRE(std::abs(model.translation[row] - reference_from_world.translation()[row]) < 1e-6);
            for (size_t column = 0; column < 3; ++column) {
                REQUIRE(std::abs(model.rotation[row][column] - expected[row][column]) < 1e-6);
            }
        }
    }

    {
        // Every camera proposes, and the proposal most points agree with wins: the reference camera's points are mostly
        // wrong here, so its consensus explains fewer points across the rig than the others'.
        core::random_pcg random(11);
        std::vector<solver::correspondence> correspondences;
        observe(reference_from_world, camera_from_reference[0], 0, 12, 5, random, correspondences);
        observe(reference_from_world, camera_from_reference[1], 1, 30, 3, random, correspondences);
        observe(reference_from_world, camera_from_reference[2], 2, 30, 3, random, correspondences);
        std::vector<size_t> inliers(correspondences.size());
        size_t inliers_size = 0;
        solver::model model;
        REQUIRE(solver::solve(correspondences.data(), correspondences.size(), &camera_from_reference[0], 3, inliers.data(), inliers_size, model, threshold));
        REQUIRE(inliers_size == 7 + 27 + 27);
        for (size_t row = 0; row < 3; ++row) {
            REQUIRE(std::abs(model.translation[row] - reference_from_world.translation()[row]) < 1e-6);
        }
    }

    {
        // Too few points in every camera.
        core::random_pcg random(3);
        std::vector<solver::correspondence> correspondences;
        observe(reference_from_world, camera_from_reference[0], 0, 5, 0, random, correspondences);
        observe(reference_from_world, camera_from_reference[2], 2, 5, 0, random, correspondences);
        std::vector<size_t> inliers(correspondences.size());
        size_t inliers_size = 0;
        solver::model model;
        REQUIRE(!solver::solve(correspondences.data(), correspondences.size(), &camera_from_reference[0], 3, inliers.data(), inliers_size, model, threshold));
        REQUIRE(inliers_size == 0);
    }

    return EXIT_SUCCESS;
}
