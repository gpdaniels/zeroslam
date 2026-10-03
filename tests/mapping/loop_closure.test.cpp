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

#include "mapping/loop_closure.hpp"

#include "core/random_pcg.hpp"
#include "feature/descriptor/binary.hpp"
#include "mapping/covisibility.hpp"
#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "sensor/camera.hpp"

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

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-8) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

static const sensor::model& test_camera() {
    static const double parameters[sensor::model::parameter_count] = { 525.0, 525.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    static const sensor::model camera(&parameters[0], sensor::model::parameter_count);
    return camera;
}

static mapping::loop_closure::record random_record(core::random_pcg& random, const int landmark_id) {
    mapping::loop_closure::record record;
    record.landmark_id = landmark_id;
    record.descriptor = feature::descriptor::stored{};
    for (size_t j = 0; j < 32; ++j) {
        record.descriptor[j] = static_cast<unsigned char>(random.get_random_raw() % 256);
    }
    record.location = math::matrix<double, 3, 1>{ { random.get_random(-3.0, 3.0), random.get_random(-2.0, 2.0), random.get_random(6.0, 12.0) } };
    record.pixel_x = 0.0f;
    record.pixel_y = 0.0f;
    return record;
}

static void observe_records(std::vector<mapping::loop_closure::record>& records, const math::se3<double>& pose) {
    for (mapping::loop_closure::record& record : records) {
        const math::matrix<double, 3, 1> in_camera = pose * record.location;
        double projected[2] = {};
        REQUIRE(test_camera().project(in_camera.data(), &projected[0]));
        record.pixel_x = static_cast<float>(projected[0]);
        record.pixel_y = static_cast<float>(projected[1]);
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    core::random_pcg random(0x5eed0055ull);
    const math::se3<double> identity = math::se3<double>::identity();
    const mapping::covisibility unconnected;
    const math::se3<double> current_pose(math::so3<double>::exp({ { 0.02, -0.01, 0.03 } }), { { 0.2, -0.3, 0.1 } });

    std::vector<mapping::loop_closure::record> first;
    for (int landmark_id = 0; landmark_id < 40; ++landmark_id) {
        first.push_back(random_record(random, landmark_id));
    }
    observe_records(first, identity);
    mapping::loop_closure closure;
    REQUIRE(closure.num_keyframes() == 0);

    REQUIRE(!closure.detect(0, identity, test_camera(), unconnected, first.data(), first.size()).found);
    closure.add_keyframe(0, identity, test_camera(), first.data(), first.size());
    REQUIRE(closure.num_keyframes() == 1);
    REQUIRE(!closure.detect(0, identity, test_camera(), unconnected, first.data(), first.size()).found);

    int next_landmark_id = 40;
    for (int keyframe_id = 1; keyframe_id < 40; ++keyframe_id) {
        std::vector<mapping::loop_closure::record> records;
        for (int i = 0; i < 30; ++i) {
            records.push_back(random_record(random, next_landmark_id++));
        }
        observe_records(records, identity);
        REQUIRE(!closure.detect(keyframe_id, identity, test_camera(), unconnected, records.data(), records.size()).found);
        closure.add_keyframe(keyframe_id, identity, test_camera(), records.data(), records.size());
    }
    REQUIRE(closure.num_keyframes() == 40);

    const math::sim3<double> drift(math::se3<double>(math::so3<double>::exp({ { 0.0, 0.0, 0.05 } }), { { 0.4, -0.3, 0.2 } }), 1.1);
    std::vector<mapping::loop_closure::record> revisit = first;
    for (size_t i = 0; i < revisit.size(); ++i) {
        revisit[i].landmark_id = 1000 + static_cast<int>(i);
        revisit[i].location = drift * first[i].location;
        if (i >= 35) {
            revisit[i].location = revisit[i].location + math::matrix<double, 3, 1>{ { 3.0, -2.0, 1.5 } };
        }
    }
    observe_records(revisit, current_pose);

    REQUIRE(!closure.detect(0 + mapping::loop_closure::min_keyframe_gap - 1, identity, test_camera(), unconnected, revisit.data(), revisit.size()).found);

    REQUIRE(!closure.detect(40, current_pose, test_camera(), unconnected, first.data(), first.size()).found);

    {
        const math::se3<double> covisible_drift(math::so3<double>::exp({ { 0.0, 0.0, 0.05 } }), { { 0.4, -0.3, 0.2 } });
        std::vector<mapping::loop_closure::record> covisible = first;
        for (size_t i = 0; i < covisible.size(); ++i) {
            covisible[i].location = covisible_drift * first[i].location;
            covisible[i].landmark_id = static_cast<int>(i);
        }
        const math::se3<double> covisible_pose = current_pose;
        observe_records(covisible, covisible_pose);
        // Without covisible revisits the drifted landmarks shared by id are not a loop.
        closure.set_covisible_revisits(false);
        REQUIRE(!closure.detect(40, covisible_pose, test_camera(), unconnected, covisible.data(), covisible.size()).found);
        closure.set_covisible_revisits(true);
        const mapping::loop_closure::result result = closure.detect(40, covisible_pose, test_camera(), unconnected, covisible.data(), covisible.size());
        REQUIRE(result.found);
        REQUIRE(result.keyframe_id == 0);
        REQUIRE(result.correspondences == 40);
        REQUIRE(result.inliers == 40);
        REQUIRE(result.matches.size() == 40);
        for (const mapping::loop_closure::correspondence& match : result.matches) {
            REQUIRE(match.landmark_id == match.recorded_landmark_id);
        }
        REQUIRE(is_value_approx(result.correction.scale(), 1.0, 1e-9));
        for (size_t i = 0; i < 40; ++i) {
            const math::matrix<double, 3, 1> corrected = result.correction * covisible[i].location;
            const math::matrix<double, 3, 1> in_loop_camera = result.relative * (covisible_pose * covisible[i].location);
            for (size_t axis = 0; axis < 3; ++axis) {
                REQUIRE(is_value_approx(corrected[axis], first[i].location[axis], 1e-6));
                REQUIRE(is_value_approx(in_loop_camera[axis], first[i].location[axis], 1e-6));
            }
        }
        REQUIRE(!closure.detect(40, current_pose, test_camera(), unconnected, first.data(), first.size()).found);
    }

    {
        const mapping::loop_closure::result result = closure.detect(40, current_pose, test_camera(), unconnected, revisit.data(), revisit.size());
        REQUIRE(result.found);
        REQUIRE(result.keyframe_id == 0);
        REQUIRE(result.correspondences == 36);
        REQUIRE(result.inliers == 35);
        REQUIRE(result.matches.size() == 35);
        for (const mapping::loop_closure::correspondence& match : result.matches) {
            REQUIRE(match.landmark_id == match.recorded_landmark_id + 1000);
            REQUIRE(match.recorded_landmark_id < 35);
        }
        REQUIRE(is_value_approx(result.correction.scale(), 1.0 / 1.1, 1e-6));
        for (size_t i = 0; i < 35; ++i) {
            const math::matrix<double, 3, 1> corrected = result.correction * revisit[i].location;
            const math::matrix<double, 3, 1> in_loop_camera = result.relative * (current_pose * revisit[i].location);
            for (size_t axis = 0; axis < 3; ++axis) {
                REQUIRE(is_value_approx(corrected[axis], first[i].location[axis], 1e-6));
                REQUIRE(is_value_approx(in_loop_camera[axis], first[i].location[axis], 1e-6));
            }
        }
    }

    // Scored by reprojection the same revisit verifies to the same similarity.
    {
        closure.set_reprojection_hypotheses(true);
        const mapping::loop_closure::result result = closure.detect(40, current_pose, test_camera(), unconnected, revisit.data(), revisit.size());
        closure.set_reprojection_hypotheses(false);
        REQUIRE(result.found);
        REQUIRE(result.keyframe_id == 0);
        REQUIRE(result.inliers == 35);
        REQUIRE(is_value_approx(result.correction.scale(), 1.0 / 1.1, 1e-6));
        for (size_t i = 0; i < 35; ++i) {
            const math::matrix<double, 3, 1> corrected = result.correction * revisit[i].location;
            for (size_t axis = 0; axis < 3; ++axis) {
                REQUIRE(is_value_approx(corrected[axis], first[i].location[axis], 1e-6));
            }
        }
    }

    {
        std::vector<mapping::loop_closure::record> misobserved = revisit;
        for (size_t i = 0; i < misobserved.size(); ++i) {
            misobserved[i].landmark_id = 5000 + static_cast<int>(i);
        }
        observe_records(misobserved, math::se3<double>(math::so3<double>::exp({ { 0.0, 0.05, 0.0 } }), { { 0.0, 0.0, 0.0 } }) * current_pose);
        const mapping::loop_closure::result result = closure.detect(40, current_pose, test_camera(), unconnected, misobserved.data(), misobserved.size());
        REQUIRE(!result.found);
        REQUIRE(result.inliers < mapping::loop_closure::min_inliers);
        closure.set_reprojection_hypotheses(true);
        REQUIRE(!closure.detect(40, current_pose, test_camera(), unconnected, misobserved.data(), misobserved.size()).found);
        closure.set_reprojection_hypotheses(false);
    }

    {
        const mapping::loop_closure::result sparse = closure.detect(40, identity, test_camera(), unconnected, revisit.data(), 2);
        REQUIRE(!sparse.found);
        REQUIRE(sparse.correspondences == 2);
        std::vector<mapping::loop_closure::record> scattered = revisit;
        for (size_t i = 0; i < scattered.size(); ++i) {
            scattered[i].location = math::matrix<double, 3, 1>{ { random.get_random(-3.0, 3.0), random.get_random(-2.0, 2.0), random.get_random(6.0, 12.0) } };
        }
        observe_records(scattered, identity);
        const mapping::loop_closure::result inconsistent = closure.detect(40, identity, test_camera(), unconnected, scattered.data(), scattered.size());
        REQUIRE(!inconsistent.found);
        REQUIRE(inconsistent.correspondences == 0);
        REQUIRE(inconsistent.inliers < mapping::loop_closure::min_inliers);
    }

    {
        std::vector<mapping::loop_closure::record> redetected = revisit;
        for (size_t i = 0; i < redetected.size(); ++i) {
            redetected[i].landmark_id = 2000 + static_cast<int>(i);
        }
        const mapping::loop_closure::result result = closure.detect(40, current_pose, test_camera(), unconnected, redetected.data(), redetected.size());
        REQUIRE(result.found);
        REQUIRE(result.keyframe_id == 0);
        REQUIRE(result.correspondences == 36);
        REQUIRE(result.inliers == 35);
        REQUIRE(result.matches.size() == 35);
        for (const mapping::loop_closure::correspondence& match : result.matches) {
            REQUIRE(match.landmark_id >= 2000);
            REQUIRE(match.recorded_landmark_id == match.landmark_id - 2000);
        }
        REQUIRE(is_value_approx(result.correction.scale(), 1.0 / 1.1, 1e-6));

        std::vector<mapping::loop_closure::record> mixed = revisit;
        for (size_t i = 0; i < mixed.size(); i += 4) {
            mixed[i].landmark_id = static_cast<int>(i);
        }
        const mapping::loop_closure::result result_mixed = closure.detect(40, current_pose, test_camera(), unconnected, mixed.data(), mixed.size());
        REQUIRE(result_mixed.found);
        REQUIRE(result_mixed.correspondences == 40);
        REQUIRE(result_mixed.inliers == 35);
        size_t by_descriptor = 0;
        for (const mapping::loop_closure::correspondence& match : result_mixed.matches) {
            by_descriptor += (match.landmark_id != match.recorded_landmark_id) ? 1 : 0;
        }
        REQUIRE(by_descriptor == 26);

        std::vector<mapping::loop_closure::record> foreign;
        for (int i = 0; i < 40; ++i) {
            foreign.push_back(random_record(random, 3000 + i));
        }
        observe_records(foreign, current_pose);
        const mapping::loop_closure::result result_foreign = closure.detect(40, current_pose, test_camera(), unconnected, foreign.data(), foreign.size());
        REQUIRE(!result_foreign.found);
    }

    {
        mapping::covisibility connected;
        for (int landmark = 0; landmark < mapping::loop_closure::max_covisible_landmarks; ++landmark) {
            const int frames[2] = { 0, 40 };
            connected.add(&frames[0], 2);
        }
        REQUIRE(!closure.detect(40, current_pose, test_camera(), connected, revisit.data(), revisit.size()).found);
        mapping::covisibility weakly_connected;
        for (int landmark = 0; landmark + 1 < mapping::loop_closure::max_covisible_landmarks; ++landmark) {
            const int frames[2] = { 0, 40 };
            weakly_connected.add(&frames[0], 2);
        }
        REQUIRE(closure.detect(40, current_pose, test_camera(), weakly_connected, revisit.data(), revisit.size()).found);
    }

    closure.add_keyframe(41, identity, test_camera(), nullptr, 0);
    REQUIRE(closure.num_keyframes() == 40);
    REQUIRE(!closure.detect(42, identity, test_camera(), unconnected, nullptr, 0).found);

    {
        std::vector<feature::descriptor::stored> descriptors;
        for (const mapping::loop_closure::record& record : first) {
            descriptors.push_back(record.descriptor);
        }
        const std::vector<int> recalled = closure.recall(descriptors.data(), descriptors.size(), 40);
        REQUIRE(!recalled.empty());
        REQUIRE(recalled.front() == 0);
        closure.remove_keyframe(0);
        REQUIRE(closure.num_keyframes() == 39);
        for (const int keyframe_id : closure.recall(descriptors.data(), descriptors.size(), 40)) {
            REQUIRE(keyframe_id != 0);
        }
        REQUIRE(!closure.detect(40, current_pose, test_camera(), unconnected, revisit.data(), revisit.size()).found);
    }

    // An alias that outranks the true loop, with the same descriptors on unrelated geometry, fails verification and the true loop ranked behind it is still found.
    {
        mapping::loop_closure aliased;
        std::vector<mapping::loop_closure::record> alias = first;
        for (size_t i = 0; i < alias.size(); ++i) {
            alias[i].landmark_id = 6000 + static_cast<int>(i);
            alias[i].location = math::matrix<double, 3, 1>{ { random.get_random(-3.0, 3.0), random.get_random(-2.0, 2.0), random.get_random(6.0, 12.0) } };
        }
        observe_records(alias, identity);
        aliased.add_keyframe(0, identity, test_camera(), alias.data(), alias.size());
        aliased.add_keyframe(1, identity, test_camera(), first.data(), first.size());
        const mapping::loop_closure::result result = aliased.detect(40, current_pose, test_camera(), unconnected, revisit.data(), revisit.size());
        REQUIRE(result.found);
        REQUIRE(result.keyframe_id == 1);
        REQUIRE(result.inliers == 35);
        REQUIRE(is_value_approx(result.correction.scale(), 1.0 / 1.1, 1e-6));
    }

    // A covisible revisit far from the world origin with a small scale drift: the correction keeps the scale at one, so the translation that assumed the fitted scale would misplace the map by the scale error times the distance from the origin.
    {
        const math::se3<double> far_pose(math::so3<double>::exp({ { 0.1, 0.4, -0.2 } }), { { 60.0, -20.0, 90.0 } });
        const math::se3<double> far_pose_inverse = far_pose.inverse();
        std::vector<mapping::loop_closure::record> far_first;
        for (int landmark_id = 0; landmark_id < 40; ++landmark_id) {
            mapping::loop_closure::record record = random_record(random, landmark_id);
            record.location = far_pose_inverse * record.location;
            far_first.push_back(record);
        }
        observe_records(far_first, far_pose);
        mapping::loop_closure far_closure;
        far_closure.add_keyframe(0, far_pose, test_camera(), far_first.data(), far_first.size());

        // The map drifted by a similarity, and the drifted camera sees the same image, each point at 1.005 times its depth.
        const double scale = 1.005;
        const math::so3<double> drift_rotation = math::so3<double>::exp({ { 0.0, 0.0, 0.01 } });
        const math::matrix<double, 3, 1> drift_translation{ { 0.3, -0.2, 0.1 } };
        const math::sim3<double> far_drift(math::se3<double>(drift_rotation, drift_translation), scale);
        const math::so3<double> revisit_rotation = far_pose.rotation() * drift_rotation.inverse();
        const math::se3<double> revisit_pose(revisit_rotation, (far_pose.translation() * scale) - (revisit_rotation * drift_translation));
        std::vector<mapping::loop_closure::record> far_revisit = far_first;
        for (size_t i = 0; i < far_revisit.size(); ++i) {
            far_revisit[i].location = far_drift * far_first[i].location;
        }
        observe_records(far_revisit, revisit_pose);
        for (size_t i = 0; i < far_revisit.size(); ++i) {
            REQUIRE(std::abs(far_revisit[i].pixel_x - far_first[i].pixel_x) < 1.0e-3f);
            REQUIRE(std::abs(far_revisit[i].pixel_y - far_first[i].pixel_y) < 1.0e-3f);
        }
        const mapping::loop_closure::result result = far_closure.detect(40, revisit_pose, test_camera(), unconnected, far_revisit.data(), far_revisit.size());
        REQUIRE(result.found);
        REQUIRE(result.keyframe_id == 0);
        REQUIRE(result.inliers == 40);
        REQUIRE(is_value_approx(result.correction.scale(), 1.0, 1e-9));
        math::matrix<double, 3, 1> centroid = math::matrix<double, 3, 1>::zero();
        for (const mapping::loop_closure::record& record : far_first) {
            centroid = centroid + record.location;
        }
        centroid = centroid * (1.0 / static_cast<double>(far_first.size()));
        for (size_t i = 0; i < far_revisit.size(); ++i) {
            // With the scale held at one, what remains is the scale error about the cloud's centroid, (s - 1)(X - c).
            const math::matrix<double, 3, 1> error = (result.correction * far_revisit[i].location) - far_first[i].location;
            const math::matrix<double, 3, 1> expected = (far_first[i].location - centroid) * (scale - 1.0);
            REQUIRE(std::sqrt((error - expected).get_length_squared()) < 1.0e-3);
        }
    }

    // The ibow place recognition proposes the keyframes of an earlier visit, its landmarks detected again under new ids,
    // and verification closes the loop; switching to it indexes the keyframes already held again.
    {
        core::random_pcg sequence_random(0x1b0e5eedull);
        mapping::loop_closure sequence;
        std::vector<std::vector<mapping::loop_closure::record>> places;
        int landmark_id = 0;
        for (int place = 0; place < 12; ++place) {
            std::vector<mapping::loop_closure::record> seen;
            for (int i = 0; i < 40; ++i) {
                seen.push_back(random_record(sequence_random, landmark_id++));
            }
            observe_records(seen, identity);
            places.push_back(seen);
            for (int view = 0; view < 3; ++view) {
                const int keyframe_id = ((place * 3) + view) * 5;
                if ((place == 6) && (view == 0)) {
                    // Half way the place recognition switches, keeping what it held.
                    REQUIRE(sequence.get_place_recognition() == mapping::place_recognition::engine::hbst);
                    sequence.set_place_recognition(mapping::place_recognition::engine::ibow);
                    REQUIRE(sequence.get_place_recognition() == mapping::place_recognition::engine::ibow);
                    REQUIRE(sequence.num_keyframes() == static_cast<size_t>(place * 3 + view));
                }
                REQUIRE(!sequence.detect(keyframe_id, identity, test_camera(), unconnected, seen.data(), seen.size()).found);
                sequence.add_keyframe(keyframe_id, identity, test_camera(), seen.data(), seen.size());
            }
        }
        REQUIRE(sequence.num_keyframes() == 36);
        std::vector<mapping::loop_closure::record> again = places[1];
        for (size_t i = 0; i < again.size(); ++i) {
            again[i].landmark_id = 9000 + static_cast<int>(i);
            again[i].location = drift * places[1][i].location;
            for (int flip = 0; flip < 3; ++flip) {
                again[i].descriptor[sequence_random.get_random_raw() % 32] ^= static_cast<unsigned char>(1u << (sequence_random.get_random_raw() % 8));
            }
        }
        observe_records(again, current_pose);
        const mapping::loop_closure::result result = sequence.detect(400, current_pose, test_camera(), unconnected, again.data(), again.size());
        REQUIRE(result.found);
        REQUIRE((result.keyframe_id >= 15) && (result.keyframe_id <= 25));
        REQUIRE(result.inliers >= mapping::loop_closure::min_inliers);
        REQUIRE(is_value_approx(result.correction.scale(), 1.0 / 1.1, 1e-6));
        for (const mapping::loop_closure::correspondence& match : result.matches) {
            REQUIRE(match.landmark_id == match.recorded_landmark_id + 9000 - 40);
        }
    }

    return EXIT_SUCCESS;
}
