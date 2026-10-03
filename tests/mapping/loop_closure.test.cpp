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
    // The checks on this detector are of the verification scored in 3D and accepted by the share of its initial pairs, which
    // each check of another way switches back to; the defaults are checked on their own below.
    closure.set_reprojection_hypotheses(false);
    closure.set_accept_by_inliers(false);
    closure.set_refine_from_hypothesis(false);
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

    // The default verification closes the same revisit to the same similarity.
    {
        mapping::loop_closure defaults;
        defaults.add_keyframe(0, identity, test_camera(), first.data(), first.size());
        const mapping::loop_closure::result result = defaults.detect(40, current_pose, test_camera(), unconnected, revisit.data(), revisit.size());
        REQUIRE(result.found);
        REQUIRE(result.keyframe_id == 0);
        REQUIRE(result.inliers == 35);
        for (const mapping::loop_closure::correspondence& match : result.matches) {
            REQUIRE(match.landmark_id == match.recorded_landmark_id + 1000);
            REQUIRE(match.recorded_landmark_id < 35);
        }
        REQUIRE(is_value_approx(result.correction.scale(), 1.0 / 1.1, 1e-6));
    }

    // A revisit most of whose inliers are found by the guided search, their descriptors too far from the recorded ones to be
    // paired at first, while more than half of the initial pairs are landmarks misplaced along their rays (monocular depth),
    // which still match by descriptor and image motion: refused for the share of the initial pairs unless the inliers are
    // accepted whatever share they are.
    {
        core::random_pcg accepting_random(0xacce97ull);
        constexpr static const size_t paired_first = 20;
        constexpr static const size_t guided_only = 30;
        constexpr static const size_t misplaced = 25;
        std::vector<mapping::loop_closure::record> earlier;
        for (int landmark_id = 0; landmark_id < static_cast<int>(paired_first + guided_only + misplaced); ++landmark_id) {
            earlier.push_back(random_record(accepting_random, landmark_id));
        }
        observe_records(earlier, identity);
        mapping::loop_closure accepting;
        accepting.set_reprojection_hypotheses(false);
        accepting.set_accept_by_inliers(false);
        accepting.set_refine_from_hypothesis(false);
        accepting.add_keyframe(0, identity, test_camera(), earlier.data(), earlier.size());
        std::vector<mapping::loop_closure::record> later = earlier;
        for (size_t i = 0; i < later.size(); ++i) {
            later[i].landmark_id = 2000 + static_cast<int>(i);
            later[i].location = drift * earlier[i].location;
        }
        observe_records(later, current_pose);
        for (size_t i = paired_first; i < paired_first + guided_only; ++i) {
            for (size_t bit = 0; bit < 60; ++bit) {
                later[i].descriptor[bit / 8] = static_cast<unsigned char>(later[i].descriptor[bit / 8] ^ (1u << (bit % 8)));
            }
        }
        // Each by its own factor, as one factor about the camera centre would be a similarity they all fit.
        const math::matrix<double, 3, 1> centre = current_pose.inverse().translation();
        for (size_t i = paired_first + guided_only; i < later.size(); ++i) {
            later[i].location = centre + ((later[i].location - centre) * (1.5 + (0.25 * static_cast<double>(i % 7))));
        }
        const mapping::loop_closure::result refused = accepting.detect(40, current_pose, test_camera(), unconnected, later.data(), later.size());
        REQUIRE(!refused.found);
        REQUIRE(refused.inliers == paired_first + guided_only);
        accepting.set_accept_by_inliers(true);
        const mapping::loop_closure::result result = accepting.detect(40, current_pose, test_camera(), unconnected, later.data(), later.size());
        REQUIRE(result.found);
        REQUIRE(result.keyframe_id == 0);
        REQUIRE(result.correspondences == later.size());
        REQUIRE(result.inliers == paired_first + guided_only);
        REQUIRE(result.inliers >= mapping::loop_closure::min_inliers_any_share);
        for (const mapping::loop_closure::correspondence& match : result.matches) {
            REQUIRE(match.landmark_id == match.recorded_landmark_id + 2000);
            REQUIRE(match.recorded_landmark_id < static_cast<int>(paired_first + guided_only));
        }
        REQUIRE(is_value_approx(result.correction.scale(), 1.0 / 1.1, 1e-6));
    }

    // A verification with fewer inliers than min_inliers_any_share, under half of its initial pairs, is refused unless loops
    // may be taken provisionally, and then the loop is held until two more keyframes verify it.
    {
        core::random_pcg provisional_random(0x9a0715ull);
        constexpr static const size_t paired_first = 10;
        constexpr static const size_t guided_only = 20;
        constexpr static const size_t misplaced = 15;
        std::vector<mapping::loop_closure::record> earlier;
        for (int landmark_id = 0; landmark_id < static_cast<int>(paired_first + guided_only + misplaced); ++landmark_id) {
            earlier.push_back(random_record(provisional_random, landmark_id));
        }
        observe_records(earlier, identity);
        const auto later_from = [&](const math::se3<double>& from) {
            std::vector<mapping::loop_closure::record> later = earlier;
            for (size_t i = 0; i < later.size(); ++i) {
                later[i].landmark_id = 7000 + static_cast<int>(i);
                later[i].location = drift * earlier[i].location;
            }
            observe_records(later, from);
            for (size_t i = paired_first; i < paired_first + guided_only; ++i) {
                for (size_t bit = 0; bit < 60; ++bit) {
                    later[i].descriptor[bit / 8] = static_cast<unsigned char>(later[i].descriptor[bit / 8] ^ (1u << (bit % 8)));
                }
            }
            const math::matrix<double, 3, 1> centre = from.inverse().translation();
            for (size_t i = paired_first + guided_only; i < later.size(); ++i) {
                later[i].location = centre + ((later[i].location - centre) * (1.5 + (0.25 * static_cast<double>(i % 7))));
            }
            return later;
        };
        for (const bool provisional : { false, true }) {
            mapping::loop_closure taking;
            taking.set_provisional_loops(provisional);
            taking.add_keyframe(0, identity, test_camera(), earlier.data(), earlier.size());
            for (int keyframe_id = 40; keyframe_id < 43; ++keyframe_id) {
                const math::se3<double> from = math::se3<double>(math::so3<double>::identity(), { { 0.05 * static_cast<double>(keyframe_id - 40), 0.0, 0.0 } }) * current_pose;
                const std::vector<mapping::loop_closure::record> later = later_from(from);
                const mapping::loop_closure::result result = taking.detect(keyframe_id, from, test_camera(), unconnected, later.data(), later.size());
                REQUIRE(result.found == (provisional && (keyframe_id == 42)));
                if (result.found) {
                    REQUIRE(result.keyframe_id == 0);
                    REQUIRE(is_value_approx(result.correction.scale(), 1.0 / 1.1, 1e-6));
                }
                taking.add_keyframe(keyframe_id, from, test_camera(), later.data(), later.size());
            }
        }
    }

    // When most of the initial pairs are landmarks misplaced along their rays, a refinement over every pair is pulled away
    // from the inliers, ending with fewer than the first similarity had, and starts again from the pairs that similarity
    // explains, which keeps them; started from those pairs at once it keeps them too.
    {
        core::random_pcg crowded_random(0xc0ded5eedull);
        constexpr static const size_t placed = mapping::loop_closure::min_inliers_any_share + 5;
        std::vector<mapping::loop_closure::record> earlier;
        for (int landmark_id = 0; landmark_id < 100; ++landmark_id) {
            earlier.push_back(random_record(crowded_random, landmark_id));
        }
        observe_records(earlier, identity);
        mapping::loop_closure crowded;
        crowded.set_reprojection_hypotheses(false);
        crowded.set_accept_by_inliers(true);
        crowded.set_refine_from_hypothesis(false);
        crowded.add_keyframe(0, identity, test_camera(), earlier.data(), earlier.size());
        std::vector<mapping::loop_closure::record> later = earlier;
        for (size_t i = 0; i < later.size(); ++i) {
            later[i].landmark_id = 3000 + static_cast<int>(i);
            later[i].location = drift * earlier[i].location;
        }
        observe_records(later, current_pose);
        const math::matrix<double, 3, 1> centre = current_pose.inverse().translation();
        for (size_t i = placed; i < later.size(); ++i) {
            later[i].location = centre + ((later[i].location - centre) * (1.5 + (0.25 * static_cast<double>(i % 7))));
        }
        const mapping::loop_closure::result recovered = crowded.detect(40, current_pose, test_camera(), unconnected, later.data(), later.size());
        REQUIRE(recovered.found);
        REQUIRE(recovered.inliers == placed);
        REQUIRE(is_value_approx(recovered.correction.scale(), 1.0 / 1.1, 1e-6));
        crowded.set_refine_from_hypothesis(true);
        const mapping::loop_closure::result result = crowded.detect(40, current_pose, test_camera(), unconnected, later.data(), later.size());
        REQUIRE(result.found);
        REQUIRE(result.keyframe_id == 0);
        REQUIRE(result.correspondences == later.size());
        REQUIRE(result.inliers == placed);
        for (const mapping::loop_closure::correspondence& match : result.matches) {
            REQUIRE(match.landmark_id == match.recorded_landmark_id + 3000);
            REQUIRE(match.recorded_landmark_id < static_cast<int>(placed));
        }
        REQUIRE(is_value_approx(result.correction.scale(), 1.0 / 1.1, 1e-6));
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

    // Correcting a held keyframe carries its records through its corrected camera, pixels as they were, and a revisit seen
    // in the corrected world verifies against it to the same scale.
    {
        mapping::loop_closure moved;
        moved.set_reprojection_hypotheses(false);
        moved.set_accept_by_inliers(false);
        moved.add_keyframe(0, identity, test_camera(), first.data(), first.size());
        const math::sim3<double> correction(math::se3<double>(math::so3<double>::exp({ { 0.1, -0.2, 0.05 } }), { { 1.0, 2.0, -0.5 } }), 2.0);
        moved.correct([&correction](const int keyframe_id, math::sim3<double>& camera_to_world) {
            if (keyframe_id != 0) {
                return false;
            }
            camera_to_world = correction;
            return true;
        });
        size_t moved_size = 0;
        const mapping::loop_closure::record* const moved_records = moved.records_of(0, moved_size);
        REQUIRE(moved_size == first.size());
        for (size_t i = 0; i < moved_size; ++i) {
            const math::matrix<double, 3, 1> expected = correction * first[i].location;
            for (size_t axis = 0; axis < 3; ++axis) {
                REQUIRE(is_value_approx(moved_records[i].location[axis], expected[axis], 1e-9));
            }
            REQUIRE(moved_records[i].pixel_x == first[i].pixel_x);
            REQUIRE(moved_records[i].pixel_y == first[i].pixel_y);
        }
        std::vector<mapping::loop_closure::record> revisit_moved = revisit;
        for (mapping::loop_closure::record& held : revisit_moved) {
            held.location = correction * held.location;
        }
        const math::se3<double> moved_pose = (correction * math::sim3<double>(current_pose.inverse(), 1.0)).transformation().inverse();
        const mapping::loop_closure::result result = moved.detect(40, moved_pose, test_camera(), unconnected, revisit_moved.data(), revisit_moved.size());
        REQUIRE(result.found);
        REQUIRE(result.keyframe_id == 0);
        REQUIRE(result.inliers == 35);
        REQUIRE(is_value_approx(result.correction.scale(), 1.0 / 1.1, 1e-6));
    }

    // Refreshing takes each record's location from the map and drops the records whose landmark is gone.
    {
        mapping::loop_closure refreshed;
        refreshed.add_keyframe(0, identity, test_camera(), first.data(), first.size());
        const math::matrix<double, 3, 1> shift = { { 0.5, -0.25, 1.0 } };
        const auto pose_of = [](const int, math::se3<double>&) {
            return false;
        };
        const auto location_of = [&first, &shift](const int landmark_id, math::matrix<double, 3, 1>& location) {
            if ((landmark_id % 2) != 0) {
                return false;
            }
            location = first[static_cast<size_t>(landmark_id)].location + shift;
            return true;
        };
        refreshed.refresh(pose_of, location_of);
        size_t records_size = 0;
        const mapping::loop_closure::record* const records = refreshed.records_of(0, records_size);
        REQUIRE(records_size == first.size() / 2);
        for (size_t i = 0; i < records_size; ++i) {
            REQUIRE((records[i].landmark_id % 2) == 0);
            const math::matrix<double, 3, 1> expected = first[static_cast<size_t>(records[i].landmark_id)].location + shift;
            for (size_t axis = 0; axis < 3; ++axis) {
                REQUIRE(is_value_approx(records[i].location[axis], expected[axis], 1e-12));
            }
        }
    }

    // A loop must be verified by three keyframes before it is reported: here two keyframes covisible with the one that finds
    // it see the revisit too and confirm it at once; without them the loop is held until the next keyframes confirm it; and
    // two keyframes in a row that do not verify it drop it, so that it has to be found and confirmed afresh.
    {
        core::random_pcg confirming_random(0xc0f1ae3ull);
        std::vector<mapping::loop_closure::record> place;
        for (int landmark_id = 0; landmark_id < 60; ++landmark_id) {
            place.push_back(random_record(confirming_random, landmark_id));
        }
        const auto seen_from = [](std::vector<mapping::loop_closure::record> records, const math::se3<double>& from) {
            observe_records(records, from);
            return records;
        };
        const auto shifted = [](const math::se3<double>& from, const double x) {
            return math::se3<double>(math::so3<double>::identity(), { { x, 0.0, 0.0 } }) * from;
        };
        std::vector<mapping::loop_closure::record> revisited = place;
        for (size_t i = 0; i < revisited.size(); ++i) {
            revisited[i].landmark_id = 4000 + static_cast<int>(i);
            revisited[i].location = drift * place[i].location;
        }
        std::vector<std::vector<mapping::loop_closure::record>> fillers;
        for (int keyframe_id = 3; keyframe_id < 38; ++keyframe_id) {
            std::vector<mapping::loop_closure::record> records;
            for (int i = 0; i < 30; ++i) {
                records.push_back(random_record(confirming_random, 6000 + (100 * keyframe_id) + i));
            }
            fillers.push_back(seen_from(records, identity));
        }
        std::vector<mapping::loop_closure::record> unrelated;
        for (int i = 0; i < 60; ++i) {
            unrelated.push_back(random_record(confirming_random, 9000 + i));
        }
        const auto earlier_visit = [&](mapping::loop_closure& closure_under_test) {
            closure_under_test.set_loop_confirmations(3);
            for (int keyframe_id = 0; keyframe_id < 3; ++keyframe_id) {
                const std::vector<mapping::loop_closure::record> records = seen_from(place, shifted(identity, 0.15 * static_cast<double>(keyframe_id - 1)));
                closure_under_test.add_keyframe(keyframe_id, shifted(identity, 0.15 * static_cast<double>(keyframe_id - 1)), test_camera(), records.data(), records.size());
            }
            for (int keyframe_id = 3; keyframe_id < 38; ++keyframe_id) {
                const std::vector<mapping::loop_closure::record>& records = fillers[static_cast<size_t>(keyframe_id - 3)];
                closure_under_test.add_keyframe(keyframe_id, identity, test_camera(), records.data(), records.size());
            }
        };

        // Confirmed at once by the two covisible keyframes.
        {
            mapping::loop_closure confirming;
            earlier_visit(confirming);
            mapping::covisibility graph;
            for (size_t i = 0; i < revisited.size(); ++i) {
                const int observers[3] = { 38, 39, 40 };
                graph.update(revisited[i].landmark_id, &observers[0], 3);
            }
            for (int keyframe_id = 38; keyframe_id < 40; ++keyframe_id) {
                const math::se3<double> from = shifted(current_pose, 0.1 * static_cast<double>(keyframe_id - 40));
                const std::vector<mapping::loop_closure::record> records = seen_from(revisited, from);
                confirming.add_keyframe(keyframe_id, from, test_camera(), records.data(), records.size());
            }
            const std::vector<mapping::loop_closure::record> records = seen_from(revisited, current_pose);
            const mapping::loop_closure::result result = confirming.detect(40, current_pose, test_camera(), graph, records.data(), records.size());
            REQUIRE(result.found);
            REQUIRE(result.keyframe_id < 3);
            REQUIRE(result.inliers == revisited.size());
            REQUIRE(is_value_approx(result.correction.scale(), 1.0 / 1.1, 1e-6));
        }

        // Held, then confirmed by the next two keyframes, the loop reported for the last of them.
        {
            mapping::loop_closure holding;
            earlier_visit(holding);
            for (int keyframe_id = 40; keyframe_id < 43; ++keyframe_id) {
                const math::se3<double> from = shifted(current_pose, 0.1 * static_cast<double>(keyframe_id - 40));
                const std::vector<mapping::loop_closure::record> records = seen_from(revisited, from);
                const mapping::loop_closure::result result = holding.detect(keyframe_id, from, test_camera(), unconnected, records.data(), records.size());
                REQUIRE(result.found == (keyframe_id == 42));
                if (result.found) {
                    REQUIRE(result.keyframe_id < 3);
                    REQUIRE(result.inliers == revisited.size());
                    for (const mapping::loop_closure::correspondence& match : result.matches) {
                        REQUIRE(match.landmark_id == match.recorded_landmark_id + 4000);
                    }
                    REQUIRE(is_value_approx(result.correction.scale(), 1.0 / 1.1, 1e-6));
                    const math::sim3<double> expected_relative = math::sim3<double>(shifted(identity, 0.15 * static_cast<double>(result.keyframe_id - 1)), 1.0) * result.correction * math::sim3<double>(from.inverse(), 1.0);
                    for (size_t axis = 0; axis < 3; ++axis) {
                        REQUIRE(is_value_approx(result.relative.transformation().translation()[axis], expected_relative.transformation().translation()[axis], 1e-9));
                    }
                }
                holding.add_keyframe(keyframe_id, from, test_camera(), records.data(), records.size());
            }
        }

        // A loop that rescales the map threefold, as a monocular map drifts to over a long loop, is held for three keyframes
        // even when one is enough for others, and then closes rather than being refused as implausible.
        {
            const math::sim3<double> rescaled(math::se3<double>(math::so3<double>::exp({ { 0.0, 0.0, 0.05 } }), { { 0.4, -0.3, 0.2 } }), 3.0);
            std::vector<mapping::loop_closure::record> far = place;
            for (size_t i = 0; i < far.size(); ++i) {
                far[i].landmark_id = 5000 + static_cast<int>(i);
                far[i].location = rescaled * place[i].location;
            }
            mapping::loop_closure rescaling;
            earlier_visit(rescaling);
            rescaling.set_loop_confirmations(1);
            for (int keyframe_id = 40; keyframe_id < 43; ++keyframe_id) {
                const math::se3<double> from = shifted(current_pose, 0.3 * static_cast<double>(keyframe_id - 40));
                const std::vector<mapping::loop_closure::record> records = seen_from(far, from);
                const mapping::loop_closure::result result = rescaling.detect(keyframe_id, from, test_camera(), unconnected, records.data(), records.size());
                REQUIRE(result.found == (keyframe_id == 42));
                if (result.found) {
                    REQUIRE(result.keyframe_id < 3);
                    REQUIRE(is_value_approx(result.correction.scale(), 1.0 / 3.0, 1e-6));
                }
                rescaling.add_keyframe(keyframe_id, from, test_camera(), records.data(), records.size());
            }
        }

        // Held, dropped after two keyframes that do not verify it, then found and confirmed afresh.
        {
            mapping::loop_closure dropping;
            earlier_visit(dropping);
            for (int keyframe_id = 40; keyframe_id < 46; ++keyframe_id) {
                const math::se3<double> from = shifted(current_pose, 0.1 * static_cast<double>(keyframe_id - 40));
                const bool elsewhere = (keyframe_id == 41) || (keyframe_id == 42);
                const std::vector<mapping::loop_closure::record> records = seen_from(elsewhere ? unrelated : revisited, from);
                const mapping::loop_closure::result result = dropping.detect(keyframe_id, from, test_camera(), unconnected, records.data(), records.size());
                REQUIRE(result.found == (keyframe_id == 45));
                if (!elsewhere) {
                    dropping.add_keyframe(keyframe_id, from, test_camera(), records.data(), records.size());
                }
            }
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
