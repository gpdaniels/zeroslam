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
#ifndef ZEROSLAM_MAPPING_LOOP_CLOSURE_HPP
#define ZEROSLAM_MAPPING_LOOP_CLOSURE_HPP

#include "estimation/correspondence_3d_3d.hpp"
#include "feature/descriptor/binary.hpp"
#include "mapping/covisibility.hpp"
#include "mapping/place_recognition.hpp"
#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "sensor/camera.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <functional>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace mapping {
    class loop_closure final {
    public:
        class record final {
        public:
            int landmark_id;
            feature::descriptor::stored descriptor;
            math::matrix<double, 3, 1> location;
            float pixel_x;
            float pixel_y;
        };

        class correspondence final {
        public:
            int landmark_id;
            int recorded_landmark_id;
        };

        class result final {
        public:
            bool found;
            int keyframe_id;
            size_t correspondences;
            size_t inliers;
            math::sim3<double> correction;
            math::sim3<double> relative;
            std::vector<correspondence> matches;
            // Verified by min_provisional_inliers rather than by the share of its initial pairs or min_inliers_any_share, so
            // reported only once rescaling_confirmations keyframes verify it.
            bool provisional;
        };

        constexpr static const size_t max_candidates = 20;
        constexpr static const size_t max_verified_candidates = 5;
        constexpr static const int min_keyframe_gap = 30;
        constexpr static const int max_covisible_landmarks = 15;
        constexpr static const double max_covisible_fraction = 0.5;
        constexpr static const double covisible_loop_min_translation = 0.25;
        constexpr static const double covisible_loop_min_relative_translation = 0.1;
        constexpr static const float match_hamming_maximum = 50.0f;
        constexpr static const float match_ratio = 0.8f;
        constexpr static const size_t gms_minimum_pairs = 40;
        constexpr static const double inlier_radius_fraction = 0.05;
        constexpr static const double guided_search_radius = 10.0;
        constexpr static const float guided_hamming_maximum = 80.0f;
        constexpr static const int refine_rounds = 10;
        constexpr static const double reprojection_inlier_bound_squared = 9.210;
        constexpr static const size_t min_inliers = 15;
        constexpr static const double min_inlier_fraction = 0.5;
        constexpr static const size_t foreign_min_inliers = 25;
        constexpr static const double foreign_min_inlier_fraction = 0.3;
        constexpr static const double max_scale_ratio = 2.0;
        // A loop between keyframes of one submap whose scale is beyond max_scale_ratio, as a monocular map drifts to over a
        // long loop (KITTI 06 and 07, 2.5 and 3.3 over 1.2 and 0.7 km), is taken only once rescaling_confirmations
        // keyframes verify it, each within max_confirmation_scale_change of the loop's scale, and never beyond
        // max_rescaling_ratio.
        constexpr static const double max_rescaling_ratio = 10.0;
        constexpr static const size_t rescaling_confirmations = 3;
        constexpr static const double max_confirmation_scale_change = 1.2;
        // With accept_by_inliers, a similarity this many reprojection inliers support is accepted whatever share of the
        // initial pairs they are: the guided search finds most of a true loop's inliers, and the share of the initial pairs
        // does not count them (79 of the 224 true pairs verification refused for inliers on EuRoC had 15 or more, up to 83
        // of 274).
        constexpr static const size_t min_inliers_any_share = 40;
        // With provisional_loops, a similarity this many reprojection inliers support whatever share of the initial pairs
        // they are is taken provisionally, to be confirmed by rescaling_confirmations keyframes: a true loop on KITTI 07,
        // matched against one keyframe's records, keeps 26-28 inliers once the refinement starts from the hypothesis.
        constexpr static const size_t min_provisional_inliers = 25;
        // A loop that waits for confirmation is checked against at most this many keyframes covisible with the keyframe that
        // found it, and dropped when this many keyframes in a row then fail to verify it (ORB-SLAM3's verification in
        // covisible keyframes and in time).
        constexpr static const size_t max_confirming_covisibles = 5;
        constexpr static const size_t max_confirmation_failures = 2;

    private:
        class keyframe final {
        public:
            math::se3<double> pose;
            sensor::model camera;
            std::vector<record> records;
        };

        // A loop verified by fewer keyframes than loop_confirmations, held until the next keyframes confirm or drop it: its
        // similarity as the camera of the last keyframe that verified it to the candidate's camera.
        class pending_loop final {
        public:
            bool active;
            int candidate_id;
            int keyframe_id;
            math::sim3<double> correction;
            math::sim3<double> relative;
            size_t confirmations;
            size_t required;
            size_t failures;
        };

        place_recognition recognition;
        std::unordered_map<int, keyframe> keyframes;
        float hamming_scale = 1.0f;
        bool covisible_revisits = true;
        // Whether the first similarity of a verification comes from hypotheses scored by reprojection into both keyframes,
        // as ORB-SLAM3 scores them, rather than by 3D distance, whose bound of a share of the cloud's radius suits monocular
        // depth poorly: the best 3D hypothesis of a true EuRoC loop held a median 8 of its 35 pairs.
        bool reprojection_hypotheses = true;
        // Whether min_inliers_any_share inliers are enough whatever share of the initial pairs they are. With these two and
        // the held keyframes refreshed (slam's refresh_loop_records), full EuRoC gave 0.63 of the default's error per run
        // and the 35 scene harness 0.83; without the refresh the two of them bent ETH3D planar_2 from 0.16 to 55 cm.
        bool accept_by_inliers = true;
        // Whether the refinement starts from the pairs the first similarity reprojects within the guided search's radius in
        // both keyframes, as ORB-SLAM3 optimises its similarity over the matches it finds by projection with it, rather
        // than from every pair. Off: on the harness it kept ETH3D planar_2 whole but closed a loop that bent LaMAria R_01
        // from 9.6 to 81 cm.
        bool refine_from_hypothesis = false;
        // Whether a verification that meets neither the share of its initial pairs nor min_inliers_any_share is still taken,
        // as provisional, with min_provisional_inliers. Off: it closes KITTI 07's loop to the start of the sequence (21.2 ->
        // 0.9 m), but confirmed loops between nearby places taken so bent LaMAria R_01 from 7.3 to 21 cm.
        bool provisional_loops = false;
        // How many keyframes must verify a loop before it is reported, the keyframe that found it included.
        size_t loop_confirmations = 1;
        pending_loop pending;

        // Similarities rhs = s R lhs + t from three pairs at a time, each supported by the pairs that reproject within the
        // inlier bound in both keyframes, and the best refitted on its supporters.
        static bool reprojection_consensus(const math::se3<double>& pose, const sensor::model& camera, const keyframe& candidate, const record* const keyframe_records, const std::vector<estimation::correspondence_3d_3d<double>>& correspondences, const std::vector<std::pair<size_t, size_t>>& pair_records, double (&rotation)[3][3], double (&translation)[3], double& scale, size_t& supporters);

        // The candidate's records of the keyframe's landmarks, by landmark id.
        static void shared_landmarks(const record* const keyframe_records, const size_t keyframe_records_size, const keyframe& candidate, std::vector<estimation::correspondence_3d_3d<double>>& correspondences, std::vector<correspondence>& pairs, std::vector<std::pair<size_t, size_t>>& pair_records);

        bool covisible_revisit_loop(const int keyframe_id, const math::se3<double>& pose, const sensor::model& camera, const record* const keyframe_records, const int candidate_id, const keyframe& candidate, const std::vector<estimation::correspondence_3d_3d<double>>& correspondences, const std::vector<correspondence>& pairs, const std::vector<std::pair<size_t, size_t>>& pair_records, result& outcome) const;

        // The pairs the similarity predicts within the guided search's radius, mutual nearest descriptors in both keyframes,
        // among the records neither side has paired yet; returns how many were added.
        size_t guided_pairs(const math::se3<double>& pose, const sensor::model& camera, const record* const keyframe_records, const size_t keyframe_records_size, const keyframe& candidate, const math::sim3<double>& correction, std::vector<unsigned char>& current_paired, std::vector<unsigned char>& recorded_paired, std::vector<estimation::correspondence_3d_3d<double>>& correspondences, std::vector<correspondence>& pairs, std::vector<std::pair<size_t, size_t>>& pair_records) const;

        // Whether a pair reprojects through the similarity within the bound in both keyframes.
        static bool reprojects(const math::se3<double>& pose, const sensor::model& camera, const record& current_record, const keyframe& candidate, const record& recorded_record, const estimation::correspondence_3d_3d<double>& correspondence, const math::sim3<double>& similarity, const math::sim3<double>& similarity_inverse, const double bound_squared);

        // The similarity refined over the seeded pairs by their reprojection into both keyframes, the pairs then beyond the
        // inlier bound dropped and the rest solved again.
        static math::sim3<double> refine_similarity(const math::se3<double>& pose, const sensor::model& camera, const record* const keyframe_records, const keyframe& candidate, const std::vector<estimation::correspondence_3d_3d<double>>& correspondences, const std::vector<std::pair<size_t, size_t>>& pair_records, const std::vector<unsigned char>& seeded, const size_t seeded_count, const math::sim3<double>& initial);

        // How many of a keyframe's records the similarity explains against the candidate's: the pairs shared by id and those
        // the guided search finds, the similarity refined over them and the inliers counted, which matches receives; none
        // when the refined scale strays beyond max_confirmation_scale_change of the given one.
        size_t confirm(const math::se3<double>& pose, const sensor::model& camera, const record* const keyframe_records, const size_t keyframe_records_size, const keyframe& candidate, math::sim3<double>& correction, std::vector<correspondence>& matches) const;

        // A verified loop as reported: at once when loop_confirmations is one or enough keyframes covisible with the keyframe
        // verify it as well, otherwise held as the pending loop and not found. A loop within one submap that rescales the map
        // beyond max_scale_ratio needs rescaling_confirmations whatever loop_confirmations is.
        result confirm_or_hold(const int keyframe_id, const int submap_start_id, const covisibility& graph, const result& verified);

        bool verify_candidate(const int keyframe_id, const math::se3<double>& pose, const sensor::model& camera, const record* const keyframe_records, const size_t keyframe_records_size, const std::vector<feature::descriptor::stored>& query, const int submap_start_id, const int candidate_id, const keyframe& candidate, result& outcome) const;

    public:
        loop_closure();

    public:
        // With seek_foreign the keyframes before the submap's start are searched as well on their own, so that the recent
        // keyframes of a submap that stands apart, which the place recognition ranks highest, cannot hide the map it lost.
        // With the ibow place recognition the islands' best keyframes, recent and covisible ones left out, are verified
        // first and the covisible keyframes it ranks are then tried as revisits; with hbst the ranked candidates are taken
        // in turn, covisible ones as revisits.
        result detect(const int keyframe_id, const math::se3<double>& pose, const sensor::model& camera, const covisibility& graph, const record* const keyframe_records, const size_t keyframe_records_size, const int submap_start_id = 0, const bool seek_foreign = false);

        void add_keyframe(const int keyframe_id, const math::se3<double>& pose, const sensor::model& camera, const record* const keyframe_records, const size_t keyframe_records_size);

        void remove_keyframe(const int keyframe_id);

        // Brings the held keyframes up to date with the map: each keyframe's pose from pose_of, and each record's location
        // from location_of, a record whose landmark is gone dropped; a keyframe pose_of does not know keeps its pose.
        void refresh(const std::function<bool(const int, math::se3<double>&)>& pose_of, const std::function<bool(const int, math::matrix<double, 3, 1>&)>& location_of);

        // Carries the held keyframes with the map when a loop closes: each keyframe camera_to_world knows takes the pose of
        // its corrected camera-to-world similarity, and its records go through it from the camera as they were seen, so
        // that they keep reprojecting into its image rather than staying where the map was before the correction.
        void correct(const std::function<bool(const int, math::sim3<double>&)>& camera_to_world);

        size_t num_keyframes() const;

        void set_hamming_scale(const float scale);

        // Switching the place recognition empties it, so it is chosen before any keyframe is added.
        void set_place_recognition(const place_recognition::engine engine);

        place_recognition::engine get_place_recognition() const;

        // Whether a candidate that shares landmarks with the keyframe closes a loop when the map has drifted between the
        // visits. The drift is measured in the world frame, so a map without a fixed one, as relative adjustment holds
        // it, has to leave these to the appearance loops, whose similarity does not depend on the world frame.
        void set_covisible_revisits(const bool enabled);

        // Whether a verification's first similarity is scored by reprojection into both keyframes (see
        // reprojection_hypotheses) rather than by 3D distance.
        void set_reprojection_hypotheses(const bool enabled);

        // Whether a verification accepts min_inliers_any_share inliers whatever share of the initial pairs they are.
        void set_accept_by_inliers(const bool enabled);

        // Whether a verification's refinement starts from the pairs its first similarity explains (see
        // refine_from_hypothesis) rather than from every pair.
        void set_refine_from_hypothesis(const bool enabled);

        // Whether a verification may take a loop provisionally (see provisional_loops).
        void set_provisional_loops(const bool enabled);

        // How many keyframes must verify a loop before detect reports it, as ORB-SLAM3 confirms a loop by three: the keyframe
        // that found it, then up to max_confirming_covisibles keyframes covisible with it, then the keyframes that follow,
        // each matched by projection through the loop's similarity; one reports a loop as soon as it verifies.
        void set_loop_confirmations(const size_t count);

        std::vector<int> recall(const feature::descriptor::stored* const descriptors, const size_t descriptors_size, const size_t max_recalled) const;

        const record* records_of(const int keyframe_id, size_t& records_size) const;
    };
}

#endif // ZEROSLAM_MAPPING_LOOP_CLOSURE_HPP
