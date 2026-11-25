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
#ifndef ZEROSLAM_MAPPING_GYROSCOPE_INTRINSICS_HPP
#define ZEROSLAM_MAPPING_GYROSCOPE_INTRINSICS_HPP

#include "math/matrix.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "sensor/imu/measurement.hpp"
#include "sensor/imu/preintegration.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace mapping {
    // The matrix of an imu's gyroscope (sensor::imu::intrinsics) as a visual inertial map holds and refines it, after Van Meter and
    // Paul, "SLAM Dunk: A Recipe for Long-Horizon Egocentric SLAM in the Wild" (2026): one matrix M for the whole run, the angular
    // velocity being (I + M) * w_measured - b_g, zero until an adjustment has refined it, with a zero mean Gaussian prior on each
    // of its nine entries (optimisation::edges::gyroscope_intrinsics_prior).
    //
    // Once the matrix is asked for, every inertial edge of an adjustment takes its vertex, one shared by them all, and the frames
    // the imu predicts and the keyframes preintegrated from then on are integrated at the estimate. The vertex is free only where
    // the matrix can be told from the rest of the state, and held at the estimate elsewhere (mapping::inertial::add_factors):
    //   - In an adjustment of the whole map with the landmarks free. A window holds a few seconds of motion between poses held at
    //     its edge, which shows little of a matrix, and what each window made of it would differ from the last.
    //   - Once the inertial initialisation and its refinements are done. Until then gravity, the scale and the accelerometer bias
    //     still move, by more than a matrix of a percent moves the measurements.
    //   - Once the motion has turned about all three axes. A column of the matrix multiplies one axis of the angular velocity, so a
    //     motion that does not turn about an axis shows nothing of its column, and a steady turn is the gyroscope bias's. The
    //     measure (excitation) is the turn over each inertial edge less the mean rate's share of it, the outer products summed:
    //     the smallest eigenvalue of that sum, the sum of the squared turns about the least turned axis, must reach
    //     excitation_minimum.
    // The prior then holds what the motion shows least, and the whole of the matrix near zero where the map disagrees with the imu
    // for another reason.
    class gyroscope_intrinsics final {
    public:
        // The turns of a set of preintegrated measurements about the axes of the imu.
        class excitation final {
        private:
            std::vector<std::pair<math::matrix<double, 3, 1>, double>> turns;

        public:
            // Takes the rotation of a measurement and the time it spans.
            void add(const sensor::imu::preintegration& measurement);

            size_t count() const;

            // The sum over the measurements of the squared turn about the least turned axis, each turn less the mean rate over its
            // time: the smallest eigenvalue of the sum of the outer products, rad^2. Zero with fewer than three measurements.
            double weakest() const;
        };

        // The least turning that frees the matrix, rad^2. The turn between two keyframes is good to about half a milliradian, the
        // cameras' share of it, so turns of 0.05 rad^2 in sum show an entry of the matrix as well as the prior's 0.002 holds it:
        // from there the estimate is the motion's as much as the prior's, and below it the prior's and whatever else the inertial
        // edges disagree with the map about. Three simulated runs of thirty seconds, the slam test's and two more motions, with
        // entries of 0.004 to 0.012, gave the matrix within 0.0012 to 0.0017 and the frames 1.5 to 2.2 mrad from the truth, for
        // 2.5 to 4.9 mrad with the matrix held at zero. A minimum of 0.25 rad^2 left the first two at 2.4 and 1.7 mrad, for 2.1
        // and 1.5 mrad, and the third, whose turns had come to 0.21 rad^2 by its end, as it was.
        constexpr static const double excitation_minimum = 0.05;

    private:
        bool refined_ = false;
        bool settled_ = false;
        sensor::imu::intrinsics estimate_;
        size_t refinements_ = 0;

    public:
        // Whether the adjustments are asked to refine the matrix.
        void set_refined(const bool refined);
        bool refined() const;

        // Whether the inertial initialisation and its refinements are done.
        void set_settled(const bool settled);
        bool settled() const;

        // The matrix as it is held, and how many adjustments have refined it.
        const sensor::imu::intrinsics& estimate() const;
        void set_estimate(const sensor::imu::intrinsics& estimate);
        size_t refinements() const;

        // Whether the inertial edges take the matrix's vertex: when it is refined, or was.
        bool active() const;

        // Whether an adjustment frees the matrix: one of the whole map that does not hold the calibrations, over measurements that
        // turn enough.
        bool frees(const bool whole_map, const bool held, const excitation& turned) const;

        // The vertex of the matrix at the estimate, free or held, and the prior of a free one.
        optimisation::vertex vertex(const bool free) const;
        void add_prior(optimisation::factor_graph& graph, optimisation::vertex* const node) const;

        // Takes the matrix an adjustment left in a free vertex.
        void read(const optimisation::vertex& node);

        // Integrates a measurement again when it was not integrated at the estimate.
        void align(sensor::imu::preintegration& measurement) const;
    };
}

#endif // ZEROSLAM_MAPPING_GYROSCOPE_INTRINSICS_HPP
