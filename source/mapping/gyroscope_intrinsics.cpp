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

#include "mapping/gyroscope_intrinsics.hpp"

#include "core/logger.hpp"
#include "math/math.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/gyroscope_intrinsics_prior.hpp"
#include "optimisation/vertices/gyroscope_intrinsics.hpp"

namespace {
    // The smallest eigenvalue of a symmetric 3x3 matrix, in closed form (Smith, "Eigenvalues of a symmetric 3x3 matrix", 1961).
    double smallest_eigenvalue(const math::matrix<double, 3, 3>& value) {
        const double off_diagonal = (value[0][1] * value[0][1]) + (value[0][2] * value[0][2]) + (value[1][2] * value[1][2]);
        if (!(off_diagonal > 0.0)) {
            return math::min(value[0][0], math::min(value[1][1], value[2][2]));
        }
        const double mean = (value[0][0] + value[1][1] + value[2][2]) / 3.0;
        const double spread = ((value[0][0] - mean) * (value[0][0] - mean)) + ((value[1][1] - mean) * (value[1][1] - mean)) + ((value[2][2] - mean) * (value[2][2] - mean)) + (2.0 * off_diagonal);
        const double scale = math::sqrt(spread / 6.0);
        math::matrix<double, 3, 3> shifted = value - (mean * math::matrix<double, 3, 3>::identity());
        shifted = (1.0 / scale) * shifted;
        const double determinant = (shifted[0][0] * ((shifted[1][1] * shifted[2][2]) - (shifted[1][2] * shifted[2][1]))) - (shifted[0][1] * ((shifted[1][0] * shifted[2][2]) - (shifted[1][2] * shifted[2][0]))) + (shifted[0][2] * ((shifted[1][0] * shifted[2][1]) - (shifted[1][1] * shifted[2][0])));
        const double angle = math::acos(math::max(-1.0, math::min(1.0, 0.5 * determinant))) / 3.0;
        // The eigenvalues are mean + 2 scale cos(angle + 2 pi k / 3), the smallest at k = 1.
        return mean + (2.0 * scale * math::cos(angle + (2.0 * math::pi<double>() / 3.0)));
    }
}

namespace mapping {
    void gyroscope_intrinsics::excitation::add(const sensor::imu::preintegration& measurement) {
        this->turns.push_back({ measurement.get_delta_rotation().log(), measurement.get_delta_time() });
    }

    size_t gyroscope_intrinsics::excitation::count() const {
        return this->turns.size();
    }

    double gyroscope_intrinsics::excitation::weakest() const {
        if (this->turns.size() < 3) {
            return 0.0;
        }
        math::matrix<double, 3, 1> total = math::matrix<double, 3, 1>::zero();
        double duration = 0.0;
        for (const std::pair<math::matrix<double, 3, 1>, double>& turn : this->turns) {
            total = total + turn.first;
            duration += turn.second;
        }
        if (!(duration > 0.0)) {
            return 0.0;
        }
        const math::matrix<double, 3, 1> rate = (1.0 / duration) * total;
        math::matrix<double, 3, 3> sum = math::matrix<double, 3, 3>::zero();
        for (const std::pair<math::matrix<double, 3, 1>, double>& turn : this->turns) {
            const math::matrix<double, 3, 1> varying = turn.first - (turn.second * rate);
            sum = sum + (varying * math::transpose(varying));
        }
        const double smallest = smallest_eigenvalue(sum);
        return math::isfinite(smallest) ? math::max(0.0, smallest) : 0.0;
    }

    void gyroscope_intrinsics::set_refined(const bool refined) {
        this->refined_ = refined;
    }

    bool gyroscope_intrinsics::refined() const {
        return this->refined_;
    }

    void gyroscope_intrinsics::set_settled(const bool settled) {
        this->settled_ = settled;
    }

    bool gyroscope_intrinsics::settled() const {
        return this->settled_;
    }

    const sensor::imu::intrinsics& gyroscope_intrinsics::estimate() const {
        return this->estimate_;
    }

    void gyroscope_intrinsics::set_estimate(const sensor::imu::intrinsics& estimate) {
        this->estimate_ = estimate;
    }

    size_t gyroscope_intrinsics::refinements() const {
        return this->refinements_;
    }

    bool gyroscope_intrinsics::active() const {
        return this->refined_ || (this->estimate_.gyroscope != (math::matrix<double, 3, 3>::zero()));
    }

    bool gyroscope_intrinsics::frees(const bool whole_map, const bool held, const excitation& turned) const {
        return this->refined_ && this->settled_ && whole_map && !held && (turned.weakest() >= gyroscope_intrinsics::excitation_minimum);
    }

    optimisation::vertex gyroscope_intrinsics::vertex(const bool free) const {
        optimisation::vertex node{ optimisation::vertices::gyroscope_intrinsics() };
        double parameters[optimisation::vertices::gyroscope_intrinsics::parameter_count];
        optimisation::vertices::gyroscope_intrinsics::to_parameters(this->estimate_, parameters);
        static_cast<void>(node.set_parameters(&parameters[0], static_cast<size_t>(optimisation::vertices::gyroscope_intrinsics::parameter_count)));
        node.set_fixed(!free);
        return node;
    }

    void gyroscope_intrinsics::add_prior(optimisation::factor_graph& graph, optimisation::vertex* const node) const {
        optimisation::edge prior{ optimisation::edges::gyroscope_intrinsics_prior() };
        prior.set_information(optimisation::edges::gyroscope_intrinsics_prior::information());
        static_cast<void>(prior.add_vertex(node));
        static_cast<void>(graph.add_edge(static_cast<optimisation::edge&&>(prior)));
    }

    void gyroscope_intrinsics::read(const optimisation::vertex& node) {
        const sensor::imu::intrinsics refined = optimisation::vertices::gyroscope_intrinsics::from_parameters(node.get_parameters());
        for (size_t i = 0; i < 9; ++i) {
            if (!math::isfinite(refined.gyroscope.data()[i])) {
                return;
            }
        }
        this->estimate_ = refined;
        ++this->refinements_;
        const math::matrix<double, 3, 3>& m = this->estimate_.gyroscope;
        core::logger::log(core::logger::level::info, "Gyroscope matrix refined (%zu): [%.6f %.6f %.6f; %.6f %.6f %.6f; %.6f %.6f %.6f].", this->refinements_, m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);
    }

    void gyroscope_intrinsics::align(sensor::imu::preintegration& measurement) const {
        if (measurement.get_linearisation_intrinsics().gyroscope != this->estimate_.gyroscope) {
            measurement.reintegrate(measurement.get_linearisation(), this->estimate_);
        }
    }
}
