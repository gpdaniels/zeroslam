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

#include "mapping/odometry.hpp"

#include "core/logger.hpp"
#include "math/math.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/calibration_prior.hpp"
#include "optimisation/edges/extrinsic_prior.hpp"
#include "optimisation/edges/odometry.hpp"
#include "optimisation/vertices/calibration.hpp"
#include "optimisation/vertices/extrinsic.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cstdio>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mapping {
    bool odometry::configure(const settings& configuration_value) {
        if (!configuration_value.kinematics.is_valid() || (configuration_value.calibration.count != configuration_value.kinematics.parameter_count())) {
            this->enabled = false;
            return false;
        }
        this->configuration = configuration_value;
        for (size_t i = 0; i < sensor::odometry::model::maximum_parameters; ++i) {
            this->values[i] = configuration_value.calibration.values[i];
            this->values_before[i] = this->values[i];
        }
        this->motions.clear();
        this->extrinsic_prior_set = false;
        this->enabled = true;
        return true;
    }

    bool odometry::is_enabled() const {
        return this->enabled;
    }

    const odometry::settings& odometry::get_settings() const {
        return this->configuration;
    }

    const double* odometry::calibration() const {
        return &this->values[0];
    }

    size_t odometry::parameter_count() const {
        return this->enabled ? this->configuration.calibration.count : 0;
    }

    bool odometry::add_sample(const sensor::odometry::sample& value) {
        if (!this->samples.empty() && (value.timestamp <= this->samples.back().timestamp)) {
            return false;
        }
        this->samples.push_back(value);
        return true;
    }

    size_t odometry::sample_count() const {
        return this->samples.size();
    }

    void odometry::set_frame_time(const int frame_id, const long long int timestamp) {
        this->frame_times[frame_id] = timestamp;
    }

    bool odometry::frame_time(const int frame_id, long long int& timestamp) const {
        const std::unordered_map<int, long long int>::const_iterator found = this->frame_times.find(frame_id);
        if (found == this->frame_times.end()) {
            return false;
        }
        timestamp = found->second;
        return true;
    }

    long long int odometry::key(const int from_frame_id, const int to_frame_id) {
        return (static_cast<long long int>(from_frame_id) * 4294967296ll) + static_cast<long long int>(static_cast<unsigned int>(to_frame_id));
    }

    bool odometry::integrate(const long long int start, const long long int end, sensor::odometry::preintegration& result, bool& covered) const {
        covered = false;
        if (this->samples.empty() || (this->samples.front().timestamp > start) || (end < start)) {
            return false;
        }
        const auto later = [](const sensor::odometry::sample& lhs, const long long int time) {
            return lhs.timestamp < time;
        };
        // The samples from the last at or before the start to the first at or after the end.
        size_t first = static_cast<size_t>(std::lower_bound(this->samples.begin(), this->samples.end(), start, later) - this->samples.begin());
        if ((first == this->samples.size()) || (this->samples[first].timestamp > start)) {
            first = (first > 0) ? (first - 1) : 0;
        }
        const size_t last = static_cast<size_t>(std::lower_bound(this->samples.begin(), this->samples.end(), end, later) - this->samples.begin());
        if (last < this->samples.size()) {
            covered = true;
            return result.integrate(this->samples.data() + first, (last - first) + 1, start, end);
        }
        // Past the newest sample, carried on at the speed between the newest two.
        const sensor::odometry::sample& newest = this->samples.back();
        if ((this->samples.size() < 2) || (static_cast<double>(end - newest.timestamp) * 1.0e-9 > odometry::maximum_extrapolation)) {
            return false;
        }
        const sensor::odometry::sample& before = this->samples[this->samples.size() - 2];
        const double ratio = static_cast<double>(end - newest.timestamp) / static_cast<double>(newest.timestamp - before.timestamp);
        std::vector<sensor::odometry::sample> extended(this->samples.begin() + static_cast<long long int>(first), this->samples.end());
        sensor::odometry::sample extrapolated = newest;
        extrapolated.timestamp = end;
        for (size_t wheel = 0; wheel < sensor::odometry::sample::maximum_wheels; ++wheel) {
            extrapolated.rotation[wheel] = newest.rotation[wheel] + (ratio * (newest.rotation[wheel] - before.rotation[wheel]));
        }
        if (end > newest.timestamp) {
            extended.push_back(extrapolated);
        }
        return result.integrate(extended.data(), extended.size(), start, end);
    }

    bool odometry::motion(const int from_frame_id, const int to_frame_id, sensor::odometry::preintegration& result) {
        long long int start = 0;
        long long int end = 0;
        if (!this->enabled || !this->frame_time(from_frame_id, start) || !this->frame_time(to_frame_id, end) || (end < start) || (static_cast<double>(end - start) * 1.0e-9 > odometry::maximum_interval)) {
            return false;
        }
        const size_t count = this->configuration.calibration.count;
        const long long int wanted = odometry::key(from_frame_id, to_frame_id);
        std::unordered_map<long long int, sensor::odometry::preintegration>::iterator found = this->motions.find(wanted);
        if (found == this->motions.end()) {
            // A keyframe culled between two keyframes leaves the motions on either side of it, which merge into this one.
            for (const auto& [first_key, first_motion] : this->motions) {
                if ((first_key / 4294967296ll) != static_cast<long long int>(from_frame_id) || (first_motion.get_end() >= end)) {
                    continue;
                }
                const int middle_frame_id = static_cast<int>(static_cast<unsigned int>(first_key - (static_cast<long long int>(from_frame_id) * 4294967296ll)));
                const std::unordered_map<long long int, sensor::odometry::preintegration>::const_iterator second = this->motions.find(odometry::key(middle_frame_id, to_frame_id));
                if (second == this->motions.end()) {
                    continue;
                }
                sensor::odometry::preintegration merged = first_motion;
                if (merged.merge(second->second)) {
                    found = this->motions.emplace(wanted, merged).first;
                    break;
                }
            }
        }
        if (found == this->motions.end()) {
            sensor::odometry::preintegration integrated(this->configuration.kinematics, this->configuration.noise, &this->values[0], count);
            bool covered = false;
            if (!this->integrate(start, end, integrated, covered)) {
                return false;
            }
            if (!covered) {
                result = integrated;
                return true;
            }
            found = this->motions.emplace(wanted, integrated).first;
        }
        if (found->second.needs_reintegration(&this->values[0], count) && !found->second.reintegrate(&this->values[0], count)) {
            this->motions.erase(found);
            return false;
        }
        result = found->second;
        return true;
    }

    bool odometry::find_sensor(const mapping::rig& rig, size_t& index) const {
        return this->enabled && (rig.reference() >= 0) && rig.find(this->configuration.sensor_id, index) && (rig.at(index).kind == mapping::rig::sensor_kind::odometry);
    }

    bool odometry::camera_motion(const int from_frame_id, const int to_frame_id, const mapping::rig& rig, math::se3<double>& result) {
        size_t index = 0;
        sensor::odometry::preintegration preintegrated;
        if (!this->find_sensor(rig, index) || !this->motion(from_frame_id, to_frame_id, preintegrated)) {
            return false;
        }
        const math::se3<double> reference_from_odometry = rig.reference_from_sensor(index);
        const math::matrix<double, 3, 1> planar = preintegrated.predict(&this->values[0], this->configuration.calibration.count);
        result = reference_from_odometry * optimisation::edges::odometry::lift(planar).inverse() * reference_from_odometry.inverse();
        return true;
    }

    bool odometry::scale_initialisation(const int anchor_frame_id, const int frame_id, const mapping::rig& rig, math::matrix<double, 3, 1>& translation) {
        math::se3<double> moved;
        if (!this->camera_motion(anchor_frame_id, frame_id, rig, moved)) {
            return false;
        }
        const double distance = math::sqrt(moved.translation().get_length_squared());
        const double length = math::sqrt(translation.get_length_squared());
        if ((distance < odometry::minimum_initialisation_distance) || !(length > 0.0)) {
            return false;
        }
        translation = translation * (distance / length);
        core::logger::log(core::logger::level::note, "Odometry scales the initialisation between frames %d and %d to %.3f m.", anchor_frame_id, frame_id, distance);
        return true;
    }

    odometry::graph_vertices odometry::add_to_graph(optimisation::factor_graph& graph, const std::unordered_map<int, optimisation::vertex*>& frame_vertices, const mapping::rig& rig) {
        graph_vertices result;
        if (!this->find_sensor(rig, result.sensor_index)) {
            return result;
        }
        std::vector<int> frame_ids;
        frame_ids.reserve(frame_vertices.size());
        for (const auto& [frame_id, node] : frame_vertices) {
            static_cast<void>(node);
            frame_ids.push_back(frame_id);
        }
        std::sort(frame_ids.begin(), frame_ids.end());
        std::vector<std::pair<int, sensor::odometry::preintegration>> intervals;
        for (size_t i = 1; i < frame_ids.size(); ++i) {
            sensor::odometry::preintegration preintegrated;
            if (this->motion(frame_ids[i - 1], frame_ids[i], preintegrated)) {
                intervals.push_back({ static_cast<int>(i), preintegrated });
            }
        }
        if (intervals.empty()) {
            return result;
        }

        const sensor::odometry::calibration& declared = this->configuration.calibration;
        const size_t count = declared.count;
        result.calibration = graph.add_vertex(optimisation::vertices::calibration::make(&this->values[0], count, declared.free_mask()));
        bool any_prior = false;
        for (size_t i = 0; i < count; ++i) {
            any_prior = any_prior || (declared.free[i] && (declared.prior_sigma[i] > 0.0));
        }
        if (any_prior) {
            optimisation::edge prior{ optimisation::edges::calibration_prior() };
            prior.set_observation(optimisation::edges::calibration_prior::observation(&declared.values[0], count));
            prior.set_information(optimisation::edges::calibration_prior::information(&declared.prior_sigma[0], count));
            prior.add_vertex(result.calibration);
            graph.add_edge(static_cast<optimisation::edge&&>(prior));
        }

        if (!this->extrinsic_prior_set) {
            this->extrinsic_prior = rig.reference_from_sensor(result.sensor_index);
            this->extrinsic_prior_set = true;
        }
        result.extrinsic = graph.add_vertex(rig.extrinsic_vertex(result.sensor_index));
        bool any_extrinsic_prior = false;
        for (const double deviation : this->configuration.extrinsic_deviations) {
            any_extrinsic_prior = any_extrinsic_prior || (deviation > 0.0);
        }
        if (!result.extrinsic->is_fixed() && any_extrinsic_prior) {
            double parameters[optimisation::vertices::extrinsic::parameter_count];
            optimisation::vertices::extrinsic::to_parameters(this->extrinsic_prior, parameters);
            optimisation::edge prior{ optimisation::edges::extrinsic_prior() };
            prior.set_observation(optimisation::edges::extrinsic_prior::observation(&parameters[0]));
            prior.set_information(optimisation::edges::extrinsic_prior::information(&this->configuration.extrinsic_deviations[0]));
            prior.add_vertex(result.extrinsic);
            graph.add_edge(static_cast<optimisation::edge&&>(prior));
        }

        for (const auto& [index, preintegrated] : intervals) {
            optimisation::edge factor{ optimisation::edges::odometry() };
            factor.set_observation(optimisation::edges::odometry::observation(preintegrated));
            factor.set_information(optimisation::edges::odometry::information(preintegrated));
            factor.add_vertex(frame_vertices.at(frame_ids[static_cast<size_t>(index) - 1]));
            factor.add_vertex(frame_vertices.at(frame_ids[static_cast<size_t>(index)]));
            factor.add_vertex(result.extrinsic);
            factor.add_vertex(result.calibration);
            graph.add_edge(static_cast<optimisation::edge&&>(factor));
            ++result.edges;
        }
        return result;
    }

    void odometry::read_from_graph(const graph_vertices& vertices, mapping::rig& rig) {
        if (vertices.calibration == nullptr) {
            return;
        }
        const size_t count = this->configuration.calibration.count;
        for (size_t i = 0; i < count; ++i) {
            this->values_before[i] = this->values[i];
            this->values[i] = vertices.calibration->get_parameters()[i];
        }
        if (vertices.extrinsic != nullptr) {
            rig.read_extrinsic_vertex(vertices.sensor_index, *vertices.extrinsic);
        }
        if (core::logger::enabled(core::logger::level::info)) {
            char text[256] = "";
            size_t offset = 0;
            for (size_t i = 0; (i < count) && (offset < sizeof(text)); ++i) {
                const int length = std::snprintf(text + offset, sizeof(text) - offset, " %s %.6f", this->configuration.kinematics.parameter_name(i), this->values[i]);
                offset += (length > 0) ? static_cast<size_t>(length) : 0;
            }
            core::logger::log(core::logger::level::info, "Odometry: %zu edges, calibration%s.", vertices.edges, text);
        }
    }

    void odometry::restore(const graph_vertices& vertices) {
        if (vertices.calibration == nullptr) {
            return;
        }
        for (size_t i = 0; i < sensor::odometry::model::maximum_parameters; ++i) {
            this->values[i] = this->values_before[i];
        }
    }
}
