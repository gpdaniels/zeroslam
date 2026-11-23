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

#include "mapping/inertial.hpp"

#include "math/math.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/bias_random_walk.hpp"
#include "optimisation/edges/inertial.hpp"
#include "optimisation/vertices/imu_bias.hpp"
#include "optimisation/vertices/pose.hpp"
#include "optimisation/vertices/velocity.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <iterator>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    optimisation::vertex make_pose_vertex(const mapping::frame& frame) {
        const math::se3<double> pose(frame.rotation, frame.translation);
        optimisation::vertex node{ optimisation::vertices::pose() };
        const double parameters[7] = { pose.translation()[0], pose.translation()[1], pose.translation()[2], pose.rotation().get_quaternion()[1], pose.rotation().get_quaternion()[2], pose.rotation().get_quaternion()[3], pose.rotation().get_quaternion()[0] };
        static_cast<void>(node.set_parameters(&parameters[0], 7));
        node.set_fixed(true);
        return node;
    }
}

namespace mapping {
    void inertial::declare(const int sensor_id, const sensor::imu::parameters& noise) {
        this->declared_ = true;
        this->sensor_id_ = sensor_id;
        this->noise_ = noise;
    }

    bool inertial::declared() const {
        return this->declared_;
    }

    int inertial::sensor_id() const {
        return this->sensor_id_;
    }

    const sensor::imu::parameters& inertial::noise() const {
        return this->noise_;
    }

    bool inertial::index_in(const mapping::rig& rig, size_t& index) const {
        return this->declared_ && rig.find(this->sensor_id_, index) && (rig.reference() >= 0);
    }

    bool inertial::add_sample(const sensor::imu::sample& sample) {
        if (!this->samples_.empty() && !(sample.timestamp > this->samples_.back().timestamp)) {
            return false;
        }
        this->samples_.push_back(sample);
        return true;
    }

    const std::vector<sensor::imu::sample>& inertial::samples() const {
        return this->samples_;
    }

    void inertial::set_frame_time(const int frame_id, const long long timestamp) {
        this->frame_times_[frame_id] = timestamp;
    }

    bool inertial::frame_time(const int frame_id, long long& timestamp) const {
        const std::map<int, long long>::const_iterator found = this->frame_times_.find(frame_id);
        if (found == this->frame_times_.end()) {
            return false;
        }
        timestamp = found->second;
        return true;
    }

    bool inertial::preintegrate(const long long start, const long long end, const sensor::imu::bias& linearisation, sensor::imu::preintegration& result) const {
        result = sensor::imu::preintegration(this->noise_, linearisation);
        if (this->samples_.empty() || (end < start)) {
            return false;
        }
        // The samples from the last one at or before the start to the first one at or after the end.
        size_t first = 0;
        size_t last = this->samples_.size();
        for (size_t k = 0; k < this->samples_.size(); ++k) {
            if (this->samples_[k].timestamp <= start) {
                first = k;
            }
            if ((this->samples_[k].timestamp >= end) && (last == this->samples_.size())) {
                last = k;
            }
        }
        if (this->samples_[first].timestamp > start) {
            return false;
        }
        if (last < this->samples_.size()) {
            return result.integrate(&this->samples_[first], (last - first) + 1, start, end);
        }
        const long long hold = static_cast<long long>(inertial::hold_periods * 1.0e9 / this->noise_.rate);
        if ((end - this->samples_.back().timestamp) > hold) {
            return false;
        }
        std::vector<sensor::imu::sample> held(this->samples_.begin() + static_cast<std::ptrdiff_t>(first), this->samples_.end());
        sensor::imu::sample extended = held.back();
        extended.timestamp = end;
        if (end > held.back().timestamp) {
            held.push_back(extended);
        }
        return result.integrate(held, start, end);
    }

    void inertial::add_keyframe(const int frame_id, const std::unordered_map<int, mapping::frame>& frames, const mapping::rig& rig) {
        long long timestamp = 0;
        if (!this->declared_ || !this->frame_time(frame_id, timestamp) || (this->keyframes_.count(frame_id) != 0)) {
            return;
        }
        keyframe_state state;
        state.timestamp = timestamp;
        state.chain = this->chain_;
        state.from_previous = sensor::imu::preintegration(this->noise_);
        // The keyframe before it in its chain.
        const keyframe_state* previous = nullptr;
        int previous_id = -1;
        for (const auto& [id, candidate] : this->keyframes_) {
            if ((id < frame_id) && (candidate.chain == this->chain_)) {
                previous = &candidate;
                previous_id = id;
            }
        }
        if (previous != nullptr) {
            state.bias = previous->bias;
            state.linked = (timestamp > previous->timestamp) && this->preintegrate(previous->timestamp, timestamp, previous->bias, state.from_previous);
            size_t imu_index = 0;
            const std::unordered_map<int, mapping::frame>::const_iterator previous_frame = frames.find(previous_id);
            if (state.linked && previous->estimated && this->chain_initialised(this->chain_) && (previous_frame != frames.end()) && this->index_in(rig, imu_index)) {
                const sensor::imu::preintegration::state start = inertial::state_at(previous_frame->second, previous->velocity, rig.reference_from_sensor(imu_index));
                state.velocity = state.from_previous.predict(start, previous->bias, this->gravity_).velocity;
                state.estimated = true;
            }
        }
        this->keyframes_[frame_id] = state;
        // Only the samples since the newest keyframe are integrated again; the preintegrations keep their own steps.
        const long long keep_from = timestamp - inertial::sample_history;
        size_t drop = 0;
        while ((drop + 1 < this->samples_.size()) && (this->samples_[drop + 1].timestamp <= keep_from)) {
            ++drop;
        }
        if (drop > 0) {
            this->samples_.erase(this->samples_.begin(), this->samples_.begin() + static_cast<std::ptrdiff_t>(drop));
        }
        while (!this->frame_times_.empty() && (this->frame_times_.begin()->second < keep_from) && (this->keyframes_.count(this->frame_times_.begin()->first) == 0)) {
            this->frame_times_.erase(this->frame_times_.begin());
        }
    }

    void inertial::break_chain() {
        ++this->chain_;
    }

    size_t inertial::chain() const {
        return this->chain_;
    }

    void inertial::synchronise(const std::unordered_set<int>& keyframe_ids, const std::unordered_map<int, mapping::frame>& frames) {
        for (std::map<int, keyframe_state>::iterator it = this->keyframes_.begin(); it != this->keyframes_.end();) {
            if ((keyframe_ids.count(it->first) != 0) && (frames.count(it->first) != 0)) {
                ++it;
                continue;
            }
            std::map<int, keyframe_state>::iterator next = std::next(it);
            while ((next != this->keyframes_.end()) && (next->second.chain != it->second.chain)) {
                ++next;
            }
            if ((next != this->keyframes_.end()) && next->second.linked) {
                if (it->second.linked) {
                    sensor::imu::preintegration merged = it->second.from_previous;
                    merged.merge(next->second.from_previous);
                    next->second.from_previous = merged;
                }
                else {
                    next->second.linked = false;
                }
            }
            it = this->keyframes_.erase(it);
        }
    }

    std::map<int, inertial::keyframe_state>& inertial::keyframes() {
        return this->keyframes_;
    }

    const std::map<int, inertial::keyframe_state>& inertial::keyframes() const {
        return this->keyframes_;
    }

    std::vector<int> inertial::chain_window() const {
        std::vector<int> window;
        for (std::map<int, keyframe_state>::const_reverse_iterator it = this->keyframes_.rbegin(); it != this->keyframes_.rend(); ++it) {
            if (it->second.chain != this->chain_) {
                continue;
            }
            window.push_back(it->first);
            if (!it->second.linked) {
                break;
            }
        }
        std::reverse(window.begin(), window.end());
        return window;
    }

    bool inertial::chain_initialised(const size_t chain) const {
        const std::map<size_t, bool>::const_iterator found = this->chain_initialised_.find(chain);
        return (found != this->chain_initialised_.end()) && found->second;
    }

    void inertial::set_chain_initialised(const size_t chain) {
        this->chain_initialised_[chain] = true;
    }

    bool inertial::initialised() const {
        for (const auto& [chain, initialised] : this->chain_initialised_) {
            static_cast<void>(chain);
            if (initialised) {
                return true;
            }
        }
        return false;
    }

    const math::matrix<double, 3, 1>& inertial::gravity() const {
        return this->gravity_;
    }

    void inertial::set_gravity(const math::matrix<double, 3, 1>& gravity_world) {
        this->gravity_ = gravity_world;
        this->gravity_known_ = true;
    }

    sensor::imu::preintegration::state inertial::state_at(const mapping::frame& frame, const math::matrix<double, 3, 1>& velocity, const math::se3<double>& reference_from_imu) {
        const math::se3<double> world_from_camera = math::se3<double>(math::so3<double>(frame.rotation), frame.translation).inverse();
        sensor::imu::preintegration::state state;
        state.rotation = (world_from_camera.rotation() * reference_from_imu.rotation()).normalised();
        state.position = world_from_camera.translation() + (world_from_camera.rotation() * reference_from_imu.translation());
        state.velocity = velocity;
        return state;
    }

    void inertial::pose_of(const sensor::imu::preintegration::state& state, const math::se3<double>& reference_from_imu, math::matrix<double, 3, 3>& rotation, math::matrix<double, 3, 1>& translation) {
        const math::so3<double> world_from_camera = (state.rotation * reference_from_imu.rotation().inverse()).normalised();
        const math::matrix<double, 3, 1> centre = state.position - (world_from_camera * reference_from_imu.translation());
        const math::so3<double> camera_from_world = world_from_camera.inverse();
        rotation = camera_from_world.get_matrix();
        translation = -1.0 * (camera_from_world * centre);
    }

    bool inertial::predict(const long long timestamp, const std::unordered_map<int, mapping::frame>& frames, const mapping::rig& rig, math::matrix<double, 3, 3>& rotation, math::matrix<double, 3, 1>& translation, math::matrix<double, 3, 1>& velocity) const {
        size_t imu_index = 0;
        if (!this->gravity_known_ || !this->chain_initialised(this->chain_) || !this->index_in(rig, imu_index)) {
            return false;
        }
        for (std::map<int, keyframe_state>::const_reverse_iterator it = this->keyframes_.rbegin(); it != this->keyframes_.rend(); ++it) {
            if ((it->second.chain != this->chain_) || !it->second.estimated) {
                continue;
            }
            const std::unordered_map<int, mapping::frame>::const_iterator frame = frames.find(it->first);
            if ((frame == frames.end()) || (it->second.timestamp > timestamp)) {
                return false;
            }
            sensor::imu::preintegration preintegration;
            if (!this->preintegrate(it->second.timestamp, timestamp, it->second.bias, preintegration)) {
                return false;
            }
            const math::se3<double> reference_from_imu = rig.reference_from_sensor(imu_index);
            const sensor::imu::preintegration::state predicted = preintegration.predict(inertial::state_at(frame->second, it->second.velocity, reference_from_imu), it->second.bias, this->gravity_);
            inertial::pose_of(predicted, reference_from_imu, rotation, translation);
            velocity = predicted.velocity;
            return true;
        }
        return false;
    }

    inertial::factors inertial::add_factors(optimisation::factor_graph& graph, const std::unordered_map<int, optimisation::vertex*>& poses, const std::unordered_map<int, mapping::frame>& frames, const mapping::rig& rig, const bool hold_extrinsic) const {
        factors held;
        size_t imu_index = 0;
        if (!this->gravity_known_ || !this->index_in(rig, imu_index)) {
            return held;
        }
        const math::se3<double> reference_from_imu = rig.reference_from_sensor(imu_index);
        std::unordered_map<int, optimisation::vertex*> velocities;
        std::unordered_map<int, optimisation::vertex*> biases;
        std::unordered_map<int, optimisation::vertex*> extra_poses;
        const auto pose_of_keyframe = [&](const int id) -> optimisation::vertex* {
            const std::unordered_map<int, optimisation::vertex*>::const_iterator found = poses.find(id);
            if (found != poses.end()) {
                return found->second;
            }
            const std::unordered_map<int, optimisation::vertex*>::const_iterator extra = extra_poses.find(id);
            if (extra != extra_poses.end()) {
                return extra->second;
            }
            const std::unordered_map<int, mapping::frame>::const_iterator frame = frames.find(id);
            if (frame == frames.end()) {
                return nullptr;
            }
            optimisation::vertex* const node = graph.add_vertex(make_pose_vertex(frame->second));
            extra_poses[id] = node;
            return node;
        };
        const auto state_vertices = [&](const int id, const keyframe_state& state, const bool fixed, optimisation::vertex*& velocity, optimisation::vertex*& bias) {
            if (velocities.count(id) == 0) {
                optimisation::vertex velocity_node{ optimisation::vertices::velocity() };
                static_cast<void>(velocity_node.set_parameters(state.velocity.data(), 3));
                velocity_node.set_fixed(fixed);
                velocities[id] = graph.add_vertex(static_cast<optimisation::vertex&&>(velocity_node));
                optimisation::vertex bias_node{ optimisation::vertices::imu_bias() };
                const double values[6] = { state.bias.gyroscope[0], state.bias.gyroscope[1], state.bias.gyroscope[2], state.bias.accelerometer[0], state.bias.accelerometer[1], state.bias.accelerometer[2] };
                static_cast<void>(bias_node.set_parameters(&values[0], 6));
                bias_node.set_fixed(fixed);
                biases[id] = graph.add_vertex(static_cast<optimisation::vertex&&>(bias_node));
                if (!fixed) {
                    held.velocities.push_back({ id, velocities[id] });
                    held.biases.push_back({ id, biases[id] });
                }
            }
            velocity = velocities.at(id);
            bias = biases.at(id);
        };
        std::map<size_t, int> last_of_chain;
        for (const auto& [id, state] : this->keyframes_) {
            const std::map<size_t, int>::const_iterator previous_entry = last_of_chain.find(state.chain);
            const int previous_id = (previous_entry != last_of_chain.end()) ? previous_entry->second : -1;
            last_of_chain[state.chain] = id;
            if ((previous_id < 0) || !state.linked || !state.estimated || !this->chain_initialised(state.chain)) {
                continue;
            }
            const keyframe_state& previous = this->keyframes_.at(previous_id);
            const std::unordered_map<int, optimisation::vertex*>::const_iterator current_pose = poses.find(id);
            if (!previous.estimated || (current_pose == poses.end())) {
                continue;
            }
            optimisation::vertex* const previous_pose = pose_of_keyframe(previous_id);
            if ((previous_pose == nullptr) || (previous_pose->is_fixed() && current_pose->second->is_fixed())) {
                continue;
            }
            math::matrix<double, 0, 0> information;
            if (!optimisation::edges::inertial::information(state.from_previous, information)) {
                continue;
            }
            if ((held.extrinsic == nullptr) && rig.refines(imu_index) && !hold_extrinsic) {
                held.extrinsic = graph.add_vertex(rig.extrinsic_vertex(imu_index));
            }
            optimisation::vertex* previous_velocity = nullptr;
            optimisation::vertex* previous_bias = nullptr;
            optimisation::vertex* current_velocity = nullptr;
            optimisation::vertex* current_bias = nullptr;
            state_vertices(previous_id, previous, previous_pose->is_fixed(), previous_velocity, previous_bias);
            state_vertices(id, state, current_pose->second->is_fixed(), current_velocity, current_bias);
            optimisation::edge factor{ optimisation::edges::inertial(this->gravity_, reference_from_imu, held.extrinsic != nullptr) };
            factor.set_observation(optimisation::edges::inertial::observation(state.from_previous));
            factor.set_information(information);
            static_cast<void>(factor.add_vertex(previous_pose));
            static_cast<void>(factor.add_vertex(previous_velocity));
            static_cast<void>(factor.add_vertex(previous_bias));
            static_cast<void>(factor.add_vertex(current_pose->second));
            static_cast<void>(factor.add_vertex(current_velocity));
            if (held.extrinsic != nullptr) {
                static_cast<void>(factor.add_vertex(held.extrinsic));
            }
            static_cast<void>(graph.add_edge(static_cast<optimisation::edge&&>(factor)));
            optimisation::edge walk{ optimisation::edges::bias_random_walk() };
            walk.set_information(optimisation::edges::bias_random_walk::information(this->noise_, state.from_previous.get_delta_time()));
            static_cast<void>(walk.add_vertex(previous_bias));
            static_cast<void>(walk.add_vertex(current_bias));
            static_cast<void>(graph.add_edge(static_cast<optimisation::edge&&>(walk)));
            ++held.edges;
        }
        return held;
    }

    void inertial::read_factors(const factors& held, mapping::rig& rig) {
        for (const std::pair<int, optimisation::vertex*>& entry : held.velocities) {
            const std::map<int, keyframe_state>::iterator found = this->keyframes_.find(entry.first);
            if (found != this->keyframes_.end()) {
                const double* const values = entry.second->get_parameters();
                found->second.velocity = { { values[0], values[1], values[2] } };
            }
        }
        for (const std::pair<int, optimisation::vertex*>& entry : held.biases) {
            const std::map<int, keyframe_state>::iterator found = this->keyframes_.find(entry.first);
            if (found != this->keyframes_.end()) {
                const double* const values = entry.second->get_parameters();
                found->second.bias.gyroscope = { { values[0], values[1], values[2] } };
                found->second.bias.accelerometer = { { values[3], values[4], values[5] } };
            }
        }
        size_t imu_index = 0;
        if ((held.extrinsic != nullptr) && this->index_in(rig, imu_index)) {
            rig.read_extrinsic_vertex(imu_index, *held.extrinsic);
        }
        // Each edge takes the bias of the keyframe before it, at which its measurements are integrated again once it has moved.
        std::map<size_t, const keyframe_state*> last_of_chain;
        for (auto& [id, state] : this->keyframes_) {
            static_cast<void>(id);
            const std::map<size_t, const keyframe_state*>::const_iterator previous = last_of_chain.find(state.chain);
            if ((previous != last_of_chain.end()) && state.linked && state.from_previous.requires_reintegration(previous->second->bias)) {
                state.from_previous.reintegrate(previous->second->bias);
            }
            last_of_chain[state.chain] = &state;
        }
    }
}
