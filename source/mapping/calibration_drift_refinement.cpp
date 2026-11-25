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

#include "mapping/calibration_drift_refinement.hpp"

#include "core/logger.hpp"
#include "math/math.hpp"
#include "math/matrix.hpp"
#include "optimisation/edges/baseline.hpp"
#include "optimisation/edges/extrinsic_random_walk.hpp"
#include "optimisation/edges/intrinsics_prior.hpp"
#include "optimisation/edges/intrinsics_random_walk.hpp"
#include "optimisation/edges/intrinsics_ratio_prior.hpp"
#include "optimisation/vertices/extrinsic.hpp"
#include "optimisation/vertices/intrinsics.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    optimisation::vertex extrinsic_vertex_of(const math::se3<double>& reference_from_camera, const bool fixed) {
        optimisation::vertex node{ optimisation::vertices::extrinsic() };
        double parameters[optimisation::vertices::extrinsic::parameter_count];
        optimisation::vertices::extrinsic::to_parameters(reference_from_camera, parameters);
        node.set_parameters(&parameters[0], optimisation::vertices::extrinsic::parameter_count);
        node.set_fixed(fixed);
        return node;
    }

    math::matrix<double, 0, 0> information_of(const double sigma) {
        return math::matrix<double, 0, 0>::identity(1, 1) * (1.0 / (sigma * sigma));
    }
}

namespace mapping {
    calibration_drift_refinement::calibration_drift_refinement(calibration_drift::request* const request_value, optimisation::factor_graph& graph_value, std::unordered_map<int, mapping::frame>& frames_value, const bool landmarks_free)
        : request(request_value)
        , graph(graph_value)
        , frames(frames_value)
        , following((request_value != nullptr) && landmarks_free)
        , nodes()
        , handed()
        , reference_vertex(nullptr)
        , known()
        , before() {
        if (this->request != nullptr) {
            this->request->refined = false;
            for (const calibration_drift::chunk& held : this->request->chunks) {
                this->known[held.index] = held;
            }
        }
    }

    math::se3<double> calibration_drift_refinement::reference_from_camera(const mapping::rig& rig, const mapping::frame* const frame, const int camera) {
        if ((frame != nullptr) && (camera > 0) && (static_cast<size_t>(camera - 1) < frame->views.size()) && (frame->views[static_cast<size_t>(camera - 1)].calibration_chunk >= 0)) {
            return frame->views[static_cast<size_t>(camera - 1)].reference_from_camera;
        }
        return rig.reference_from_sensor(rig.camera(static_cast<size_t>(camera)));
    }

    optimisation::vertex calibration_drift_refinement::extrinsic_vertex(const mapping::rig& rig, const mapping::frame& frame, const int camera) {
        if ((camera > 0) && (static_cast<size_t>(camera - 1) < frame.views.size()) && (frame.views[static_cast<size_t>(camera - 1)].calibration_chunk >= 0)) {
            return extrinsic_vertex_of(frame.views[static_cast<size_t>(camera - 1)].reference_from_camera, true);
        }
        return rig.extrinsic_vertex(rig.camera(static_cast<size_t>(camera)));
    }

    const mapping::frame* calibration_drift_refinement::followed_view(const int frame_id, const int camera) const {
        const std::unordered_map<int, mapping::frame>::const_iterator found = this->frames.find(frame_id);
        if ((found == this->frames.end()) || (camera < 0)) {
            return nullptr;
        }
        const mapping::frame& frame = found->second;
        if (camera == 0) {
            return (frame.calibration_chunk >= 0) ? &frame : nullptr;
        }
        if (static_cast<size_t>(camera - 1) >= frame.views.size()) {
            return nullptr;
        }
        const mapping::frame& view = frame.views[static_cast<size_t>(camera - 1)];
        return (view.calibration_chunk >= 0) ? &view : nullptr;
    }

    calibration_drift_refinement::node& calibration_drift_refinement::node_of(const int camera, const long long chunk) {
        if (this->nodes.size() <= static_cast<size_t>(camera)) {
            this->nodes.resize(static_cast<size_t>(camera) + 1);
        }
        return this->nodes[static_cast<size_t>(camera)][chunk];
    }

    bool calibration_drift_refinement::takes(const int frame_id, const int camera) {
        if (!this->following || (camera < 0) || (static_cast<size_t>(camera) >= this->request->cameras.size()) || !this->request->cameras[static_cast<size_t>(camera)].follow_focal) {
            return false;
        }
        const mapping::frame* const view = this->followed_view(frame_id, camera);
        if (view == nullptr) {
            return false;
        }
        node& held = this->node_of(camera, view->calibration_chunk);
        if (held.intrinsics == nullptr) {
            // Only the focal lengths are free, each by a local dimension of its own: the ratio between them has a prior.
            const size_t count = view->camera.get_parameter_count();
            std::vector<int> locals(count, -1);
            int next = 0;
            for (const size_t index : calibration_drift::focal_parameters(view->camera)) {
                locals[index] = next++;
            }
            held.intrinsics = this->graph.add_vertex(optimisation::vertices::intrinsics::make(view->camera, locals.data(), count));
        }
        return held.intrinsics != nullptr;
    }

    void calibration_drift_refinement::attach(optimisation::edge& factor, const int frame_id, const int camera) {
        const mapping::frame* const view = this->followed_view(frame_id, camera);
        if (view == nullptr) {
            return;
        }
        node& held = this->node_of(camera, view->calibration_chunk);
        if (held.intrinsics == nullptr) {
            return;
        }
        factor.add_vertex(held.intrinsics);
        ++held.observations;
    }

    const std::vector<optimisation::vertex*>& calibration_drift_refinement::extrinsics(const int frame_id, const int camera, const std::vector<optimisation::vertex*>& rig_extrinsics) {
        if (camera <= 0) {
            return rig_extrinsics;
        }
        const mapping::frame* const view = this->followed_view(frame_id, camera);
        if (view == nullptr) {
            return rig_extrinsics;
        }
        const bool asked = this->following && (static_cast<size_t>(camera) < this->request->cameras.size());
        const bool free = asked && this->request->cameras[static_cast<size_t>(camera)].follow_extrinsic;
        node& held = this->node_of(camera, view->calibration_chunk);
        if (held.extrinsic == nullptr) {
            held.extrinsic = this->graph.add_vertex(extrinsic_vertex_of(view->reference_from_camera, !free));
        }
        // An observation is counted once, with its intrinsics vertex when it takes one.
        if (free && !this->request->cameras[static_cast<size_t>(camera)].follow_focal) {
            ++held.observations;
        }
        this->handed.assign(static_cast<size_t>(camera) + 1, nullptr);
        this->handed[static_cast<size_t>(camera)] = held.extrinsic;
        return this->handed;
    }

    void calibration_drift_refinement::add_ties() {
        if (!this->following) {
            return;
        }
        const calibration_drift::settings& walk = this->request->walk;
        for (size_t camera = 0; (camera < this->nodes.size()) && (camera < this->request->cameras.size()); ++camera) {
            const calibration_drift::request::camera& asked = this->request->cameras[camera];
            const std::vector<size_t> focal = calibration_drift::focal_parameters(asked.given);
            std::vector<double> given(asked.given.get_parameter_count(), 0.0);
            if (!given.empty() && !asked.given.get_parameters(given.data(), given.size())) {
                given.clear();
            }
            size_t chunks_held = 0;
            for (const auto& [chunk, held] : this->nodes[camera]) {
                static_cast<void>(chunk);
                chunks_held += (held.intrinsics != nullptr) ? 1u : 0u;
            }
            // The focal lengths: each chunk's on the calibration given, the priors of all the chunks together weighing as one
            // on their mean, the ratio of the two in each chunk on the ratio given, and each chunk's on the chunk's before it.
            const node* previous = nullptr;
            long long previous_chunk = 0;
            for (const auto& [chunk, held] : this->nodes[camera]) {
                if ((held.intrinsics == nullptr) || given.empty()) {
                    continue;
                }
                for (const size_t index : focal) {
                    const double sigma = this->request->focal_prior_fraction * math::abs(given[index]) * math::sqrt(static_cast<double>(chunks_held));
                    if (!(sigma > 0.0)) {
                        continue;
                    }
                    optimisation::edge prior{ optimisation::edges::intrinsics_prior(static_cast<int>(index), given[index]) };
                    prior.add_vertex(held.intrinsics);
                    prior.set_information(information_of(sigma));
                    this->graph.add_edge(static_cast<optimisation::edge&&>(prior));
                }
                if ((focal.size() == 2) && (math::abs(given[focal[0]]) > 0.0) && (math::abs(given[focal[1]]) > 0.0)) {
                    optimisation::edge aspect{ optimisation::edges::intrinsics_ratio_prior(static_cast<int>(focal[0]), static_cast<int>(focal[1]), given[focal[0]] / given[focal[1]]) };
                    aspect.add_vertex(held.intrinsics);
                    aspect.set_information(information_of(calibration_drift::aspect_prior_fraction * math::abs(given[focal[0]])));
                    this->graph.add_edge(static_cast<optimisation::edge&&>(aspect));
                }
                if (previous != nullptr) {
                    const double seconds = static_cast<double>(chunk - previous_chunk) * walk.chunk_seconds;
                    for (const size_t index : focal) {
                        optimisation::edge step{ optimisation::edges::intrinsics_random_walk(static_cast<int>(index)) };
                        step.add_vertex(previous->intrinsics);
                        step.add_vertex(held.intrinsics);
                        step.set_information(optimisation::edges::intrinsics_random_walk::information(walk.focal_walk * math::abs(given[index]), calibration_drift::walk_seconds, seconds));
                        this->graph.add_edge(static_cast<optimisation::edge&&>(step));
                    }
                }
                previous = &held;
                previous_chunk = chunk;
            }
            // The extrinsics: each chunk's at its declared distance from the reference, a metric rig's scale, which the
            // reprojections do not measure once the extrinsic is free, and on the chunk's before it.
            previous = nullptr;
            for (const auto& [chunk, held] : this->nodes[camera]) {
                if ((held.extrinsic == nullptr) || held.extrinsic->is_fixed()) {
                    continue;
                }
                if (asked.baseline >= mapping::rig::metric_baseline_minimum) {
                    if (this->reference_vertex == nullptr) {
                        this->reference_vertex = this->graph.add_vertex(extrinsic_vertex_of(math::se3<double>::identity(), true));
                    }
                    optimisation::edge length{ optimisation::edges::baseline(asked.baseline) };
                    length.add_vertex(held.extrinsic);
                    length.add_vertex(this->reference_vertex);
                    length.set_information(information_of(calibration_drift::baseline_sigma));
                    this->graph.add_edge(static_cast<optimisation::edge&&>(length));
                }
                if (previous != nullptr) {
                    const double seconds = static_cast<double>(chunk - previous_chunk) * walk.chunk_seconds;
                    optimisation::edge step{ optimisation::edges::extrinsic_random_walk() };
                    step.add_vertex(previous->extrinsic);
                    step.add_vertex(held.extrinsic);
                    step.set_information(optimisation::edges::extrinsic_random_walk::information(walk.rotation_walk, walk.translation_walk, calibration_drift::walk_seconds, seconds));
                    this->graph.add_edge(static_cast<optimisation::edge&&>(step));
                }
                previous = &held;
                previous_chunk = chunk;
            }
        }
    }

    calibration_drift::camera_state calibration_drift_refinement::solved(const size_t camera, const long long chunk, const calibration_drift::camera_state& was) const {
        calibration_drift::camera_state result = was;
        result.observations = 0;
        if (camera >= this->nodes.size()) {
            return result;
        }
        // The nearest chunks at or before this one, and after it, that hold the camera's focal lengths, and its extrinsic.
        const node* focal_below = nullptr;
        const node* focal_above = nullptr;
        const node* extrinsic_below = nullptr;
        const node* extrinsic_above = nullptr;
        long long focal_chunks[2] = { 0, 0 };
        long long extrinsic_chunks[2] = { 0, 0 };
        for (const auto& [index, held] : this->nodes[camera]) {
            if (held.intrinsics != nullptr) {
                if (index <= chunk) {
                    focal_below = &held;
                    focal_chunks[0] = index;
                }
                else if (focal_above == nullptr) {
                    focal_above = &held;
                    focal_chunks[1] = index;
                }
            }
            if ((held.extrinsic != nullptr) && !held.extrinsic->is_fixed()) {
                if (index <= chunk) {
                    extrinsic_below = &held;
                    extrinsic_chunks[0] = index;
                }
                else if (extrinsic_above == nullptr) {
                    extrinsic_above = &held;
                    extrinsic_chunks[1] = index;
                }
            }
            if (index == chunk) {
                result.observations = held.observations;
            }
        }
        // The camera an intrinsics vertex holds, at the parameters the solve left it.
        const auto camera_of = [&was](const node& held) {
            const optimisation::vertices::intrinsics* const policy = held.intrinsics->get_policy<optimisation::vertices::intrinsics>();
            return (policy != nullptr) ? sensor::model(policy->get_camera()) : was.camera;
        };
        if ((focal_below != nullptr) && ((focal_chunks[0] == chunk) || (focal_above == nullptr))) {
            result.camera = camera_of(*focal_below);
        }
        else if ((focal_below != nullptr) && (focal_above != nullptr)) {
            result.camera = calibration_drift::camera_between(camera_of(*focal_below), camera_of(*focal_above), static_cast<double>(chunk - focal_chunks[0]) / static_cast<double>(focal_chunks[1] - focal_chunks[0]));
        }
        else if (focal_above != nullptr) {
            result.camera = camera_of(*focal_above);
        }
        const auto extrinsic_of = [](const node& held) {
            return optimisation::vertices::extrinsic::from_parameters(held.extrinsic->get_parameters());
        };
        if ((extrinsic_below != nullptr) && ((extrinsic_chunks[0] == chunk) || (extrinsic_above == nullptr))) {
            result.reference_from_camera = extrinsic_of(*extrinsic_below);
        }
        else if ((extrinsic_below != nullptr) && (extrinsic_above != nullptr)) {
            result.reference_from_camera = calibration_drift::extrinsic_between(extrinsic_of(*extrinsic_below), extrinsic_of(*extrinsic_above), static_cast<double>(chunk - extrinsic_chunks[0]) / static_cast<double>(extrinsic_chunks[1] - extrinsic_chunks[0]));
        }
        else if (extrinsic_above != nullptr) {
            result.reference_from_camera = extrinsic_of(*extrinsic_above);
        }
        return result;
    }

    void calibration_drift_refinement::apply() {
        if (!this->following) {
            return;
        }
        bool any = false;
        for (const std::map<long long, node>& camera_nodes : this->nodes) {
            for (const auto& [chunk, held] : camera_nodes) {
                static_cast<void>(chunk);
                any = any || (held.intrinsics != nullptr) || ((held.extrinsic != nullptr) && !held.extrinsic->is_fixed());
            }
        }
        if (!any) {
            return;
        }
        std::vector<int> ordered;
        ordered.reserve(this->frames.size());
        for (const auto& [frame_id, frame] : this->frames) {
            if (frame.calibration_chunk >= 0) {
                ordered.push_back(frame_id);
            }
        }
        std::sort(ordered.begin(), ordered.end());
        // The chunks before the solve: those the request came with, each with every camera, and for a frame in a chunk it did
        // not come with, the chunk as its frames have it. A frame that is not a keyframe has let its views go, so a camera
        // of such a chunk is taken from the first of its frames that still has the view.
        this->before = this->known;
        for (const int frame_id : ordered) {
            const mapping::frame& frame = this->frames.at(frame_id);
            if (this->known.count(frame.calibration_chunk) != 0) {
                continue;
            }
            calibration_drift::chunk& was = this->before[frame.calibration_chunk];
            was.index = frame.calibration_chunk;
            if (was.cameras.empty()) {
                was.cameras.push_back(calibration_drift::camera_state{ frame.camera, math::se3<double>::identity(), 0 });
            }
            for (size_t index = 0; index < frame.views.size(); ++index) {
                const mapping::frame& view = frame.views[index];
                if (view.calibration_chunk < 0) {
                    continue;
                }
                if (was.cameras.size() < (index + 2)) {
                    was.cameras.resize(index + 2);
                }
                if (!was.cameras[index + 1].camera.is_valid()) {
                    was.cameras[index + 1] = calibration_drift::camera_state{ view.camera, view.reference_from_camera, 0 };
                }
            }
        }
        std::map<long long, calibration_drift::chunk> after;
        for (const auto& [chunk, was] : this->before) {
            calibration_drift::chunk now = was;
            for (size_t camera = 0; camera < now.cameras.size(); ++camera) {
                now.cameras[camera] = this->solved(camera, chunk, was.cameras[camera]);
            }
            after.emplace(chunk, static_cast<calibration_drift::chunk&&>(now));
        }
        for (const int frame_id : ordered) {
            mapping::frame& frame = this->frames.at(frame_id);
            const calibration_drift::chunk& now = after.at(frame.calibration_chunk);
            if (now.cameras.empty()) {
                continue;
            }
            frame.camera = now.cameras[0].camera;
            for (size_t index = 0; (index < frame.views.size()) && ((index + 1) < now.cameras.size()); ++index) {
                mapping::frame& view = frame.views[index];
                if (view.calibration_chunk >= 0) {
                    view.camera = now.cameras[index + 1].camera;
                    view.reference_from_camera = now.cameras[index + 1].reference_from_camera;
                }
            }
        }
        this->request->refined = true;
        this->request->chunks.clear();
        for (const auto& [chunk, now] : after) {
            static_cast<void>(chunk);
            this->request->chunks.push_back(now);
        }
        if (!core::logger::enabled(core::logger::level::note)) {
            return;
        }
        // What was followed, each camera's focal length as a multiple of the one given and its extrinsic by how far it turns.
        for (size_t camera = 0; camera < this->request->cameras.size(); ++camera) {
            const calibration_drift::request::camera& asked = this->request->cameras[camera];
            const std::vector<size_t> focal = calibration_drift::focal_parameters(asked.given);
            std::vector<double> given(asked.given.get_parameter_count(), 0.0);
            if (focal.empty() || !asked.given.get_parameters(given.data(), given.size())) {
                continue;
            }
            double ratios[2] = { 1.0e300, -1.0e300 };
            double turn = 0.0;
            double lengths[2] = { 1.0e300, -1.0e300 };
            size_t chunks_held = 0;
            size_t observations = 0;
            const calibration_drift::chunk* first = nullptr;
            for (const calibration_drift::chunk& now : this->request->chunks) {
                if (camera >= now.cameras.size()) {
                    continue;
                }
                const calibration_drift::camera_state& state = now.cameras[camera];
                std::vector<double> parameters(given.size(), 0.0);
                if ((state.camera.get_parameter_count() != given.size()) || !state.camera.get_parameters(parameters.data(), parameters.size())) {
                    continue;
                }
                first = (first == nullptr) ? &now : first;
                const double ratio = parameters[focal[0]] / given[focal[0]];
                ratios[0] = math::min(ratios[0], ratio);
                ratios[1] = math::max(ratios[1], ratio);
                const double length = math::sqrt(state.reference_from_camera.translation().get_length_squared());
                lengths[0] = math::min(lengths[0], length);
                lengths[1] = math::max(lengths[1], length);
                turn = math::max(turn, math::sqrt((first->cameras[camera].reference_from_camera.rotation().inverse() * state.reference_from_camera.rotation()).log().get_length_squared()));
                chunks_held += (state.observations > 0) ? 1u : 0u;
                observations += state.observations;
            }
            if (first == nullptr) {
                continue;
            }
            if (asked.follow_extrinsic) {
                core::logger::log(core::logger::level::note, "Calibration of camera %zu followed over %zu of %zu chunks and %zu observations: focal length %.5f to %.5f of the one given, extrinsic turning up to %.4f degrees, baseline %.5f to %.5f m.", camera, chunks_held, this->request->chunks.size(), observations, ratios[0], ratios[1], turn * 180.0 / math::pi<double>(), lengths[0], lengths[1]);
            }
            else {
                core::logger::log(core::logger::level::note, "Calibration of camera %zu followed over %zu of %zu chunks and %zu observations: focal length %.5f to %.5f of the one given.", camera, chunks_held, this->request->chunks.size(), observations, ratios[0], ratios[1]);
            }
        }
    }

    const sensor::model& calibration_drift_refinement::camera_before(const int frame_id, const sensor::model& otherwise) const {
        if (this->before.empty()) {
            return otherwise;
        }
        const std::unordered_map<int, mapping::frame>::const_iterator found = this->frames.find(frame_id);
        if ((found == this->frames.end()) || (found->second.calibration_chunk < 0)) {
            return otherwise;
        }
        const std::map<long long, calibration_drift::chunk>::const_iterator was = this->before.find(found->second.calibration_chunk);
        return ((was != this->before.end()) && !was->second.cameras.empty()) ? was->second.cameras[0].camera : otherwise;
    }

    void calibration_drift_refinement::restore() {
        for (auto& [frame_id, frame] : this->frames) {
            static_cast<void>(frame_id);
            const std::map<long long, calibration_drift::chunk>::const_iterator was = (frame.calibration_chunk >= 0) ? this->before.find(frame.calibration_chunk) : this->before.end();
            if ((was == this->before.end()) || was->second.cameras.empty()) {
                continue;
            }
            frame.camera = was->second.cameras[0].camera;
            for (size_t index = 0; (index < frame.views.size()) && ((index + 1) < was->second.cameras.size()); ++index) {
                mapping::frame& view = frame.views[index];
                if (view.calibration_chunk >= 0) {
                    view.camera = was->second.cameras[index + 1].camera;
                    view.reference_from_camera = was->second.cameras[index + 1].reference_from_camera;
                }
            }
        }
        this->before.clear();
        if (this->request != nullptr) {
            this->request->refined = false;
            this->request->chunks.clear();
            for (const auto& [chunk, held] : this->known) {
                static_cast<void>(chunk);
                this->request->chunks.push_back(held);
            }
        }
    }
}
