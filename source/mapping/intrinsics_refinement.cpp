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

#include "mapping/intrinsics_refinement.hpp"

#include "core/logger.hpp"
#include "math/matrix.hpp"
#include "optimisation/edges/intrinsics_prior.hpp"
#include "optimisation/vertices/intrinsics.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cstdio>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mapping {
    intrinsics_refinement::intrinsics_refinement(calibration::refinement* const request_value, optimisation::factor_graph& graph_value, const bool landmarks_free)
        : request(request_value)
        , graph(graph_value)
        , enabled((request_value != nullptr) && landmarks_free)
        , node(nullptr)
        , request_parameters()
        , frame_ids()
        , edge_count(0)
        , refined_frame_ids()
        , previous_camera() {
        if (this->request != nullptr) {
            this->request->refined = false;
            this->request->parameters.clear();
            this->request_parameters = intrinsics_refinement::camera_parameters(this->request->camera);
        }
        this->enabled = this->enabled && !this->request_parameters.empty() && (this->request_parameters.size() <= optimisation::vertex::maximum_parameters) && (this->request->free_parameters.size() == this->request_parameters.size());
    }

    bool intrinsics_refinement::make_request(const calibration& camera_calibration, const std::unordered_map<int, mapping::frame>& frames, const std::unordered_set<int>& keyframe_ids, calibration::refinement& request) {
        if (camera_calibration.get_level() == calibration::level::off) {
            return false;
        }
        // The keyframes in id order, so the spread does not depend on the set's order.
        std::vector<int> ordered(keyframe_ids.begin(), keyframe_ids.end());
        std::sort(ordered.begin(), ordered.end());
        std::vector<math::matrix<double, 3, 3>> rotations;
        rotations.reserve(ordered.size());
        for (const int keyframe_id : ordered) {
            const std::unordered_map<int, mapping::frame>::const_iterator found = frames.find(keyframe_id);
            if (found != frames.end()) {
                rotations.push_back(found->second.rotation);
            }
        }
        const double spread = calibration::view_spread_degrees(rotations);
        if (!camera_calibration.ready(rotations.size(), spread)) {
            core::logger::log(core::logger::level::info, "Calibration waits: %zu keyframes spanning %.1f degrees, %zu and %.1f needed.", rotations.size(), spread, calibration::keyframes_minimum, calibration::view_spread_degrees_minimum);
            return false;
        }
        if (!camera_calibration.make_refinement(request)) {
            core::logger::log(core::logger::level::warn, "Calibration: the camera model does not say which of its parameters the configuration asks to refine.");
            return false;
        }
        return true;
    }

    void intrinsics_refinement::add_vertex(const std::unordered_map<int, mapping::frame>& frames, const std::unordered_map<int, optimisation::vertex*>& camera_vertexes) {
        if (!this->enabled || (this->node != nullptr)) {
            return;
        }
        this->node = this->graph.add_vertex(optimisation::vertices::intrinsics::make(this->request->camera, this->request->free_parameters.data(), this->request_parameters.size()));
        if (this->node == nullptr) {
            return;
        }
        for (const auto& [frame_id, vertex] : camera_vertexes) {
            static_cast<void>(vertex);
            const std::unordered_map<int, mapping::frame>::const_iterator found = frames.find(frame_id);
            if ((found != frames.end()) && (intrinsics_refinement::camera_parameters(found->second.camera) == this->request_parameters)) {
                this->frame_ids.insert(frame_id);
            }
        }
    }

    bool intrinsics_refinement::takes(const int frame_id) const {
        return (this->node != nullptr) && (this->frame_ids.count(frame_id) != 0);
    }

    void intrinsics_refinement::attach(optimisation::edge& factor) {
        if (this->node == nullptr) {
            return;
        }
        factor.add_vertex(this->node);
        ++this->edge_count;
    }

    void intrinsics_refinement::add_priors() {
        if (this->node == nullptr) {
            return;
        }
        if (this->edge_count == 0) {
            this->graph.remove_vertex(this->node);
            this->node = nullptr;
            return;
        }
        for (size_t index = 0; index < this->request_parameters.size(); ++index) {
            const double sigma = (index < this->request->prior_sigmas.size()) ? this->request->prior_sigmas[index] : 0.0;
            if ((this->request->free_parameters[index] < 0) || !(sigma > 0.0) || (index >= this->request->prior_values.size())) {
                continue;
            }
            optimisation::edge prior{ optimisation::edges::intrinsics_prior(static_cast<int>(index), this->request->prior_values[index]) };
            prior.add_vertex(this->node);
            prior.set_information(math::matrix<double, 0, 0>::identity(1, 1) * (1.0 / (sigma * sigma)));
            this->graph.add_edge(static_cast<optimisation::edge&&>(prior));
        }
    }

    void intrinsics_refinement::apply(std::unordered_map<int, mapping::frame>& frames) {
        if ((this->node == nullptr) || this->frame_ids.empty()) {
            return;
        }
        const int first = *std::min_element(this->frame_ids.begin(), this->frame_ids.end());
        this->previous_camera = frames.at(first).camera;
        sensor::model refined_camera = this->previous_camera;
        const size_t count = this->request_parameters.size();
        if (!refined_camera.set_parameters(this->node->get_parameters(), count)) {
            return;
        }
        std::vector<int> ordered;
        ordered.reserve(frames.size());
        for (const auto& [frame_id, frame] : frames) {
            static_cast<void>(frame);
            ordered.push_back(frame_id);
        }
        std::sort(ordered.begin(), ordered.end());
        for (const int frame_id : ordered) {
            mapping::frame& frame = frames.at(frame_id);
            if (intrinsics_refinement::camera_parameters(frame.camera) == this->request_parameters) {
                frame.camera = refined_camera;
                this->refined_frame_ids.push_back(frame_id);
            }
        }
        this->request->refined = true;
        this->request->parameters.assign(this->node->get_parameters(), this->node->get_parameters() + count);
        std::vector<char> text((32 * count) + 1, 0);
        size_t used = 0;
        for (size_t index = 0; (index < count) && ((used + 1) < text.size()); ++index) {
            const int written = std::snprintf(text.data() + used, text.size() - used, " %.6g", this->request->parameters[index]);
            used += (written > 0) ? static_cast<size_t>(written) : 0;
        }
        core::logger::log(core::logger::level::note, "Calibration refined over %zu observations:%s.", this->edge_count, text.data());
    }

    const sensor::model& intrinsics_refinement::camera_before(const int frame_id, const mapping::frame& frame) const {
        return std::binary_search(this->refined_frame_ids.begin(), this->refined_frame_ids.end(), frame_id) ? this->previous_camera : frame.camera;
    }

    void intrinsics_refinement::restore(std::unordered_map<int, mapping::frame>& frames) {
        for (const int frame_id : this->refined_frame_ids) {
            frames.at(frame_id).camera = this->previous_camera;
        }
        this->refined_frame_ids.clear();
        if (this->request != nullptr) {
            this->request->refined = false;
            this->request->parameters.clear();
        }
    }
}
