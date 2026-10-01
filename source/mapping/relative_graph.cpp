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

#include "mapping/relative_graph.hpp"

#include "core/logger.hpp"
#include "core/timestamp.hpp"
#include "geometry/plucker.hpp"
#include "math/math.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/relative_similarity.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/similarity.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <unordered_set>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mapping {
    namespace {
        using step_type = optimisation::relative_bundle_adjustment::step;

        // The frame's camera to world transform, in the given units.
        math::sim3<double> world_from_frame(const mapping::frame& frame, const double scale) {
            return math::sim3<double>(math::se3<double>(frame.rotation, frame.translation).inverse(), scale);
        }

        // A similarity on a homogeneous point (x, w): (scale * R * x + t * w, w).
        void transform_point(const math::sim3<double>& similarity, const double (&point)[4], double (&result)[4]) {
            const math::matrix<double, 3, 3> rotation = similarity.transformation().rotation().get_matrix();
            const math::matrix<double, 3, 1>& translation = similarity.transformation().translation();
            const double scale = similarity.scale();
            for (size_t row = 0; row < 3; ++row) {
                result[row] = (scale * ((rotation[row][0] * point[0]) + (rotation[row][1] * point[1]) + (rotation[row][2] * point[2]))) + (translation[row] * point[3]);
            }
            result[3] = point[3];
        }

        // The landmark as a homogeneous world point, a direction when it is at infinity.
        void homogeneous_of(const mapping::point& landmark, double (&point)[4]) {
            if (landmark.inverse_depth) {
                const double rho = landmark.inverse_parameters[2];
                const math::matrix<double, 3, 1> bearing = landmark.anchor_rotation * math::matrix<double, 3, 1>({ landmark.inverse_parameters[0], landmark.inverse_parameters[1], 1.0 });
                for (size_t row = 0; row < 3; ++row) {
                    point[row] = bearing[row] + (landmark.anchor_translation[row] * rho);
                }
                point[3] = rho;
                return;
            }
            for (size_t row = 0; row < 3; ++row) {
                point[row] = landmark.location[row];
            }
            point[3] = 1.0;
        }

        // The child's transform into the parent's coordinates from the frames' embeddings, a chain link keeping a scale of one.
        math::sim3<double> relative_transform(const math::sim3<double>& parent, const math::sim3<double>& child, const bool similarity) {
            math::sim3<double> transform = parent.inverse() * child;
            if (!similarity) {
                transform.scale() = 1.0;
            }
            transform.transformation().rotation() = transform.transformation().rotation().normalised();
            return transform;
        }

        void apply_step(const relative_graph::link& joined, const bool inverse, double (&point)[4]) {
            double result[4];
            transform_point(inverse ? joined.transform.inverse() : joined.transform, point, result);
            for (size_t row = 0; row < 4; ++row) {
                point[row] = result[row];
            }
        }

        template <typename value_type>
        std::vector<int> sorted_keys(const std::unordered_map<int, value_type>& entries) {
            std::vector<int> keys;
            keys.reserve(entries.size());
            for (const auto& entry : entries) {
                keys.push_back(entry.first);
            }
            std::sort(keys.begin(), keys.end());
            return keys;
        }

        double squared_error(const optimisation::relative_bundle_adjustment& problem, const optimisation::relative_bundle_adjustment::observation& measured, double& pixels) {
            double predicted[2];
            if (!problem.predict(measured, predicted) || !math::isfinite(predicted[0]) || !math::isfinite(predicted[1])) {
                pixels = relative_graph::unprojected_error;
                return 1.0e300;
            }
            const double dx = measured.pixel[0] - predicted[0];
            const double dy = measured.pixel[1] - predicted[1];
            pixels = math::sqrt((dx * dx) + (dy * dy));
            return ((dx * dx) + (dy * dy)) / (measured.sigma * measured.sigma);
        }
    }

    int relative_graph::add_link(const int parent, const int child, const math::sim3<double>& transform, const bool similarity) {
        const int index = static_cast<int>(this->links.size());
        link added;
        added.parent = parent;
        added.child = child;
        added.transform = transform;
        added.similarity = similarity;
        this->links.push_back(added);
        this->nodes.at(parent).links.push_back(index);
        this->nodes.at(child).links.push_back(index);
        return index;
    }

    void relative_graph::remove_link(const int index) {
        link& removed = this->links[static_cast<size_t>(index)];
        const int ends[2] = { removed.parent, removed.child };
        for (const int end : ends) {
            const std::unordered_map<int, node>::iterator found = this->nodes.find(end);
            if (found == this->nodes.end()) {
                continue;
            }
            std::vector<int>& incident = found->second.links;
            incident.erase(std::remove(incident.begin(), incident.end(), index), incident.end());
            if (found->second.parent_link == index) {
                found->second.parent_link = -1;
            }
        }
        removed.parent = -1;
        removed.child = -1;
    }

    void relative_graph::remove_node(const int frame_id) {
        const node removed = this->nodes.at(frame_id);
        // The neighbour taking the frame's place: its parent, or for the first frame of a component its oldest child.
        int joining = removed.parent_link;
        int replacement = (joining >= 0) ? this->links[static_cast<size_t>(joining)].parent : -1;
        if (joining < 0) {
            for (const int index : removed.links) {
                const link& candidate = this->links[static_cast<size_t>(index)];
                if ((candidate.parent == frame_id) && (this->nodes.at(candidate.child).parent_link == index) && ((replacement < 0) || (candidate.child < replacement))) {
                    replacement = candidate.child;
                    joining = index;
                }
            }
        }
        if (replacement >= 0) {
            const link joined = this->links[static_cast<size_t>(joining)];
            const math::sim3<double> into_replacement = (joined.parent == replacement) ? joined.transform : joined.transform.inverse();
            for (const int index : removed.links) {
                if (index == joining) {
                    continue;
                }
                const link old = this->links[static_cast<size_t>(index)];
                const int other = (old.parent == frame_id) ? old.child : old.parent;
                if (other == replacement) {
                    continue;
                }
                const bool similarity = old.similarity || joined.similarity;
                if (old.parent == frame_id) {
                    const bool was_parent_link = this->nodes.at(other).parent_link == index;
                    math::sim3<double> transform = into_replacement * old.transform;
                    transform.transformation().rotation() = transform.transformation().rotation().normalised();
                    const int added = this->add_link(replacement, other, transform, similarity);
                    if (was_parent_link) {
                        this->nodes.at(other).parent_link = added;
                    }
                }
                else {
                    math::sim3<double> transform = old.transform * into_replacement.inverse();
                    transform.transformation().rotation() = transform.transformation().rotation().normalised();
                    this->add_link(other, replacement, transform, similarity);
                }
            }
            if (joined.parent == frame_id) {
                // The oldest child becomes the first frame of the component.
                this->nodes.at(replacement).parent_link = -1;
            }
        }
        for (const int index : removed.links) {
            this->remove_link(index);
        }
        this->nodes.erase(frame_id);
        for (std::unordered_map<int, based_point>::iterator it = this->points.begin(); it != this->points.end();) {
            if (it->second.base == frame_id) {
                it = this->points.erase(it);
            }
            else {
                ++it;
            }
        }
    }

    bool relative_graph::base_point(const mapping::point& landmark, const int base, based_point& based) const {
        const std::unordered_map<int, node>::const_iterator found = this->nodes.find(base);
        if (found == this->nodes.end()) {
            return false;
        }
        double world[4];
        homogeneous_of(landmark, world);
        double local[4];
        transform_point(found->second.embedding.inverse(), world, local);
        if (!(local[2] > 0.0)) {
            return false;
        }
        based.base = base;
        based.parameters[0] = local[0] / local[2];
        based.parameters[1] = local[1] / local[2];
        based.parameters[2] = math::max(0.0, local[3] / local[2]);
        based.location = landmark.location;
        return true;
    }

    void relative_graph::synchronise(const mapping::map& reconstruction, const std::vector<std::pair<int, size_t>>& keyframes) {
        std::unordered_set<int> present;
        for (const std::pair<int, size_t>& keyframe : keyframes) {
            if (reconstruction.frames.count(keyframe.first) != 0) {
                present.insert(keyframe.first);
            }
        }
        for (const int frame_id : sorted_keys(this->nodes)) {
            if (present.count(frame_id) == 0) {
                this->remove_node(frame_id);
            }
        }
        size_t moved = 0;
        for (const int frame_id : sorted_keys(this->nodes)) {
            const mapping::frame& frame = reconstruction.frames.at(frame_id);
            node& current = this->nodes.at(frame_id);
            if ((frame.rotation == current.rotation) && (frame.translation == current.translation)) {
                continue;
            }
            current.embedding = world_from_frame(frame, current.embedding.scale());
            current.rotation = frame.rotation;
            current.translation = frame.translation;
            if (current.parent_link >= 0) {
                link& joined = this->links[static_cast<size_t>(current.parent_link)];
                joined.transform = relative_transform(this->nodes.at(joined.parent).embedding, current.embedding, joined.similarity);
            }
            ++moved;
        }
        size_t added = 0;
        int previous_id = -1;
        size_t previous_submap = 0;
        for (const std::pair<int, size_t>& keyframe : keyframes) {
            if (present.count(keyframe.first) == 0) {
                continue;
            }
            const bool chained = (previous_id >= 0) && (previous_submap == keyframe.second);
            if (this->nodes.count(keyframe.first) == 0) {
                const mapping::frame& frame = reconstruction.frames.at(keyframe.first);
                node joining;
                joining.embedding = world_from_frame(frame, chained ? this->nodes.at(previous_id).embedding.scale() : 1.0);
                joining.origin = joining.embedding;
                joining.rotation = frame.rotation;
                joining.translation = frame.translation;
                this->nodes[keyframe.first] = joining;
                if (chained) {
                    const math::sim3<double> transform = relative_transform(this->nodes.at(previous_id).embedding, this->nodes.at(keyframe.first).embedding, false);
                    this->nodes.at(keyframe.first).parent_link = this->add_link(previous_id, keyframe.first, transform, false);
                }
                ++added;
            }
            previous_id = keyframe.first;
            previous_submap = keyframe.second;
        }

        for (std::unordered_map<int, based_point>::iterator it = this->points.begin(); it != this->points.end();) {
            if (reconstruction.landmarks.count(it->first) == 0) {
                it = this->points.erase(it);
            }
            else {
                ++it;
            }
        }
        size_t based = 0;
        for (const int landmark_id : sorted_keys(reconstruction.landmarks)) {
            const mapping::point& landmark = reconstruction.landmarks.at(landmark_id);
            const std::unordered_map<int, based_point>::iterator existing = this->points.find(landmark_id);
            const bool held = (existing != this->points.end()) && (this->nodes.count(existing->second.base) != 0);
            if (held && (existing->second.location == landmark.location)) {
                continue;
            }
            based_point rebased;
            bool found = held && this->base_point(landmark, existing->second.base, rebased);
            const std::unordered_map<int, std::vector<mapping::map::observation>>::const_iterator observations_it = reconstruction.observations.find(landmark_id);
            if (observations_it != reconstruction.observations.end()) {
                for (const mapping::map::observation& obs : observations_it->second) {
                    if (found) {
                        break;
                    }
                    found = this->base_point(landmark, obs.frame_id, rebased);
                }
            }
            if (found) {
                this->points[landmark_id] = rebased;
                ++based;
            }
            else if (existing != this->points.end()) {
                this->points.erase(existing);
            }
        }
        if ((added > 0) || (moved > 0) || (based > 0) || (present.size() != this->nodes.size())) {
            core::logger::log(core::logger::level::debug, "Relative graph: %zu frames (%zu new, %zu moved), %zu points (%zu based).", this->nodes.size(), added, moved, this->points.size(), based);
        }
    }

    int relative_graph::add_loop(const int parent, const int child, const math::sim3<double>& child_to_parent) {
        if ((parent == child) || (this->nodes.count(parent) == 0) || (this->nodes.count(child) == 0)) {
            return -1;
        }
        math::sim3<double> transform = child_to_parent;
        transform.transformation().rotation() = transform.transformation().rotation().normalised();
        return this->add_link(parent, child, transform, true);
    }

    void relative_graph::search(const int source, const std::vector<int>* const targets) {
        this->search_towards.clear();
        this->search_queue.clear();
        if (this->nodes.count(source) == 0) {
            return;
        }
        this->search_towards[source] = -1;
        this->search_queue.push_back(source);
        size_t remaining = 0;
        if (targets != nullptr) {
            for (const int target : *targets) {
                remaining += static_cast<size_t>((target != source) ? 1 : 0);
            }
            if (remaining == 0) {
                return;
            }
        }
        for (size_t head = 0; head < this->search_queue.size(); ++head) {
            const int current = this->search_queue[head];
            for (const int index : this->nodes.at(current).links) {
                const link& joined = this->links[static_cast<size_t>(index)];
                const int other = (joined.parent == current) ? joined.child : joined.parent;
                if (this->search_towards.count(other) != 0) {
                    continue;
                }
                this->search_towards[other] = index;
                this->search_queue.push_back(other);
                if ((targets != nullptr) && std::binary_search(targets->begin(), targets->end(), other) && (--remaining == 0)) {
                    return;
                }
            }
        }
    }

    void relative_graph::steps_to_source(const int from, std::vector<step_type>& steps) const {
        steps.clear();
        int current = from;
        for (int index = this->search_towards.at(current); index >= 0; index = this->search_towards.at(current)) {
            const link& joined = this->links[static_cast<size_t>(index)];
            step_type taken;
            taken.transform = index;
            taken.inverse = (joined.child == current);
            steps.push_back(taken);
            current = (joined.parent == current) ? joined.child : joined.parent;
        }
    }

    void relative_graph::steps_from_source(const int to, std::vector<step_type>& steps) const {
        steps.clear();
        int current = to;
        for (int index = this->search_towards.at(current); index >= 0; index = this->search_towards.at(current)) {
            const link& joined = this->links[static_cast<size_t>(index)];
            const int next = (joined.parent == current) ? joined.child : joined.parent;
            step_type taken;
            taken.transform = index;
            taken.inverse = (joined.child == next);
            steps.push_back(taken);
            current = next;
        }
        std::reverse(steps.begin(), steps.end());
    }

    bool relative_graph::path(const int observer, const int base, std::vector<step_type>& steps) {
        const std::vector<int> targets = { observer };
        this->search(base, &targets);
        if (this->search_towards.count(observer) == 0) {
            steps.clear();
            return false;
        }
        this->steps_to_source(observer, steps);
        return true;
    }

    bool relative_graph::point_in_frame(const int point_id, const int observer, math::matrix<double, 3, 1>& point) {
        const std::unordered_map<int, based_point>::const_iterator found = this->points.find(point_id);
        std::vector<step_type> steps;
        if ((found == this->points.end()) || !this->path(observer, found->second.base, steps)) {
            return false;
        }
        double current[4] = { found->second.parameters[0], found->second.parameters[1], 1.0, found->second.parameters[2] };
        for (size_t a = steps.size(); a-- > 0;) {
            apply_step(this->links[static_cast<size_t>(steps[a].transform)], steps[a].inverse, current);
        }
        point = math::matrix<double, 3, 1>({ current[0], current[1], current[2] });
        return true;
    }

    std::vector<int> relative_graph::component(const int frame_id) {
        this->search(frame_id, nullptr);
        std::vector<int> members(this->search_queue);
        std::sort(members.begin(), members.end());
        return members;
    }

    double relative_graph::mean_error(const mapping::map& reconstruction, const int frame_id, const std::vector<std::pair<int, const mapping::map::observation*>>& observed) {
        std::vector<int> bases;
        bases.reserve(observed.size());
        for (const std::pair<int, const mapping::map::observation*>& entry : observed) {
            bases.push_back(this->points.at(entry.first).base);
        }
        std::sort(bases.begin(), bases.end());
        bases.erase(std::unique(bases.begin(), bases.end()), bases.end());
        this->search(frame_id, &bases);
        const mapping::frame& frame = reconstruction.frames.at(frame_id);
        std::vector<step_type> steps;
        double total = 0.0;
        size_t count = 0;
        for (const std::pair<int, const mapping::map::observation*>& entry : observed) {
            const based_point& based = this->points.at(entry.first);
            if (this->search_towards.count(based.base) == 0) {
                continue;
            }
            this->steps_from_source(based.base, steps);
            double current[4] = { based.parameters[0], based.parameters[1], 1.0, based.parameters[2] };
            for (size_t a = steps.size(); a-- > 0;) {
                apply_step(this->links[static_cast<size_t>(steps[a].transform)], steps[a].inverse, current);
            }
            const double scaled[3] = { current[0], current[1], current[2] };
            double pixel[2];
            double error = relative_graph::unprojected_error;
            if (frame.camera.project(&scaled[0], &pixel[0]) && math::isfinite(pixel[0]) && math::isfinite(pixel[1])) {
                const double dx = pixel[0] - entry.second->point[0];
                const double dy = pixel[1] - entry.second->point[1];
                error = math::sqrt((dx * dx) + (dy * dy));
            }
            total += error;
            ++count;
        }
        return (count > 0) ? (total / static_cast<double>(count)) : 0.0;
    }

    relative_graph::summary relative_graph::adjust(mapping::map& reconstruction, const std::vector<int>& seeds, const std::vector<int>& fresh_links, const int rounds) {
        const long long int started = core::timestamp();
        summary result;

        // Each frame's observations of the points the graph holds.
        std::vector<int> landmark_ids;
        landmark_ids.reserve(reconstruction.observations.size());
        for (const auto& [landmark_id, landmark_observations] : reconstruction.observations) {
            static_cast<void>(landmark_observations);
            if (this->points.count(landmark_id) != 0) {
                landmark_ids.push_back(landmark_id);
            }
        }
        std::sort(landmark_ids.begin(), landmark_ids.end());
        std::unordered_map<int, std::vector<std::pair<int, const mapping::map::observation*>>> observed;
        for (const int landmark_id : landmark_ids) {
            for (const mapping::map::observation& obs : reconstruction.observations.at(landmark_id)) {
                if (this->nodes.count(obs.frame_id) != 0) {
                    observed[obs.frame_id].push_back({ landmark_id, &obs });
                }
            }
        }
        const std::vector<std::pair<int, const mapping::map::observation*>> nothing_observed;
        const auto observed_by = [&observed, &nothing_observed](const int frame_id) -> const std::vector<std::pair<int, const mapping::map::observation*>>& {
            const std::unordered_map<int, std::vector<std::pair<int, const mapping::map::observation*>>>::const_iterator found = observed.find(frame_id);
            return (found != observed.end()) ? found->second : nothing_observed;
        };

        // The region starts as the frames nearest the seeds, and grows outwards through every frame whose error has changed
        // since it was last adjusted.
        std::vector<int> queue;
        std::unordered_set<int> reached;
        for (const int seed : seeds) {
            if ((this->nodes.count(seed) != 0) && reached.insert(seed).second) {
                queue.push_back(seed);
            }
        }
        const size_t seed_count = queue.size();
        const size_t active_limit = seed_count + this->region.frames_maximum;
        std::vector<int> active_frames;
        std::unordered_set<int> active;
        for (size_t head = 0; (head < queue.size()) && (active_frames.size() < active_limit); ++head) {
            const int frame_id = queue[head];
            bool activate = (head < seed_count) || (active_frames.size() < this->region.frames_minimum);
            if (!activate) {
                const double baseline = this->nodes.at(frame_id).error;
                activate = (baseline < 0.0) || (math::abs(this->mean_error(reconstruction, frame_id, observed_by(frame_id)) - baseline) > this->region.error_change);
            }
            if (!activate) {
                continue;
            }
            active.insert(frame_id);
            active_frames.push_back(frame_id);
            for (const int index : this->nodes.at(frame_id).links) {
                const link& joined = this->links[static_cast<size_t>(index)];
                const int other = (joined.parent == frame_id) ? joined.child : joined.parent;
                if (reached.insert(other).second) {
                    queue.push_back(other);
                }
            }
        }
        if (active_frames.empty()) {
            return result;
        }

        class entry final {
        public:
            int landmark_id;
            int frame_id;
        };

        optimisation::relative_bundle_adjustment problem;
        std::vector<entry> entries;
        std::vector<int> point_ids;
        std::unordered_map<int, int> transform_of_link;
        std::vector<int> link_of_transform;
        std::vector<int> static_frames;
        std::vector<step_type> steps;
        std::vector<int> targets;

        // The problem for the active frames: the transforms into them and every loop touching one are solved for, with the
        // points they see, which every other frame seeing those points holds as it is.
        const auto build = [&]() {
            problem = optimisation::relative_bundle_adjustment();
            entries.clear();
            point_ids.clear();
            transform_of_link.clear();
            link_of_transform.clear();
            static_frames.clear();
            std::vector<char> link_active(this->links.size(), 0);
            for (const int frame_id : active_frames) {
                const node& current = this->nodes.at(frame_id);
                if (current.parent_link >= 0) {
                    link_active[static_cast<size_t>(current.parent_link)] = 1;
                }
                for (const int index : current.links) {
                    const link& joined = this->links[static_cast<size_t>(index)];
                    if (this->nodes.at(joined.child).parent_link != index) {
                        link_active[static_cast<size_t>(index)] = 1;
                    }
                }
            }
            for (const int index : fresh_links) {
                if ((index >= 0) && (static_cast<size_t>(index) < this->links.size()) && (this->links[static_cast<size_t>(index)].parent >= 0)) {
                    link_active[static_cast<size_t>(index)] = 1;
                }
            }
            // The points grouped by base, so one search finds the chains to all of their observers.
            std::vector<std::pair<int, int>> by_base;
            for (const int frame_id : active_frames) {
                for (const std::pair<int, const mapping::map::observation*>& seen : observed_by(frame_id)) {
                    by_base.push_back({ this->points.at(seen.first).base, seen.first });
                }
            }
            std::sort(by_base.begin(), by_base.end());
            by_base.erase(std::unique(by_base.begin(), by_base.end()), by_base.end());
            std::unordered_map<int, int> camera_of_frame;
            std::unordered_set<int> static_set;
            size_t scale_links = 0;
            for (size_t group = 0; group < by_base.size();) {
                const int base = by_base[group].first;
                size_t group_end = group;
                targets.clear();
                while ((group_end < by_base.size()) && (by_base[group_end].first == base)) {
                    for (const mapping::map::observation& obs : reconstruction.observations.at(by_base[group_end].second)) {
                        if (this->nodes.count(obs.frame_id) != 0) {
                            targets.push_back(obs.frame_id);
                        }
                    }
                    ++group_end;
                }
                std::sort(targets.begin(), targets.end());
                targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
                this->search(base, &targets);
                for (size_t member = group; member < group_end; ++member) {
                    const int landmark_id = by_base[member].second;
                    const based_point& based = this->points.at(landmark_id);
                    const size_t first_observation = problem.observations.size();
                    size_t static_observers = 0;
                    for (const mapping::map::observation& obs : reconstruction.observations.at(landmark_id)) {
                        if ((this->nodes.count(obs.frame_id) == 0) || (this->search_towards.count(obs.frame_id) == 0)) {
                            continue;
                        }
                        this->steps_to_source(obs.frame_id, steps);
                        const mapping::frame& frame = reconstruction.frames.at(obs.frame_id);
                        const std::unordered_map<int, int>::const_iterator camera_it = camera_of_frame.find(obs.frame_id);
                        int camera = (camera_it != camera_of_frame.end()) ? camera_it->second : -1;
                        if (camera < 0) {
                            camera = static_cast<int>(problem.cameras.size());
                            camera_of_frame[obs.frame_id] = camera;
                            problem.cameras.push_back(frame.camera);
                        }
                        optimisation::relative_bundle_adjustment::observation measured;
                        measured.landmark = static_cast<int>(problem.landmarks.size());
                        measured.camera = camera;
                        measured.pixel[0] = obs.point[0];
                        measured.pixel[1] = obs.point[1];
                        measured.sigma = mapping::map::observation_sigma(frame, obs);
                        for (const step_type& taken : steps) {
                            const std::unordered_map<int, int>::const_iterator transform_it = transform_of_link.find(taken.transform);
                            int transform = (transform_it != transform_of_link.end()) ? transform_it->second : -1;
                            if (transform < 0) {
                                transform = static_cast<int>(problem.transforms.size());
                                transform_of_link[taken.transform] = transform;
                                link_of_transform.push_back(taken.transform);
                                const link& joined = this->links[static_cast<size_t>(taken.transform)];
                                optimisation::relative_bundle_adjustment::transform added;
                                added.estimate = joined.transform;
                                added.similarity = joined.similarity;
                                added.active = link_active[static_cast<size_t>(taken.transform)] != 0;
                                problem.transforms.push_back(added);
                            }
                            step_type mapped = taken;
                            mapped.transform = transform;
                            measured.path.push_back(mapped);
                        }
                        problem.observations.push_back(measured);
                        entries.push_back({ landmark_id, obs.frame_id });
                        if (active.count(obs.frame_id) == 0) {
                            static_observers += 1;
                            if (static_set.insert(obs.frame_id).second) {
                                static_frames.push_back(obs.frame_id);
                            }
                        }
                    }
                    optimisation::relative_bundle_adjustment::landmark point;
                    point.parameters[0] = based.parameters[0];
                    point.parameters[1] = based.parameters[1];
                    point.parameters[2] = based.parameters[2];
                    point.active = (problem.observations.size() - first_observation) >= 2;
                    problem.landmarks.push_back(point);
                    point_ids.push_back(landmark_id);
                    scale_links += static_cast<size_t>((point.active && (static_observers >= 2)) ? 1 : 0);
                }
                group = group_end;
            }
            std::sort(static_frames.begin(), static_frames.end());

            // Without two static frames seeing enough of the active points the scale of the region is free, so the length of
            // the transform joining its oldest frame to the map, or failing that the first it solves for, is held.
            if (scale_links < relative_graph::scale_links_minimum) {
                int held = -1;
                int oldest = active_frames.front();
                for (const int frame_id : active_frames) {
                    oldest = math::min(oldest, frame_id);
                }
                const int oldest_link = this->nodes.at(oldest).parent_link;
                const std::unordered_map<int, int>::const_iterator oldest_it = transform_of_link.find(oldest_link);
                if ((oldest_link >= 0) && (oldest_it != transform_of_link.end()) && problem.transforms[static_cast<size_t>(oldest_it->second)].active && !problem.transforms[static_cast<size_t>(oldest_it->second)].similarity) {
                    held = oldest_it->second;
                }
                for (size_t index = 0; (held < 0) && (index < problem.transforms.size()); ++index) {
                    const optimisation::relative_bundle_adjustment::transform& candidate = problem.transforms[index];
                    if (candidate.active && !candidate.similarity) {
                        held = static_cast<int>(index);
                    }
                }
                if (held >= 0) {
                    const double length = math::sqrt(problem.transforms[static_cast<size_t>(held)].estimate.transformation().translation().get_length_squared());
                    if (length > 1.0e-9) {
                        optimisation::relative_bundle_adjustment::length_prior prior;
                        prior.transform = held;
                        prior.length = length;
                        prior.information = relative_graph::scale_prior_information / (length * length);
                        problem.priors.push_back(prior);
                    }
                }
            }
        };

        // The state before the adjustment, for a diverged one to restore.
        const std::vector<link> links_before = this->links;
        std::unordered_map<int, based_point> points_before;
        const auto restore = [&]() {
            this->links = links_before;
            for (const auto& [landmark_id, based] : points_before) {
                this->points[landmark_id] = based;
            }
        };

        std::vector<double> errors_before;
        std::vector<double> errors_after;
        std::vector<double> squared;
        std::vector<entry> removed;
        std::unordered_map<int, double> changes;
        int iterations = 0;
        for (;;) {
            ++iterations;
            build();
            const bool solving_transforms = std::any_of(problem.transforms.begin(), problem.transforms.end(), [](const optimisation::relative_bundle_adjustment::transform& transform) {
                return transform.active;
            });
            const bool solving_points = std::any_of(problem.landmarks.begin(), problem.landmarks.end(), [](const optimisation::relative_bundle_adjustment::landmark& point) {
                return point.active;
            });
            if (!solving_transforms && !solving_points) {
                restore();
                return result;
            }
            errors_before.resize(problem.observations.size());
            for (size_t index = 0; index < problem.observations.size(); ++index) {
                static_cast<void>(squared_error(problem, problem.observations[index], errors_before[index]));
            }

            // Two passes as the absolute adjustment makes them: the observations beyond the bound after the first leave the second.
            removed.clear();
            const int first_rounds = (rounds > mapping::map::first_pass_rounds) ? mapping::map::first_pass_rounds : rounds;
            optimisation::relative_bundle_adjustment::summary pass = problem.solve(first_rounds);
            if (iterations == 1) {
                result.initial_cost = pass.initial_cost;
            }
            result.accepted += pass.accepted;
            if (rounds > first_rounds) {
                size_t write = 0;
                std::vector<int> remaining(problem.landmarks.size(), 0);
                for (size_t read = 0; read < problem.observations.size(); ++read) {
                    double pixels = 0.0;
                    if (squared_error(problem, problem.observations[read], pixels) > mapping::map::inlier_bound_squared) {
                        removed.push_back(entries[read]);
                        continue;
                    }
                    ++remaining[static_cast<size_t>(problem.observations[read].landmark)];
                    problem.observations[write] = problem.observations[read];
                    entries[write] = entries[read];
                    errors_before[write] = errors_before[read];
                    ++write;
                }
                problem.observations.resize(write);
                entries.resize(write);
                errors_before.resize(write);
                for (size_t index = 0; index < problem.landmarks.size(); ++index) {
                    problem.landmarks[index].active = problem.landmarks[index].active && (remaining[index] >= 2);
                }
                pass = problem.solve(rounds - first_rounds);
                result.accepted += pass.accepted;
            }
            result.final_cost = pass.final_cost;

            // An adjustment leaving a quarter of its observations grossly wrong has diverged, and the graph keeps its state.
            size_t gross = 0;
            squared.resize(problem.observations.size());
            errors_after.resize(problem.observations.size());
            for (size_t index = 0; index < problem.observations.size(); ++index) {
                squared[index] = squared_error(problem, problem.observations[index], errors_after[index]);
                gross += static_cast<size_t>((squared[index] > mapping::map::gross_error_squared) ? 1 : 0);
            }
            if ((problem.observations.size() >= mapping::map::divergence_minimum_observations) && (gross * mapping::map::divergence_gross_fraction_denominator > problem.observations.size())) {
                core::logger::log(core::logger::level::warn, "Relative adjustment diverged: %zu of %zu observations beyond %.0f px; the graph keeps its state before it.", gross, problem.observations.size(), math::sqrt(mapping::map::gross_error_squared));
                restore();
                result.diverged = true;
                return result;
            }

            for (size_t index = 0; index < problem.transforms.size(); ++index) {
                if (problem.transforms[index].active) {
                    this->links[static_cast<size_t>(link_of_transform[index])].transform = problem.transforms[index].estimate;
                }
            }
            for (size_t index = 0; index < problem.landmarks.size(); ++index) {
                if (!problem.landmarks[index].active) {
                    continue;
                }
                based_point& based = this->points.at(point_ids[index]);
                points_before.insert({ point_ids[index], based });
                based.parameters[0] = problem.landmarks[index].parameters[0];
                based.parameters[1] = problem.landmarks[index].parameters[1];
                based.parameters[2] = math::max(0.0, problem.landmarks[index].parameters[2]);
            }

            // The adjustment ripples out: a static frame whose mean error it has changed by more than the threshold joins the
            // region, which is solved again, until the changes die away.
            std::unordered_map<int, double> sums;
            for (size_t index = 0; index < problem.observations.size(); ++index) {
                if (active.count(entries[index].frame_id) == 0) {
                    sums[entries[index].frame_id] += errors_after[index] - errors_before[index];
                }
            }
            std::vector<int> joining;
            for (const int frame_id : static_frames) {
                const size_t count = observed_by(frame_id).size();
                double& change = changes[frame_id];
                change += (count > 0) ? (sums[frame_id] / static_cast<double>(count)) : 0.0;
                if (math::abs(change) > this->region.error_change) {
                    joining.push_back(frame_id);
                }
            }
            if (joining.empty() || (iterations >= this->region.ripples_maximum) || (active_frames.size() >= active_limit)) {
                break;
            }
            // The newest of them first, as far as the region has room.
            std::sort(joining.begin(), joining.end(), [](const int lhs, const int rhs) {
                return lhs > rhs;
            });
            for (const int frame_id : joining) {
                if (active_frames.size() >= active_limit) {
                    break;
                }
                active.insert(frame_id);
                active_frames.push_back(frame_id);
            }
        }

        result.active_frames = static_cast<int>(active_frames.size());
        result.static_frames = static_cast<int>(static_frames.size());
        for (const optimisation::relative_bundle_adjustment::transform& transform : problem.transforms) {
            result.active_links += transform.active ? 1 : 0;
        }
        for (size_t index = 0; index < problem.landmarks.size(); ++index) {
            if (problem.landmarks[index].active) {
                ++result.active_points;
                result.adjusted_points.push_back(point_ids[index]);
            }
        }
        result.observations = static_cast<int>(problem.observations.size());
        result.priors = static_cast<int>(problem.priors.size());

        // The active frames' errors are the baselines their next changes are measured from.
        std::unordered_map<int, std::pair<double, size_t>> frame_errors;
        for (size_t index = 0; index < problem.observations.size(); ++index) {
            if (squared[index] > mapping::map::inlier_bound_squared) {
                removed.push_back(entries[index]);
                continue;
            }
            std::pair<double, size_t>& accumulated = frame_errors[entries[index].frame_id];
            accumulated.first += errors_after[index];
            ++accumulated.second;
        }
        for (const int frame_id : active_frames) {
            const std::unordered_map<int, std::pair<double, size_t>>::const_iterator found = frame_errors.find(frame_id);
            this->nodes.at(frame_id).error = ((found != frame_errors.end()) && (found->second.second > 0)) ? (found->second.first / static_cast<double>(found->second.second)) : 0.0;
        }

        // Observations beyond the bound leave the map, and with them the landmarks left with fewer than two.
        std::sort(removed.begin(), removed.end(), [](const entry& lhs, const entry& rhs) {
            return (lhs.landmark_id != rhs.landmark_id) ? (lhs.landmark_id < rhs.landmark_id) : (lhs.frame_id < rhs.frame_id);
        });
        for (size_t index = 0; index < removed.size();) {
            const int landmark_id = removed[index].landmark_id;
            const std::unordered_map<int, std::vector<mapping::map::observation>>::iterator observations_it = reconstruction.observations.find(landmark_id);
            size_t next = index;
            while ((next < removed.size()) && (removed[next].landmark_id == landmark_id)) {
                ++next;
            }
            if (observations_it != reconstruction.observations.end()) {
                std::vector<mapping::map::observation>& landmark_observations = observations_it->second;
                size_t write = 0;
                for (size_t read = 0; read < landmark_observations.size(); ++read) {
                    bool drop = false;
                    for (size_t candidate = index; candidate < next; ++candidate) {
                        drop = drop || (removed[candidate].frame_id == landmark_observations[read].frame_id);
                    }
                    if (drop) {
                        ++result.removed;
                    }
                    else {
                        landmark_observations[write++] = landmark_observations[read];
                    }
                }
                landmark_observations.resize(write);
                if (landmark_observations.size() < 2) {
                    reconstruction.observations.erase(observations_it);
                    reconstruction.landmarks.erase(landmark_id);
                    this->points.erase(landmark_id);
                }
            }
            index = next;
        }
        result.adjusted_points.erase(std::remove_if(result.adjusted_points.begin(), result.adjusted_points.end(), [this](const int landmark_id) {
                                         return this->points.count(landmark_id) == 0;
                                     }),
                                     result.adjusted_points.end());
        result.iterations = iterations;
        core::logger::log(core::logger::level::info, "Relative adjustment: %f to %f [frames: %d active, %d static; links: %d; points: %d; observations: %d; priors: %d] [%d accepted rounds over %d regions], %d outlier observations removed, %.1f ms.", result.initial_cost, result.final_cost, result.active_frames, result.static_frames, result.active_links, result.active_points, result.observations, result.priors, result.accepted, result.iterations, result.removed, static_cast<double>(core::timestamp() - started) * 1.0e-6);
        return result;
    }

    void relative_graph::write(mapping::map& reconstruction, const std::unordered_map<int, math::sim3<double>>& embeddings) {
        std::unordered_map<int, math::sim3<double>> moved;
        for (const auto& [frame_id, embedding] : embeddings) {
            node& current = this->nodes.at(frame_id);
            moved[frame_id] = embedding * current.embedding.inverse();
            current.embedding = embedding;
            current.embedding.transformation().rotation() = current.embedding.transformation().rotation().normalised();
            const std::unordered_map<int, mapping::frame>::iterator frame_it = reconstruction.frames.find(frame_id);
            if (frame_it == reconstruction.frames.end()) {
                continue;
            }
            const math::se3<double> pose = current.embedding.transformation().inverse();
            frame_it->second.rotation = pose.rotation().get_matrix();
            frame_it->second.translation = pose.translation();
            current.rotation = frame_it->second.rotation;
            current.translation = frame_it->second.translation;
        }
        for (auto& [landmark_id, based] : this->points) {
            const std::unordered_map<int, math::sim3<double>>::const_iterator embedding_it = embeddings.find(based.base);
            const std::unordered_map<int, mapping::point>::iterator landmark_it = reconstruction.landmarks.find(landmark_id);
            if ((embedding_it == embeddings.end()) || (landmark_it == reconstruction.landmarks.end())) {
                continue;
            }
            const math::sim3<double>& embedding = this->nodes.at(based.base).embedding;
            mapping::point& landmark = landmark_it->second;
            landmark.anchor_rotation = embedding.transformation().rotation().get_matrix();
            landmark.anchor_translation = embedding.transformation().translation();
            landmark.inverse_parameters = math::matrix<double, 3, 1>({ based.parameters[0], based.parameters[1], based.parameters[2] / embedding.scale() });
            landmark.inverse_depth = true;
            landmark.update_location_from_inverse_depth();
            based.location = landmark.location;
        }
        for (const auto& [landmark_id, landmark_observations] : reconstruction.line_observations) {
            if (landmark_observations.empty()) {
                continue;
            }
            const std::unordered_map<int, math::sim3<double>>::const_iterator moved_it = moved.find(landmark_observations.front().frame_id);
            const std::unordered_map<int, mapping::line>::iterator landmark_it = reconstruction.line_landmarks.find(landmark_id);
            if ((moved_it == moved.end()) || (landmark_it == reconstruction.line_landmarks.end())) {
                continue;
            }
            mapping::line& landmark = landmark_it->second;
            landmark.locations[0] = moved_it->second * landmark.locations[0];
            landmark.locations[1] = moved_it->second * landmark.locations[1];
            geometry::plucker moved_line;
            if (geometry::plucker::from_points(landmark.locations[0], landmark.locations[1], moved_line)) {
                landmark.plucker_line = moved_line;
            }
        }
    }

    void relative_graph::embed_component(const int root, const math::sim3<double>& root_embedding, std::unordered_map<int, math::sim3<double>>& embeddings) {
        this->search(root, nullptr);
        embeddings[root] = root_embedding;
        for (size_t index = 1; index < this->search_queue.size(); ++index) {
            const int frame_id = this->search_queue[index];
            const link& joined = this->links[static_cast<size_t>(this->search_towards.at(frame_id))];
            const int previous = (joined.parent == frame_id) ? joined.child : joined.parent;
            const math::sim3<double>& previous_embedding = embeddings.at(previous);
            embeddings[frame_id] = (joined.parent == previous) ? (previous_embedding * joined.transform) : (previous_embedding * joined.transform.inverse());
        }
    }

    void relative_graph::embed(mapping::map& reconstruction, const int root) {
        const std::unordered_map<int, mapping::frame>::const_iterator root_frame = reconstruction.frames.find(root);
        if ((this->nodes.count(root) == 0) || (root_frame == reconstruction.frames.end())) {
            return;
        }
        std::unordered_map<int, math::sim3<double>> embeddings;
        this->embed_component(root, world_from_frame(root_frame->second, this->nodes.at(root).embedding.scale()), embeddings);
        this->write(reconstruction, embeddings);
    }

    bool relative_graph::relax(mapping::map& reconstruction, const int rounds) {
        // Each component starts embedded from the frame the pose graph holds: the gauge frame where it joined the graph, or
        // the component's newest frame where it is now.
        const std::vector<int> frame_ids = sorted_keys(this->nodes);
        std::vector<int> roots;
        if (this->nodes.count(reconstruction.gauge_frame_id) != 0) {
            roots.push_back(reconstruction.gauge_frame_id);
        }
        roots.insert(roots.end(), frame_ids.rbegin(), frame_ids.rend());
        std::unordered_map<int, math::sim3<double>> embeddings;
        std::unordered_set<int> held;
        for (const int root : roots) {
            if ((embeddings.count(root) != 0) || (reconstruction.frames.count(root) == 0)) {
                continue;
            }
            held.insert(root);
            this->embed_component(root, (root == reconstruction.gauge_frame_id) ? this->nodes.at(root).origin : world_from_frame(reconstruction.frames.at(root), this->nodes.at(root).embedding.scale()), embeddings);
        }
        const auto similarity_parameters = [](const math::sim3<double>& similarity, double (&parameters)[8]) {
            const math::matrix<double, 4, 1> quaternion = similarity.transformation().rotation().get_quaternion();
            parameters[0] = similarity.transformation().translation()[0];
            parameters[1] = similarity.transformation().translation()[1];
            parameters[2] = similarity.transformation().translation()[2];
            parameters[3] = quaternion[1];
            parameters[4] = quaternion[2];
            parameters[5] = quaternion[3];
            parameters[6] = quaternion[0];
            parameters[7] = similarity.scale();
        };
        optimisation::factor_graph graph;
        std::unordered_map<int, optimisation::vertex*> vertices;
        for (const int frame_id : frame_ids) {
            const std::unordered_map<int, math::sim3<double>>::const_iterator embedding = embeddings.find(frame_id);
            if (embedding == embeddings.end()) {
                continue;
            }
            optimisation::vertex vertex{ optimisation::vertices::similarity() };
            double parameters[8];
            similarity_parameters(embedding->second, parameters);
            vertex.set_parameters(&parameters[0], 8);
            vertex.set_fixed(held.count(frame_id) != 0);
            vertices[frame_id] = graph.add_vertex(static_cast<optimisation::vertex&&>(vertex));
        }
        size_t loops = 0;
        for (size_t index = 0; index < this->links.size(); ++index) {
            const link& joined = this->links[index];
            if ((joined.parent < 0) || (vertices.count(joined.parent) == 0) || (vertices.count(joined.child) == 0)) {
                continue;
            }
            optimisation::edge constraint{ optimisation::edges::relative_similarity() };
            double parameters[8];
            similarity_parameters(joined.transform, parameters);
            constraint.set_observation(math::matrix<double, 0, 0>(8, 1, &parameters[0]));
            constraint.add_vertex(vertices.at(joined.parent));
            constraint.add_vertex(vertices.at(joined.child));
            graph.add_edge(static_cast<optimisation::edge&&>(constraint));
            loops += static_cast<size_t>((this->nodes.at(joined.child).parent_link != static_cast<int>(index)) ? 1 : 0);
        }
        // A graph without loops is consistent as it is embedded.
        if ((loops > 0) && (graph.solve(rounds, true) > 0)) {
            for (const auto& [frame_id, vertex] : vertices) {
                const double* const p = vertex->get_parameters();
                embeddings[frame_id] = math::sim3<double>(math::se3<double>(math::so3<double>(p[6], p[3], p[4], p[5]), { { p[0], p[1], p[2] } }), p[7]);
            }
        }
        this->write(reconstruction, embeddings);
        core::logger::log(core::logger::level::note, "Relaxed the relative graph: %zu frames, %zu components, %zu loops.", embeddings.size(), held.size(), loops);
        return true;
    }
}
