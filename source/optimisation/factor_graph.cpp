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

#include "optimisation/factor_graph.hpp"

#include "core/assert.hpp"
#include "core/logger.hpp"
#include "core/thread_pool.hpp"
#include "math/math.hpp"
#include "math/matrix_conjugate_gradient.hpp"
#include "math/matrix_decomposition_cholesky.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <utility>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace optimisation {
    factor_graph::factor_graph() {
    }

    const factor_graph::diagnostics& factor_graph::get_diagnostics() const {
        return this->last_diagnostics;
    }

    void factor_graph::set_strategy(const strategy solve_strategy_value) {
        this->solve_strategy = solve_strategy_value;
    }

    void factor_graph::set_precision(const precision solve_precision_value) {
        this->solve_precision = solve_precision_value;
    }

    void factor_graph::set_square_root_parameter_threshold(const int threshold) {
        this->square_root_parameter_threshold = threshold;
    }

    void factor_graph::set_conjugate_gradient_tolerance(const double tolerance) {
        this->conjugate_gradient_tolerance = tolerance;
    }

    void factor_graph::set_conjugate_gradient_iteration_limit(const int limit) {
        this->conjugate_gradient_iteration_limit = limit;
    }

    vertex* factor_graph::add_vertex(vertex&& node) {
        if (!node.is_valid()) {
            return nullptr;
        }
        this->vertex_storage.push_back(static_cast<vertex&&>(node));
        vertex* const stored = &this->vertex_storage.back();
        this->vertex_set.insert(stored);
        if (stored->is_marginalised()) {
            this->vertices_marginalised.push_back(stored);
        }
        else {
            this->vertices_general.push_back(stored);
        }
        return stored;
    }

    bool factor_graph::remove_vertex(vertex* node) {
        if (this->vertex_set.count(node) == 0) {
            return false;
        }
        int found_index = -1;
        for (int i = 0; i < static_cast<int>(this->vertices_general.size()); ++i) {
            if (this->vertices_general[static_cast<size_t>(i)] == node) {
                found_index = i;
                break;
            }
        }
        if (found_index == -1) {
            for (int i = 0; i < static_cast<int>(this->vertices_marginalised.size()); ++i) {
                if (this->vertices_marginalised[static_cast<size_t>(i)] == node) {
                    found_index = i;
                    break;
                }
            }
        }
        if (found_index == -1) {
            return false;
        }
        std::vector<edge*> remove_edges = this->get_connected_edges(node);
        for (size_t i = 0; i < remove_edges.size(); i++) {
            this->remove_edge(remove_edges[i]);
        }
        if (node->is_marginalised()) {
            this->vertices_marginalised.erase(this->vertices_marginalised.begin() + found_index);
        }
        else {
            this->vertices_general.erase(this->vertices_general.begin() + found_index);
        }
        node->set_ordering_id(-1);
        this->vertex_to_edge.erase(node);
        this->vertex_set.erase(node);
        return true;
    }

    edge* factor_graph::add_edge(edge&& factor) {
        if (!factor.is_valid()) {
            return nullptr;
        }
        this->edge_storage.push_back(static_cast<edge&&>(factor));
        edge* const stored = &this->edge_storage.back();
        this->edge_set.insert(stored);
        this->edges.push_back(stored);
        for (vertex* node : stored->get_vertices()) {
            this->vertex_to_edge.insert({ node, stored });
        }
        return stored;
    }

    bool factor_graph::remove_edge(edge* factor) {
        if (this->edge_set.count(factor) == 0) {
            return false;
        }
        int found_index = -1;
        for (int i = 0; i < static_cast<int>(this->edges.size()); ++i) {
            if (this->edges[static_cast<size_t>(i)] == factor) {
                found_index = i;
                break;
            }
        }
        if (found_index == -1) {
            return false;
        }
        this->edges.erase(this->edges.begin() + found_index);
        this->edge_set.erase(factor);
        for (vertex* node : factor->get_vertices()) {
            const std::pair<typename std::unordered_multimap<vertex*, edge*>::iterator, typename std::unordered_multimap<vertex*, edge*>::iterator> range = this->vertex_to_edge.equal_range(node);
            for (typename std::unordered_multimap<vertex*, edge*>::iterator iterator = range.first; iterator != range.second;) {
                if (iterator->second == factor) {
                    iterator = this->vertex_to_edge.erase(iterator);
                }
                else {
                    ++iterator;
                }
            }
        }
        return true;
    }

    std::vector<edge*> factor_graph::get_connected_edges(vertex* node) const {
        std::vector<edge*> edges_connected;
        const std::pair<typename std::unordered_multimap<vertex*, edge*>::const_iterator, typename std::unordered_multimap<vertex*, edge*>::const_iterator> range = this->vertex_to_edge.equal_range(node);
        edges_connected.reserve(this->vertex_to_edge.count(node));
        for (typename std::unordered_multimap<vertex*, edge*>::const_iterator iterator = range.first; iterator != range.second; ++iterator) {
            edges_connected.push_back(iterator->second);
        }
        return edges_connected;
    }

    int factor_graph::solve(int iterations, bool use_relative_convergence) {
        if ((this->edges.size() == 0) || ((this->vertices_general.size() == 0) && (this->vertices_marginalised.size() == 0))) {
            core::logger::log(core::logger::level::warn, "Cannot solve problem without edges or vertices");
            return 0;
        }

        for (edge* factor : this->edges) {
            factor->compute_residual();
        }

        this->set_ordering();

        this->last_diagnostics = diagnostics();

        this->linearise();

        this->damping_factor = 2.0;
        this->chi_squared = this->get_current_chi();
        double max_diagonal = 0;
        for (size_t i = 0; i < this->hessian_diagonal.size(); ++i) {
            const double value = this->damping_diagonal.empty() ? this->hessian_diagonal[i] : (this->hessian_diagonal[i] / this->damping_diagonal[i]);
            max_diagonal = std::max(math::abs(value), max_diagonal);
        }
        max_diagonal = std::min(5e10, max_diagonal);
        const double tau = 1e-5;
        this->damping_lambda = tau * max_diagonal;

        core::logger::log(core::logger::level::debug, "[INIT] iter: XXX, attempt XXX, chi = % 7.7f, rho = XXXX.XXXXXXX, base lambda = % 8.7f", this->chi_squared, this->damping_lambda);

        const int max_failures = 10;
        const double legacy_stop_threshold = 1e-10 * this->chi_squared;
        const double relative_tolerance = 1e-6;
        const double chi_squared_floor = 1e-12;
        int success_count = 0;
        for (int iter = 0; iter < iterations; ++iter) {
            double last_chi_squared = this->chi_squared;
            int failure_count = 0;
            while (failure_count < max_failures) {
                if (!this->compute_step()) {
                    this->damping_lambda *= this->damping_factor;
                    this->damping_factor *= 2.0;
                    ++failure_count;
                    ++this->last_diagnostics.rejected_attempts;
                    if (!math::isfinite(this->damping_lambda)) {
                        break;
                    }
                    continue;
                }

                this->update_states();

                double scale = 0;
                for (int i = 0; i < (this->count_general_params + this->count_marginalised_params); ++i) {
                    scale += this->delta_x[static_cast<size_t>(i)][0] * (this->damping_lambda * this->damping_weight(i) * this->delta_x[static_cast<size_t>(i)][0] + this->vector_b[static_cast<size_t>(i)][0]);
                }
                scale += 1e-3;

                for (edge* factor : this->edges) {
                    factor->compute_residual();
                }
                double tempChi = this->get_current_chi();
                double rho = (this->chi_squared - tempChi) / scale;

                bool good_step = false;
                if ((rho > 0) && (math::isfinite(tempChi))) {
                    double alpha = 1.0 - math::pow((2.0 * rho - 1.0), 3.0);
                    alpha = math::min(alpha, 2.0 / 3.0);
                    double scaleFactor = math::max(1.0 / 3.0, alpha);
                    this->damping_lambda *= scaleFactor;
                    this->damping_factor = 2.0;
                    this->chi_squared = tempChi;
                    good_step = true;
                    ++success_count;
                    core::logger::log(core::logger::level::debug, "[GOOD] iter: % 3d, attempt % 3d, chi2 = % 7.7f, rho = % 7.7f, next lambda = % 8.7f", iter, failure_count, tempChi, rho, this->damping_lambda);
                }
                else {
                    this->damping_lambda *= this->damping_factor;
                    this->damping_factor *= 2.0;
                    good_step = false;
                    ++this->last_diagnostics.rejected_attempts;
                    core::logger::log(core::logger::level::debug, "[ BAD] iter: % 3d, attempt % 3d, chi2 = % 7.7f, rho = % 7.7f, next lambda = % 8.7f", iter, failure_count, tempChi, rho, this->damping_lambda);
                }
                if (good_step) {
                    this->linearise();
                    break;
                }
                else {
                    this->rollback_states();
                    ++failure_count;
                }
                if (!math::isfinite(this->damping_lambda)) {
                    break;
                }
            }
            const double stop_threshold = use_relative_convergence ? (relative_tolerance * std::max(last_chi_squared, chi_squared_floor)) : legacy_stop_threshold;
            if ((failure_count >= max_failures) || ((last_chi_squared - this->chi_squared) < stop_threshold) || (!math::isfinite(this->damping_lambda))) {
                break;
            }
        }

        return success_count;
    }

    double factor_graph::get_current_chi(bool recompute_residuals) {
        if (recompute_residuals) {
            for (edge* factor : this->edges) {
                factor->compute_residual();
            }
        }
        double current_chi = 0.0;
        for (const edge* factor : this->edges) {
            current_chi += factor->robust_chi2();
        }
        return current_chi;
    }

    bool factor_graph::compute_damped_step(double lambda, math::matrix<double, 0, 0>& step) {
        if ((this->edges.size() == 0) || ((this->vertices_general.size() == 0) && (this->vertices_marginalised.size() == 0))) {
            return false;
        }
        for (edge* factor : this->edges) {
            factor->compute_residual();
        }
        this->set_ordering();
        this->last_diagnostics = diagnostics();
        this->linearise();
        this->damping_lambda = lambda;
        const bool solved = this->compute_step();
        step = this->delta_x;
        return solved;
    }

    double factor_graph::damping_weight(int index) const {
        return this->damping_diagonal.empty() ? 1.0 : this->damping_diagonal[static_cast<size_t>(index)];
    }

    void factor_graph::update_scaling() {
        const size_t total = this->hessian_diagonal.size();
        this->column_scale.assign(total, 1.0);
        this->damping_diagonal.assign(total, 1.0);
        for (size_t i = 0; i < total; ++i) {
            const double norm = math::sqrt(math::max(this->hessian_diagonal[i], 0.0));
            this->column_scale[i] = 1.0 / (1.0 + norm);
            this->damping_diagonal[i] = (1.0 + norm) * (1.0 + norm);
        }
    }

    bool factor_graph::select_square_root_path() const {
        if (this->solve_strategy == strategy::dense_schur) {
            return false;
        }
        if (this->solve_strategy == strategy::automatic) {
            if ((this->count_marginalised_params == 0) || (this->count_general_params < this->square_root_parameter_threshold)) {
                return false;
            }
        }
        for (const vertex* node : this->vertices_general) {
            if (!node->is_fixed() && (node->get_local_dimensions() != 6)) {
                return false;
            }
        }
        for (const edge* factor : this->edges) {
            int marginalised_count = 0;
            int general_count = 0;
            int marginalised_dimensions = 0;
            for (const vertex* node : factor->get_vertices()) {
                if (node->is_marginalised()) {
                    ++marginalised_count;
                    marginalised_dimensions = node->get_local_dimensions();
                }
                else {
                    ++general_count;
                }
            }
            if (marginalised_count == 0) {
                continue;
            }
            if ((marginalised_count != 1) || (general_count > 1) || (marginalised_dimensions > factor_graph::maximum_landmark_dimensions) || (factor->get_residual().rows() != 2)) {
                return false;
            }
        }
        return true;
    }

    void factor_graph::linearise() {
        this->square_root_active = this->select_square_root_path();
        this->last_diagnostics.used_square_root = this->square_root_active;
        this->last_diagnostics.used_single_precision = this->square_root_active && (this->solve_precision == precision::single_precision);
        if (this->square_root_active) {
            if (this->solve_precision == precision::single_precision) {
                this->linearise_landmark_blocks(this->square_root_single);
            }
            else {
                this->linearise_landmark_blocks(this->square_root_double);
            }
            return;
        }
        this->make_hessian();
    }

    bool factor_graph::compute_step() {
        if (this->square_root_active) {
            if (this->solve_precision == precision::single_precision) {
                return this->solve_linear_system_qr(this->square_root_single);
            }
            return this->solve_linear_system_qr(this->square_root_double);
        }
        return this->solve_linear_system();
    }

    void factor_graph::set_ordering() {
        this->count_general_params = 0;
        this->count_marginalised_params = 0;
        for (const auto& node : this->vertices_general) {
            if (node->is_fixed()) {
                node->set_ordering_id(-1);
                continue;
            }
            node->set_ordering_id(this->count_general_params);
            this->count_general_params += node->get_local_dimensions();
        }
        for (const auto& node : this->vertices_marginalised) {
            if (node->is_fixed()) {
                node->set_ordering_id(-1);
                continue;
            }
            node->set_ordering_id(this->count_marginalised_params + this->count_general_params);
            this->count_marginalised_params += node->get_local_dimensions();
        }
    }

    void factor_graph::add_block(math::matrix<double, 0, 0>& target, size_t row, size_t col, const math::matrix<double, 0, 0>& block, double scale) {
        for (size_t i = 0; i < block.rows(); ++i) {
            for (size_t j = 0; j < block.cols(); ++j) {
                target[row + i][col + j] = target[row + i][col + j] + (scale * block[i][j]);
            }
        }
    }

    void factor_graph::accumulate_general_edge(edge* factor) {
        factor->compute_jacobians();
        const std::vector<vertex*>& edge_vertices = factor->get_vertices();
        const std::vector<math::matrix<double, 0, 0>>& jacobians = factor->get_jacobians();
        ASSERT(edge_vertices.size() == jacobians.size(), "Mismatching sizes between edge vertices and edge jacobians.");
        double drho;
        math::matrix<double, 0, 0> robust_information;
        factor->robust_info(drho, robust_information, false);
        const math::matrix<double, 0, 0> weighted_residual = drho * (factor->get_information() * factor->get_residual());
        for (size_t i = 0; i < edge_vertices.size(); ++i) {
            const vertex* v_i = edge_vertices[i];
            if (v_i->is_fixed()) {
                continue;
            }
            const int index_i = v_i->get_ordering_id();
            ASSERT(index_i != -1, "Missing ordering id value of edge vertex.");
            const math::matrix<double, 0, 0> jacobian_i_transposed = math::transpose(jacobians[i]);
            const math::matrix<double, 0, 0> jtw = jacobian_i_transposed * robust_information;
            factor_graph::add_block(this->vector_b, static_cast<size_t>(index_i), 0, jacobian_i_transposed * weighted_residual, -1.0);
            for (size_t j = i; j < edge_vertices.size(); ++j) {
                const vertex* v_j = edge_vertices[j];
                if (v_j->is_fixed()) {
                    continue;
                }
                const int index_j = v_j->get_ordering_id();
                ASSERT(index_j != -1, "Missing ordering id value of edge vertex.");
                const math::matrix<double, 0, 0> hessian_block = jtw * jacobians[j];
                factor_graph::add_block(this->h_pp, static_cast<size_t>(index_i), static_cast<size_t>(index_j), hessian_block, 1.0);
                if (i != j) {
                    factor_graph::add_block(this->h_pp, static_cast<size_t>(index_j), static_cast<size_t>(index_i), math::transpose(hessian_block), 1.0);
                }
            }
        }
    }

    template <typename scalar>
    void factor_graph::linearise_landmark_blocks(square_root_state<scalar>& state) {
        const size_t general_count = static_cast<size_t>(this->count_general_params);
        const size_t total_count = static_cast<size_t>(this->count_general_params + this->count_marginalised_params);
        this->h_pp = math::matrix<double, 0, 0>::zero(general_count, general_count);
        this->vector_b = math::matrix<double, 0, 0>::zero(total_count, 1);
        this->delta_x = math::matrix<double, 0, 0>::zero(total_count, 1);
        this->hessian_diagonal.assign(total_count, 0.0);

        std::unordered_map<const vertex*, size_t>& landmark_indices = state.landmark_indices;
        landmark_indices.clear();
        for (size_t i = 0; i < this->vertices_marginalised.size(); ++i) {
            landmark_indices[this->vertices_marginalised[i]] = i;
        }
        std::vector<std::vector<edge*>>& grouped_edges = state.grouped_edges;
        grouped_edges.resize(this->vertices_marginalised.size());
        for (std::vector<edge*>& group : grouped_edges) {
            group.clear();
        }
        for (edge* factor : this->edges) {
            const vertex* marginalised_vertex = nullptr;
            for (const vertex* node : factor->get_vertices()) {
                if (node->is_marginalised()) {
                    marginalised_vertex = node;
                }
            }
            if (marginalised_vertex == nullptr) {
                this->accumulate_general_edge(factor);
                continue;
            }
            const std::unordered_map<const vertex*, size_t>::const_iterator found = landmark_indices.find(marginalised_vertex);
            ASSERT(found != landmark_indices.end(), "Edge references a marginalised vertex that is not in the graph.");
            if (found == landmark_indices.end()) {
                continue;
            }
            grouped_edges[found->second].push_back(factor);
        }
        state.general_right_hand_side.assign(general_count, static_cast<scalar>(0));
        for (size_t i = 0; i < general_count; ++i) {
            state.general_right_hand_side[i] = static_cast<scalar>(this->vector_b[i][0]);
            this->hessian_diagonal[i] = this->h_pp[i][i];
        }

        state.block_vertices.clear();
        for (size_t index = 0; index < this->vertices_marginalised.size(); ++index) {
            const std::vector<edge*>& group = grouped_edges[index];
            if (group.empty()) {
                continue;
            }
            if (this->vertices_marginalised[index]->is_fixed()) {
                bool free_pose = false;
                for (const edge* factor : group) {
                    for (const vertex* node : factor->get_vertices()) {
                        free_pose = free_pose || (!node->is_marginalised() && !node->is_fixed());
                    }
                }
                if (!free_pose) {
                    continue;
                }
            }
            state.block_vertices.push_back(index);
        }
        const size_t block_count = state.block_vertices.size();
        state.blocks.resize(block_count);

        core::thread_pool::instance().parallel_for(block_count, 16, [this, &state, &grouped_edges](const size_t block_index) {
            const std::vector<edge*>& group = grouped_edges[state.block_vertices[block_index]];
            const vertex* const landmark_vertex = this->vertices_marginalised[state.block_vertices[block_index]];
            const bool landmark_is_free = !landmark_vertex->is_fixed();
            const int landmark_dimension = landmark_vertex->get_local_dimensions();
            std::vector<int> pose_offsets;
            for (const edge* factor : group) {
                for (const vertex* node : factor->get_vertices()) {
                    if (node->is_marginalised() || node->is_fixed()) {
                        continue;
                    }
                    pose_offsets.push_back(node->get_ordering_id());
                }
            }
            std::sort(pose_offsets.begin(), pose_offsets.end());
            pose_offsets.erase(std::unique(pose_offsets.begin(), pose_offsets.end()), pose_offsets.end());
            landmark_block<scalar>& block = state.blocks[block_index];
            block.configure(static_cast<int>(group.size()), landmark_dimension, landmark_is_free, landmark_is_free ? landmark_vertex->get_ordering_id() : 0, pose_offsets.data(), static_cast<int>(pose_offsets.size()));
            for (size_t slot = 0; slot < group.size(); ++slot) {
                edge* const factor = group[slot];
                factor->compute_jacobians();
                const std::vector<vertex*>& edge_vertices = factor->get_vertices();
                const std::vector<math::matrix<double, 0, 0>>& jacobians = factor->get_jacobians();
                ASSERT(edge_vertices.size() == jacobians.size(), "Mismatching sizes between edge vertices and edge jacobians.");
                const math::matrix<double, 0, 0>* jacobian_pose = nullptr;
                const math::matrix<double, 0, 0>* jacobian_landmark = nullptr;
                int pose_slot = -1;
                for (size_t i = 0; i < edge_vertices.size(); ++i) {
                    if (edge_vertices[i]->is_marginalised()) {
                        jacobian_landmark = &jacobians[i];
                    }
                    else if (!edge_vertices[i]->is_fixed()) {
                        jacobian_pose = &jacobians[i];
                        pose_slot = static_cast<int>(std::lower_bound(pose_offsets.begin(), pose_offsets.end(), edge_vertices[i]->get_ordering_id()) - pose_offsets.begin());
                    }
                }
                double drho = 1.0;
                math::matrix<double, 0, 0> robust_information;
                factor->robust_info(drho, robust_information, false);
                scalar whitening[4] = { 0, 0, 0, 0 };
                if ((robust_information[0][1] == 0.0) && (robust_information[1][0] == 0.0) && (robust_information[0][0] == robust_information[1][1])) {
                    const double root = math::sqrt(math::max(robust_information[0][0], 0.0));
                    whitening[0] = static_cast<scalar>(root);
                    whitening[3] = static_cast<scalar>(root);
                }
                else {
                    math::matrix<double, 2, 2> root;
                    if (!math::sqrt_symmetric_2x2(math::matrix<double, 2, 2>(robust_information.data()), root)) {
                        root = math::matrix<double, 2, 2>::zero();
                    }
                    for (size_t i = 0; i < 4; ++i) {
                        whitening[i] = static_cast<scalar>(root.data()[i]);
                    }
                }
                scalar pose_values[12] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                scalar landmark_values[2 * factor_graph::maximum_landmark_dimensions] = {};
                scalar residual_values[2] = { 0, 0 };
                if (jacobian_pose != nullptr) {
                    for (size_t i = 0; i < 12; ++i) {
                        pose_values[i] = static_cast<scalar>(jacobian_pose->data()[i]);
                    }
                }
                if ((jacobian_landmark != nullptr) && landmark_is_free) {
                    for (size_t i = 0; i < static_cast<size_t>(2 * landmark_dimension); ++i) {
                        landmark_values[i] = static_cast<scalar>(jacobian_landmark->data()[i]);
                    }
                }
                residual_values[0] = static_cast<scalar>(factor->get_residual()[0][0]);
                residual_values[1] = static_cast<scalar>(factor->get_residual()[1][0]);
                block.set_observation(static_cast<int>(slot), (jacobian_pose != nullptr) ? pose_slot : -1, pose_values, landmark_values, residual_values, whitening);
            }
        });

        const size_t general_groups = general_count / 6;
        state.pose_block_begin.assign(general_groups + 1, 0);
        for (const landmark_block<scalar>& block : state.blocks) {
            for (int slot = 0; slot < block.pose_count(); ++slot) {
                ++state.pose_block_begin[static_cast<size_t>(block.get_pose_offset(slot) / 6) + 1];
            }
        }
        for (size_t group = 0; group < general_groups; ++group) {
            state.pose_block_begin[group + 1] += state.pose_block_begin[group];
        }
        state.pose_block_index.assign(static_cast<size_t>(state.pose_block_begin[general_groups]), 0);
        state.pose_block_slot.assign(static_cast<size_t>(state.pose_block_begin[general_groups]), 0);
        {
            std::vector<int> fill(state.pose_block_begin.begin(), state.pose_block_begin.end() - 1);
            for (size_t block_index = 0; block_index < block_count; ++block_index) {
                const landmark_block<scalar>& block = state.blocks[block_index];
                for (int slot = 0; slot < block.pose_count(); ++slot) {
                    const size_t group = static_cast<size_t>(block.get_pose_offset(slot) / 6);
                    state.pose_block_index[static_cast<size_t>(fill[group])] = static_cast<int>(block_index);
                    state.pose_block_slot[static_cast<size_t>(fill[group])] = slot;
                    ++fill[group];
                }
            }
        }

        if (total_count > 0) {
            core::thread_pool::instance().parallel_for(general_groups, 1, [this, &state](const size_t group) {
                double* const gradient = this->vector_b.data() + (6 * group);
                double* const diagonal = this->hessian_diagonal.data() + (6 * group);
                for (int k = state.pose_block_begin[group]; k < state.pose_block_begin[group + 1]; ++k) {
                    const landmark_block<scalar>& block = state.blocks[static_cast<size_t>(state.pose_block_index[static_cast<size_t>(k)])];
                    block.accumulate_gradient_slot(state.pose_block_slot[static_cast<size_t>(k)], gradient);
                    block.accumulate_hessian_diagonal_slot(state.pose_block_slot[static_cast<size_t>(k)], diagonal);
                }
            });
            core::thread_pool::instance().parallel_for(block_count, 16, [this, &state](const size_t block_index) {
                const landmark_block<scalar>& block = state.blocks[block_index];
                if (!block.is_marginalised()) {
                    return;
                }
                block.accumulate_gradient_landmark(this->vector_b.data() + block.get_landmark_offset());
                block.accumulate_hessian_diagonal_landmark(this->hessian_diagonal.data() + block.get_landmark_offset());
            });
        }
        this->update_scaling();
        core::thread_pool::instance().parallel_for(block_count, 16, [this, &state](const size_t block_index) {
            landmark_block<scalar>& block = state.blocks[block_index];
            block.scale_columns(this->column_scale.data());
            block.perform_qr();
        });
        this->last_diagnostics.landmark_blocks = static_cast<int>(block_count);
    }

    template <typename scalar>
    bool factor_graph::solve_linear_system_qr(square_root_state<scalar>& state) {
        const int general_params = this->count_general_params;
        const int general_groups = general_params / 6;
        const size_t total_count = static_cast<size_t>(this->count_general_params + this->count_marginalised_params);
        const scalar lambda = static_cast<scalar>(this->damping_lambda);
        const bool general_edges = !state.general_right_hand_side.empty() && (this->h_pp.rows() > 0);
        core::thread_pool& pool = core::thread_pool::instance();

        pool.parallel_for(state.blocks.size(), 16, [&state, lambda](const size_t block_index) {
            state.blocks[block_index].set_damping(lambda);
        });

        state.preconditioner.assign(static_cast<size_t>(general_groups) * 36u, static_cast<scalar>(0));
        state.right_hand_side.assign(static_cast<size_t>(general_params), static_cast<scalar>(0));
        pool.parallel_for(static_cast<size_t>(general_groups), 1, [this, &state, lambda, general_edges](const size_t group) {
            scalar* const entries = state.preconditioner.data() + (group * 36u);
            scalar* const right_hand_side = state.right_hand_side.data() + (6 * group);
            for (int k = state.pose_block_begin[group]; k < state.pose_block_begin[group + 1]; ++k) {
                const landmark_block<scalar>& block = state.blocks[static_cast<size_t>(state.pose_block_index[static_cast<size_t>(k)])];
                block.add_reduced_diagonal_slot(state.pose_block_slot[static_cast<size_t>(k)], entries);
            }
            if (general_edges) {
                for (int i = 0; i < 6; ++i) {
                    for (int j = 0; j < 6; ++j) {
                        const size_t row = (6 * group) + static_cast<size_t>(i);
                        const size_t col = (6 * group) + static_cast<size_t>(j);
                        entries[(6 * i) + j] += static_cast<scalar>(this->column_scale[row] * this->h_pp[row][col] * this->column_scale[col]);
                    }
                }
            }
            math::matrix<scalar, 6, 6> damped(entries);
            for (size_t i = 0; i < 6; ++i) {
                damped[i][i] += lambda;
            }
            math::matrix<scalar, 6, 6> inverse;
            if (math::invert(damped, inverse)) {
                for (size_t i = 0; i < 36; ++i) {
                    entries[i] = inverse.data()[i];
                }
            }
            else {
                for (size_t i = 0; i < 36; ++i) {
                    entries[i] = ((i % 7u) == 0u) ? static_cast<scalar>(1) : static_cast<scalar>(0);
                }
            }
            for (int k = state.pose_block_begin[group]; k < state.pose_block_begin[group + 1]; ++k) {
                const landmark_block<scalar>& block = state.blocks[static_cast<size_t>(state.pose_block_index[static_cast<size_t>(k)])];
                block.add_right_hand_side_slot(state.pose_block_slot[static_cast<size_t>(k)], right_hand_side);
            }
            if (general_edges) {
                for (size_t i = 0; i < 6; ++i) {
                    right_hand_side[i] += static_cast<scalar>(this->column_scale[(6 * group) + i]) * state.general_right_hand_side[(6 * group) + i];
                }
            }
        });

        state.increment.assign(static_cast<size_t>(general_params), static_cast<scalar>(0));
        state.scratch.assign(static_cast<size_t>(4 * general_params), static_cast<scalar>(0));
        const auto apply_operator = [this, &state, &pool, general_groups, general_params, lambda, general_edges](const scalar* input, scalar* output) {
            pool.parallel_for(state.blocks.size(), 16, [&state, input](const size_t block_index) {
                state.blocks[block_index].compute_operator_image(input);
            });
            pool.parallel_for(static_cast<size_t>(general_groups), 1, [this, &state, input, output, general_params, lambda, general_edges](const size_t group) {
                scalar* const target = output + (6 * group);
                for (size_t i = 0; i < 6; ++i) {
                    target[i] = lambda * input[(6 * group) + i];
                }
                for (int k = state.pose_block_begin[group]; k < state.pose_block_begin[group + 1]; ++k) {
                    const landmark_block<scalar>& block = state.blocks[static_cast<size_t>(state.pose_block_index[static_cast<size_t>(k)])];
                    if (block.reduced_rows() > 0) {
                        block.add_operator_slot(state.pose_block_slot[static_cast<size_t>(k)], target);
                    }
                }
                if (general_edges) {
                    for (size_t i = 0; i < 6; ++i) {
                        const size_t row = (6 * group) + i;
                        double sum = 0.0;
                        for (int j = 0; j < general_params; ++j) {
                            sum += this->h_pp[row][static_cast<size_t>(j)] * this->column_scale[static_cast<size_t>(j)] * static_cast<double>(input[j]);
                        }
                        target[i] += static_cast<scalar>(this->column_scale[row] * sum);
                    }
                }
            });
        };
        const auto apply_preconditioner = [&state, general_groups](const scalar* input, scalar* output) {
            for (int group = 0; group < general_groups; ++group) {
                const scalar* const entries = state.preconditioner.data() + (static_cast<size_t>(group) * 36u);
                for (int i = 0; i < 6; ++i) {
                    scalar sum = 0;
                    for (int j = 0; j < 6; ++j) {
                        sum += entries[(6 * i) + j] * input[(6 * group) + j];
                    }
                    output[(6 * group) + i] = sum;
                }
            }
        };
        const double tolerance_floor = (sizeof(scalar) < sizeof(double)) ? 1e-5 : 0.0;
        const scalar tolerance = static_cast<scalar>(math::max(this->conjugate_gradient_tolerance, tolerance_floor));
        const int iteration_limit = math::max(1, (this->conjugate_gradient_iteration_limit > 0) ? this->conjugate_gradient_iteration_limit : math::min(500, 2 * general_params));
        const math::conjugate_gradient_result result = math::conjugate_gradient(apply_operator, apply_preconditioner, state.right_hand_side.data(), state.increment.data(), general_params, iteration_limit, tolerance, state.scratch.data());
        ++this->last_diagnostics.reduced_solves;
        this->last_diagnostics.reduced_iterations += result.iterations;
        if (!result.converged) {
            ++this->last_diagnostics.reduced_failures;
        }

        this->delta_x = math::matrix<double, 0, 0>::zero(total_count, 1);
        for (int i = 0; i < general_params; ++i) {
            const double value = static_cast<double>(state.increment[static_cast<size_t>(i)]);
            if (!math::isfinite(value)) {
                core::logger::log(core::logger::level::warn, "Square root solver produced a non-finite pose step!");
                this->delta_x = math::matrix<double, 0, 0>::zero(total_count, 1);
                return false;
            }
            this->delta_x[static_cast<size_t>(i)][0] = this->column_scale[static_cast<size_t>(i)] * value;
        }
        for (landmark_block<scalar>& block : state.blocks) {
            if (!block.is_marginalised()) {
                continue;
            }
            scalar landmark_delta[factor_graph::maximum_landmark_dimensions] = {};
            if (!block.back_substitute(state.increment.data(), landmark_delta)) {
                core::logger::log(core::logger::level::warn, "Square root solver failed to back substitute a landmark!");
                this->delta_x = math::matrix<double, 0, 0>::zero(total_count, 1);
                return false;
            }
            const size_t offset = static_cast<size_t>(block.get_landmark_offset());
            for (size_t i = 0; i < static_cast<size_t>(block.get_landmark_dimension()); ++i) {
                const double value = static_cast<double>(landmark_delta[i]);
                if (!math::isfinite(value)) {
                    this->delta_x = math::matrix<double, 0, 0>::zero(total_count, 1);
                    return false;
                }
                this->delta_x[offset + i][0] = this->column_scale[offset + i] * value;
            }
        }
        return true;
    }

    void factor_graph::make_hessian() {
        const size_t general_count = static_cast<size_t>(this->count_general_params);
        this->h_pp = math::matrix<double, 0, 0>::zero(general_count, general_count);
        this->h_ll.clear();
        this->landmark_block_of_parameter.assign(static_cast<size_t>(this->count_marginalised_params), 0);
        for (const vertex* node : this->vertices_marginalised) {
            if (node->is_fixed()) {
                continue;
            }
            ASSERT(node->get_local_dimensions() <= factor_graph::maximum_landmark_dimensions, "The marginalised vertex has more local dimensions than the landmark blocks hold.");
            landmark_diagonal diagonal = {};
            diagonal.offset = node->get_ordering_id();
            diagonal.dimensions = node->get_local_dimensions();
            for (int i = 0; i < diagonal.dimensions; ++i) {
                this->landmark_block_of_parameter[static_cast<size_t>(diagonal.offset - this->count_general_params + i)] = this->h_ll.size();
            }
            this->h_ll.push_back(diagonal);
        }
        this->couplings.clear();
        this->landmark_couplings.clear();
        this->landmark_couplings.resize(this->h_ll.size());
        this->vector_b = math::matrix<double, 0, 0>::zero(static_cast<size_t>(this->count_general_params + this->count_marginalised_params), 1);
        this->delta_x = math::matrix<double, 0, 0>::zero(static_cast<size_t>(this->count_general_params + this->count_marginalised_params), 1);

        for (const auto& factor : this->edges) {
            bool touches_marginalised = false;
            for (const vertex* node : factor->get_vertices()) {
                touches_marginalised = touches_marginalised || node->is_marginalised();
            }
            if (!touches_marginalised) {
                this->accumulate_general_edge(factor);
                continue;
            }
            factor->compute_jacobians();
            const std::vector<vertex*>& edge_vertices = factor->get_vertices();
            const std::vector<math::matrix<double, 0, 0>>& jacobians = factor->get_jacobians();
            ASSERT(edge_vertices.size() == jacobians.size(), "Mismatching sizes between edge vertices and edge jacobians.");
            const size_t residuals = factor->get_residual().rows();
            ASSERT(residuals <= factor_graph::maximum_residuals, "The edge has more residuals than the linearisation scratch holds.");
            const double drho = factor->robust_weight();
            const math::matrix<double, 0, 0>& information = factor->get_information();
            const math::matrix<double, 0, 0>& residual = factor->get_residual();
            double robust_information[factor_graph::maximum_residuals][factor_graph::maximum_residuals];
            double weighted_residual[factor_graph::maximum_residuals];
            for (size_t r = 0; r < residuals; ++r) {
                for (size_t c = 0; c < residuals; ++c) {
                    robust_information[r][c] = drho * information[r][c];
                }
                double sum = 0;
                for (size_t c = 0; c < residuals; ++c) {
                    sum += information[r][c] * residual[c][0];
                }
                weighted_residual[r] = drho * sum;
            }
            for (size_t i = 0; i < edge_vertices.size(); ++i) {
                const vertex* v_i = edge_vertices[i];
                if (v_i->is_fixed()) {
                    continue;
                }
                const int index_i = v_i->get_ordering_id();
                ASSERT(index_i != -1, "Missing ordering id value of edge vertex.");
                const math::matrix<double, 0, 0>& jacobian_i = jacobians[i];
                const size_t dimensions_i = jacobian_i.cols();
                ASSERT(dimensions_i <= vertex::maximum_parameters, "The vertex has more local dimensions than the linearisation scratch holds.");
                double jtw[vertex::maximum_parameters][factor_graph::maximum_residuals];
                for (size_t a = 0; a < dimensions_i; ++a) {
                    for (size_t c = 0; c < residuals; ++c) {
                        double sum = 0;
                        for (size_t r = 0; r < residuals; ++r) {
                            sum += jacobian_i[r][a] * robust_information[r][c];
                        }
                        jtw[a][c] = sum;
                    }
                    double sum = 0;
                    for (size_t r = 0; r < residuals; ++r) {
                        sum += jacobian_i[r][a] * weighted_residual[r];
                    }
                    this->vector_b[static_cast<size_t>(index_i) + a][0] = this->vector_b[static_cast<size_t>(index_i) + a][0] + (-1.0 * sum);
                }
                for (size_t j = i; j < edge_vertices.size(); ++j) {
                    const vertex* v_j = edge_vertices[j];
                    if (v_j->is_fixed()) {
                        continue;
                    }
                    const int index_j = v_j->get_ordering_id();
                    ASSERT(index_j != -1, "Missing ordering id value of edge vertex.");
                    const math::matrix<double, 0, 0>& jacobian_j = jacobians[j];
                    const size_t dimensions_j = jacobian_j.cols();
                    ASSERT(dimensions_j <= vertex::maximum_parameters, "The vertex has more local dimensions than the linearisation scratch holds.");
                    double hessian_block[vertex::maximum_parameters][vertex::maximum_parameters];
                    for (size_t a = 0; a < dimensions_i; ++a) {
                        for (size_t b = 0; b < dimensions_j; ++b) {
                            double sum = 0;
                            for (size_t c = 0; c < residuals; ++c) {
                                sum += jtw[a][c] * jacobian_j[c][b];
                            }
                            hessian_block[a][b] = sum;
                        }
                    }
                    if (!v_i->is_marginalised() && !v_j->is_marginalised()) {
                        for (size_t a = 0; a < dimensions_i; ++a) {
                            for (size_t b = 0; b < dimensions_j; ++b) {
                                this->h_pp[static_cast<size_t>(index_i) + a][static_cast<size_t>(index_j) + b] = this->h_pp[static_cast<size_t>(index_i) + a][static_cast<size_t>(index_j) + b] + (1.0 * hessian_block[a][b]);
                            }
                        }
                        if (i != j) {
                            for (size_t a = 0; a < dimensions_i; ++a) {
                                for (size_t b = 0; b < dimensions_j; ++b) {
                                    this->h_pp[static_cast<size_t>(index_j) + b][static_cast<size_t>(index_i) + a] = this->h_pp[static_cast<size_t>(index_j) + b][static_cast<size_t>(index_i) + a] + (1.0 * hessian_block[a][b]);
                                }
                            }
                        }
                    }
                    else if (!v_i->is_marginalised()) {
                        const size_t landmark_block = this->landmark_block_of_parameter[static_cast<size_t>(index_j - this->count_general_params)];
                        this->landmark_couplings[landmark_block].push_back(this->couplings.size());
                        coupling created;
                        created.general_index = index_i;
                        created.general_dimensions = v_i->get_local_dimensions();
                        created.landmark_block = landmark_block;
                        for (size_t a = 0; a < dimensions_i; ++a) {
                            for (size_t b = 0; b < dimensions_j; ++b) {
                                created.block[a][b] = hessian_block[a][b];
                            }
                        }
                        this->couplings.push_back(created);
                    }
                    else if (!v_j->is_marginalised()) {
                        const size_t landmark_block = this->landmark_block_of_parameter[static_cast<size_t>(index_i - this->count_general_params)];
                        this->landmark_couplings[landmark_block].push_back(this->couplings.size());
                        coupling created;
                        created.general_index = index_j;
                        created.general_dimensions = v_j->get_local_dimensions();
                        created.landmark_block = landmark_block;
                        for (size_t a = 0; a < dimensions_i; ++a) {
                            for (size_t b = 0; b < dimensions_j; ++b) {
                                created.block[b][a] = hessian_block[a][b];
                            }
                        }
                        this->couplings.push_back(created);
                    }
                    else {
                        ASSERT(i == j, "An edge may only touch one marginalised vertex.");
                        landmark_diagonal& diagonal = this->h_ll[this->landmark_block_of_parameter[static_cast<size_t>(index_i - this->count_general_params)]];
                        for (size_t a = 0; a < dimensions_i; ++a) {
                            for (size_t b = 0; b < dimensions_i; ++b) {
                                diagonal.block[a][b] = diagonal.block[a][b] + hessian_block[a][b];
                            }
                        }
                    }
                }
            }
        }
        this->b_pp = (this->count_general_params > 0) ? math::get_block(this->vector_b, 0, 0, general_count, 1) : math::matrix<double, 0, 0>(0, 1);
        this->b_ll = (this->count_marginalised_params > 0) ? math::get_block(this->vector_b, general_count, 0, static_cast<size_t>(this->count_marginalised_params), 1) : math::matrix<double, 0, 0>(0, 1);
        this->hessian_diagonal.assign(static_cast<size_t>(this->count_general_params + this->count_marginalised_params), 0.0);
        for (size_t i = 0; i < general_count; ++i) {
            this->hessian_diagonal[i] = this->h_pp[i][i];
        }
        for (const landmark_diagonal& diagonal : this->h_ll) {
            for (int i = 0; i < diagonal.dimensions; ++i) {
                this->hessian_diagonal[static_cast<size_t>(diagonal.offset + i)] = diagonal.block[i][i];
            }
        }
        this->update_scaling();
    }

    bool factor_graph::invert_landmark_block(const double (&damped)[maximum_landmark_dimensions][maximum_landmark_dimensions], const int dimensions, double (&inverse)[maximum_landmark_dimensions][maximum_landmark_dimensions]) {
        if (dimensions == 3) {
            math::matrix<double, 3, 3> block;
            for (size_t a = 0; a < 3; ++a) {
                for (size_t b = 0; b < 3; ++b) {
                    block[a][b] = damped[a][b];
                }
            }
            math::matrix<double, 3, 3> block_inverse;
            if (math::invert(block, block_inverse)) {
                for (size_t a = 0; a < 3; ++a) {
                    for (size_t b = 0; b < 3; ++b) {
                        inverse[a][b] = block_inverse[a][b];
                    }
                }
                return true;
            }
        }
        double packed[maximum_landmark_dimensions * maximum_landmark_dimensions] = {};
        double lower[maximum_landmark_dimensions * maximum_landmark_dimensions] = {};
        for (int a = 0; a < dimensions; ++a) {
            for (int b = 0; b < dimensions; ++b) {
                packed[(a * dimensions) + b] = damped[a][b];
            }
        }
        if (!math::decompose_cholesky(packed, dimensions, dimensions, lower)) {
            return false;
        }
        for (int column = 0; column < dimensions; ++column) {
            double unit[maximum_landmark_dimensions] = {};
            double solution[maximum_landmark_dimensions] = {};
            unit[column] = 1.0;
            if (!math::solve_cholesky(lower, unit, dimensions, dimensions, solution)) {
                return false;
            }
            for (int row = 0; row < dimensions; ++row) {
                inverse[row][column] = solution[row];
            }
        }
        return true;
    }

    bool factor_graph::solve_linear_system() {
        const size_t general_count = static_cast<size_t>(this->count_general_params);

        math::matrix<double, 0, 0> h_pp_schur = this->h_pp;
        for (size_t i = 0; i < general_count; ++i) {
            h_pp_schur[i][i] += this->damping_lambda * this->damping_weight(static_cast<int>(i));
        }
        math::matrix<double, 0, 0> b_pp_schur = this->b_pp;
        for (size_t landmark_block = 0; landmark_block < this->h_ll.size(); ++landmark_block) {
            landmark_diagonal& diagonal = this->h_ll[landmark_block];
            const size_t dimensions = static_cast<size_t>(diagonal.dimensions);
            const size_t offset = static_cast<size_t>(diagonal.offset);
            double damped_block[maximum_landmark_dimensions][maximum_landmark_dimensions];
            for (size_t i = 0; i < dimensions; ++i) {
                for (size_t j = 0; j < dimensions; ++j) {
                    damped_block[i][j] = diagonal.block[i][j];
                }
                damped_block[i][i] += this->damping_lambda * this->damping_weight(diagonal.offset + static_cast<int>(i));
            }
            if (!factor_graph::invert_landmark_block(damped_block, diagonal.dimensions, diagonal.inverse)) {
                core::logger::log(core::logger::level::warn, "Landmark block inversion failed!");
                this->delta_x = math::matrix<double, 0, 0>::zero(static_cast<size_t>(this->count_general_params + this->count_marginalised_params), 1);
                return false;
            }
            const double (&inverse_block)[maximum_landmark_dimensions][maximum_landmark_dimensions] = diagonal.inverse;
            double b_landmark[maximum_landmark_dimensions];
            for (size_t i = 0; i < dimensions; ++i) {
                b_landmark[i] = this->b_ll[offset - general_count + i][0];
            }
            for (const size_t e : this->landmark_couplings[landmark_block]) {
                const coupling& coupling_e = this->couplings[e];
                const size_t rows_e = static_cast<size_t>(coupling_e.general_dimensions);
                const size_t index_e = static_cast<size_t>(coupling_e.general_index);
                double block_e_inverse[vertex::maximum_parameters][maximum_landmark_dimensions];
                for (size_t i = 0; i < rows_e; ++i) {
                    for (size_t k = 0; k < dimensions; ++k) {
                        double sum = 0;
                        for (size_t m = 0; m < dimensions; ++m) {
                            sum += coupling_e.block[i][m] * inverse_block[m][k];
                        }
                        block_e_inverse[i][k] = sum;
                    }
                    double sum = 0;
                    for (size_t k = 0; k < dimensions; ++k) {
                        sum += block_e_inverse[i][k] * b_landmark[k];
                    }
                    b_pp_schur[index_e + i][0] = b_pp_schur[index_e + i][0] + (-1.0 * sum);
                }
                for (const size_t f : this->landmark_couplings[landmark_block]) {
                    const coupling& coupling_f = this->couplings[f];
                    const size_t rows_f = static_cast<size_t>(coupling_f.general_dimensions);
                    const size_t index_f = static_cast<size_t>(coupling_f.general_index);
                    for (size_t i = 0; i < rows_e; ++i) {
                        for (size_t j = 0; j < rows_f; ++j) {
                            double sum = 0;
                            for (size_t k = 0; k < dimensions; ++k) {
                                sum += block_e_inverse[i][k] * coupling_f.block[j][k];
                            }
                            h_pp_schur[index_e + i][index_f + j] = h_pp_schur[index_e + i][index_f + j] + (-1.0 * sum);
                        }
                    }
                }
            }
        }

        math::matrix<double, 0, 0> matrix_lower(general_count, general_count);
        math::matrix<double, 0, 0> delta_xpp = math::matrix<double, 0, 0>::zero(general_count, 1);
        if (!math::decompose_cholesky(h_pp_schur.data(), this->count_general_params, this->count_general_params, matrix_lower.data()) || !math::solve_cholesky(matrix_lower.data(), b_pp_schur.data(), this->count_general_params, this->count_general_params, delta_xpp.data())) {
            core::logger::log(core::logger::level::warn, "Cholesky solver failed!");
            this->delta_x = math::matrix<double, 0, 0>::zero(static_cast<size_t>(this->count_general_params + this->count_marginalised_params), 1);
            return false;
        }

        if (this->count_general_params > 0) {
            math::set_block(this->delta_x, 0, 0, delta_xpp);
        }

        for (size_t landmark_block = 0; landmark_block < this->h_ll.size(); ++landmark_block) {
            const landmark_diagonal& diagonal = this->h_ll[landmark_block];
            const size_t dimensions = static_cast<size_t>(diagonal.dimensions);
            const size_t offset = static_cast<size_t>(diagonal.offset);
            double b_landmark[maximum_landmark_dimensions];
            for (size_t i = 0; i < dimensions; ++i) {
                b_landmark[i] = this->b_ll[offset - general_count + i][0];
            }
            for (const size_t e : this->landmark_couplings[landmark_block]) {
                const coupling& coupling_e = this->couplings[e];
                const size_t rows_e = static_cast<size_t>(coupling_e.general_dimensions);
                const size_t index_e = static_cast<size_t>(coupling_e.general_index);
                for (size_t m = 0; m < dimensions; ++m) {
                    double sum = 0;
                    for (size_t k = 0; k < rows_e; ++k) {
                        sum += coupling_e.block[k][m] * delta_xpp[index_e + k][0];
                    }
                    b_landmark[m] = b_landmark[m] + (-1.0 * sum);
                }
            }
            for (size_t i = 0; i < dimensions; ++i) {
                double sum = 0;
                for (size_t k = 0; k < dimensions; ++k) {
                    sum += diagonal.inverse[i][k] * b_landmark[k];
                }
                this->delta_x[offset + i][0] = sum;
            }
        }
        return true;
    }

    void factor_graph::update_states() {
        for (auto node : this->vertices_general) {
            if (node->is_fixed()) {
                continue;
            }
            node->backup();
            const size_t index = static_cast<size_t>(node->get_ordering_id());
            const size_t dimensions = static_cast<size_t>(node->get_local_dimensions());
            double delta[vertex::maximum_parameters] = {};
            for (size_t i = 0; i < dimensions; ++i) {
                delta[i] = this->delta_x[index + i][0];
            }
            node->plus(&delta[0]);
        }
        for (auto node : this->vertices_marginalised) {
            if (node->is_fixed()) {
                continue;
            }
            node->backup();
            const size_t index = static_cast<size_t>(node->get_ordering_id());
            const size_t dimensions = static_cast<size_t>(node->get_local_dimensions());
            double delta[vertex::maximum_parameters] = {};
            for (size_t i = 0; i < dimensions; ++i) {
                delta[i] = this->delta_x[index + i][0];
            }
            node->plus(&delta[0]);
        }
    }

    void factor_graph::rollback_states() {
        for (auto node : this->vertices_general) {
            if (node->is_fixed()) {
                continue;
            }
            node->restore();
        }
        for (auto node : this->vertices_marginalised) {
            if (node->is_fixed()) {
                continue;
            }
            node->restore();
        }
    }
}
