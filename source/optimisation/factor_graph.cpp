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
    namespace {
        // Fixed size copies of the block products for the common shapes, which the compiler can unroll; each sum keeps the
        // order of the general loops, so the results are the same.
        template <size_t rows, size_t cols, size_t residuals>
        void edge_block_fixed(const double* const first_data, const double* const second_data, const double* const weight, double* const product) {
            double weighted[rows * residuals];
            for (size_t a = 0; a < rows; ++a) {
                for (size_t c = 0; c < residuals; ++c) {
                    double sum = 0;
                    for (size_t r = 0; r < residuals; ++r) {
                        sum += first_data[(r * rows) + a] * weight[(r * residuals) + c];
                    }
                    weighted[(a * residuals) + c] = sum;
                }
            }
            for (size_t a = 0; a < rows; ++a) {
                for (size_t b = 0; b < cols; ++b) {
                    double sum = 0;
                    for (size_t c = 0; c < residuals; ++c) {
                        sum += weighted[(a * residuals) + c] * second_data[(c * cols) + b];
                    }
                    product[(a * cols) + b] = sum;
                }
            }
        }

        template <size_t rows, size_t cols, size_t inner>
        void subtract_schur_term_fixed(const double* const product, const double* const values, double* const target) {
            for (size_t a = 0; a < rows; ++a) {
                for (size_t b = 0; b < cols; ++b) {
                    double sum = 0;
                    for (size_t m = 0; m < inner; ++m) {
                        sum += product[(a * inner) + m] * values[(b * inner) + m];
                    }
                    target[(a * cols) + b] = target[(a * cols) + b] + (-1.0 * sum);
                }
            }
        }

        void subtract_schur_term(const double* const product, const double* const values, const size_t rows, const size_t cols, const size_t inner, double* const target) {
            if ((rows == 6) && (cols == 6) && (inner == 3)) {
                subtract_schur_term_fixed<6, 6, 3>(product, values, target);
                return;
            }
            for (size_t a = 0; a < rows; ++a) {
                for (size_t b = 0; b < cols; ++b) {
                    double sum = 0;
                    for (size_t m = 0; m < inner; ++m) {
                        sum += product[(a * inner) + m] * values[(b * inner) + m];
                    }
                    target[(a * cols) + b] = target[(a * cols) + b] + (-1.0 * sum);
                }
            }
        }

        template <size_t rows, size_t dimensions, size_t stride>
        void form_coupling_products_fixed(const double* const values, const double (&inverse)[stride][stride], const double* const landmark_gradient, double* const product, double* const gradient) {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t k = 0; k < dimensions; ++k) {
                    double sum = 0;
                    for (size_t m = 0; m < dimensions; ++m) {
                        sum += values[(i * dimensions) + m] * inverse[m][k];
                    }
                    product[(i * dimensions) + k] = sum;
                }
                double sum = 0;
                for (size_t k = 0; k < dimensions; ++k) {
                    sum += product[(i * dimensions) + k] * landmark_gradient[k];
                }
                gradient[i] = sum;
            }
        }

        template <size_t stride>
        void form_coupling_products(const double* const values, const double (&inverse)[stride][stride], const double* const landmark_gradient, const size_t rows, const size_t dimensions, double* const product, double* const gradient) {
            if ((rows == 6) && (dimensions == 3)) {
                form_coupling_products_fixed<6, 3, stride>(values, inverse, landmark_gradient, product, gradient);
                return;
            }
            for (size_t i = 0; i < rows; ++i) {
                for (size_t k = 0; k < dimensions; ++k) {
                    double sum = 0;
                    for (size_t m = 0; m < dimensions; ++m) {
                        sum += values[(i * dimensions) + m] * inverse[m][k];
                    }
                    product[(i * dimensions) + k] = sum;
                }
                double sum = 0;
                for (size_t k = 0; k < dimensions; ++k) {
                    sum += product[(i * dimensions) + k] * landmark_gradient[k];
                }
                gradient[i] = sum;
            }
        }
    }

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
        return stored;
    }

    bool factor_graph::remove_vertex(vertex* node) {
        if (this->vertex_set.count(node) == 0) {
            return false;
        }
        for (edge* const factor : this->get_connected_edges(node)) {
            this->remove_edge(factor);
        }
        // The general and marginalised lists are rebuilt from the vertex set by every solve.
        node->set_ordering_id(-1);
        this->vertex_to_edge.erase(node);
        this->vertex_set.erase(node);
        return true;
    }

    edge* factor_graph::add_edge(edge&& factor) {
        if (!factor.is_valid()) {
            return nullptr;
        }
        if (factor.num_vertices() != factor.required_vertices()) {
            core::logger::log(core::logger::level::warn, "Cannot add a '%s' edge holding %zu vertices, it takes %zu.", factor.name(), factor.num_vertices(), factor.required_vertices());
            return nullptr;
        }
        // A vertex of another graph has no ordering in this one, so its rows would alias the first vertex's.
        for (vertex* const node : factor.get_vertices()) {
            if (this->vertex_set.count(node) == 0) {
                core::logger::log(core::logger::level::warn, "Cannot add a '%s' edge holding a vertex that is not in the graph.", factor.name());
                return nullptr;
            }
        }
        this->edge_storage.push_back(static_cast<edge&&>(factor));
        edge* const stored = &this->edge_storage.back();
        this->edge_positions.insert({ stored, this->edges.size() });
        this->edges.push_back(stored);
        for (vertex* const node : stored->get_vertices()) {
            this->vertex_to_edge[node].edges.push_back(stored);
        }
        return stored;
    }

    bool factor_graph::remove_edge(edge* factor) {
        // The edge leaves a gap in the edge list, closed in order before the list is next used, and its vertices' lists
        // drop it once half of their entries are gone, so removing an edge costs a constant amortised time.
        const std::unordered_map<edge*, size_t>::iterator found = this->edge_positions.find(factor);
        if (found == this->edge_positions.end()) {
            return false;
        }
        this->edges[found->second] = nullptr;
        this->edge_positions.erase(found);
        ++this->removed_edge_count;
        for (vertex* const node : factor->get_vertices()) {
            const std::unordered_map<vertex*, adjacency>::iterator connected = this->vertex_to_edge.find(node);
            if (connected == this->vertex_to_edge.end()) {
                continue;
            }
            adjacency& list = connected->second;
            ++list.removed;
            if ((2 * list.removed) > list.edges.size()) {
                size_t kept = 0;
                for (edge* const other : list.edges) {
                    if (this->edge_positions.count(other) != 0) {
                        list.edges[kept] = other;
                        ++kept;
                    }
                }
                list.edges.resize(kept);
                list.removed = 0;
            }
        }
        return true;
    }

    std::vector<edge*> factor_graph::get_connected_edges(vertex* node) const {
        const std::unordered_map<vertex*, adjacency>::const_iterator found = this->vertex_to_edge.find(node);
        if (found == this->vertex_to_edge.end()) {
            return {};
        }
        const adjacency& list = found->second;
        if (list.removed == 0) {
            return list.edges;
        }
        std::vector<edge*> edges_connected;
        edges_connected.reserve(list.edges.size());
        for (edge* const factor : list.edges) {
            if (this->edge_positions.count(factor) != 0) {
                edges_connected.push_back(factor);
            }
        }
        return edges_connected;
    }

    void factor_graph::compact_edges() {
        if (this->removed_edge_count == 0) {
            return;
        }
        size_t kept = 0;
        for (size_t index = 0; index < this->edges.size(); ++index) {
            edge* const factor = this->edges[index];
            if (factor == nullptr) {
                continue;
            }
            if (kept != index) {
                this->edges[kept] = factor;
                this->edge_positions[factor] = kept;
            }
            ++kept;
        }
        this->edges.resize(kept);
        this->removed_edge_count = 0;
    }

    int factor_graph::solve(int iterations, bool use_relative_convergence) {
        this->compact_edges();
        if (this->edges.empty() || this->vertex_set.empty()) {
            core::logger::log(core::logger::level::warn, "Cannot solve problem without edges or vertices");
            return 0;
        }

        this->compute_residuals();

        this->set_ordering();
        if (!this->linearisation_limits_hold()) {
            return 0;
        }

        this->last_diagnostics = diagnostics();

        this->analyse();
        this->linearise();

        this->damping_factor = 2.0;
        // The residuals were computed at the start of the solve, before the linearisation.
        this->chi_squared = this->get_current_chi(false);
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
        bool residuals_current = true;
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

                double predicted_reduction = 0;
                for (int i = 0; i < (this->count_general_params + this->count_marginalised_params); ++i) {
                    predicted_reduction += this->delta_x[static_cast<size_t>(i)][0] * (this->damping_lambda * this->damping_weight(i) * this->delta_x[static_cast<size_t>(i)][0] + this->vector_b[static_cast<size_t>(i)][0]);
                }

                double tempChi = this->get_current_chi(true);
                double rho = factor_graph::gain_ratio(this->chi_squared, tempChi, predicted_reduction);

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
                    residuals_current = true;
                    break;
                }
                else {
                    this->rollback_states();
                    residuals_current = false;
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

        // A rejected last attempt rolled the vertices back but left its residuals, and the robust weights they give, in the edges.
        if (!residuals_current) {
            this->compute_residuals();
        }

        return success_count;
    }

    double factor_graph::gain_ratio(const double current_chi, const double evaluated_chi, const double predicted_reduction) {
        // A step the model does not predict to reduce the cost is rejected, and the epsilon, relative to the cost, only guards
        // the ratio of a problem whose cost is tiny.
        if (!(predicted_reduction > 0.0)) {
            return 0.0;
        }
        const double epsilon = math::isfinite(current_chi) ? (factor_graph::gain_ratio_epsilon * math::abs(current_chi)) : 0.0;
        return (current_chi - evaluated_chi) / (predicted_reduction + epsilon);
    }

    void factor_graph::compute_residuals() {
        core::thread_pool::instance().parallel_for(this->edges.size(), factor_graph::residual_grain, [this](const size_t index) {
            this->edges[index]->compute_residual();
        });
    }

    double factor_graph::get_current_chi(bool recompute_residuals) {
        this->compact_edges();
        // Each edge's cost in parallel, and their sum in edge order as a single pass over the edges gives it.
        const size_t count = this->edges.size();
        this->edge_costs.resize(count);
        core::thread_pool::instance().parallel_for(count, factor_graph::residual_grain, [this, recompute_residuals](const size_t index) {
            edge* const factor = this->edges[index];
            if (recompute_residuals) {
                factor->compute_residual();
            }
            this->edge_costs[index] = factor->robust_chi2();
        });
        double current_chi = 0.0;
        for (const double cost : this->edge_costs) {
            current_chi += cost;
        }
        return current_chi;
    }

    bool factor_graph::compute_damped_step(double lambda, math::matrix<double, 0, 0>& step) {
        this->compact_edges();
        if (this->edges.empty() || this->vertex_set.empty()) {
            return false;
        }
        this->compute_residuals();
        this->set_ordering();
        if (!this->linearisation_limits_hold()) {
            return false;
        }
        this->last_diagnostics = diagnostics();
        this->analyse();
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
        this->linearise_edges();
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

    bool factor_graph::linearisation_limits_hold() const {
        for (const vertex* node : this->vertices_marginalised) {
            if (!node->is_fixed() && (node->get_local_dimensions() > factor_graph::maximum_landmark_dimensions)) {
                core::logger::log(core::logger::level::error, "Cannot solve: a marginalised vertex has %d local dimensions, more than the %d the landmark blocks hold.", node->get_local_dimensions(), factor_graph::maximum_landmark_dimensions);
                return false;
            }
        }
        for (const edge* factor : this->edges) {
            if (factor->get_residual().rows() > factor_graph::maximum_residuals) {
                core::logger::log(core::logger::level::error, "Cannot solve: an edge has %zu residuals, more than the %zu the linearisation holds.", factor->get_residual().rows(), factor_graph::maximum_residuals);
                return false;
            }
            size_t marginalised_count = 0;
            for (const vertex* node : factor->get_vertices()) {
                marginalised_count += node->is_marginalised() ? 1u : 0u;
            }
            if (marginalised_count > 1) {
                core::logger::log(core::logger::level::error, "Cannot solve: an edge touches %zu marginalised vertices, at most one is supported.", marginalised_count);
                return false;
            }
        }
        return true;
    }

    void factor_graph::partition_vertices() {
        this->vertices_general.clear();
        this->vertices_marginalised.clear();
        for (vertex& node : this->vertex_storage) {
            if (this->vertex_set.count(&node) == 0) {
                continue;
            }
            if (node.is_marginalised()) {
                this->vertices_marginalised.push_back(&node);
            }
            else {
                this->vertices_general.push_back(&node);
            }
        }
    }

    void factor_graph::set_ordering() {
        this->partition_vertices();
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

    void factor_graph::gradient_term(const math::matrix<double, 0, 0>& jacobian, const double* weighted_residual, const size_t residuals, double* term) {
        for (size_t a = 0; a < jacobian.cols(); ++a) {
            double sum = 0;
            for (size_t r = 0; r < residuals; ++r) {
                sum += jacobian[r][a] * weighted_residual[r];
            }
            term[a] = sum;
        }
    }

    void factor_graph::edge_block(const math::matrix<double, 0, 0>& first, const math::matrix<double, 0, 0>& second, const double* weight, const size_t residuals, double* product) {
        // (J_first^T W) J_second, summed in the order the dense accumulation always used.
        const size_t rows = first.cols();
        const size_t cols = second.cols();
        const double* const first_data = first.data();
        const double* const second_data = second.data();
        if (residuals == 2) {
            if ((rows == 6) && (cols == 6)) {
                edge_block_fixed<6, 6, 2>(first_data, second_data, weight, product);
                return;
            }
            if ((rows == 6) && (cols == 3)) {
                edge_block_fixed<6, 3, 2>(first_data, second_data, weight, product);
                return;
            }
            if ((rows == 3) && (cols == 6)) {
                edge_block_fixed<3, 6, 2>(first_data, second_data, weight, product);
                return;
            }
            if ((rows == 3) && (cols == 3)) {
                edge_block_fixed<3, 3, 2>(first_data, second_data, weight, product);
                return;
            }
        }
        double weighted[vertex::maximum_parameters * factor_graph::maximum_residuals];
        for (size_t a = 0; a < rows; ++a) {
            for (size_t c = 0; c < residuals; ++c) {
                double sum = 0;
                for (size_t r = 0; r < residuals; ++r) {
                    sum += first_data[(r * rows) + a] * weight[(r * residuals) + c];
                }
                weighted[(a * residuals) + c] = sum;
            }
        }
        for (size_t a = 0; a < rows; ++a) {
            for (size_t b = 0; b < cols; ++b) {
                double sum = 0;
                for (size_t c = 0; c < residuals; ++c) {
                    sum += weighted[(a * residuals) + c] * second_data[(c * cols) + b];
                }
                product[(a * cols) + b] = sum;
            }
        }
    }

    int factor_graph::general_block_of(const vertex* node) const {
        if (node->is_fixed() || node->is_marginalised()) {
            return -1;
        }
        return this->block_of_offset[static_cast<size_t>(node->get_ordering_id())];
    }

    int factor_graph::landmark_block_of(const vertex* node) const {
        if (node->is_fixed() || !node->is_marginalised()) {
            return -1;
        }
        return this->block_of_offset[static_cast<size_t>(node->get_ordering_id())];
    }

    void factor_graph::analyse() {
        this->square_root_active = this->select_square_root_path();
        this->last_diagnostics.used_square_root = this->square_root_active;
        this->last_diagnostics.used_single_precision = this->square_root_active && (this->solve_precision == precision::single_precision);
        this->numeric_jacobians = std::any_of(this->edges.begin(), this->edges.end(), [](const edge* factor) {
            return !factor->has_analytic_jacobians();
        });

        // Each edge's robust information and weighted residual, filled at every linearisation.
        const size_t edge_count = this->edges.size();
        this->edge_weights_begin.assign(edge_count + 1, 0);
        for (size_t index = 0; index < edge_count; ++index) {
            const size_t residuals = this->edges[index]->get_residual().rows();
            this->edge_weights_begin[index + 1] = this->edge_weights_begin[index] + (residuals * residuals) + residuals;
        }
        this->edge_weights.assign(this->edge_weights_begin[edge_count], 0.0);

        // The free vertices as blocks, found from their ordering ids.
        this->block_of_offset.assign(static_cast<size_t>(this->count_general_params + this->count_marginalised_params), -1);
        this->general_blocks.clear();
        size_t hessian_cursor = 0;
        for (const vertex* node : this->vertices_general) {
            if (node->is_fixed()) {
                continue;
            }
            this->block_of_offset[static_cast<size_t>(node->get_ordering_id())] = static_cast<int>(this->general_blocks.size());
            free_block entry;
            entry.offset = node->get_ordering_id();
            entry.dimensions = node->get_local_dimensions();
            entry.hessian = hessian_cursor;
            hessian_cursor += static_cast<size_t>(entry.dimensions * entry.dimensions);
            this->general_blocks.push_back(entry);
        }
        int landmark_count = 0;
        for (const vertex* node : this->vertices_marginalised) {
            if (node->is_fixed()) {
                continue;
            }
            this->block_of_offset[static_cast<size_t>(node->get_ordering_id())] = landmark_count;
            ++landmark_count;
        }

        // The edge slots of each block in edge order, so every sum over the edges of a block keeps the order of the edges.
        const size_t general_count = this->general_blocks.size();
        const size_t landmarks = static_cast<size_t>(landmark_count);
        this->general_incidence_begin.assign(general_count + 1, 0);
        this->landmark_incidence_begin.assign(landmarks + 1, 0);
        // The square root path puts every edge of a marginalised vertex in a landmark block, so it sums the others only.
        const auto summed = [this](const size_t index) {
            if (!this->square_root_active) {
                return true;
            }
            for (const vertex* node : this->edges[index]->get_vertices()) {
                if (node->is_marginalised()) {
                    return false;
                }
            }
            return true;
        };
        for (size_t index = 0; index < edge_count; ++index) {
            const std::vector<vertex*>& edge_vertices = this->edges[index]->get_vertices();
            const bool included = summed(index);
            for (size_t slot = 0; slot < edge_vertices.size(); ++slot) {
                const int general = included ? this->general_block_of(edge_vertices[slot]) : -1;
                const int landmark = this->landmark_block_of(edge_vertices[slot]);
                if (general >= 0) {
                    ++this->general_incidence_begin[static_cast<size_t>(general) + 1];
                }
                if (landmark >= 0) {
                    ++this->landmark_incidence_begin[static_cast<size_t>(landmark) + 1];
                }
            }
        }
        for (size_t general = 0; general < general_count; ++general) {
            this->general_incidence_begin[general + 1] += this->general_incidence_begin[general];
        }
        for (size_t landmark = 0; landmark < landmarks; ++landmark) {
            this->landmark_incidence_begin[landmark + 1] += this->landmark_incidence_begin[landmark];
        }
        this->general_incidence.assign(this->general_incidence_begin[general_count], incidence());
        this->landmark_incidence.assign(this->landmark_incidence_begin[landmarks], incidence());
        {
            std::vector<size_t> general_fill(this->general_incidence_begin.begin(), this->general_incidence_begin.end() - 1);
            std::vector<size_t> landmark_fill(this->landmark_incidence_begin.begin(), this->landmark_incidence_begin.end() - 1);
            for (size_t index = 0; index < edge_count; ++index) {
                const std::vector<vertex*>& edge_vertices = this->edges[index]->get_vertices();
                const bool included = summed(index);
                for (size_t slot = 0; slot < edge_vertices.size(); ++slot) {
                    const int general = included ? this->general_block_of(edge_vertices[slot]) : -1;
                    const int landmark = this->landmark_block_of(edge_vertices[slot]);
                    if (general >= 0) {
                        this->general_incidence[general_fill[static_cast<size_t>(general)]++] = { static_cast<int>(index), static_cast<int>(slot) };
                    }
                    if (landmark >= 0) {
                        this->landmark_incidence[landmark_fill[static_cast<size_t>(landmark)]++] = { static_cast<int>(index), static_cast<int>(slot) };
                    }
                }
            }
        }

        // Where each edge slot writes its terms when its edge is linearised: a general incidence its gradient and one Hessian
        // term per slot of the same block, a landmark incidence its gradient and diagonal block.
        this->edge_slot_begin.assign(edge_count + 1, 0);
        for (size_t index = 0; index < edge_count; ++index) {
            this->edge_slot_begin[index + 1] = this->edge_slot_begin[index] + this->edges[index]->get_vertices().size();
        }
        const size_t slot_count = this->edge_slot_begin[edge_count];
        this->slot_general_block.assign(slot_count, -1);
        this->slot_general_term.assign(slot_count, -1);
        this->slot_landmark_term.assign(slot_count, -1);
        this->slot_coupling.assign(slot_count, -1);
        for (size_t index = 0; index < edge_count; ++index) {
            const std::vector<vertex*>& edge_vertices = this->edges[index]->get_vertices();
            for (size_t slot = 0; slot < edge_vertices.size(); ++slot) {
                this->slot_general_block[this->edge_slot_begin[index] + slot] = this->general_block_of(edge_vertices[slot]);
            }
        }
        this->general_term_begin.assign(this->general_incidence.size() + 1, 0);
        for (size_t general = 0; general < general_count; ++general) {
            const size_t dimensions = static_cast<size_t>(this->general_blocks[general].dimensions);
            for (size_t k = this->general_incidence_begin[general]; k < this->general_incidence_begin[general + 1]; ++k) {
                const incidence& link = this->general_incidence[k];
                const size_t first_slot = this->edge_slot_begin[static_cast<size_t>(link.edge)];
                const size_t slots = this->edge_slot_begin[static_cast<size_t>(link.edge) + 1] - first_slot;
                size_t terms = 0;
                for (size_t other = 0; other < slots; ++other) {
                    terms += (this->slot_general_block[first_slot + other] == static_cast<int>(general)) ? 1u : 0u;
                }
                this->slot_general_term[first_slot + static_cast<size_t>(link.slot)] = static_cast<int>(k);
                this->general_term_begin[k + 1] = this->general_term_begin[k] + dimensions + (terms * dimensions * dimensions);
            }
        }
        this->general_terms.assign(this->general_term_begin[this->general_incidence.size()], 0.0);
        // The square root path builds its landmark blocks from the Jacobians, so only the Schur path writes landmark terms.
        this->landmark_term_begin.assign(this->landmark_incidence.size() + 1, 0);
        for (size_t k = 0; k < this->landmark_incidence.size(); ++k) {
            const incidence& link = this->landmark_incidence[k];
            const size_t dimensions = this->square_root_active ? 0u : static_cast<size_t>(this->edges[static_cast<size_t>(link.edge)]->get_vertices()[static_cast<size_t>(link.slot)]->get_local_dimensions());
            if (!this->square_root_active) {
                this->slot_landmark_term[this->edge_slot_begin[static_cast<size_t>(link.edge)] + static_cast<size_t>(link.slot)] = static_cast<int>(k);
            }
            this->landmark_term_begin[k + 1] = this->landmark_term_begin[k] + dimensions + (dimensions * dimensions);
        }
        this->landmark_terms.assign(this->landmark_term_begin[this->landmark_incidence.size()], 0.0);

        if (!this->square_root_active) {
            this->analyse_schur();
        }
        else {
            this->analyse_pairs();
        }
    }

    void factor_graph::analyse_schur() {
        const size_t general_count = this->general_blocks.size();
        const size_t landmark_count = this->landmark_incidence_begin.size() - 1;

        this->analyse_pairs();

        // The couplings, per landmark in edge order and per edge in slot order, and the couplings of each general block.
        this->couplings.clear();
        this->landmark_coupling_begin.assign(landmark_count + 1, 0);
        size_t values_cursor = 0;
        size_t gradient_cursor = 0;
        for (size_t landmark = 0; landmark < landmark_count; ++landmark) {
            for (size_t k = this->landmark_incidence_begin[landmark]; k < this->landmark_incidence_begin[landmark + 1]; ++k) {
                const incidence& link = this->landmark_incidence[k];
                const std::vector<vertex*>& edge_vertices = this->edges[static_cast<size_t>(link.edge)]->get_vertices();
                const size_t landmark_dimensions = static_cast<size_t>(edge_vertices[static_cast<size_t>(link.slot)]->get_local_dimensions());
                for (size_t slot = 0; slot < edge_vertices.size(); ++slot) {
                    const int general = this->general_block_of(edge_vertices[slot]);
                    if (general < 0) {
                        continue;
                    }
                    coupling created;
                    created.general = general;
                    created.landmark = static_cast<int>(landmark);
                    created.edge = link.edge;
                    created.general_slot = static_cast<int>(slot);
                    created.landmark_slot = link.slot;
                    created.values = values_cursor;
                    created.gradient = gradient_cursor;
                    values_cursor += static_cast<size_t>(this->general_blocks[static_cast<size_t>(general)].dimensions) * landmark_dimensions;
                    gradient_cursor += static_cast<size_t>(this->general_blocks[static_cast<size_t>(general)].dimensions);
                    this->slot_coupling[this->edge_slot_begin[static_cast<size_t>(link.edge)] + slot] = static_cast<int>(this->couplings.size());
                    this->couplings.push_back(created);
                }
            }
            this->landmark_coupling_begin[landmark + 1] = this->couplings.size();
        }
        this->coupling_values.assign(values_cursor, 0.0);
        this->coupling_products.assign(values_cursor, 0.0);
        this->coupling_gradients.assign(gradient_cursor, 0.0);
        this->landmark_failures.assign(landmark_count, 0);
        this->general_coupling_begin.assign(general_count + 1, 0);
        for (const coupling& link : this->couplings) {
            ++this->general_coupling_begin[static_cast<size_t>(link.general) + 1];
        }
        for (size_t general = 0; general < general_count; ++general) {
            this->general_coupling_begin[general + 1] += this->general_coupling_begin[general];
        }
        this->general_couplings.assign(this->couplings.size(), 0);
        {
            std::vector<size_t> fill(this->general_coupling_begin.begin(), this->general_coupling_begin.end() - 1);
            for (size_t index = 0; index < this->couplings.size(); ++index) {
                this->general_couplings[fill[static_cast<size_t>(this->couplings[index].general)]++] = static_cast<int>(index);
            }
        }

        // The reduced camera system couples the blocks of each pair and every two blocks that see the same landmark.
        std::vector<int> dimensions(general_count, 0);
        std::vector<std::vector<int>> neighbours(general_count);
        for (size_t general = 0; general < general_count; ++general) {
            dimensions[general] = this->general_blocks[general].dimensions;
        }
        for (const pair_block& pair : this->pair_blocks) {
            neighbours[static_cast<size_t>(pair.lower)].push_back(pair.upper);
        }
        std::vector<int> clique;
        for (size_t landmark = 0; landmark < landmark_count; ++landmark) {
            clique.clear();
            for (size_t index = this->landmark_coupling_begin[landmark]; index < this->landmark_coupling_begin[landmark + 1]; ++index) {
                clique.push_back(this->couplings[index].general);
            }
            std::sort(clique.begin(), clique.end());
            clique.erase(std::unique(clique.begin(), clique.end()), clique.end());
            for (size_t i = 0; i < clique.size(); ++i) {
                for (size_t j = i + 1; j < clique.size(); ++j) {
                    neighbours[static_cast<size_t>(clique[i])].push_back(clique[j]);
                }
            }
        }
        this->reduced.analyse(dimensions, neighbours);

        // Where each pair block goes in the factor, and the pairs of each factor column.
        const size_t columns = general_count;
        this->column_pair_begin.assign(columns + 1, 0);
        std::vector<int> pair_columns(this->pair_blocks.size(), 0);
        for (size_t index = 0; index < this->pair_blocks.size(); ++index) {
            pair_block& pair = this->pair_blocks[index];
            const int lower_position = this->reduced.position_of(pair.lower);
            const int upper_position = this->reduced.position_of(pair.upper);
            pair.transposed = lower_position < upper_position;
            const int row = pair.transposed ? upper_position : lower_position;
            const int column = pair.transposed ? lower_position : upper_position;
            const bool found = this->reduced.find_offset(row, column, pair.target);
            ASSERT(found, "A pair block must be in the pattern of the factor.");
            static_cast<void>(found);
            pair_columns[index] = column;
            ++this->column_pair_begin[static_cast<size_t>(column) + 1];
        }
        for (size_t column = 0; column < columns; ++column) {
            this->column_pair_begin[column + 1] += this->column_pair_begin[column];
        }
        this->column_pairs.assign(this->pair_blocks.size(), 0);
        {
            std::vector<size_t> fill(this->column_pair_begin.begin(), this->column_pair_begin.end() - 1);
            for (size_t index = 0; index < this->pair_blocks.size(); ++index) {
                this->column_pairs[fill[static_cast<size_t>(pair_columns[index])]++] = static_cast<int>(index);
            }
        }

        // The terms of each factor column: for each coupling of its block, every coupling of the same landmark at or below it.
        this->schur_term_begin.assign(columns + 1, 0);
        this->schur_terms.clear();
        for (size_t position = 0; position < columns; ++position) {
            const size_t general = static_cast<size_t>(this->reduced.block_at(static_cast<int>(position)));
            for (size_t k = this->general_coupling_begin[general]; k < this->general_coupling_begin[general + 1]; ++k) {
                const int column_coupling = this->general_couplings[k];
                const size_t landmark = static_cast<size_t>(this->couplings[static_cast<size_t>(column_coupling)].landmark);
                for (size_t row_coupling = this->landmark_coupling_begin[landmark]; row_coupling < this->landmark_coupling_begin[landmark + 1]; ++row_coupling) {
                    const int row_position = this->reduced.position_of(this->couplings[row_coupling].general);
                    if (row_position < static_cast<int>(position)) {
                        continue;
                    }
                    schur_term term;
                    term.row = static_cast<int>(row_coupling);
                    term.column = column_coupling;
                    if (row_position == static_cast<int>(position)) {
                        term.target = this->reduced.diagonal_offset(static_cast<int>(position));
                    }
                    else {
                        const bool found = this->reduced.find_offset(row_position, static_cast<int>(position), term.target);
                        ASSERT(found, "Two blocks that see the same landmark must share a block of the factor.");
                        static_cast<void>(found);
                    }
                    this->schur_terms.push_back(term);
                }
            }
            this->schur_term_begin[position + 1] = this->schur_terms.size();
        }
        this->reduced_right_hand_side.assign(static_cast<size_t>(this->count_general_params), 0.0);
        this->reduced_solution.assign(static_cast<size_t>(this->count_general_params), 0.0);
    }

    void factor_graph::analyse_pairs() {
        const size_t edge_count = this->edges.size();

        // The pair blocks, every two different general blocks sharing an edge, with their terms in edge order.
        class candidate final {
        public:
            int lower = 0;
            int upper = 0;
            pair_term term;
        };

        std::vector<candidate> candidates;
        for (size_t index = 0; index < edge_count; ++index) {
            const std::vector<vertex*>& edge_vertices = this->edges[index]->get_vertices();
            bool marginalised = false;
            for (const vertex* node : edge_vertices) {
                marginalised = marginalised || node->is_marginalised();
            }
            if (this->square_root_active && marginalised) {
                continue;
            }
            for (size_t first = 0; first < edge_vertices.size(); ++first) {
                const int first_block = this->general_block_of(edge_vertices[first]);
                if (first_block < 0) {
                    continue;
                }
                for (size_t second = first + 1; second < edge_vertices.size(); ++second) {
                    const int second_block = this->general_block_of(edge_vertices[second]);
                    if ((second_block < 0) || (second_block == first_block)) {
                        continue;
                    }
                    candidate entry;
                    entry.lower = math::min(first_block, second_block);
                    entry.upper = math::max(first_block, second_block);
                    entry.term = { static_cast<int>(index), static_cast<int>(first), static_cast<int>(second), first_block > second_block };
                    candidates.push_back(entry);
                }
            }
        }
        std::stable_sort(candidates.begin(), candidates.end(), [](const candidate& lhs, const candidate& rhs) {
            return (lhs.lower != rhs.lower) ? (lhs.lower < rhs.lower) : (lhs.upper < rhs.upper);
        });
        size_t cursor = 0;
        for (const free_block& entry : this->general_blocks) {
            cursor = math::max(cursor, entry.hessian + static_cast<size_t>(entry.dimensions * entry.dimensions));
        }
        this->pair_blocks.clear();
        this->pair_terms.clear();
        this->pair_term_begin.assign(1, 0);
        for (size_t i = 0; i < candidates.size(); ++i) {
            if ((i == 0) || (candidates[i].lower != candidates[i - 1].lower) || (candidates[i].upper != candidates[i - 1].upper)) {
                if (i != 0) {
                    this->pair_term_begin.push_back(this->pair_terms.size());
                }
                pair_block pair;
                pair.lower = candidates[i].lower;
                pair.upper = candidates[i].upper;
                pair.values = cursor;
                cursor += static_cast<size_t>(this->general_blocks[static_cast<size_t>(pair.lower)].dimensions * this->general_blocks[static_cast<size_t>(pair.upper)].dimensions);
                this->pair_blocks.push_back(pair);
            }
            this->pair_terms.push_back(candidates[i].term);
        }
        if (!candidates.empty()) {
            this->pair_term_begin.push_back(this->pair_terms.size());
        }
        this->hessian_values.assign(cursor, 0.0);

        // The row of each general block in the square root operator, its own block and its pairs, by neighbouring block.
        const size_t general_count = this->general_blocks.size();
        std::vector<std::vector<neighbour>> rows(general_count);
        for (size_t general = 0; general < general_count; ++general) {
            if (this->general_incidence_begin[general + 1] > this->general_incidence_begin[general]) {
                rows[general].push_back({ static_cast<int>(general), this->general_blocks[general].hessian, false });
            }
        }
        for (const pair_block& pair : this->pair_blocks) {
            rows[static_cast<size_t>(pair.lower)].push_back({ pair.upper, pair.values, false });
            rows[static_cast<size_t>(pair.upper)].push_back({ pair.lower, pair.values, true });
        }
        this->operator_neighbour_begin.assign(general_count + 1, 0);
        this->operator_neighbours.clear();
        for (size_t general = 0; general < general_count; ++general) {
            std::sort(rows[general].begin(), rows[general].end(), [](const neighbour& lhs, const neighbour& rhs) {
                return lhs.block < rhs.block;
            });
            this->operator_neighbours.insert(this->operator_neighbours.end(), rows[general].begin(), rows[general].end());
            this->operator_neighbour_begin[general + 1] = this->operator_neighbours.size();
        }
    }

    void factor_graph::linearise_edges() {
        // Jacobians from numeric differences move shared vertices, so those edges are linearised one at a time.
        const size_t count = this->edges.size();
        core::thread_pool::instance().parallel_for(count, this->numeric_jacobians ? count : factor_graph::edge_grain, [this](const size_t index) {
            edge* const factor = this->edges[index];
            factor->compute_jacobians();
            const math::matrix<double, 0, 0>& information = factor->get_information();
            const math::matrix<double, 0, 0>& residual = factor->get_residual();
            const size_t residuals = residual.rows();
            const double drho = factor->robust_weight();
            double* const weights = this->edge_weights.data() + this->edge_weights_begin[index];
            for (size_t r = 0; r < residuals; ++r) {
                double sum = 0;
                for (size_t c = 0; c < residuals; ++c) {
                    weights[(r * residuals) + c] = drho * information[r][c];
                    sum += information[r][c] * residual[c][0];
                }
                weights[(residuals * residuals) + r] = drho * sum;
            }

            // The edge's terms of its blocks and its couplings, formed while its Jacobians are at hand.
            const double* const weighted_residual = weights + (residuals * residuals);
            const std::vector<math::matrix<double, 0, 0>>& jacobians = factor->get_jacobians();
            const size_t first_slot = this->edge_slot_begin[index];
            const size_t slots = this->edge_slot_begin[index + 1] - first_slot;
            ASSERT(jacobians.size() == slots, "Mismatching sizes between edge vertices and edge jacobians.");
            double product[vertex::maximum_parameters * vertex::maximum_parameters];
            for (size_t slot = 0; slot < slots; ++slot) {
                const math::matrix<double, 0, 0>& jacobian = jacobians[slot];
                const size_t size = jacobian.cols();
                // Only the Jacobian of a free vertex is used, and its terms are laid out by the vertex dimensions.
                ASSERT(factor->get_vertices()[slot]->is_fixed() || ((jacobian.rows() == residuals) && (size == static_cast<size_t>(factor->get_vertices()[slot]->get_local_dimensions()))), "An edge Jacobian must have a row per residual and a column per local dimension of its vertex.");
                const int general_term = this->slot_general_term[first_slot + slot];
                if (general_term >= 0) {
                    double* term = this->general_terms.data() + this->general_term_begin[static_cast<size_t>(general_term)];
                    factor_graph::gradient_term(jacobian, weighted_residual, residuals, term);
                    term += size;
                    for (size_t other = 0; other < slots; ++other) {
                        if (this->slot_general_block[first_slot + other] != this->slot_general_block[first_slot + slot]) {
                            continue;
                        }
                        if (other >= slot) {
                            factor_graph::edge_block(jacobian, jacobians[other], weights, residuals, term);
                        }
                        else {
                            factor_graph::edge_block(jacobians[other], jacobian, weights, residuals, &product[0]);
                            for (size_t a = 0; a < size; ++a) {
                                for (size_t b = 0; b < size; ++b) {
                                    term[(a * size) + b] = product[(b * size) + a];
                                }
                            }
                        }
                        term += size * size;
                    }
                }
                const int landmark_term = this->slot_landmark_term[first_slot + slot];
                if (landmark_term >= 0) {
                    double* const term = this->landmark_terms.data() + this->landmark_term_begin[static_cast<size_t>(landmark_term)];
                    factor_graph::gradient_term(jacobian, weighted_residual, residuals, term);
                    factor_graph::edge_block(jacobian, jacobian, weights, residuals, term + size);
                }
                const int coupling_index = this->slot_coupling[first_slot + slot];
                if (coupling_index >= 0) {
                    const coupling& link = this->couplings[static_cast<size_t>(coupling_index)];
                    const math::matrix<double, 0, 0>& landmark_jacobian = jacobians[static_cast<size_t>(link.landmark_slot)];
                    const size_t dimensions = landmark_jacobian.cols();
                    double* const values = this->coupling_values.data() + link.values;
                    if (link.general_slot < link.landmark_slot) {
                        factor_graph::edge_block(jacobian, landmark_jacobian, weights, residuals, values);
                    }
                    else {
                        factor_graph::edge_block(landmark_jacobian, jacobian, weights, residuals, &product[0]);
                        for (size_t a = 0; a < size; ++a) {
                            for (size_t b = 0; b < dimensions; ++b) {
                                values[(a * dimensions) + b] = product[(b * size) + a];
                            }
                        }
                    }
                }
            }
        });
    }

    template <typename scalar>
    void factor_graph::linearise_landmark_blocks(square_root_state<scalar>& state) {
        const size_t general_count = static_cast<size_t>(this->count_general_params);
        const size_t total_count = static_cast<size_t>(this->count_general_params + this->count_marginalised_params);
        this->vector_b = math::matrix<double, 0, 0>::zero(total_count, 1);
        this->delta_x = math::matrix<double, 0, 0>::zero(total_count, 1);
        this->hessian_diagonal.assign(total_count, 0.0);
        this->accumulate_general_blocks();
        this->accumulate_pair_blocks();

        std::unordered_map<const vertex*, size_t>& landmark_indices = state.landmark_indices;
        landmark_indices.clear();
        for (size_t i = 0; i < this->vertices_marginalised.size(); ++i) {
            landmark_indices[this->vertices_marginalised[i]] = i;
        }
        std::vector<std::vector<int>>& grouped_edges = state.grouped_edges;
        grouped_edges.resize(this->vertices_marginalised.size());
        for (std::vector<int>& group : grouped_edges) {
            group.clear();
        }
        for (size_t index = 0; index < this->edges.size(); ++index) {
            const edge* const factor = this->edges[index];
            const vertex* marginalised_vertex = nullptr;
            for (const vertex* node : factor->get_vertices()) {
                if (node->is_marginalised()) {
                    marginalised_vertex = node;
                }
            }
            if (marginalised_vertex == nullptr) {
                continue;
            }
            const std::unordered_map<const vertex*, size_t>::const_iterator found = landmark_indices.find(marginalised_vertex);
            ASSERT(found != landmark_indices.end(), "Edge references a marginalised vertex that is not in the graph.");
            if (found == landmark_indices.end()) {
                continue;
            }
            grouped_edges[found->second].push_back(static_cast<int>(index));
        }
        state.general_right_hand_side.assign(general_count, static_cast<scalar>(0));
        for (size_t i = 0; i < general_count; ++i) {
            state.general_right_hand_side[i] = static_cast<scalar>(this->vector_b[i][0]);
        }
        for (const free_block& entry : this->general_blocks) {
            for (size_t i = 0; i < 6; ++i) {
                this->hessian_diagonal[static_cast<size_t>(entry.offset) + i] = this->hessian_values[entry.hessian + (i * 6) + i];
            }
        }

        state.block_vertices.clear();
        for (size_t index = 0; index < this->vertices_marginalised.size(); ++index) {
            const std::vector<int>& group = grouped_edges[index];
            if (group.empty()) {
                continue;
            }
            if (this->vertices_marginalised[index]->is_fixed()) {
                bool free_pose = false;
                for (const int edge_index : group) {
                    for (const vertex* node : this->edges[static_cast<size_t>(edge_index)]->get_vertices()) {
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
            const std::vector<int>& group = grouped_edges[state.block_vertices[block_index]];
            const vertex* const landmark_vertex = this->vertices_marginalised[state.block_vertices[block_index]];
            const bool landmark_is_free = !landmark_vertex->is_fixed();
            const int landmark_dimension = landmark_vertex->get_local_dimensions();
            std::vector<int> pose_offsets;
            for (const int edge_index : group) {
                for (const vertex* node : this->edges[static_cast<size_t>(edge_index)]->get_vertices()) {
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
                const size_t edge_index = static_cast<size_t>(group[slot]);
                const edge* const factor = this->edges[edge_index];
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
                const double* const robust_information = this->edge_weights.data() + this->edge_weights_begin[edge_index];
                scalar whitening[4] = { 0, 0, 0, 0 };
                if ((robust_information[1] == 0.0) && (robust_information[2] == 0.0) && (robust_information[0] == robust_information[3])) {
                    const double root = math::sqrt(math::max(robust_information[0], 0.0));
                    whitening[0] = static_cast<scalar>(root);
                    whitening[3] = static_cast<scalar>(root);
                }
                else {
                    math::matrix<double, 2, 2> root;
                    if (!math::sqrt_symmetric_2x2(math::matrix<double, 2, 2>(robust_information), root)) {
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
        const bool general_edges = !state.general_right_hand_side.empty();
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
                const double* const hessian = this->hessian_values.data() + this->general_blocks[group].hessian;
                for (int i = 0; i < 6; ++i) {
                    for (int j = 0; j < 6; ++j) {
                        const size_t row = (6 * group) + static_cast<size_t>(i);
                        const size_t col = (6 * group) + static_cast<size_t>(j);
                        entries[(6 * i) + j] += static_cast<scalar>(this->column_scale[row] * hessian[(6 * i) + j] * this->column_scale[col]);
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
        const auto apply_operator = [this, &state, &pool, general_groups, lambda, general_edges](const scalar* input, scalar* output) {
            pool.parallel_for(state.blocks.size(), 16, [&state, input](const size_t block_index) {
                state.blocks[block_index].compute_operator_image(input);
            });
            pool.parallel_for(static_cast<size_t>(general_groups), 1, [this, &state, input, output, lambda, general_edges](const size_t group) {
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
                if (general_edges && (this->operator_neighbour_begin[group + 1] > this->operator_neighbour_begin[group])) {
                    for (size_t i = 0; i < 6; ++i) {
                        const size_t row = (6 * group) + i;
                        double sum = 0.0;
                        for (size_t k = this->operator_neighbour_begin[group]; k < this->operator_neighbour_begin[group + 1]; ++k) {
                            const neighbour& other = this->operator_neighbours[k];
                            const double* const values = this->hessian_values.data() + other.values;
                            for (size_t j = 0; j < 6; ++j) {
                                const size_t column = (6 * static_cast<size_t>(other.block)) + j;
                                sum += (other.transposed ? values[(j * 6) + i] : values[(i * 6) + j]) * this->column_scale[column] * static_cast<double>(input[column]);
                            }
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

    void factor_graph::accumulate_general_blocks() {
        // Each block sums the terms of its own edges in edge order, so the blocks are those of a serial pass over the edges.
        core::thread_pool& pool = core::thread_pool::instance();
        pool.parallel_for(this->general_blocks.size(), factor_graph::general_grain, [this](const size_t general) {
            const free_block& entry = this->general_blocks[general];
            const size_t offset = static_cast<size_t>(entry.offset);
            const size_t rows = static_cast<size_t>(entry.dimensions);
            const size_t size = rows * rows;
            double* const hessian = this->hessian_values.data() + entry.hessian;
            for (size_t i = 0; i < size; ++i) {
                hessian[i] = 0.0;
            }
            for (size_t k = this->general_incidence_begin[general]; k < this->general_incidence_begin[general + 1]; ++k) {
                const double* term = this->general_terms.data() + this->general_term_begin[k];
                const double* const end = this->general_terms.data() + this->general_term_begin[k + 1];
                for (size_t a = 0; a < rows; ++a) {
                    this->vector_b[offset + a][0] = this->vector_b[offset + a][0] + (-1.0 * term[a]);
                }
                for (term += rows; term < end; term += size) {
                    for (size_t i = 0; i < size; ++i) {
                        hessian[i] = hessian[i] + (1.0 * term[i]);
                    }
                }
            }
        });
    }

    void factor_graph::accumulate_pair_blocks() {
        core::thread_pool& pool = core::thread_pool::instance();
        pool.parallel_for(this->pair_blocks.size(), factor_graph::general_grain, [this](const size_t index) {
            const pair_block& pair = this->pair_blocks[index];
            const size_t rows = static_cast<size_t>(this->general_blocks[static_cast<size_t>(pair.lower)].dimensions);
            const size_t cols = static_cast<size_t>(this->general_blocks[static_cast<size_t>(pair.upper)].dimensions);
            double* const values = this->hessian_values.data() + pair.values;
            for (size_t i = 0; i < (rows * cols); ++i) {
                values[i] = 0.0;
            }
            double product[vertex::maximum_parameters * vertex::maximum_parameters];
            for (size_t k = this->pair_term_begin[index]; k < this->pair_term_begin[index + 1]; ++k) {
                const pair_term& term = this->pair_terms[k];
                const edge* const factor = this->edges[static_cast<size_t>(term.edge)];
                const std::vector<math::matrix<double, 0, 0>>& jacobians = factor->get_jacobians();
                const size_t residuals = factor->get_residual().rows();
                const double* const weight = this->edge_weights.data() + this->edge_weights_begin[static_cast<size_t>(term.edge)];
                factor_graph::edge_block(jacobians[static_cast<size_t>(term.first)], jacobians[static_cast<size_t>(term.second)], weight, residuals, &product[0]);
                for (size_t a = 0; a < rows; ++a) {
                    for (size_t b = 0; b < cols; ++b) {
                        values[(a * cols) + b] = values[(a * cols) + b] + (1.0 * (term.transposed ? product[(b * rows) + a] : product[(a * cols) + b]));
                    }
                }
            }
        });
    }

    void factor_graph::make_hessian() {
        const size_t total_count = static_cast<size_t>(this->count_general_params + this->count_marginalised_params);
        const size_t landmark_count = this->landmark_incidence_begin.size() - 1;
        this->vector_b = math::matrix<double, 0, 0>::zero(total_count, 1);
        this->delta_x = math::matrix<double, 0, 0>::zero(total_count, 1);
        this->h_ll.assign(landmark_count, landmark_diagonal());
        for (const vertex* node : this->vertices_marginalised) {
            const int landmark = this->landmark_block_of(node);
            if (landmark < 0) {
                continue;
            }
            ASSERT(node->get_local_dimensions() <= factor_graph::maximum_landmark_dimensions, "The marginalised vertex has more local dimensions than the landmark blocks hold.");
            this->h_ll[static_cast<size_t>(landmark)].offset = node->get_ordering_id();
            this->h_ll[static_cast<size_t>(landmark)].dimensions = node->get_local_dimensions();
        }

        this->accumulate_general_blocks();
        this->accumulate_pair_blocks();
        // The couplings were written when the edges were linearised.
        core::thread_pool::instance().parallel_for(landmark_count, factor_graph::landmark_grain, [this](const size_t landmark) {
            landmark_diagonal& diagonal = this->h_ll[landmark];
            const size_t offset = static_cast<size_t>(diagonal.offset);
            const size_t dimensions = static_cast<size_t>(diagonal.dimensions);
            for (size_t k = this->landmark_incidence_begin[landmark]; k < this->landmark_incidence_begin[landmark + 1]; ++k) {
                const double* const term = this->landmark_terms.data() + this->landmark_term_begin[k];
                for (size_t a = 0; a < dimensions; ++a) {
                    this->vector_b[offset + a][0] = this->vector_b[offset + a][0] + (-1.0 * term[a]);
                }
                for (size_t a = 0; a < dimensions; ++a) {
                    for (size_t b = 0; b < dimensions; ++b) {
                        diagonal.block[a][b] = diagonal.block[a][b] + term[dimensions + (a * dimensions) + b];
                    }
                }
            }
        });
        this->hessian_diagonal.assign(total_count, 0.0);
        for (const free_block& entry : this->general_blocks) {
            const size_t size = static_cast<size_t>(entry.dimensions);
            for (size_t i = 0; i < size; ++i) {
                this->hessian_diagonal[static_cast<size_t>(entry.offset) + i] = this->hessian_values[entry.hessian + (i * size) + i];
            }
        }
        for (const landmark_diagonal& diagonal : this->h_ll) {
            for (int i = 0; i < diagonal.dimensions; ++i) {
                this->hessian_diagonal[static_cast<size_t>(diagonal.offset + i)] = diagonal.block[i][i];
            }
        }
        this->update_scaling();
    }

    bool factor_graph::invert_landmark_block(const double (&damped)[maximum_landmark_dimensions][maximum_landmark_dimensions], const int dimensions, double (&inverse)[maximum_landmark_dimensions][maximum_landmark_dimensions]) {
        if ((dimensions <= 0) || (dimensions > maximum_landmark_dimensions)) {
            return false;
        }
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
        const size_t total_count = static_cast<size_t>(this->count_general_params + this->count_marginalised_params);
        const size_t landmark_count = this->h_ll.size();
        const double lambda = this->damping_lambda;
        core::thread_pool& pool = core::thread_pool::instance();

        // Invert each damped landmark block and form Y = W * H_ll^-1 and Y * b_ll for its couplings.
        pool.parallel_for(landmark_count, factor_graph::landmark_grain, [this, lambda](const size_t landmark) {
            landmark_diagonal& diagonal = this->h_ll[landmark];
            const size_t dimensions = static_cast<size_t>(diagonal.dimensions);
            const size_t offset = static_cast<size_t>(diagonal.offset);
            double damped[maximum_landmark_dimensions][maximum_landmark_dimensions];
            for (size_t i = 0; i < dimensions; ++i) {
                for (size_t j = 0; j < dimensions; ++j) {
                    damped[i][j] = diagonal.block[i][j];
                }
                damped[i][i] += lambda * this->damping_weight(diagonal.offset + static_cast<int>(i));
            }
            this->landmark_failures[landmark] = factor_graph::invert_landmark_block(damped, diagonal.dimensions, diagonal.inverse) ? 0 : 1;
            if (this->landmark_failures[landmark] != 0) {
                return;
            }
            double landmark_gradient[maximum_landmark_dimensions];
            for (size_t k = 0; k < dimensions; ++k) {
                landmark_gradient[k] = this->vector_b[offset + k][0];
            }
            for (size_t index = this->landmark_coupling_begin[landmark]; index < this->landmark_coupling_begin[landmark + 1]; ++index) {
                const coupling& link = this->couplings[index];
                const size_t rows = static_cast<size_t>(this->general_blocks[static_cast<size_t>(link.general)].dimensions);
                form_coupling_products(this->coupling_values.data() + link.values, diagonal.inverse, &landmark_gradient[0], rows, dimensions, this->coupling_products.data() + link.values, this->coupling_gradients.data() + link.gradient);
            }
        });
        for (size_t landmark = 0; landmark < landmark_count; ++landmark) {
            if (this->landmark_failures[landmark] != 0) {
                core::logger::log(core::logger::level::warn, "Landmark block inversion failed!");
                this->delta_x = math::matrix<double, 0, 0>::zero(total_count, 1);
                return false;
            }
        }

        // Assemble the damped reduced camera system into the factor, a column at a time, each block's terms in a fixed order.
        double* const factor = this->reduced.get_values();
        const size_t columns = static_cast<size_t>(this->reduced.block_count());
        pool.parallel_for(columns, 1, [this, lambda, factor](const size_t column) {
            const int position = static_cast<int>(column);
            const free_block& entry = this->general_blocks[static_cast<size_t>(this->reduced.block_at(position))];
            const size_t size = static_cast<size_t>(entry.dimensions);
            double* const diagonal = factor + this->reduced.diagonal_offset(position);
            const double* const hessian = this->hessian_values.data() + entry.hessian;
            for (size_t i = 0; i < (size * size); ++i) {
                diagonal[i] = hessian[i];
            }
            for (size_t i = 0; i < size; ++i) {
                diagonal[(i * size) + i] += lambda * this->damping_weight(entry.offset + static_cast<int>(i));
            }
            for (size_t k = this->reduced.column_entries_begin(position); k < this->reduced.column_entries_end(position); ++k) {
                const size_t rows = static_cast<size_t>(this->reduced.dimensions_at(this->reduced.entry_row(k)));
                double* const target = factor + this->reduced.entry_offset(k);
                for (size_t i = 0; i < (rows * size); ++i) {
                    target[i] = 0.0;
                }
            }
            for (size_t k = this->column_pair_begin[column]; k < this->column_pair_begin[column + 1]; ++k) {
                const pair_block& pair = this->pair_blocks[static_cast<size_t>(this->column_pairs[k])];
                const size_t rows = static_cast<size_t>(this->general_blocks[static_cast<size_t>(pair.lower)].dimensions);
                const size_t cols = static_cast<size_t>(this->general_blocks[static_cast<size_t>(pair.upper)].dimensions);
                const double* const values = this->hessian_values.data() + pair.values;
                double* const target = factor + pair.target;
                for (size_t a = 0; a < rows; ++a) {
                    for (size_t b = 0; b < cols; ++b) {
                        if (pair.transposed) {
                            target[(b * rows) + a] = values[(a * cols) + b];
                        }
                        else {
                            target[(a * cols) + b] = values[(a * cols) + b];
                        }
                    }
                }
            }
            for (size_t k = this->schur_term_begin[column]; k < this->schur_term_begin[column + 1]; ++k) {
                const schur_term& term = this->schur_terms[k];
                const coupling& row_coupling = this->couplings[static_cast<size_t>(term.row)];
                const size_t rows = static_cast<size_t>(this->general_blocks[static_cast<size_t>(row_coupling.general)].dimensions);
                const size_t inner = static_cast<size_t>(this->h_ll[static_cast<size_t>(row_coupling.landmark)].dimensions);
                const double* const product = this->coupling_products.data() + row_coupling.values;
                const double* const values = this->coupling_values.data() + this->couplings[static_cast<size_t>(term.column)].values;
                subtract_schur_term(product, values, rows, size, inner, factor + term.target);
            }
        });
        pool.parallel_for(this->general_blocks.size(), factor_graph::general_grain, [this](const size_t general) {
            const free_block& entry = this->general_blocks[general];
            const size_t offset = static_cast<size_t>(entry.offset);
            double* const right_hand_side = this->reduced_right_hand_side.data() + offset;
            for (size_t i = 0; i < static_cast<size_t>(entry.dimensions); ++i) {
                right_hand_side[i] = this->vector_b[offset + i][0];
            }
            for (size_t k = this->general_coupling_begin[general]; k < this->general_coupling_begin[general + 1]; ++k) {
                const double* const gradient = this->coupling_gradients.data() + this->couplings[static_cast<size_t>(this->general_couplings[k])].gradient;
                for (size_t i = 0; i < static_cast<size_t>(entry.dimensions); ++i) {
                    right_hand_side[i] = right_hand_side[i] + (-1.0 * gradient[i]);
                }
            }
        });

        if ((columns > 0) && (!this->reduced.factorise() || !this->reduced.solve(this->reduced_right_hand_side.data(), this->reduced_solution.data()))) {
            core::logger::log(core::logger::level::warn, "Cholesky solver failed!");
            this->delta_x = math::matrix<double, 0, 0>::zero(total_count, 1);
            return false;
        }
        for (size_t i = 0; i < static_cast<size_t>(this->count_general_params); ++i) {
            this->delta_x[i][0] = this->reduced_solution[i];
        }

        // Back substitute each landmark from the pose steps.
        pool.parallel_for(landmark_count, factor_graph::landmark_grain, [this](const size_t landmark) {
            const landmark_diagonal& diagonal = this->h_ll[landmark];
            const size_t dimensions = static_cast<size_t>(diagonal.dimensions);
            const size_t offset = static_cast<size_t>(diagonal.offset);
            double b_landmark[maximum_landmark_dimensions];
            for (size_t i = 0; i < dimensions; ++i) {
                b_landmark[i] = this->vector_b[offset + i][0];
            }
            for (size_t index = this->landmark_coupling_begin[landmark]; index < this->landmark_coupling_begin[landmark + 1]; ++index) {
                const coupling& link = this->couplings[index];
                const size_t rows = static_cast<size_t>(this->general_blocks[static_cast<size_t>(link.general)].dimensions);
                const size_t general_offset = static_cast<size_t>(this->general_blocks[static_cast<size_t>(link.general)].offset);
                const double* const values = this->coupling_values.data() + link.values;
                for (size_t m = 0; m < dimensions; ++m) {
                    double sum = 0;
                    for (size_t k = 0; k < rows; ++k) {
                        sum += values[(k * dimensions) + m] * this->delta_x[general_offset + k][0];
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
        });
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
