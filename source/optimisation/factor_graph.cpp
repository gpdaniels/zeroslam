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
#include "math/math.hpp"
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
                if (!math::isfinite(this->damping_lambda)) {
                    break;
                }

                if (good_step) {
                    this->linearise();
                    break;
                }
                else {
                    this->rollback_states();
                    ++failure_count;
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

    void factor_graph::linearise() {
        this->make_hessian();
    }

    bool factor_graph::compute_step() {
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

    void factor_graph::make_hessian() {
        this->h_pp.diagonal().clear();
        this->h_pp.diagonal().resize(static_cast<size_t>(this->count_general_params / 6), math::matrix<double, 6, 6>::zero());
        this->h_ll.diagonal().clear();
        this->h_ll.diagonal().resize(static_cast<size_t>(this->count_marginalised_params / 3), math::matrix<double, 3, 3>::zero());
        this->h_pl.blocks().clear();
        this->h_pl.set_size(static_cast<size_t>(this->count_general_params), static_cast<size_t>(this->count_marginalised_params));
        this->h_lp.blocks().clear();
        this->h_lp.set_size(static_cast<size_t>(this->count_marginalised_params), static_cast<size_t>(this->count_general_params));
        this->h_ll_inverse.diagonal().clear();
        this->h_ll_inverse.diagonal().resize(static_cast<size_t>(this->count_marginalised_params / 3), math::matrix<double, 3, 3>::zero());
        this->vector_b = math::matrix<double, 0, 0>::zero(static_cast<size_t>(this->count_general_params + this->count_marginalised_params), 1);
        this->delta_x = math::matrix<double, 0, 0>::zero(static_cast<size_t>(this->count_general_params + this->count_marginalised_params), 1);

        for (const auto& factor : this->edges) {
            factor->compute_jacobians();
            const std::vector<vertex*>& edge_vertices = factor->get_vertices();
            const std::vector<math::matrix<double, 0, 0>>& jacobians = factor->get_jacobians();
            ASSERT(edge_vertices.size() == jacobians.size(), "Mismatching sizes between edge vertices and edge jacobians.");
            const math::matrix<double, 2, 2> information_fixed(factor->get_information().data());
            const math::matrix<double, 2, 1> residual_fixed(factor->get_residual().data());
            for (size_t i = 0; i < edge_vertices.size(); ++i) {
                const vertex* v_i = edge_vertices[i];
                if (v_i->is_fixed()) {
                    continue;
                }
                const math::matrix<double, 0, 0>& jacobian_i = jacobians[i];
                const int index_i = v_i->get_ordering_id();
                double drho;
                const math::matrix<double, 2, 2> robust_information = factor->robust_info_2x2(drho, false);
                if (v_i->is_marginalised() == false) {
                    ASSERT(v_i->get_local_dimensions() == 6, "Currently only pose vertices of local dimension 6 are supported.");
                    const math::matrix<double, 2, 6> jacobian_i_fixed(jacobian_i.data());
                    const math::matrix<double, 6, 2> jtw = math::transpose(jacobian_i_fixed) * robust_information;
                    for (size_t j = i; j < edge_vertices.size(); ++j) {
                        const vertex* v_j = edge_vertices[j];
                        if (v_j->is_fixed()) {
                            continue;
                        }
                        const math::matrix<double, 0, 0>& jacobian_j = jacobians[j];
                        const int index_j = v_j->get_ordering_id();
                        ASSERT(v_j->get_ordering_id() != -1, "Missing ordering id value of edge vertex.");
                        if (v_j->is_marginalised() == false) {
                            ASSERT(index_i == index_j, "Attempting to add a pose-pose block that is not on the block diagonal.");
                            const math::matrix<double, 2, 6> jacobian_j_fixed(jacobian_j.data());
                            const math::matrix<double, 6, 6> hessian_block = jtw * jacobian_j_fixed;
                            this->h_pp.diagonal()[static_cast<size_t>(index_i / 6)] = this->h_pp.diagonal()[static_cast<size_t>(index_i / 6)] + hessian_block;
                        }
                        else {
                            ASSERT(index_i != index_j, "Attempting to add a pose-landmark block that is on the block diagonal.");
                            const math::matrix<double, 2, 3> jacobian_j_fixed(jacobian_j.data());
                            const math::matrix<double, 6, 3> hessian_block = jtw * jacobian_j_fixed;
                            const size_t block_row = static_cast<size_t>(index_i / 6);
                            const size_t block_col = static_cast<size_t>((index_j - this->count_general_params) / 3);
                            if (this->h_pl.blocks().find({ block_row, block_col }) == this->h_pl.blocks().end()) {
                                this->h_pl.blocks()[{ block_row, block_col }] = hessian_block;
                            }
                            else {
                                this->h_pl.blocks()[{ block_row, block_col }] = this->h_pl.blocks()[{ block_row, block_col }] + hessian_block;
                            }
                        }
                    }
                    const math::matrix<double, 6, 1> vector_b_block = math::get_block<double, 6, 1>(this->vector_b, static_cast<size_t>(index_i), 0) - drho * math::transpose(jacobian_i_fixed) * information_fixed * residual_fixed;
                    math::set_block(this->vector_b, static_cast<size_t>(index_i), static_cast<size_t>(0), vector_b_block);
                }
                else {
                    ASSERT(v_i->get_local_dimensions() == 3, "Currently only landmark vertices of local dimension 3 are supported.");
                    const math::matrix<double, 2, 3> jacobian_i_fixed(jacobian_i.data());
                    const math::matrix<double, 3, 2> jtw = math::transpose(jacobian_i_fixed) * robust_information;
                    for (size_t j = i; j < edge_vertices.size(); ++j) {
                        const vertex* v_j = edge_vertices[j];
                        if (v_j->is_fixed()) {
                            continue;
                        }
                        const math::matrix<double, 0, 0>& jacobian_j = jacobians[j];
                        const int index_j = v_j->get_ordering_id();
                        ASSERT(v_j->get_ordering_id() != -1, "Missing ordering id value of edge vertex.");
                        if (v_j->is_marginalised() == false) {
                            ASSERT(false, "This code path is unexpected for current edge vertex ordering conventions.");
                        }
                        else {
                            ASSERT(index_i == index_j, "Attempting to add a landmark-landmark block that is not on the block diagonal.");
                            const math::matrix<double, 2, 3> jacobian_j_fixed(jacobian_j.data());
                            const math::matrix<double, 3, 3> hessian_block = jtw * jacobian_j_fixed;
                            this->h_ll.diagonal()[static_cast<size_t>((index_j - this->count_general_params) / 3)] = this->h_ll.diagonal()[static_cast<size_t>((index_j - this->count_general_params) / 3)] + hessian_block;
                        }
                    }
                    const math::matrix<double, 3, 1> vector_b_block = math::get_block<double, 3, 1>(this->vector_b, static_cast<size_t>(index_i), 0) - drho * math::transpose(jacobian_i_fixed) * information_fixed * residual_fixed;
                    math::set_block(this->vector_b, static_cast<size_t>(index_i), static_cast<size_t>(0), vector_b_block);
                }
            }
        }
        this->h_lp = this->h_pl.get_transpose();
        this->b_pp = (this->count_general_params > 0) ? math::get_block(this->vector_b, 0, 0, static_cast<size_t>(this->count_general_params), 1) : math::matrix<double, 0, 0>::zero(0, 1);
        this->b_ll = (this->count_marginalised_params > 0) ? math::get_block(this->vector_b, static_cast<size_t>(this->count_general_params), 0, static_cast<size_t>(this->count_marginalised_params), 1) : math::matrix<double, 0, 0>::zero(0, 1);
        this->hessian_diagonal.assign(static_cast<size_t>(this->count_general_params + this->count_marginalised_params), 0.0);
        for (int i = 0; i < this->count_general_params; ++i) {
            this->hessian_diagonal[static_cast<size_t>(i)] = this->h_pp(static_cast<size_t>(i), static_cast<size_t>(i));
        }
        for (int i = 0; i < this->count_marginalised_params; ++i) {
            this->hessian_diagonal[static_cast<size_t>(this->count_general_params + i)] = this->h_ll(static_cast<size_t>(i), static_cast<size_t>(i));
        }
        this->update_scaling();
    }

    bool factor_graph::solve_linear_system() {
        const math::sparse_block_diagonal<6> h_pp_backup = this->h_pp;
        const math::sparse_block_diagonal<3> h_ll_backup = this->h_ll;

        for (size_t d = 0; d < this->h_pp.diagonal().size(); ++d) {
            for (size_t i = 0; i < 6; ++i) {
                this->h_pp.diagonal()[d][i][i] += this->damping_lambda * this->damping_weight(static_cast<int>((6 * d) + i));
            }
        }
        for (size_t d = 0; d < this->h_ll.diagonal().size(); ++d) {
            for (size_t i = 0; i < 3; ++i) {
                this->h_ll.diagonal()[d][i][i] += this->damping_lambda * this->damping_weight(this->count_general_params + static_cast<int>((3 * d) + i));
            }
        }

        for (const auto& landmark_vertex : this->vertices_marginalised) {
            if (landmark_vertex->is_fixed()) {
                continue;
            }
            const int idx = landmark_vertex->get_ordering_id() - this->count_general_params;
            const int size = landmark_vertex->get_local_dimensions();
            ASSERT(size == 3, "Currently only landmarks of size 3 are supported.");
            ASSERT(idx % 3 == 0, "Landmark indexes must be aligned in sets of 3.");
            math::matrix<double, 3, 3> inverse_block;
            math::invert(this->h_ll.diagonal()[static_cast<size_t>(idx / size)], inverse_block);
            this->h_ll_inverse.diagonal()[static_cast<size_t>(idx / size)] = inverse_block;
        }

        const math::sparse_block<6, 3> h_p_m_times_hll_inv = math::multiply(this->h_pl, this->h_ll_inverse);
        const math::matrix<double, 0, 0> b_pp_schur = this->b_pp - h_p_m_times_hll_inv.multiply(this->b_ll);
        const math::sparse_block<6, 6> h_pp_schur = math::subtract(this->h_pp, h_p_m_times_hll_inv.multiply(this->h_lp));

        math::matrix<double, 0, 0> hpp_schur_dense = math::matrix<double, 0, 0>::zero(h_pp_schur.rows(), h_pp_schur.cols());
        for (std::unordered_map<typename math::sparse_block<6, 6>::block_key<size_t, size_t>, typename math::sparse_block<6, 6>::block_type, typename math::sparse_block<6, 6>::block_key<size_t, size_t>>::const_iterator iterator = h_pp_schur.blocks().begin(); iterator != h_pp_schur.blocks().end(); ++iterator) {
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 6; ++j) {
                    hpp_schur_dense[(iterator->first.first * 6) + static_cast<size_t>(i)][(iterator->first.second * 6) + static_cast<size_t>(j)] = hpp_schur_dense[(iterator->first.first * 6) + static_cast<size_t>(i)][(iterator->first.second * 6) + static_cast<size_t>(j)] + iterator->second[static_cast<size_t>(i)][static_cast<size_t>(j)];
                }
            }
        }
        math::matrix<double, 0, 0> matrix_lower(static_cast<size_t>(this->count_general_params), static_cast<size_t>(this->count_general_params));
        math::matrix<double, 0, 0> delta_xpp = math::matrix<double, 0, 0>::zero(static_cast<size_t>(this->count_general_params), 1);
        if (!math::decompose_cholesky(hpp_schur_dense.data(), this->count_general_params, this->count_general_params, matrix_lower.data()) || !math::solve_cholesky(matrix_lower.data(), b_pp_schur.data(), this->count_general_params, this->count_general_params, delta_xpp.data())) {
            core::logger::log(core::logger::level::warn, "Cholesky solver failed!");
            this->h_pp = h_pp_backup;
            this->h_ll = h_ll_backup;
            this->delta_x = math::matrix<double, 0, 0>::zero(static_cast<size_t>(this->count_general_params + this->count_marginalised_params), 1);
            return false;
        }

        if (this->count_general_params > 0) {
            math::set_block(this->delta_x, 0, 0, delta_xpp);
        }

        if (this->count_marginalised_params > 0) {
            const math::matrix<double, 0, 0> delta_xll = this->h_ll_inverse.multiply(this->b_ll - this->h_lp.multiply(delta_xpp));
            math::set_block(this->delta_x, static_cast<size_t>(this->count_general_params), 0, delta_xll);
        }

        this->h_pp = h_pp_backup;
        this->h_ll = h_ll_backup;
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
