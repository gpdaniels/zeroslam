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
#ifndef ZEROSLAM_OPTIMISATION_FACTOR_GRAPH_HPP
#define ZEROSLAM_OPTIMISATION_FACTOR_GRAPH_HPP

#include "math/matrix.hpp"
#include "optimisation/block_cholesky.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/landmark_block.hpp"
#include "optimisation/vertex.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace optimisation {
    class factor_graph final {
    public:
        enum class strategy {
            dense_schur,
            square_root,
            automatic
        };

        enum class precision {
            double_precision,
            single_precision
        };

        class diagnostics final {
        public:
            int rejected_attempts = 0;
            bool used_square_root = false;
            bool used_single_precision = false;
            int landmark_blocks = 0;
            int reduced_solves = 0;
            int reduced_iterations = 0;
            int reduced_failures = 0;
        };

    private:
        double damping_lambda;
        double damping_factor;
        double chi_squared;

        // The edges of a vertex in the order they were added, some possibly already removed from the graph.
        class adjacency final {
        public:
            std::vector<edge*> edges;
            size_t removed = 0;
        };

        std::deque<vertex> vertex_storage;
        std::deque<edge> edge_storage;

        std::vector<vertex*> vertices_general;
        std::vector<vertex*> vertices_marginalised;
        std::vector<edge*> edges;
        size_t removed_edge_count = 0;
        std::unordered_set<vertex*> vertex_set;
        std::unordered_map<edge*, size_t> edge_positions;
        std::unordered_map<vertex*, adjacency> vertex_to_edge;

    public:
        constexpr static const int maximum_landmark_dimensions = 6;

    private:
        class landmark_diagonal final {
        public:
            int offset;
            int dimensions;
            double block[maximum_landmark_dimensions][maximum_landmark_dimensions];
            double inverse[maximum_landmark_dimensions][maximum_landmark_dimensions];
        };

        constexpr static const size_t maximum_residuals = 16;
        constexpr static const double gain_ratio_epsilon = 1e-12;

        // Work items per thread pool task, sized so a task outweighs its scheduling and small problems stay serial.
        constexpr static const size_t residual_grain = 512;
        constexpr static const size_t edge_grain = 256;
        constexpr static const size_t general_grain = 4;
        constexpr static const size_t landmark_grain = 128;

        // A free vertex of the linear system, with the offset of its diagonal Hessian block for a general one.
        class free_block final {
        public:
            int offset = 0;
            int dimensions = 0;
            size_t hessian = 0;
        };

        // The Hessian block between two different general blocks, rows of the lower block and columns of the upper one, and
        // where it goes in the factor of the reduced system, transposed when the upper block is eliminated later.
        class pair_block final {
        public:
            int lower = 0;
            int upper = 0;
            size_t values = 0;
            size_t target = 0;
            bool transposed = false;
        };

        // One edge's term of a pair block, from its slots first < second, transposed when the first slot holds the upper block.
        class pair_term final {
        public:
            int edge = 0;
            int first = 0;
            int second = 0;
            bool transposed = false;
        };

        // The block between a general block and a landmark from one edge, general rows and landmark columns, with the offsets
        // of W, of Y = W * H_ll^-1, and of Y * b_ll.
        class coupling final {
        public:
            int general = 0;
            int landmark = 0;
            int edge = 0;
            int general_slot = 0;
            int landmark_slot = 0;
            size_t values = 0;
            size_t gradient = 0;
        };

        // A block of a general block's row of the Hessian of the general edges, for the square root operator.
        class neighbour final {
        public:
            int block = 0;
            size_t values = 0;
            bool transposed = false;
        };

        // A term of the reduced system, minus the row coupling's Y times the column coupling's W^T, into a factor block.
        class schur_term final {
        public:
            int row = 0;
            int column = 0;
            size_t target = 0;
        };

        // The vertex slot of an edge that holds a block.
        class incidence final {
        public:
            int edge = 0;
            int slot = 0;
        };

        template <typename scalar>
        class square_root_state final {
        public:
            std::vector<landmark_block<scalar>> blocks;
            std::vector<std::vector<int>> grouped_edges;
            std::unordered_map<const vertex*, size_t> landmark_indices;
            std::vector<size_t> block_vertices;
            std::vector<int> pose_block_begin;
            std::vector<int> pose_block_index;
            std::vector<int> pose_block_slot;
            std::vector<scalar> preconditioner;
            std::vector<scalar> right_hand_side;
            std::vector<scalar> general_right_hand_side;
            std::vector<scalar> increment;
            std::vector<scalar> scratch;
        };

    private:
        strategy solve_strategy = strategy::dense_schur;
        precision solve_precision = precision::double_precision;
        // The block sparse direct solve beat the square root path at every benchmarked size, 50 to 1000 poses on keyframe
        // loops, by 10 times or more, so automatic only switches well beyond them.
        int square_root_parameter_threshold = 6 * 4096;
        double conjugate_gradient_tolerance = 1e-2;
        int conjugate_gradient_iteration_limit = 0;
        bool square_root_active = false;
        square_root_state<double> square_root_double;
        square_root_state<float> square_root_single;

    private:
        bool numeric_jacobians = false;
        std::vector<double> edge_costs;
        std::vector<size_t> edge_weights_begin;
        std::vector<double> edge_weights;
        std::vector<int> block_of_offset;
        std::vector<free_block> general_blocks;
        std::vector<size_t> general_incidence_begin;
        std::vector<incidence> general_incidence;
        std::vector<size_t> landmark_incidence_begin;
        std::vector<incidence> landmark_incidence;
        // Each edge's first vertex slot and, per slot, its general block and the general term, landmark term and coupling it
        // writes when its edge is linearised, -1 for none.
        std::vector<size_t> edge_slot_begin;
        std::vector<int> slot_general_block;
        std::vector<int> slot_general_term;
        std::vector<int> slot_landmark_term;
        std::vector<int> slot_coupling;
        // Each incidence's gradient and then its Hessian terms in slot order, which its block sums in edge order.
        std::vector<size_t> general_term_begin;
        std::vector<double> general_terms;
        std::vector<size_t> landmark_term_begin;
        std::vector<double> landmark_terms;
        std::vector<size_t> landmark_coupling_begin;
        std::vector<double> hessian_values;
        std::vector<pair_block> pair_blocks;
        std::vector<size_t> pair_term_begin;
        std::vector<pair_term> pair_terms;
        std::vector<size_t> operator_neighbour_begin;
        std::vector<neighbour> operator_neighbours;
        std::vector<coupling> couplings;
        std::vector<size_t> general_coupling_begin;
        std::vector<int> general_couplings;
        std::vector<double> coupling_values;
        std::vector<double> coupling_products;
        std::vector<double> coupling_gradients;
        std::vector<unsigned char> landmark_failures;
        std::vector<size_t> column_pair_begin;
        std::vector<int> column_pairs;
        std::vector<size_t> schur_term_begin;
        std::vector<schur_term> schur_terms;
        block_cholesky reduced;
        std::vector<double> reduced_right_hand_side;
        std::vector<double> reduced_solution;

    private:
        std::vector<landmark_diagonal> h_ll;
        math::matrix<double, 0, 0> vector_b;
        math::matrix<double, 0, 0> delta_x;

        int count_general_params = 0;
        int count_marginalised_params = 0;

    private:
        std::vector<double> hessian_diagonal;
        std::vector<double> column_scale;
        std::vector<double> damping_diagonal;
        diagnostics last_diagnostics;

    public:
        factor_graph();
        // The graph holds pointers into its own storage, so a copy or a move would alias the original.
        factor_graph(const factor_graph&) = delete;
        factor_graph(factor_graph&&) = delete;
        factor_graph& operator=(const factor_graph&) = delete;
        factor_graph& operator=(factor_graph&&) = delete;

    public:
        const diagnostics& get_diagnostics() const;

        void set_strategy(const strategy solve_strategy_value);
        void set_precision(const precision solve_precision_value);
        void set_square_root_parameter_threshold(const int threshold);
        void set_conjugate_gradient_tolerance(const double tolerance);
        void set_conjugate_gradient_iteration_limit(const int limit);

    public:
        vertex* add_vertex(vertex&& node);

        bool remove_vertex(vertex* node);

        edge* add_edge(edge&& factor);

        bool remove_edge(edge* factor);

        std::vector<edge*> get_connected_edges(vertex* node) const;

    public:
        int solve(int iterations = 10, bool use_relative_convergence = false);

        double get_current_chi(bool recompute_residuals = true);

        // The ratio of the actual to the predicted cost reduction of a step, 0 when the model predicts no reduction.
        static double gain_ratio(const double current_chi, const double evaluated_chi, const double predicted_reduction);

        bool compute_damped_step(double lambda, math::matrix<double, 0, 0>& step);

    private:
        void compact_edges();

        void compute_residuals();

        static void gradient_term(const math::matrix<double, 0, 0>& jacobian, const double* weighted_residual, const size_t residuals, double* term);

        static void edge_block(const math::matrix<double, 0, 0>& first, const math::matrix<double, 0, 0>& second, const double* weight, const size_t residuals, double* product);

        int general_block_of(const vertex* node) const;

        int landmark_block_of(const vertex* node) const;

        void analyse();

        void analyse_schur();

        void linearise_edges();

        void analyse_pairs();

        void accumulate_general_blocks();

        void accumulate_pair_blocks();

        bool select_square_root_path() const;

        bool linearisation_limits_hold() const;

        void partition_vertices();

        static bool invert_landmark_block(const double (&damped)[maximum_landmark_dimensions][maximum_landmark_dimensions], int dimensions, double (&inverse)[maximum_landmark_dimensions][maximum_landmark_dimensions]);

        template <typename scalar>
        void linearise_landmark_blocks(square_root_state<scalar>& state);

        template <typename scalar>
        bool solve_linear_system_qr(square_root_state<scalar>& state);

        double damping_weight(int index) const;

        void update_scaling();

        void linearise();

        bool compute_step();

        void set_ordering();

        void make_hessian();

        bool solve_linear_system();

        void update_states();

        void rollback_states();
    };
}

#endif // ZEROSLAM_OPTIMISATION_FACTOR_GRAPH_HPP
