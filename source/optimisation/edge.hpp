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
#ifndef ZEROSLAM_OPTIMISATION_EDGE_HPP
#define ZEROSLAM_OPTIMISATION_EDGE_HPP

#include "math/matrix.hpp"
#include "optimisation/loss.hpp"
#include "optimisation/vertex.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace optimisation {
    class edge;
}

inline void* operator new(size_t size, void* pointer, optimisation::edge* unused_type_tag) {
    static_cast<void>(size);
    static_cast<void>(unused_type_tag);
    return pointer;
}

inline void operator delete(void* data, void* pointer, optimisation::edge* unused_type_tag) {
    static_cast<void>(data);
    static_cast<void>(pointer);
    static_cast<void>(unused_type_tag);
}

namespace optimisation {
    class edge final {
    public:
        constexpr static const size_t maximum_size = 512;
        constexpr static const size_t maximum_alignment = 16;

    private:
        template <typename value_type>
        static value_type& reference();

        template <typename edge_type, typename = void>
        struct is_edge {
            constexpr static const bool value = false;
        };

        template <typename edge_type>
        struct is_edge<
            edge_type,
            decltype(static_cast<void>(edge_type::name), static_cast<void>(static_cast<int>(edge_type::residual_count)), static_cast<void>(static_cast<int>(edge_type::vertex_count)), static_cast<void>(edge::reference<const edge_type>().compute_residual(edge::reference<const edge>(), edge::reference<math::matrix<double, 0, 0>>())))
        > {
            constexpr static const bool value = true;
        };

        template <typename edge_type, typename = void>
        struct has_jacobians {
            constexpr static const bool value = false;
        };

        template <typename edge_type>
        struct has_jacobians<
            edge_type,
            decltype(static_cast<void>(edge::reference<const edge_type>().compute_jacobians(edge::reference<const edge>(), edge::reference<std::vector<math::matrix<double, 0, 0>>>())))
        > {
            constexpr static const bool value = true;
        };

    private:
        struct table final {
            const char* name;
            int residual_count;
            int vertex_count;
            void (*compute_residual)(const void* object, const edge& context, math::matrix<double, 0, 0>& residual);
            void (*compute_jacobians)(const void* object, const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians);
            void (*copy)(const void* from, void* to);
            void (*destroy)(void* object);
        };

        template <typename edge_type>
        static void (*jacobians_for())(const void* object, const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) {
            if constexpr (edge::has_jacobians<edge_type>::value) {
                return [](const void* object, const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobian_blocks) -> void {
                    static_cast<const edge_type*>(object)->compute_jacobians(context, jacobian_blocks);
                };
            }
            else {
                return nullptr;
            }
        }

        template <typename edge_type>
        static const table* table_for() {
            static const table entries = {
                edge_type::name,
                edge_type::residual_count,
                edge_type::vertex_count,
                [](const void* object, const edge& context, math::matrix<double, 0, 0>& residual_values) -> void {
                    static_cast<const edge_type*>(object)->compute_residual(context, residual_values);
                },
                edge::jacobians_for<edge_type>(),
                [](const void* from, void* to) -> void {
                    new (to, static_cast<edge*>(nullptr)) edge_type(*static_cast<const edge_type*>(from));
                },
                [](void* object) -> void {
                    static_cast<edge_type*>(object)->~edge_type();
                }
            };
            return &entries;
        }

    private:
        alignas(maximum_alignment) unsigned char storage[maximum_size];
        const table* functions;
        int ordering_id;
        std::vector<vertex*> vertices;
        math::matrix<double, 0, 0> residual;
        std::vector<math::matrix<double, 0, 0>> jacobians;
        math::matrix<double, 0, 0> information;
        math::matrix<double, 0, 0> observation;
        loss robust_loss;

    public:
        edge();

        template <typename edge_type>
        edge(const edge_type& factor)
            : storage{}
            , functions(edge::table_for<edge_type>())
            , ordering_id(0)
            , vertices()
            , residual(math::matrix<double, 0, 0>::zero(static_cast<size_t>(edge_type::residual_count), 1))
            , jacobians(static_cast<size_t>(edge_type::vertex_count))
            , information(math::matrix<double, 0, 0>::identity(static_cast<size_t>(edge_type::residual_count), static_cast<size_t>(edge_type::residual_count)))
            , observation()
            , robust_loss() {
            static_assert(edge::is_edge<edge_type>::value, "An edge provides name, residual_count, vertex_count and compute_residual(edge, residual).");
            static_assert(sizeof(edge_type) <= edge::maximum_size, "The edge must fit the storage of the erased edge.");
            static_assert(alignof(edge_type) <= edge::maximum_alignment, "The edge must not need more alignment than the erased edge provides.");
            this->vertices.reserve(static_cast<size_t>(edge_type::vertex_count));
            new (static_cast<void*>(this->storage), static_cast<edge*>(nullptr)) edge_type(factor);
        }

        edge(const edge& other);
        edge(edge&& other);
        edge& operator=(const edge& other);
        edge& operator=(edge&& other);
        ~edge();

    public:
        void clear();
        bool is_valid() const;
        const char* name() const;

        int get_ordering_id() const;
        void set_ordering_id(int id);
        size_t num_vertices() const;
        bool add_vertex(vertex* node);
        const std::vector<vertex*>& get_vertices() const;
        bool set_vertices(const std::vector<vertex*>& vertices_value);
        vertex* get_vertex(int i);
        const vertex* get_vertex(int i) const;
        double chi2() const;
        double robust_chi2() const;
        const math::matrix<double, 0, 0>& get_residual() const;
        const std::vector<math::matrix<double, 0, 0>>& get_jacobians() const;
        const math::matrix<double, 0, 0>& get_information() const;
        void set_information(const math::matrix<double, 0, 0>& information_value);
        const loss& get_loss() const;
        void set_loss(const loss& robust_loss_value);
        void robust_info(double& rho_delta, math::matrix<double, 0, 0>& robust_information) const;
        math::matrix<double, 2, 2> robust_info_2x2(double& rho_delta, bool apply_triggs_correction = true) const;
        const math::matrix<double, 0, 0>& get_observation() const;
        void set_observation(const math::matrix<double, 0, 0>& observation_value);

    public:
        void compute_residual();
        void compute_jacobians();
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGE_HPP
