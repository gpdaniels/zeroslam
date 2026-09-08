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
#ifndef ZEROSLAM_OPTIMISATION_VERTEX_HPP
#define ZEROSLAM_OPTIMISATION_VERTEX_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace optimisation {
    class vertex;
}

inline void* operator new(size_t size, void* pointer, optimisation::vertex* unused_type_tag) {
    static_cast<void>(size);
    static_cast<void>(unused_type_tag);
    return pointer;
}

inline void operator delete(void* data, void* pointer, optimisation::vertex* unused_type_tag) {
    static_cast<void>(data);
    static_cast<void>(pointer);
    static_cast<void>(unused_type_tag);
}

namespace optimisation {
    class vertex final {
    public:
        constexpr static const size_t maximum_size = 64;
        constexpr static const size_t maximum_alignment = 16;
        constexpr static const size_t maximum_parameters = 16;

    private:
        template <typename value_type>
        static value_type& reference();

        template <typename vertex_type, typename = void>
        struct is_vertex {
            constexpr static const bool value = false;
        };

        template <typename vertex_type>
        struct is_vertex<
            vertex_type,
            decltype(static_cast<void>(vertex_type::name), static_cast<void>(static_cast<int>(vertex_type::parameter_count)), static_cast<void>(static_cast<int>(vertex_type::local_count)), static_cast<void>(vertex::reference<const vertex_type>().plus(static_cast<double*>(nullptr), static_cast<const double*>(nullptr))))
        > {
            constexpr static const bool value = true;
        };

    private:
        struct table final {
            const char* name;
            int parameter_count;
            int local_count;
            void (*plus)(const void* object, double* const parameters, const double* const delta);
            void (*copy)(const void* from, void* to);
            void (*destroy)(void* object);
        };

        template <typename vertex_type>
        static const table* table_for() {
            static const table entries = {
                vertex_type::name,
                vertex_type::parameter_count,
                vertex_type::local_count,
                [](const void* object, double* const values, const double* const delta) -> void {
                    static_cast<const vertex_type*>(object)->plus(values, delta);
                },
                [](const void* from, void* to) -> void {
                    new (to, static_cast<vertex*>(nullptr)) vertex_type(*static_cast<const vertex_type*>(from));
                },
                [](void* object) -> void {
                    static_cast<vertex_type*>(object)->~vertex_type();
                }
            };
            return &entries;
        }

    private:
        alignas(maximum_alignment) unsigned char storage[maximum_size];
        const table* functions;
        int ordering_id;
        bool fixed;
        bool marginalised;
        double parameters[maximum_parameters];
        double parameters_backup[maximum_parameters];

    public:
        vertex();

        template <typename vertex_type>
        vertex(const vertex_type& policy)
            : storage{}
            , functions(vertex::table_for<vertex_type>())
            , ordering_id(0)
            , fixed(false)
            , marginalised(false)
            , parameters{}
            , parameters_backup{} {
            static_assert(vertex::is_vertex<vertex_type>::value, "A vertex provides name, parameter_count, local_count and plus(parameters, delta).");
            static_assert(vertex_type::parameter_count > 0, "A vertex must have at least one parameter.");
            static_assert(static_cast<size_t>(vertex_type::parameter_count) <= vertex::maximum_parameters, "The vertex parameters must fit the storage of the erased vertex.");
            static_assert(vertex_type::local_count > 0, "A vertex must have at least one local dimension.");
            static_assert(vertex_type::local_count <= vertex_type::parameter_count, "A vertex cannot have more local dimensions than parameters.");
            static_assert(sizeof(vertex_type) <= vertex::maximum_size, "The vertex must fit the storage of the erased vertex.");
            static_assert(alignof(vertex_type) <= vertex::maximum_alignment, "The vertex must not need more alignment than the erased vertex provides.");
            new (static_cast<void*>(this->storage), static_cast<vertex*>(nullptr)) vertex_type(policy);
        }

        vertex(const vertex& other);
        vertex(vertex&& other);
        vertex& operator=(const vertex& other);
        vertex& operator=(vertex&& other);
        ~vertex();

    public:
        void clear();
        bool is_valid() const;
        const char* name() const;

        void set_ordering_id(int id);
        int get_ordering_id() const;
        int get_parameter_dimensions() const;
        int get_local_dimensions() const;
        void set_fixed(bool fixed_value = true);
        bool is_fixed() const;
        void set_marginalised(bool marginalised_value = true);
        bool is_marginalised() const;
        const double* get_parameters() const;
        bool set_parameters(const double* const parameters_value, const size_t parameters_length);
        bool get_parameters(double* const parameters_value, const size_t parameters_length) const;
        void backup();
        void restore();

        void plus(const double* const delta);
    };
}

#endif // ZEROSLAM_OPTIMISATION_VERTEX_HPP
