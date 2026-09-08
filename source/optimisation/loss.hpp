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
#ifndef ZEROSLAM_OPTIMISATION_LOSS_HPP
#define ZEROSLAM_OPTIMISATION_LOSS_HPP

#include "math/matrix.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace optimisation {
    class loss;
}

inline void* operator new(size_t size, void* pointer, optimisation::loss* unused_type_tag) {
    static_cast<void>(size);
    static_cast<void>(unused_type_tag);
    return pointer;
}

inline void operator delete(void* data, void* pointer, optimisation::loss* unused_type_tag) {
    static_cast<void>(data);
    static_cast<void>(pointer);
    static_cast<void>(unused_type_tag);
}

namespace optimisation {
    class loss final {
    public:
        constexpr static const size_t maximum_size = 64;
        constexpr static const size_t maximum_alignment = 16;

    private:
        template <typename value_type>
        static value_type& reference();

        template <typename loss_type, typename = void>
        struct is_loss {
            constexpr static const bool value = false;
        };

        template <typename loss_type>
        struct is_loss<
            loss_type,
            decltype(static_cast<void>(loss_type::name), static_cast<void>(loss::reference<const loss_type>().compute(double(0), loss::reference<math::matrix<double, 3, 1>>())))
        > {
            constexpr static const bool value = true;
        };

    private:
        struct table final {
            const char* name;
            void (*compute)(const void* object, const double error_squared, math::matrix<double, 3, 1>& rho);
            void (*copy)(const void* from, void* to);
            void (*destroy)(void* object);
        };

        template <typename loss_type>
        static const table* table_for() {
            static const table entries = {
                loss_type::name,
                [](const void* object, const double error_squared, math::matrix<double, 3, 1>& rho) -> void {
                    static_cast<const loss_type*>(object)->compute(error_squared, rho);
                },
                [](const void* from, void* to) -> void {
                    new (to, static_cast<loss*>(nullptr)) loss_type(*static_cast<const loss_type*>(from));
                },
                [](void* object) -> void {
                    static_cast<loss_type*>(object)->~loss_type();
                }
            };
            return &entries;
        }

    private:
        alignas(maximum_alignment) unsigned char storage[maximum_size];
        const table* functions;

    public:
        loss();

        template <typename loss_type>
        loss(const loss_type& robust_loss)
            : storage{}
            , functions(loss::table_for<loss_type>()) {
            static_assert(loss::is_loss<loss_type>::value, "A loss provides name and compute(error_squared, rho).");
            static_assert(sizeof(loss_type) <= loss::maximum_size, "The loss must fit the storage of the erased loss.");
            static_assert(alignof(loss_type) <= loss::maximum_alignment, "The loss must not need more alignment than the erased loss provides.");
            new (static_cast<void*>(this->storage), static_cast<loss*>(nullptr)) loss_type(robust_loss);
        }

        loss(const loss& other);
        loss(loss&& other);
        loss& operator=(const loss& other);
        loss& operator=(loss&& other);
        ~loss();

    public:
        void clear();
        bool is_valid() const;
        const char* name() const;
        void compute(const double error_squared, math::matrix<double, 3, 1>& rho) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_LOSS_HPP
