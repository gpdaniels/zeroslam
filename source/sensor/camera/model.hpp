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
#ifndef ZEROSLAM_SENSOR_CAMERA_MODEL_HPP
#define ZEROSLAM_SENSOR_CAMERA_MODEL_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor::camera {
    template <typename type>
    class model;
}

template <typename type>
inline void* operator new(size_t size, void* pointer, sensor::camera::model<type>* unused_type_tag) {
    static_cast<void>(size);
    static_cast<void>(unused_type_tag);
    return pointer;
}

template <typename type>
inline void operator delete(void* data, void* pointer, sensor::camera::model<type>* unused_type_tag) {
    static_cast<void>(data);
    static_cast<void>(pointer);
    static_cast<void>(unused_type_tag);
}

namespace sensor::camera {
    template <typename type>
    class model final {
    public:
        constexpr static const size_t maximum_size = 256;
        constexpr static const size_t maximum_alignment = 16;

    private:
        template <typename value_type>
        static value_type& reference();

        template <typename model_type, typename = void>
        struct is_model {
            constexpr static const bool value = false;
        };

        template <typename model_type>
        struct is_model<
            model_type,
            decltype(static_cast<void>(model_type::name), static_cast<void>(model::reference<const model_type>().get_parameter_count()), static_cast<void>(model::reference<model_type>().set_parameters(static_cast<const type*>(nullptr), size_t(0))), static_cast<void>(model::reference<const model_type>().get_parameters(static_cast<type*>(nullptr), size_t(0))), static_cast<void>(model::reference<const model_type>().project(static_cast<const type*>(nullptr), static_cast<type*>(nullptr), static_cast<type*>(nullptr), static_cast<type*>(nullptr))), static_cast<void>(model::reference<const model_type>().unproject(static_cast<const type*>(nullptr), static_cast<type*>(nullptr), static_cast<type*>(nullptr))))
        > {
            constexpr static const bool value = true;
        };

    private:
        struct table final {
            const char* name;
            size_t (*parameter_count)(const void* storage);
            bool (*set_parameters)(void* storage, const type* const parameters, const size_t parameters_length);
            bool (*get_parameters)(const void* storage, type* const parameters, const size_t parameters_length);
            bool (*project)(const void* storage, const type* const point_xyz, type* const point_xy, type* const jacobian_projection, type* const jacobian_parameters);
            bool (*unproject)(const void* storage, const type* const point_xy, type* const ray_xyz, type* const jacobian_unprojection);
            void (*copy)(const void* from, void* to);
            void (*destroy)(void* object);
        };

        template <typename model_type>
        static const table* table_for() {
            static const table entries = {
                model_type::name,
                [](const void* object) -> size_t {
                    return static_cast<const model_type*>(object)->get_parameter_count();
                },
                [](void* object, const type* const parameters, const size_t parameters_length) -> bool {
                    return static_cast<model_type*>(object)->set_parameters(parameters, parameters_length);
                },
                [](const void* object, type* const parameters, const size_t parameters_length) -> bool {
                    return static_cast<const model_type*>(object)->get_parameters(parameters, parameters_length);
                },
                [](const void* object, const type* const point_xyz, type* const point_xy, type* const jacobian_projection, type* const jacobian_parameters) -> bool {
                    return static_cast<const model_type*>(object)->project(point_xyz, point_xy, jacobian_projection, jacobian_parameters);
                },
                [](const void* object, const type* const point_xy, type* const ray_xyz, type* const jacobian_unprojection) -> bool {
                    return static_cast<const model_type*>(object)->unproject(point_xy, ray_xyz, jacobian_unprojection);
                },
                [](const void* from, void* to) -> void {
                    new (to, static_cast<model*>(nullptr)) model_type(*static_cast<const model_type*>(from));
                },
                [](void* object) -> void {
                    static_cast<model_type*>(object)->~model_type();
                }
            };
            return &entries;
        }

    private:
        alignas(maximum_alignment) unsigned char storage[maximum_size];
        const table* functions;

    public:
        model()
            : storage{}
            , functions(nullptr) {
        }

        template <typename model_type>
        model(const model_type& camera_model)
            : storage{}
            , functions(model::table_for<model_type>()) {
            static_assert(model::is_model<model_type>::value, "A camera model provides name, get_parameter_count, set_parameters, get_parameters, project and unproject.");
            static_assert(sizeof(model_type) <= model::maximum_size, "The camera model must fit the storage of the erased model.");
            static_assert(alignof(model_type) <= model::maximum_alignment, "The camera model must not need more alignment than the erased model provides.");
            new (static_cast<void*>(this->storage), static_cast<model*>(nullptr)) model_type(camera_model);
        }

        model(const model& other)
            : storage{}
            , functions(other.functions) {
            if (this->functions != nullptr) {
                this->functions->copy(other.storage, this->storage);
            }
        }

        model(model&& other)
            : model(static_cast<const model&>(other)) {
        }

        model& operator=(const model& other) {
            if (this != &other) {
                this->clear();
                this->functions = other.functions;
                if (this->functions != nullptr) {
                    this->functions->copy(other.storage, this->storage);
                }
            }
            return *this;
        }

        model& operator=(model&& other) {
            return *this = static_cast<const model&>(other);
        }

        ~model() {
            this->clear();
        }

    public:
        void clear() {
            if (this->functions != nullptr) {
                this->functions->destroy(this->storage);
                this->functions = nullptr;
            }
        }

        bool is_valid() const {
            return this->functions != nullptr;
        }

        const char* name() const {
            return (this->functions != nullptr) ? this->functions->name : "";
        }

        size_t get_parameter_count() const {
            return (this->functions != nullptr) ? this->functions->parameter_count(this->storage) : 0;
        }

        bool set_parameters(const type* const parameters, const size_t parameters_length) {
            return (this->functions != nullptr) && this->functions->set_parameters(this->storage, parameters, parameters_length);
        }

        bool get_parameters(type* const parameters, const size_t parameters_length) const {
            return (this->functions != nullptr) && this->functions->get_parameters(this->storage, parameters, parameters_length);
        }

        // Camera frame point to image coordinates normalised by the image width, jacobians row major 2 by 3 and 2 by parameter count.
        bool project(
            const type* const point_xyz,
            type* const point_xy,
            type* const jacobian_projection = nullptr,
            type* const jacobian_parameters = nullptr
        ) const {
            return (this->functions != nullptr) && this->functions->project(this->storage, point_xyz, point_xy, jacobian_projection, jacobian_parameters);
        }

        // Image coordinates normalised by the image width to the ray with unit depth, jacobian row major 3 by 2.
        bool unproject(
            const type* const point_xy,
            type* const ray_xyz,
            type* const jacobian_unprojection = nullptr
        ) const {
            return (this->functions != nullptr) && this->functions->unproject(this->storage, point_xy, ray_xyz, jacobian_unprojection);
        }
    };
}

#endif // ZEROSLAM_SENSOR_CAMERA_MODEL_HPP
