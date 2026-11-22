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

#include "mapping/rig.hpp"

#include "core/assert.hpp"
#include "core/logger.hpp"
#include "math/math.hpp"
#include "optimisation/vertices/extrinsic.hpp"

namespace mapping {
    size_t rig::add(const int id, const sensor_kind kind, const math::se3<double>& body_from_sensor, const bool refined, const int attached_to) {
        this->sensors.push_back(sensor{ id, kind, body_from_sensor, refined, attached_to, 0.0 });
        const size_t index = this->sensors.size() - 1;
        if ((kind == sensor_kind::camera) && (this->reference_index < 0)) {
            this->reference_index = static_cast<int>(index);
            for (sensor& existing : this->sensors) {
                existing.declared_distance = math::sqrt(this->reference_from_sensor(static_cast<size_t>(&existing - this->sensors.data())).translation().get_length_squared());
            }
        }
        else if (this->reference_index >= 0) {
            this->sensors[index].declared_distance = math::sqrt(this->reference_from_sensor(index).translation().get_length_squared());
        }
        return index;
    }

    void rig::clear() {
        this->sensors.clear();
        this->reference_index = -1;
    }

    size_t rig::size() const {
        return this->sensors.size();
    }

    const rig::sensor& rig::at(const size_t index) const {
        ASSERT(index < this->sensors.size(), "Rig sensor index out of bounds.");
        return this->sensors[index];
    }

    bool rig::find(const int id, size_t& index) const {
        for (size_t i = 0; i < this->sensors.size(); ++i) {
            if (this->sensors[i].id == id) {
                index = i;
                return true;
            }
        }
        return false;
    }

    int rig::reference() const {
        return this->reference_index;
    }

    size_t rig::camera_count() const {
        size_t count = 0;
        for (const sensor& entry : this->sensors) {
            count += (entry.kind == sensor_kind::camera) ? 1u : 0u;
        }
        return count;
    }

    size_t rig::camera(const size_t camera_index) const {
        size_t count = 0;
        for (size_t i = 0; i < this->sensors.size(); ++i) {
            if (this->sensors[i].kind != sensor_kind::camera) {
                continue;
            }
            if (count == camera_index) {
                return i;
            }
            ++count;
        }
        ASSERT(false, "Rig camera index out of bounds.");
        return 0;
    }

    math::se3<double> rig::reference_from_sensor(const size_t index) const {
        ASSERT(index < this->sensors.size(), "Rig sensor index out of bounds.");
        if ((this->reference_index < 0) || (static_cast<int>(index) == this->reference_index)) {
            return math::se3<double>::identity();
        }
        return this->sensors[static_cast<size_t>(this->reference_index)].body_from_sensor.inverse() * this->sensors[index].body_from_sensor;
    }

    void rig::set_reference_from_sensor(const size_t index, const math::se3<double>& reference_from_sensor_value) {
        ASSERT(index < this->sensors.size(), "Rig sensor index out of bounds.");
        if ((this->reference_index < 0) || (static_cast<int>(index) == this->reference_index)) {
            return;
        }
        this->sensors[index].body_from_sensor = this->sensors[static_cast<size_t>(this->reference_index)].body_from_sensor * reference_from_sensor_value;
    }

    bool rig::refines(const size_t index) const {
        return (index < this->sensors.size()) && this->sensors[index].refined && (static_cast<int>(index) != this->reference_index);
    }

    void rig::set_refined(const size_t index, const bool refined) {
        ASSERT(index < this->sensors.size(), "Rig sensor index out of bounds.");
        this->sensors[index].refined = refined;
    }

    bool rig::metric() const {
        for (size_t i = 0; i < this->sensors.size(); ++i) {
            if ((this->sensors[i].kind == sensor_kind::camera) && (static_cast<int>(i) != this->reference_index) && (this->sensors[i].declared_distance >= rig::metric_baseline_minimum)) {
                return true;
            }
        }
        return false;
    }

    optimisation::vertex rig::extrinsic_vertex(const size_t index) const {
        optimisation::vertex node{ optimisation::vertices::extrinsic() };
        double parameters[optimisation::vertices::extrinsic::parameter_count];
        optimisation::vertices::extrinsic::to_parameters(this->reference_from_sensor(index), parameters);
        node.set_parameters(&parameters[0], optimisation::vertices::extrinsic::parameter_count);
        node.set_fixed(!this->refines(index));
        return node;
    }

    void rig::read_extrinsic_vertex(const size_t index, const optimisation::vertex& node) {
        if (!this->refines(index)) {
            return;
        }
        this->set_reference_from_sensor(index, optimisation::vertices::extrinsic::from_parameters(node.get_parameters()));
        const math::se3<double>& refined = this->sensors[index].body_from_sensor;
        const math::matrix<double, 4, 1> quaternion = refined.rotation().get_quaternion();
        core::logger::log(core::logger::level::info, "Extrinsic of sensor %d refined to [%.6f %.6f %.6f] [%.6f %.6f %.6f %.6f].", this->sensors[index].id, refined.translation()[0], refined.translation()[1], refined.translation()[2], quaternion[1], quaternion[2], quaternion[3], quaternion[0]);
    }
}
