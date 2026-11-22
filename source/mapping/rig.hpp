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
#ifndef ZEROSLAM_MAPPING_RIG_HPP
#define ZEROSLAM_MAPPING_RIG_HPP

#include "math/lie.hpp"
#include "math/matrix.hpp"
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

namespace mapping {
    // The sensors of a rig and where each sits on it. Three frames are kept apart:
    // - the body, the rig's own frame that the caller gives every extrinsic in (the scene's ego, or an imu's frame);
    // - each sensor's frame, posed on the body by its extrinsic, sensor to body (p_body = R * p_sensor + t);
    // - the reference, the frame the map's frames are posed in: the primary camera's, the first camera declared. The gauge
    //   holds its extrinsic, so the map's poses are the primary camera's and a monocular rig is the monocular system.
    // Every sensor is declared the same way whatever it measures (a camera, an imu, wheel odometry, a depth sensor tied to
    // a camera), and gets its extrinsic in the reference frame (reference_from_sensor) and, for an adjustment, a vertex
    // holding it (extrinsic_vertex), free when the sensor's calibration is refined.
    class rig final {
    public:
        enum class sensor_kind {
            camera,
            imu,
            odometry,
            depth,
            other
        };

        struct sensor final {
            int id;
            sensor_kind kind;
            math::se3<double> body_from_sensor;
            // Whether adjustments refine the extrinsic. The reference's never is: it is the gauge.
            bool refined;
            // The sensor whose measurements this one's belong to, a camera for a depth sensor registered to it, or -1.
            int attached_to;
            // The distance from the reference the sensor was declared at. The map's scale is a metric rig's baseline, which
            // reprojections do not measure once an extrinsic is free, so a refined extrinsic keeps this length.
            double declared_distance;
        };

        // A baseline shorter than this, in metres, gives no metric scale (a camera declared at the reference's place).
        constexpr static const double metric_baseline_minimum = 1.0e-3;

    private:
        std::vector<sensor> sensors;
        int reference_index = -1;

    public:
        // Adds a sensor and returns its index. The first camera becomes the reference.
        size_t add(const int id, const sensor_kind kind, const math::se3<double>& body_from_sensor, const bool refined, const int attached_to = -1);

        void clear();

        size_t size() const;

        const sensor& at(const size_t index) const;

        // The index of the sensor with the id, false when there is none.
        bool find(const int id, size_t& index) const;

        // The index of the reference sensor, -1 while no camera is declared.
        int reference() const;

        // The cameras in the order they were declared, the reference first: camera c of the map's observations is camera(c).
        size_t camera_count() const;
        size_t camera(const size_t camera_index) const;

        // The sensor's pose in the reference frame, reference from sensor; the identity for the reference itself.
        math::se3<double> reference_from_sensor(const size_t index) const;

        // Moves the sensor to a pose in the reference frame, keeping the reference where it is on the body.
        void set_reference_from_sensor(const size_t index, const math::se3<double>& reference_from_sensor_value);

        // Whether an adjustment refines the sensor's extrinsic: asked for, and not the reference.
        bool refines(const size_t index) const;

        // Asks for the sensor's extrinsic to be refined, or held.
        void set_refined(const size_t index, const bool refined);

        // Whether a camera sits at a distance from the reference that gives the map a metric scale.
        bool metric() const;

        // A vertex holding the sensor's extrinsic in the reference frame, fixed unless it is refined.
        optimisation::vertex extrinsic_vertex(const size_t index) const;

        // Takes the extrinsic an adjustment left in the sensor's vertex.
        void read_extrinsic_vertex(const size_t index, const optimisation::vertex& node);
    };
}

#endif // ZEROSLAM_MAPPING_RIG_HPP
