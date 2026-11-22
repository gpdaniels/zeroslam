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

#include "sensor/depth.hpp"

#include "sensor/camera.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    constexpr static const size_t rows = 6;
    constexpr static const size_t columns = 8;

    {
        // An empty depth samples nothing.
        const sensor::depth none;
        REQUIRE(none.empty());
        double value = 0.0;
        REQUIRE(!none.sample(1.5, 1.5, value));
    }

    {
        // Values outside the range, zeros, negatives and non-finite values are missing; the pixel a centre position falls in
        // is its floor.
        std::vector<float> metres(rows * columns, 2.0f);
        metres[(1 * columns) + 1] = 0.0f;
        metres[(1 * columns) + 6] = -1.0f;
        metres[(4 * columns) + 1] = std::numeric_limits<float>::quiet_NaN();
        metres[(4 * columns) + 6] = 9.0f;
        sensor::depth::settings settings;
        settings.minimum = 0.5;
        settings.maximum = 5.0;
        const sensor::depth image(rows, columns, metres.data(), settings);
        REQUIRE(!image.empty());
        REQUIRE((image.get_rows() == rows) && (image.get_cols() == columns));
        double value = 0.0;
        REQUIRE(image.sample(3.5, 2.5, value) && (value == 2.0));
        REQUIRE(image.sample(3.99, 2.01, value) && (value == 2.0));
        REQUIRE(!image.sample(1.5, 1.5, value));
        REQUIRE(!image.sample(6.5, 1.5, value));
        REQUIRE(!image.sample(1.5, 4.5, value));
        REQUIRE(!image.sample(6.5, 4.5, value));
        REQUIRE(image.at(4, 6) == 0.0f);
        REQUIRE(!image.sample(-0.1, 2.5, value));
        REQUIRE(!image.sample(8.0, 2.5, value));
        REQUIRE(!image.sample(3.5, 6.0, value));
        // A pixel beside a missing one still measures: the missing neighbour says nothing.
        REQUIRE(image.sample(2.5, 1.5, value) && (value == 2.0));
    }

    {
        // A pixel next to a depth that differs by more than the discontinuity fraction is on an occluding edge; a gradual
        // slope is not.
        std::vector<float> metres(rows * columns, 0.0f);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t column = 0; column < columns; ++column) {
                metres[(row * columns) + column] = (column < 4) ? (1.0f + (0.01f * static_cast<float>(column))) : 3.0f;
            }
        }
        const sensor::depth image(rows, columns, metres.data(), sensor::depth::settings());
        double value = 0.0;
        REQUIRE(image.sample(1.5, 2.5, value) && (std::abs(value - 1.01) < 1.0e-6));
        REQUIRE(image.sample(2.5, 2.5, value));
        REQUIRE(!image.sample(3.5, 2.5, value));
        REQUIRE(!image.sample(4.5, 2.5, value));
        REQUIRE(image.sample(5.5, 2.5, value) && (value == 3.0));
    }

    {
        // Sixteen bit values scale into metres, zero is missing.
        std::vector<unsigned short> values(rows * columns, static_cast<unsigned short>(5000));
        values[(2 * columns) + 2] = 0;
        const sensor::depth image(rows, columns, values.data(), 1.0 / 5000.0, sensor::depth::settings());
        double value = 0.0;
        REQUIRE(image.sample(5.5, 3.5, value) && (std::abs(value - 1.0) < 1.0e-6));
        REQUIRE(!image.sample(2.5, 2.5, value));
        REQUIRE(image.at(2, 2) == 0.0f);
    }

    {
        // The noise model, its floor, and the inverse depth's deviation sigma(d) / d^2.
        sensor::depth::settings settings;
        settings.noise.constant = 0.001;
        settings.noise.linear = 0.002;
        settings.noise.quadratic = 0.003;
        const sensor::depth image(1, 1, std::vector<float>{ 1.0f }.data(), settings);
        REQUIRE(std::abs(image.sigma(2.0) - (0.001 + 0.004 + 0.012)) < 1.0e-12);
        REQUIRE(std::abs(image.inverse_sigma(2.0) - (0.017 / 4.0)) < 1.0e-12);
        sensor::depth::settings silent;
        silent.noise.quadratic = 0.0;
        const sensor::depth floored(1, 1, std::vector<float>{ 1.0f }.data(), silent);
        REQUIRE(floored.sigma(1.0) == sensor::depth::sigma_minimum);
        // Releasing frees the image and keeps the settings.
        sensor::depth released = image;
        released.release();
        REQUIRE(released.empty());
        REQUIRE(released.get_settings().noise.linear == 0.002);
    }

    {
        // Back projection scales the camera's ray: an axial depth is the point's z, a radial depth its distance from the
        // centre, and each comes back as what the point measures.
        const double parameters[sensor::camera::pinhole_radial_tangential<double>::parameter_count] = { 500.0, 480.0, 320.0, 240.0, -0.2, 0.05, 0.001, -0.002, 0.0, 0.0, 0.0, 0.0 };
        const sensor::model camera(sensor::camera::pinhole_radial_tangential<double>(&parameters[0], sensor::camera::pinhole_radial_tangential<double>::parameter_count));
        const double pixel[2] = { 100.5, 400.5 };
        for (const sensor::depth::measure_kind measure : { sensor::depth::measure_kind::axial, sensor::depth::measure_kind::radial }) {
            double point[3] = { 0.0, 0.0, 0.0 };
            REQUIRE(sensor::depth::back_project(camera, measure, pixel, 2.5, point));
            REQUIRE(std::abs(sensor::depth::measured(measure, point) - 2.5) < 1.0e-9);
            double projected[2] = { 0.0, 0.0 };
            REQUIRE(camera.project(&point[0], &projected[0]));
            REQUIRE(std::abs(projected[0] - pixel[0]) < 1.0e-6);
            REQUIRE(std::abs(projected[1] - pixel[1]) < 1.0e-6);
        }
        double axial[3];
        double radial[3];
        REQUIRE(sensor::depth::back_project(camera, sensor::depth::measure_kind::axial, pixel, 2.5, axial));
        REQUIRE(sensor::depth::back_project(camera, sensor::depth::measure_kind::radial, pixel, 2.5, radial));
        REQUIRE(std::abs(axial[2] - 2.5) < 1.0e-12);
        REQUIRE(radial[2] < axial[2]);
        double unused[3];
        REQUIRE(!sensor::depth::back_project(camera, sensor::depth::measure_kind::axial, pixel, 0.0, unused));
    }

    return EXIT_SUCCESS;
}
