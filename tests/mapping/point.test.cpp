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

#include "mapping/point.hpp"

#include "feature/descriptor/binary.hpp"
#include "math/lie.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

    {
        mapping::point p;
    }
    {
        mapping::point p0(0, { { 1, 2, 3 } }, { { 4, 5, 6 } });
        REQUIRE(p0.id == 0);
        mapping::point p1(1, { { 4, 5, 6 } }, { { 7, 8, 9 } });
        REQUIRE(p1.id == 1);
    }

    {
        mapping::point p(2, { { 1.0, -0.5, 4.0 } }, { { 0, 0, 0 } });
        const math::matrix<double, 3, 3> rotation = math::so3<double>::exp({ { 0.1, -0.2, 0.05 } }).get_matrix();
        const math::matrix<double, 3, 1> translation({ 0.3, 0.1, -0.4 });
        REQUIRE(p.anchor(rotation, translation));
        REQUIRE(p.inverse_depth);
        REQUIRE(!p.at_infinity());
        const math::matrix<double, 3, 1> before = p.location;
        p.update_location_from_inverse_depth();
        REQUIRE(std::abs((p.location - before).get_length_squared()) < 1.0e-18);
        mapping::point behind(3, { { 0.0, 0.0, -1.0 } }, { { 0, 0, 0 } });
        REQUIRE(!behind.anchor(math::matrix<double, 3, 3>::identity(), math::matrix<double, 3, 1>::zero()));
        REQUIRE(!behind.inverse_depth);
        mapping::point distant(4, math::matrix<double, 3, 1>::zero(), { { 0, 0, 0 } });
        distant.anchor_at_infinity(math::matrix<double, 3, 3>::identity(), math::matrix<double, 3, 1>({ 1.0, 2.0, 3.0 }), { { 0.5, -0.25, 1.0 } });
        REQUIRE(distant.at_infinity());
        const math::matrix<double, 3, 1> centre = -(math::transpose(math::matrix<double, 3, 3>::identity()) * math::matrix<double, 3, 1>({ 1.0, 2.0, 3.0 }));
        const math::matrix<double, 3, 1> offset = distant.location - centre;
        REQUIRE(std::abs(offset[0] / offset[2] - 0.5) < 1.0e-9);
        REQUIRE(std::abs(offset[1] / offset[2] + 0.25) < 1.0e-9);
        REQUIRE(offset[2] > 1.0e3);
    }
    {
        mapping::point p(7, { { 0.0, 0.0, 4.0 } }, { { 1.0, 1.0, 1.0 } });
        math::matrix<double, 0, 0> jacobian(2, 3, math::matrix<double, 2, 3>{ { { 100.0, 0.0, 10.0 }, { 0.0, 100.0, 0.0 } } }.data());
        REQUIRE(p.uncertainty == mapping::point::uncertainty_kind::unknown);
        math::matrix<double, 2, 2> w = p.observation_information(jacobian, 2.0);
        REQUIRE(std::abs(w[0][0] - 0.25) < 1.0e-12);
        REQUIRE(std::abs(w[1][1] - 0.25) < 1.0e-12);
        REQUIRE(std::abs(w[0][1]) < 1.0e-12);

        math::matrix<double, 3, 3> information = math::matrix<double, 3, 3>::zero();
        information[0][0] = 1.0e4;
        information[1][1] = 1.0e4;
        information[2][2] = 1.0;
        p.set_information(information, { { 0.0, 0.0, 1.0 } });
        REQUIRE(p.uncertainty == mapping::point::uncertainty_kind::estimated);
        REQUIRE(std::abs(p.covariance[2][2] - 1.0) < 1.0e-12);
        w = p.observation_information(jacobian, 1.0);
        REQUIRE(std::abs(w[0][0] - 1.0) < 1.0e-12);
        REQUIRE(std::abs(w[1][1] - 1.0) < 1.0e-12);
        REQUIRE(std::abs(w[0][1]) < 1.0e-12);

        p.set_information(math::matrix<double, 3, 3>::zero(), { { 0.0, 0.0, 1.0 } });
        REQUIRE(p.uncertainty == mapping::point::uncertainty_kind::estimated);
        REQUIRE(p.anchor(math::matrix<double, 3, 3>::identity(), math::matrix<double, 3, 1>::zero()));
        p.set_information(math::matrix<double, 3, 3>::zero(), { { 0.0, 0.0, 1.0 } });
        REQUIRE(p.uncertainty == mapping::point::uncertainty_kind::unbounded);
        w = p.observation_information(jacobian, 1.0);
        REQUIRE(std::abs(w[0][0]) < 1.0e-12);
        REQUIRE(std::abs(w[1][1] - 1.0) < 1.0e-12);
        REQUIRE(std::abs(w[0][1]) < 1.0e-12);

        mapping::point distant(8, { { 0.0, 0.0, 1.0 } }, { { 1.0, 1.0, 1.0 } });
        distant.anchor_at_infinity(math::matrix<double, 3, 3>::identity(), math::matrix<double, 3, 1>::zero(), { { 0.5, -0.25, 1.0 } });
        REQUIRE(distant.uncertainty == mapping::point::uncertainty_kind::unbounded);
        REQUIRE(std::abs(distant.depth_direction[2] - 1.0) < 1.0e-12);
    }

    // The running medoid matches the medoid recomputed from the whole history, last index winning ties, through and past a full history.
    {
        constexpr static const size_t size_bytes = feature::descriptor::stored::size_bytes;
        const auto brute_medoid = [](const std::vector<std::array<unsigned char, size_bytes>>& history) {
            size_t medoid = history.size() - 1;
            unsigned int medoid_sum = 0xFFFFFFFFu;
            for (size_t i = 0; i < history.size(); ++i) {
                unsigned int sum = 0;
                for (size_t j = 0; j < history.size(); ++j) {
                    for (size_t index = 0; index < size_bytes; ++index) {
                        unsigned char difference = static_cast<unsigned char>(history[i][index] ^ history[j][index]);
                        while (difference != 0) {
                            sum += (difference & 1u);
                            difference = static_cast<unsigned char>(difference >> 1);
                        }
                    }
                }
                if (sum <= medoid_sum) {
                    medoid_sum = sum;
                    medoid = i;
                }
            }
            return medoid;
        };
        unsigned int state = 12345u;
        const auto next = [&state]() {
            state = (state * 1103515245u) + 12345u;
            return static_cast<unsigned char>(state >> 16);
        };
        mapping::point p(9, { { 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 0.0 } });
        unsigned char base[size_bytes] = {};
        for (size_t index = 0; index < size_bytes; ++index) {
            base[index] = next();
        }
        for (size_t insert = 0; insert < 3 * mapping::point::descriptor_history_maximum; ++insert) {
            unsigned char bytes[size_bytes] = {};
            for (size_t index = 0; index < size_bytes; ++index) {
                // Mostly the base descriptor with a few flipped bits, and every seventh a duplicate so ties occur.
                bytes[index] = ((insert % 7) == 3) ? base[index] : static_cast<unsigned char>(base[index] ^ (next() & next() & next()));
            }
            p.add_descriptor(&bytes[0]);
            REQUIRE(p.descriptor_history.size() == ((insert + 1 < mapping::point::descriptor_history_maximum) ? insert + 1 : mapping::point::descriptor_history_maximum));
            REQUIRE(p.descriptor_distance_sums.size() == p.descriptor_history.size());
            const size_t medoid = brute_medoid(p.descriptor_history);
            for (size_t index = 0; index < size_bytes; ++index) {
                REQUIRE(p.descriptor[index] == p.descriptor_history[medoid][index]);
            }
        }
        // A history filled without the running sums is resynchronised on the next insert.
        mapping::point copied(10, { { 0.0, 0.0, 1.0 } }, { { 0.0, 0.0, 0.0 } });
        copied.descriptor_history = p.descriptor_history;
        copied.add_descriptor(&base[0]);
        REQUIRE(copied.descriptor_distance_sums.size() == copied.descriptor_history.size());
        const size_t copied_medoid = brute_medoid(copied.descriptor_history);
        for (size_t index = 0; index < size_bytes; ++index) {
            REQUIRE(copied.descriptor[index] == copied.descriptor_history[copied_medoid][index]);
        }
    }

    return EXIT_SUCCESS;
}
