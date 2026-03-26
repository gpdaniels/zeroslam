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

#include "frame.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>

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
        frame::frame frame;
    }
    {
        const matrix::matrix<double, 3, 3> intrinsics = { { { 1.0, 0.0, 0.5 }, { 0.0, 1.0, 0.5 }, { 0.0, 0.0, 1.0 } } };
        camera::pinhole camera(std::vector<double>{ intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2] }.data(), 4);
        image::image image(256, 256);
        for (size_t i = 0; i < image.get_rows(); ++i) {
            for (size_t j = 0; j < image.get_cols(); ++j) {
                if ((i < 64) || (i >= image.get_rows() - 64) || (j < 64) || (j >= image.get_cols() - 64)) {
                    image.get_data()[i * image.get_cols() + j] = 0;
                }
                else {
                    image.get_data()[i * image.get_cols() + j] = 255;
                }
            }
        }
        frame::frame frame_0(camera, image);
        REQUIRE(frame_0.id == 0);
    }

    return EXIT_SUCCESS;
}