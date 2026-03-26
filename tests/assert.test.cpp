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

#include "assert.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdlib>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        ASSERT(true, "");
    }

    {
        ASSERT(1 == 1, "Short message.");
    }

    {
        ASSERT(2 != 3, "A much longer message string");
    }

    return EXIT_SUCCESS;
}