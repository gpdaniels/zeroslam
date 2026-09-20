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
#ifndef ZEROSLAM_OPTIMISATION_VERTICES_SIMILARITY_HPP
#define ZEROSLAM_OPTIMISATION_VERTICES_SIMILARITY_HPP

namespace optimisation::vertices {
    class similarity final {
    public:
        constexpr static const char* name = "similarity";
        constexpr static const int parameter_count = 8;
        constexpr static const int local_count = 7;

    public:
        void plus(double* const parameters, const double* const delta) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_VERTICES_SIMILARITY_HPP
