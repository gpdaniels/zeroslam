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
#ifndef ZEROSLAM_TOOLS_GUI_MAP_FILE_HPP
#define ZEROSLAM_TOOLS_GUI_MAP_FILE_HPP

#include "dataset.hpp"
#include "file.hpp"
#include "render_snapshot.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace map_file {

    constexpr static const char magic[8] = { 'Z', 'S', 'L', 'A', 'M', 'M', 'A', 'P' };
    constexpr static const unsigned int current_version = 4;

    namespace detail {
        template <typename type>
        inline void write_value(std::vector<unsigned char>& out, const type& value) {
            const unsigned char* const bytes = reinterpret_cast<const unsigned char*>(&value);
            out.insert(out.end(), bytes, bytes + sizeof(type));
        }

        template <typename type>
        inline bool read_value(const std::vector<unsigned char>& in, std::size_t& cursor, type& value) {
            if ((cursor + sizeof(type)) > in.size()) {
                return false;
            }
            std::memcpy(&value, &in[cursor], sizeof(type));
            cursor += sizeof(type);
            return true;
        }

        template <typename map_type>
        inline std::vector<int> sorted_ids(const map_type& values) {
            std::vector<int> ids;
            ids.reserve(values.size());
            for (const auto& entry : values) {
                ids.push_back(entry.first);
            }
            std::sort(ids.begin(), ids.end());
            return ids;
        }
    }

    inline bool save(const std::string& path, const gui::render_snapshot& snapshot, std::string& error) {
        std::vector<unsigned char> out;
        out.insert(out.end(), &magic[0], &magic[0] + sizeof(magic));
        detail::write_value(out, current_version);
        detail::write_value<unsigned long long>(out, snapshot.frames.size());
        detail::write_value<unsigned long long>(out, snapshot.landmarks.size());
        detail::write_value<unsigned long long>(out, snapshot.lines.size());
        detail::write_value<unsigned long long>(out, snapshot.edges.size());
        detail::write_value<unsigned long long>(out, snapshot.keyframes.size());
        detail::write_value<int>(out, snapshot.processed_frame_count);

        for (const int id : detail::sorted_ids(snapshot.frames)) {
            const gui::render_snapshot::frame& frame = snapshot.frames.at(id);
            detail::write_value<int>(out, id);
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    detail::write_value<double>(out, frame.rotation[row][column]);
                }
            }
            for (int row = 0; row < 3; ++row) {
                detail::write_value<double>(out, frame.translation[row]);
            }
            for (std::size_t i = 0; i < gui::render_snapshot::camera_parameter_count; ++i) {
                detail::write_value<double>(out, frame.camera_parameters[i]);
            }
            detail::write_value<int>(out, frame.image_width);
            detail::write_value<int>(out, frame.image_height);
            detail::write_value<long long>(out, frame.timestamp_nanoseconds);
        }
        for (const int id : detail::sorted_ids(snapshot.landmarks)) {
            const gui::render_snapshot::landmark& landmark = snapshot.landmarks.at(id);
            detail::write_value<int>(out, id);
            for (int i = 0; i < 3; ++i) {
                detail::write_value<double>(out, landmark.location[i]);
            }
            for (int i = 0; i < 3; ++i) {
                detail::write_value<double>(out, landmark.colour[i]);
            }
        }
        for (const int id : detail::sorted_ids(snapshot.lines)) {
            const gui::render_snapshot::line& line = snapshot.lines.at(id);
            detail::write_value<int>(out, id);
            for (int i = 0; i < 3; ++i) {
                detail::write_value<double>(out, line.a[i]);
            }
            for (int i = 0; i < 3; ++i) {
                detail::write_value<double>(out, line.b[i]);
            }
        }
        for (const gui::render_snapshot::edge& edge : snapshot.edges) {
            detail::write_value<int>(out, edge.frame_a);
            detail::write_value<int>(out, edge.frame_b);
            detail::write_value<int>(out, edge.kind);
            detail::write_value<int>(out, edge.weight);
        }
        {
            std::vector<int> keyframes(snapshot.keyframes.begin(), snapshot.keyframes.end());
            std::sort(keyframes.begin(), keyframes.end());
            for (const int id : keyframes) {
                detail::write_value<int>(out, id);
            }
        }
        gtl::file handle(path.c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated);
        if (!handle.is_open()) {
            error = "the map file could not be opened for writing";
            return false;
        }
        if (!dataset::write_all(handle, reinterpret_cast<const char*>(out.data()), out.size())) {
            error = "the map file could not be written";
            return false;
        }
        return true;
    }

    inline bool load(const std::string& path, gui::render_snapshot& snapshot, std::string& error) {
        std::vector<unsigned char> in;
        if (!dataset::read_file(path, in)) {
            error = "the map file could not be read";
            return false;
        }

        std::size_t cursor = 0;
        if ((in.size() < sizeof(magic)) || (std::memcmp(in.data(), &magic[0], sizeof(magic)) != 0)) {
            error = "the file is not a zeroslam map";
            return false;
        }
        cursor += sizeof(magic);
        unsigned int version = 0;
        if (!detail::read_value(in, cursor, version)) {
            error = "the map file is truncated";
            return false;
        }
        if (version != current_version) {
            error = "the map file is version " + std::to_string(version) + ", this build reads version " + std::to_string(current_version);
            return false;
        }

        unsigned long long frame_count = 0;
        unsigned long long landmark_count = 0;
        unsigned long long line_count = 0;
        unsigned long long edge_count = 0;
        unsigned long long keyframe_count = 0;
        int processed = 0;
        if (!detail::read_value(in, cursor, frame_count) || !detail::read_value(in, cursor, landmark_count) || !detail::read_value(in, cursor, line_count) || !detail::read_value(in, cursor, edge_count) || !detail::read_value(in, cursor, keyframe_count) || !detail::read_value(in, cursor, processed)) {
            error = "the map file header is truncated";
            return false;
        }

        gui::render_snapshot loaded;
        loaded.processed_frame_count = processed;
        const auto fail = [&]() {
            error = "the map file is truncated";
            return false;
        };
        for (unsigned long long i = 0; i < frame_count; ++i) {
            int id = 0;
            gui::render_snapshot::frame frame;
            if (!detail::read_value(in, cursor, id)) {
                return fail();
            }
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    if (!detail::read_value(in, cursor, frame.rotation[row][column])) {
                        return fail();
                    }
                }
            }
            for (int row = 0; row < 3; ++row) {
                if (!detail::read_value(in, cursor, frame.translation[row])) {
                    return fail();
                }
            }
            for (std::size_t index = 0; index < gui::render_snapshot::camera_parameter_count; ++index) {
                if (!detail::read_value(in, cursor, frame.camera_parameters[index])) {
                    return fail();
                }
            }
            if (!detail::read_value(in, cursor, frame.image_width) || !detail::read_value(in, cursor, frame.image_height) || !detail::read_value(in, cursor, frame.timestamp_nanoseconds)) {
                return fail();
            }
            loaded.frames[id] = frame;
        }
        for (unsigned long long i = 0; i < landmark_count; ++i) {
            int id = 0;
            gui::render_snapshot::landmark landmark = {};
            if (!detail::read_value(in, cursor, id)) {
                return fail();
            }
            for (int index = 0; index < 3; ++index) {
                if (!detail::read_value(in, cursor, landmark.location[index])) {
                    return fail();
                }
            }
            for (int index = 0; index < 3; ++index) {
                if (!detail::read_value(in, cursor, landmark.colour[index])) {
                    return fail();
                }
            }
            loaded.landmarks[id] = landmark;
        }
        for (unsigned long long i = 0; i < line_count; ++i) {
            int id = 0;
            gui::render_snapshot::line line = {};
            if (!detail::read_value(in, cursor, id)) {
                return fail();
            }
            for (int index = 0; index < 3; ++index) {
                if (!detail::read_value(in, cursor, line.a[index])) {
                    return fail();
                }
            }
            for (int index = 0; index < 3; ++index) {
                if (!detail::read_value(in, cursor, line.b[index])) {
                    return fail();
                }
            }
            loaded.lines[id] = line;
        }
        for (unsigned long long i = 0; i < edge_count; ++i) {
            gui::render_snapshot::edge edge = {};
            if (!detail::read_value(in, cursor, edge.frame_a) || !detail::read_value(in, cursor, edge.frame_b) || !detail::read_value(in, cursor, edge.kind) || !detail::read_value(in, cursor, edge.weight)) {
                return fail();
            }
            loaded.edges.push_back(edge);
        }
        for (unsigned long long i = 0; i < keyframe_count; ++i) {
            int id = 0;
            if (!detail::read_value(in, cursor, id)) {
                return fail();
            }
            loaded.keyframes.insert(id);
        }
        snapshot = std::move(loaded);
        return true;
    }
}

#endif // ZEROSLAM_TOOLS_GUI_MAP_FILE_HPP
