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
#ifndef ZEROSLAM_TOOLS_ANDROID_RENDERER_HPP
#define ZEROSLAM_TOOLS_ANDROID_RENDERER_HPP

#include "session.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <GLES2/gl2.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mobile {
    // A rectangle of the window in pixels, from the top left.
    struct rectangle final {
        float x = 0.0f;
        float y = 0.0f;
        float width = 0.0f;
        float height = 0.0f;

        bool contains(const float px, const float py) const {
            return (px >= this->x) && (py >= this->y) && (px < (this->x + this->width)) && (py < (this->y + this->height));
        }
    };

    enum class button {
        none,
        reset,
        save,
        record
    };

    // Draws the window with opengl es 2: the frame the system processed last, turned upright, with the landmarks in front
    // of it projected through its pose; a view of the map from above and behind the camera, which a drag turns and a pinch
    // zooms; and a panel of the system's state with the buttons.
    class renderer final {
    public:
        renderer() = default;
        ~renderer() = default;

        renderer(const renderer&) = delete;
        renderer& operator=(const renderer&) = delete;

        // With a current context, and again whenever the context is made anew.
        bool create(std::string& error);
        void destroy();

        // The window's size and density (pixels a density independent pixel), and the clockwise turn in degrees that
        // shows the camera's image upright on the screen as it is held.
        void resize(int width, int height, float density, int image_rotation);

        void draw(const snapshot& state, const std::string& status);

        // The button at a point of the window, if any.
        button hit(float x, float y) const;
        // Whether a point is in the map view, where touches turn and zoom it.
        bool in_map(float x, float y) const;
        // Turns the map view by a drag of so many pixels, and zooms it by a pinch's ratio of spreads.
        void orbit(float dx, float dy);
        void zoom(float ratio);

    private:
        struct vertex final {
            float x, y, z;
            float r, g, b, a;
        };

        void layout();
        void draw_image(const snapshot& state);
        void draw_projected_landmarks(const snapshot& state);
        void draw_map(const snapshot& state);
        void draw_panel(const snapshot& state, const std::string& status);
        void draw_text(const std::string& text, float x, float y, float scale, const float colour[4]);
        void draw_rectangle(const rectangle& area, const float colour[4]);
        void flush(GLenum mode, const float matrix[16], float point_size);
        void upload_image(const snapshot& state);

        GLuint colour_program = 0;
        GLint colour_matrix = -1;
        GLint colour_point_size = -1;
        GLuint texture_program = 0;
        GLint texture_matrix = -1;
        GLint texture_tint = -1;
        GLuint image_texture = 0;
        GLuint font_texture = 0;

        int width = 0;
        int height = 0;
        float density = 1.0f;
        int image_rotation = 0;
        rectangle image_area;
        rectangle map_area;
        rectangle panel_area;
        rectangle reset_button;
        rectangle save_button;
        rectangle record_button;

        int image_width = 0;
        int image_height = 0;
        unsigned long long image_sequence = 0;
        bool image_valid = false;

        // The map view's turn about the vertical (yaw) and its tilt, in radians, and its distance as a multiple of the
        // map's size.
        float yaw = 0.0f;
        float pitch = 0.6f;
        float distance = 1.5f;

        std::vector<vertex> vertices;
        std::vector<float> text_vertices;
    };
}

#endif // ZEROSLAM_TOOLS_ANDROID_RENDERER_HPP
