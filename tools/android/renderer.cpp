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

#include "renderer.hpp"

#include "font.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mobile {
    namespace {
        constexpr static const char colour_vertex_source[] = R"(
            uniform mat4 matrix;
            uniform float point_size;
            attribute vec3 position;
            attribute vec4 colour;
            varying vec4 shade;
            void main() {
                gl_Position = matrix * vec4(position, 1.0);
                gl_PointSize = point_size;
                shade = colour;
            }
        )";
        constexpr static const char colour_fragment_source[] = R"(
            precision mediump float;
            varying vec4 shade;
            void main() {
                gl_FragColor = shade;
            }
        )";
        constexpr static const char texture_vertex_source[] = R"(
            uniform mat4 matrix;
            attribute vec2 position;
            attribute vec2 coordinate;
            varying vec2 sample;
            void main() {
                gl_Position = matrix * vec4(position, 0.0, 1.0);
                sample = coordinate;
            }
        )";
        // The image and the font are both luminance textures: the image is drawn as it is and the font as coverage.
        constexpr static const char texture_fragment_source[] = R"(
            precision mediump float;
            uniform sampler2D image;
            uniform vec4 tint;
            varying vec2 sample;
            void main() {
                float value = texture2D(image, sample).r;
                gl_FragColor = (tint.a < 0.0) ? vec4(value, value, value, 1.0) : vec4(tint.rgb, tint.a * value);
            }
        )";

        constexpr static const GLuint position_attribute = 0;
        constexpr static const GLuint colour_attribute = 1;
        constexpr static const int font_columns = 16;
        constexpr static const int font_rows = 8;

        constexpr static const float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        constexpr static const float grey[4] = { 0.65f, 0.65f, 0.68f, 1.0f };
        constexpr static const float green[4] = { 0.30f, 0.85f, 0.40f, 1.0f };
        constexpr static const float amber[4] = { 1.00f, 0.70f, 0.20f, 1.0f };
        constexpr static const float red[4] = { 0.95f, 0.30f, 0.25f, 1.0f };
        constexpr static const float panel_colour[4] = { 0.09f, 0.10f, 0.12f, 1.0f };
        constexpr static const float button_colour[4] = { 0.20f, 0.22f, 0.27f, 1.0f };
        constexpr static const float recording_colour[4] = { 0.55f, 0.12f, 0.12f, 1.0f };

        GLuint compile(const GLenum type, const char* const source, std::string& error) {
            const GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &source, nullptr);
            glCompileShader(shader);
            GLint compiled = GL_FALSE;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
            if (compiled != GL_TRUE) {
                char log[512] = { 0 };
                glGetShaderInfoLog(shader, sizeof(log), nullptr, &log[0]);
                error = std::string("Failed to compile a shader: ") + &log[0];
                glDeleteShader(shader);
                return 0;
            }
            return shader;
        }

        GLuint link(const char* const vertex_source, const char* const fragment_source, const char* const second_attribute, std::string& error) {
            const GLuint vertex_shader = compile(GL_VERTEX_SHADER, vertex_source, error);
            const GLuint fragment_shader = compile(GL_FRAGMENT_SHADER, fragment_source, error);
            if ((vertex_shader == 0) || (fragment_shader == 0)) {
                return 0;
            }
            const GLuint program = glCreateProgram();
            glAttachShader(program, vertex_shader);
            glAttachShader(program, fragment_shader);
            glBindAttribLocation(program, position_attribute, "position");
            glBindAttribLocation(program, colour_attribute, second_attribute);
            glLinkProgram(program);
            glDeleteShader(vertex_shader);
            glDeleteShader(fragment_shader);
            GLint linked = GL_FALSE;
            glGetProgramiv(program, GL_LINK_STATUS, &linked);
            if (linked != GL_TRUE) {
                error = "Failed to link a shader program";
                glDeleteProgram(program);
                return 0;
            }
            return program;
        }

        // Column major, as opengl takes them.
        void multiply(const float a[16], const float b[16], float result[16]) {
            for (int column = 0; column < 4; ++column) {
                for (int row = 0; row < 4; ++row) {
                    float sum = 0.0f;
                    for (int k = 0; k < 4; ++k) {
                        sum += a[(k * 4) + row] * b[(column * 4) + k];
                    }
                    result[(column * 4) + row] = sum;
                }
            }
        }

        // Window pixels from the top left to clip space.
        void pixel_matrix(const float width, const float height, float matrix[16]) {
            const float values[16] = { 2.0f / width, 0.0f, 0.0f, 0.0f, 0.0f, -2.0f / height, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f };
            std::copy(&values[0], &values[16], matrix);
        }

        void perspective_matrix(const float vertical_fov, const float aspect, const float near_plane, const float far_plane, float matrix[16]) {
            const float f = 1.0f / std::tan(0.5f * vertical_fov);
            const float values[16] = { f / aspect, 0.0f, 0.0f, 0.0f, 0.0f, f, 0.0f, 0.0f, 0.0f, 0.0f, (far_plane + near_plane) / (near_plane - far_plane), -1.0f, 0.0f, 0.0f, (2.0f * far_plane * near_plane) / (near_plane - far_plane), 0.0f };
            std::copy(&values[0], &values[16], matrix);
        }

        void look_at_matrix(const float eye[3], const float target[3], const float up[3], float matrix[16]) {
            float forward[3] = { target[0] - eye[0], target[1] - eye[1], target[2] - eye[2] };
            const float forward_length = std::sqrt((forward[0] * forward[0]) + (forward[1] * forward[1]) + (forward[2] * forward[2]));
            for (float& value : forward) {
                value /= forward_length;
            }
            float side[3] = { (forward[1] * up[2]) - (forward[2] * up[1]), (forward[2] * up[0]) - (forward[0] * up[2]), (forward[0] * up[1]) - (forward[1] * up[0]) };
            const float side_length = std::sqrt((side[0] * side[0]) + (side[1] * side[1]) + (side[2] * side[2]));
            for (float& value : side) {
                value /= side_length;
            }
            const float upward[3] = { (side[1] * forward[2]) - (side[2] * forward[1]), (side[2] * forward[0]) - (side[0] * forward[2]), (side[0] * forward[1]) - (side[1] * forward[0]) };
            const float values[16] = {
                side[0],
                upward[0],
                -forward[0],
                0.0f,
                side[1],
                upward[1],
                -forward[1],
                0.0f,
                side[2],
                upward[2],
                -forward[2],
                0.0f,
                -((side[0] * eye[0]) + (side[1] * eye[1]) + (side[2] * eye[2])),
                -((upward[0] * eye[0]) + (upward[1] * eye[1]) + (upward[2] * eye[2])),
                (forward[0] * eye[0]) + (forward[1] * eye[1]) + (forward[2] * eye[2]),
                1.0f
            };
            std::copy(&values[0], &values[16], matrix);
        }

        // The camera-to-world rotation of a pose { x, y, z, qx, qy, qz, qw }, row major.
        void pose_rotation(const double pose[7], double rotation[3][3]) {
            const double x = pose[3];
            const double y = pose[4];
            const double z = pose[5];
            const double w = pose[6];
            rotation[0][0] = 1.0 - (2.0 * ((y * y) + (z * z)));
            rotation[0][1] = 2.0 * ((x * y) - (z * w));
            rotation[0][2] = 2.0 * ((x * z) + (y * w));
            rotation[1][0] = 2.0 * ((x * y) + (z * w));
            rotation[1][1] = 1.0 - (2.0 * ((x * x) + (z * z)));
            rotation[1][2] = 2.0 * ((y * z) - (x * w));
            rotation[2][0] = 2.0 * ((x * z) - (y * w));
            rotation[2][1] = 2.0 * ((y * z) + (x * w));
            rotation[2][2] = 1.0 - (2.0 * ((x * x) + (y * y)));
        }

        // Where a point of the image, as fractions of its width and height, is shown once the image is turned clockwise.
        void turn(const int rotation, const float u, const float v, float& x, float& y) {
            switch (rotation) {
                case 90:
                    x = 1.0f - v;
                    y = u;
                    return;
                case 180:
                    x = 1.0f - u;
                    y = 1.0f - v;
                    return;
                case 270:
                    x = v;
                    y = 1.0f - u;
                    return;
                default:
                    x = u;
                    y = v;
                    return;
            }
        }

        // From near (warm) to far (cool), for a depth's place in the spread of depths in view.
        void depth_colour(const float place, float colour[4]) {
            const float t = std::min(std::max(place, 0.0f), 1.0f);
            colour[0] = 1.0f - (0.8f * t);
            colour[1] = 0.45f + (0.4f * t);
            colour[2] = 0.15f + (0.85f * t);
            colour[3] = 1.0f;
        }
    }

    bool renderer::create(std::string& error) {
        this->destroy();
        this->colour_program = link(&colour_vertex_source[0], &colour_fragment_source[0], "colour", error);
        this->texture_program = link(&texture_vertex_source[0], &texture_fragment_source[0], "coordinate", error);
        if ((this->colour_program == 0) || (this->texture_program == 0)) {
            return false;
        }
        this->colour_matrix = glGetUniformLocation(this->colour_program, "matrix");
        this->colour_point_size = glGetUniformLocation(this->colour_program, "point_size");
        this->texture_matrix = glGetUniformLocation(this->texture_program, "matrix");
        this->texture_tint = glGetUniformLocation(this->texture_program, "tint");

        // The font's glyphs in a grid, a byte a pixel.
        std::vector<unsigned char> atlas(static_cast<std::size_t>(font_columns * gtl::font::width * font_rows * gtl::font::height), 0);
        const int atlas_width = font_columns * gtl::font::width;
        for (int glyph = 0; glyph < 128; ++glyph) {
            const int left = (glyph % font_columns) * gtl::font::width;
            const int top = (glyph / font_columns) * gtl::font::height;
            for (int row = 0; row < gtl::font::height; ++row) {
                for (int column = 0; column < gtl::font::width; ++column) {
                    const bool set = (gtl::font::glyphs[glyph][row] & (0x80 >> column)) != 0;
                    atlas[static_cast<std::size_t>(((top + row) * atlas_width) + left + column)] = set ? 255 : 0;
                }
            }
        }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glGenTextures(1, &this->font_texture);
        glBindTexture(GL_TEXTURE_2D, this->font_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, atlas_width, font_rows * gtl::font::height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, atlas.data());

        glGenTextures(1, &this->image_texture);
        glBindTexture(GL_TEXTURE_2D, this->image_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        this->image_valid = false;
        this->image_width = 0;
        this->image_height = 0;
        return true;
    }

    void renderer::destroy() {
        if (this->colour_program != 0) {
            glDeleteProgram(this->colour_program);
            this->colour_program = 0;
        }
        if (this->texture_program != 0) {
            glDeleteProgram(this->texture_program);
            this->texture_program = 0;
        }
        if (this->image_texture != 0) {
            glDeleteTextures(1, &this->image_texture);
            this->image_texture = 0;
        }
        if (this->font_texture != 0) {
            glDeleteTextures(1, &this->font_texture);
            this->font_texture = 0;
        }
        this->image_valid = false;
    }

    void renderer::resize(const int window_width, const int window_height, const float window_density, const int rotation) {
        this->width = window_width;
        this->height = window_height;
        this->density = (window_density > 0.0f) ? window_density : 1.0f;
        this->image_rotation = ((rotation % 360) + 360) % 360;
        this->layout();
    }

    void renderer::layout() {
        const float w = static_cast<float>(this->width);
        const float h = static_cast<float>(this->height);
        const bool turned = (this->image_rotation == 90) || (this->image_rotation == 270);
        const float image_w = static_cast<float>(turned ? this->image_height : this->image_width);
        const float image_h = static_cast<float>(turned ? this->image_width : this->image_height);
        const float aspect = ((image_w > 0.0f) && (image_h > 0.0f)) ? (image_w / image_h) : (turned ? 0.75f : (4.0f / 3.0f));
        if (w >= h) {
            // Side by side: the image as tall as the window, and the map above the panel beside it.
            const float image_width_shown = std::min(h * aspect, 0.62f * w);
            this->image_area = rectangle{ 0.0f, 0.0f, image_width_shown, h };
            const float side = w - image_width_shown;
            const float panel_height = std::min(0.5f * h, std::max(0.38f * h, 230.0f * this->density));
            this->map_area = rectangle{ image_width_shown, 0.0f, side, h - panel_height };
            this->panel_area = rectangle{ image_width_shown, h - panel_height, side, panel_height };
        }
        else {
            // Stacked: the image as wide as the window, then the map, then the panel.
            const float image_height_shown = std::min(w / aspect, 0.5f * h);
            this->image_area = rectangle{ 0.0f, 0.0f, w, image_height_shown };
            const float panel_height = std::min(0.3f * h, 230.0f * this->density);
            this->map_area = rectangle{ 0.0f, image_height_shown, w, h - image_height_shown - panel_height };
            this->panel_area = rectangle{ 0.0f, h - panel_height, w, panel_height };
        }
        // The buttons across the bottom of the panel, clear of the screen's rounded corners and gesture bar.
        const float margin = 12.0f * this->density;
        const float button_height = 48.0f * this->density;
        const float button_width = (this->panel_area.width - (4.0f * margin)) / 3.0f;
        const float button_top = this->panel_area.y + this->panel_area.height - button_height - (2.0f * margin);
        this->reset_button = rectangle{ this->panel_area.x + margin, button_top, button_width, button_height };
        this->save_button = rectangle{ this->panel_area.x + (2.0f * margin) + button_width, button_top, button_width, button_height };
        this->record_button = rectangle{ this->panel_area.x + (3.0f * margin) + (2.0f * button_width), button_top, button_width, button_height };
    }

    void renderer::draw(const snapshot& state, const std::string& status) {
        if ((this->width <= 0) || (this->height <= 0) || (this->colour_program == 0)) {
            return;
        }
        this->upload_image(state);
        glViewport(0, 0, this->width, this->height);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_DEPTH_TEST);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        this->draw_image(state);
        this->draw_projected_landmarks(state);
        this->draw_map(state);
        this->draw_panel(state, status);
    }

    void renderer::upload_image(const snapshot& state) {
        const int w = state.camera.width;
        const int h = state.camera.height;
        if ((w <= 0) || (h <= 0) || (state.image.size() != static_cast<std::size_t>(w) * static_cast<std::size_t>(h))) {
            return;
        }
        if (this->image_valid && (state.frame_sequence == this->image_sequence) && (w == this->image_width) && (h == this->image_height)) {
            return;
        }
        glBindTexture(GL_TEXTURE_2D, this->image_texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        if ((w != this->image_width) || (h != this->image_height) || !this->image_valid) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, state.image.data());
            this->image_width = w;
            this->image_height = h;
            this->layout();
        }
        else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_LUMINANCE, GL_UNSIGNED_BYTE, state.image.data());
        }
        this->image_sequence = state.frame_sequence;
        this->image_valid = true;
    }

    void renderer::draw_image(const snapshot& state) {
        static_cast<void>(state);
        if (!this->image_valid) {
            const float waiting[4] = { 0.05f, 0.05f, 0.06f, 1.0f };
            this->draw_rectangle(this->image_area, waiting);
            return;
        }
        // The image fitted into its area, turned upright.
        const bool turned = (this->image_rotation == 90) || (this->image_rotation == 270);
        const float shown_w = static_cast<float>(turned ? this->image_height : this->image_width);
        const float shown_h = static_cast<float>(turned ? this->image_width : this->image_height);
        const float scale = std::min(this->image_area.width / shown_w, this->image_area.height / shown_h);
        const float left = this->image_area.x + (0.5f * (this->image_area.width - (scale * shown_w)));
        const float top = this->image_area.y + (0.5f * (this->image_area.height - (scale * shown_h)));
        const float corners[4][2] = { { 0.0f, 0.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f } };
        float quad[16];
        for (int i = 0; i < 4; ++i) {
            float x = 0.0f;
            float y = 0.0f;
            turn(this->image_rotation, corners[i][0], corners[i][1], x, y);
            quad[(i * 4) + 0] = left + (x * scale * shown_w);
            quad[(i * 4) + 1] = top + (y * scale * shown_h);
            quad[(i * 4) + 2] = corners[i][0];
            quad[(i * 4) + 3] = corners[i][1];
        }
        float matrix[16];
        pixel_matrix(static_cast<float>(this->width), static_cast<float>(this->height), &matrix[0]);
        glUseProgram(this->texture_program);
        glUniformMatrix4fv(this->texture_matrix, 1, GL_FALSE, &matrix[0]);
        const float untinted[4] = { 1.0f, 1.0f, 1.0f, -1.0f };
        glUniform4fv(this->texture_tint, 1, &untinted[0]);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, this->image_texture);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glEnableVertexAttribArray(position_attribute);
        glEnableVertexAttribArray(colour_attribute);
        glVertexAttribPointer(position_attribute, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), &quad[0]);
        glVertexAttribPointer(colour_attribute, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), &quad[2]);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }

    void renderer::draw_projected_landmarks(const snapshot& state) {
        if (!this->image_valid || !state.posed || state.landmarks.empty()) {
            return;
        }
        const zeroslam_sensor_parameters_camera_struct& camera = state.camera;
        double rotation[3][3];
        pose_rotation(&state.pose[0], rotation);
        const bool turned = (this->image_rotation == 90) || (this->image_rotation == 270);
        const float shown_w = static_cast<float>(turned ? this->image_height : this->image_width);
        const float shown_h = static_cast<float>(turned ? this->image_width : this->image_height);
        const float scale = std::min(this->image_area.width / shown_w, this->image_area.height / shown_h);
        const float left = this->image_area.x + (0.5f * (this->image_area.width - (scale * shown_w)));
        const float top = this->image_area.y + (0.5f * (this->image_area.height - (scale * shown_h)));

        // The landmarks in front of the camera, and their depths, to colour them by their place among the depths in view.
        struct projected final {
            float x, y, depth;
        };

        std::vector<projected> in_view;
        in_view.reserve(state.landmarks.size() / 3);
        for (std::size_t i = 0; (i + 2) < state.landmarks.size(); i += 3) {
            const double offset[3] = { static_cast<double>(state.landmarks[i]) - state.pose[0], static_cast<double>(state.landmarks[i + 1]) - state.pose[1], static_cast<double>(state.landmarks[i + 2]) - state.pose[2] };
            // World to camera is the transpose of the camera-to-world rotation.
            const double cx = (rotation[0][0] * offset[0]) + (rotation[1][0] * offset[1]) + (rotation[2][0] * offset[2]);
            const double cy = (rotation[0][1] * offset[0]) + (rotation[1][1] * offset[1]) + (rotation[2][1] * offset[2]);
            const double cz = (rotation[0][2] * offset[0]) + (rotation[1][2] * offset[1]) + (rotation[2][2] * offset[2]);
            if (cz <= 1e-6) {
                continue;
            }
            const double x = cx / cz;
            const double y = cy / cz;
            const double r2 = (x * x) + (y * y);
            const double* const k = &camera.distortion[0];
            const double radial = 1.0 + (r2 * (k[0] + (r2 * (k[1] + (r2 * k[4])))));
            const double xd = (x * radial) + (2.0 * k[2] * x * y) + (k[3] * (r2 + (2.0 * x * x)));
            const double yd = (y * radial) + (k[2] * (r2 + (2.0 * y * y))) + (2.0 * k[3] * x * y);
            const double u = (camera.focal_x * xd) + camera.centre_x;
            const double v = (camera.focal_y * yd) + camera.centre_y;
            if ((u < 0.0) || (v < 0.0) || (u >= camera.width) || (v >= camera.height) || (r2 > 4.0)) {
                continue;
            }
            in_view.push_back(projected{ static_cast<float>(u / camera.width), static_cast<float>(v / camera.height), static_cast<float>(cz) });
        }
        if (in_view.empty()) {
            return;
        }
        std::vector<float> depths(in_view.size());
        for (std::size_t i = 0; i < in_view.size(); ++i) {
            depths[i] = in_view[i].depth;
        }
        std::sort(depths.begin(), depths.end());
        const float near_depth = std::log(depths[depths.size() / 10]);
        const float far_depth = std::log(depths[(9 * depths.size()) / 10]);
        const float spread = std::max(far_depth - near_depth, 1e-3f);
        this->vertices.clear();
        for (const projected& point : in_view) {
            float x = 0.0f;
            float y = 0.0f;
            turn(this->image_rotation, point.x, point.y, x, y);
            float colour[4];
            depth_colour((std::log(point.depth) - near_depth) / spread, &colour[0]);
            this->vertices.push_back(vertex{ left + (x * scale * shown_w), top + (y * scale * shown_h), 0.0f, colour[0], colour[1], colour[2], 0.9f });
        }
        float matrix[16];
        pixel_matrix(static_cast<float>(this->width), static_cast<float>(this->height), &matrix[0]);
        this->flush(GL_POINTS, &matrix[0], std::max(3.0f, 2.5f * this->density));
    }

    void renderer::draw_map(const snapshot& state) {
        const float background[4] = { 0.035f, 0.04f, 0.05f, 1.0f };
        this->draw_rectangle(this->map_area, background);
        // The view turns about the camera when it has a pose, else about the keyframes' middle, at a distance set by the
        // spread of the keyframes and of the landmarks about it, as a monocular map has no scale of its own.
        float target[3] = { 0.0f, 0.0f, 0.0f };
        if (state.posed) {
            for (int i = 0; i < 3; ++i) {
                target[i] = static_cast<float>(state.pose[i]);
            }
        }
        else if (!state.keyframes.empty()) {
            const std::size_t count = state.keyframes.size() / 3;
            for (std::size_t k = 0; k < count; ++k) {
                for (std::size_t i = 0; i < 3; ++i) {
                    target[i] += state.keyframes[(3 * k) + i] / static_cast<float>(count);
                }
            }
        }
        float extent = 0.0f;
        for (std::size_t i = 0; (i + 2) < state.keyframes.size(); i += 3) {
            const float dx = state.keyframes[i] - target[0];
            const float dy = state.keyframes[i + 1] - target[1];
            const float dz = state.keyframes[i + 2] - target[2];
            extent = std::max(extent, std::sqrt((dx * dx) + (dy * dy) + (dz * dz)));
        }
        if (!state.landmarks.empty()) {
            std::vector<float> distances;
            distances.reserve(state.landmarks.size() / 3);
            for (std::size_t i = 0; (i + 2) < state.landmarks.size(); i += 3) {
                const float dx = state.landmarks[i] - target[0];
                const float dy = state.landmarks[i + 1] - target[1];
                const float dz = state.landmarks[i + 2] - target[2];
                distances.push_back(std::sqrt((dx * dx) + (dy * dy) + (dz * dz)));
            }
            std::nth_element(distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(distances.size() / 2), distances.end());
            extent = std::max(extent, distances[distances.size() / 2]);
        }
        extent = std::max(extent, 0.5f);
        // The map's world is the first camera's frame, with y down, so up is -y.
        const float range = this->distance * extent;
        const float eye[3] = { target[0] + (range * std::sin(this->yaw) * std::cos(this->pitch)), target[1] - (range * std::sin(this->pitch)), target[2] - (range * std::cos(this->yaw) * std::cos(this->pitch)) };
        const float up[3] = { 0.0f, -1.0f, 0.0f };
        float view[16];
        float projection[16];
        float matrix[16];
        look_at_matrix(&eye[0], &target[0], &up[0], &view[0]);
        perspective_matrix(1.0f, this->map_area.width / std::max(this->map_area.height, 1.0f), 0.01f * range, 100.0f * range, &projection[0]);
        multiply(&projection[0], &view[0], &matrix[0]);

        const GLint left = static_cast<GLint>(this->map_area.x);
        const GLint bottom = static_cast<GLint>(static_cast<float>(this->height) - (this->map_area.y + this->map_area.height));
        glViewport(left, bottom, static_cast<GLsizei>(this->map_area.width), static_cast<GLsizei>(this->map_area.height));
        glEnable(GL_SCISSOR_TEST);
        glScissor(left, bottom, static_cast<GLsizei>(this->map_area.width), static_cast<GLsizei>(this->map_area.height));

        // The world's axes at the first camera, a tenth of the map's size.
        this->vertices.clear();
        const float axis = 0.2f * extent;
        const float axes[3][4] = { { 0.9f, 0.25f, 0.25f, 1.0f }, { 0.25f, 0.9f, 0.25f, 1.0f }, { 0.3f, 0.45f, 1.0f, 1.0f } };
        for (int i = 0; i < 3; ++i) {
            const float end[3] = { (i == 0) ? axis : 0.0f, (i == 1) ? axis : 0.0f, (i == 2) ? axis : 0.0f };
            this->vertices.push_back(vertex{ 0.0f, 0.0f, 0.0f, axes[i][0], axes[i][1], axes[i][2], 1.0f });
            this->vertices.push_back(vertex{ end[0], end[1], end[2], axes[i][0], axes[i][1], axes[i][2], 1.0f });
        }
        this->flush(GL_LINES, &matrix[0], 1.0f);

        this->vertices.clear();
        for (std::size_t i = 0; (i + 2) < state.landmarks.size(); i += 3) {
            this->vertices.push_back(vertex{ state.landmarks[i], state.landmarks[i + 1], state.landmarks[i + 2], 0.85f, 0.87f, 0.9f, 0.8f });
        }
        this->flush(GL_POINTS, &matrix[0], std::max(2.0f, 1.5f * this->density));

        this->vertices.clear();
        for (std::size_t i = 0; (i + 2) < state.keyframes.size(); i += 3) {
            this->vertices.push_back(vertex{ state.keyframes[i], state.keyframes[i + 1], state.keyframes[i + 2], amber[0], amber[1], amber[2], 1.0f });
        }
        this->flush(GL_LINE_STRIP, &matrix[0], 1.0f);
        for (std::size_t i = 0; (i + 2) < state.keyframes.size(); i += 3) {
            this->vertices.push_back(vertex{ state.keyframes[i], state.keyframes[i + 1], state.keyframes[i + 2], amber[0], amber[1], amber[2], 1.0f });
        }
        this->flush(GL_POINTS, &matrix[0], std::max(4.0f, 3.0f * this->density));

        // The camera as a frustum of its field of view.
        if (state.posed && (state.camera.focal_x > 0.0) && (state.camera.focal_y > 0.0)) {
            double rotation[3][3];
            pose_rotation(&state.pose[0], rotation);
            const double depth = 0.4 * static_cast<double>(extent);
            const double pixel_corners[4][2] = { { 0.0, 0.0 }, { static_cast<double>(state.camera.width), 0.0 }, { static_cast<double>(state.camera.width), static_cast<double>(state.camera.height) }, { 0.0, static_cast<double>(state.camera.height) } };
            float corners[4][3];
            for (int c = 0; c < 4; ++c) {
                const double ray[3] = { depth * (pixel_corners[c][0] - state.camera.centre_x) / state.camera.focal_x, depth * (pixel_corners[c][1] - state.camera.centre_y) / state.camera.focal_y, depth };
                for (int i = 0; i < 3; ++i) {
                    corners[c][i] = static_cast<float>(state.pose[i] + (rotation[i][0] * ray[0]) + (rotation[i][1] * ray[1]) + (rotation[i][2] * ray[2]));
                }
            }
            const float* const colour = (state.state == tracking_state::tracking) ? &green[0] : &red[0];
            for (int c = 0; c < 4; ++c) {
                const int next = (c + 1) % 4;
                this->vertices.push_back(vertex{ target[0], target[1], target[2], colour[0], colour[1], colour[2], 1.0f });
                this->vertices.push_back(vertex{ corners[c][0], corners[c][1], corners[c][2], colour[0], colour[1], colour[2], 1.0f });
                this->vertices.push_back(vertex{ corners[c][0], corners[c][1], corners[c][2], colour[0], colour[1], colour[2], 1.0f });
                this->vertices.push_back(vertex{ corners[next][0], corners[next][1], corners[next][2], colour[0], colour[1], colour[2], 1.0f });
            }
            this->flush(GL_LINES, &matrix[0], 1.0f);
        }
        glDisable(GL_SCISSOR_TEST);
        glViewport(0, 0, this->width, this->height);
    }

    void renderer::draw_panel(const snapshot& state, const std::string& status) {
        this->draw_rectangle(this->panel_area, panel_colour);
        const float margin = 12.0f * this->density;
        // The text as large as forty characters across the panel allow.
        const float scale = std::max(1.0f, std::floor((this->panel_area.width - (2.0f * margin)) / (40.0f * static_cast<float>(gtl::font::width))));
        const float line = 1.35f * scale * static_cast<float>(gtl::font::height);
        const float left = this->panel_area.x + margin;
        float top = this->panel_area.y + margin;
        const std::size_t columns = static_cast<std::size_t>((this->panel_area.width - (2.0f * margin)) / (scale * static_cast<float>(gtl::font::width)));
        const auto write = [&](const std::string& text, const float colour[4]) {
            // Long lines wrap at a space before the panel's edge, as long as there is room above the buttons.
            std::size_t start = 0;
            while ((start < text.size()) && ((top + line) <= this->reset_button.y) && (columns > 0)) {
                std::size_t length = std::min(columns, text.size() - start);
                if ((start + length) < text.size()) {
                    const std::size_t space = text.rfind(' ', start + length);
                    if ((space != std::string::npos) && (space > start)) {
                        length = space - start;
                    }
                }
                this->draw_text(text.substr(start, length), left, top, scale, colour);
                top += line;
                start += length;
                while ((start < text.size()) && (text[start] == ' ')) {
                    ++start;
                }
            }
        };
        const float* state_colour = &amber[0];
        const char* state_text = "INITIALISING: move the phone sideways";
        if (state.state == tracking_state::tracking) {
            state_colour = &green[0];
            state_text = "TRACKING";
        }
        else if (state.state == tracking_state::lost) {
            state_colour = &red[0];
            state_text = "LOST: return to a mapped view";
        }
        write(state_text, state_colour);
        char text[256];
        std::snprintf(&text[0], sizeof(text), "%.0f ms a frame (%.0f now), %lld of %lld frames", state.average_process_milliseconds, state.process_milliseconds, state.frames_processed, state.frames_received);
        write(&text[0], white);
        std::snprintf(&text[0], sizeof(text), "%zu landmarks, %zu keyframes", state.landmarks.size() / 3, state.keyframes.size() / 3);
        write(&text[0], white);
        if (state.recording) {
            std::snprintf(&text[0], sizeof(text), "Recording: %lld frames written, %lld skipped", state.frames_recorded, state.frames_unrecorded);
            write(&text[0], red);
        }
        if (!state.message.empty()) {
            write(state.message, grey);
        }
        if (!status.empty()) {
            write(status, grey);
        }

        const auto draw_button = [&](const rectangle& area, const char* const label, const float colour[4]) {
            this->draw_rectangle(area, colour);
            const float label_scale = std::max(1.0f, std::floor(std::min(0.5f * area.height / static_cast<float>(gtl::font::height), area.width / (static_cast<float>(gtl::font::width) * (1.0f + static_cast<float>(std::string(label).size()))))));
            const float label_width = label_scale * static_cast<float>(gtl::font::width * static_cast<int>(std::string(label).size()));
            this->draw_text(label, area.x + (0.5f * (area.width - label_width)), area.y + (0.5f * (area.height - (label_scale * static_cast<float>(gtl::font::height)))), label_scale, white);
        };
        draw_button(this->reset_button, "RESET", button_colour);
        draw_button(this->save_button, "SAVE", button_colour);
        draw_button(this->record_button, state.recording ? "STOP" : "RECORD", state.recording ? recording_colour : button_colour);
    }

    void renderer::draw_text(const std::string& text, const float x, const float y, const float scale, const float colour[4]) {
        this->text_vertices.clear();
        const float glyph_w = scale * static_cast<float>(gtl::font::width);
        const float glyph_h = scale * static_cast<float>(gtl::font::height);
        const float atlas_w = static_cast<float>(font_columns * gtl::font::width);
        const float atlas_h = static_cast<float>(font_rows * gtl::font::height);
        float pen = x;
        for (const char character : text) {
            const unsigned char byte = static_cast<unsigned char>(character);
            const int code = (byte < 127) ? static_cast<int>(byte) : static_cast<int>('?');
            const float u0 = static_cast<float>((code % font_columns) * gtl::font::width) / atlas_w;
            const float v0 = static_cast<float>((code / font_columns) * gtl::font::height) / atlas_h;
            const float u1 = u0 + (static_cast<float>(gtl::font::width) / atlas_w);
            const float v1 = v0 + (static_cast<float>(gtl::font::height) / atlas_h);
            const float quad[6][4] = { { pen, y, u0, v0 }, { pen + glyph_w, y, u1, v0 }, { pen + glyph_w, y + glyph_h, u1, v1 }, { pen, y, u0, v0 }, { pen + glyph_w, y + glyph_h, u1, v1 }, { pen, y + glyph_h, u0, v1 } };
            for (const auto& corner : quad) {
                this->text_vertices.insert(this->text_vertices.end(), &corner[0], &corner[4]);
            }
            pen += glyph_w;
        }
        if (this->text_vertices.empty()) {
            return;
        }
        float matrix[16];
        pixel_matrix(static_cast<float>(this->width), static_cast<float>(this->height), &matrix[0]);
        glUseProgram(this->texture_program);
        glUniformMatrix4fv(this->texture_matrix, 1, GL_FALSE, &matrix[0]);
        glUniform4fv(this->texture_tint, 1, colour);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, this->font_texture);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glEnableVertexAttribArray(position_attribute);
        glEnableVertexAttribArray(colour_attribute);
        glVertexAttribPointer(position_attribute, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), &this->text_vertices[0]);
        glVertexAttribPointer(colour_attribute, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), &this->text_vertices[2]);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(this->text_vertices.size() / 4));
    }

    void renderer::draw_rectangle(const rectangle& area, const float colour[4]) {
        this->vertices.clear();
        const float corners[6][2] = { { area.x, area.y }, { area.x + area.width, area.y }, { area.x + area.width, area.y + area.height }, { area.x, area.y }, { area.x + area.width, area.y + area.height }, { area.x, area.y + area.height } };
        for (const auto& corner : corners) {
            this->vertices.push_back(vertex{ corner[0], corner[1], 0.0f, colour[0], colour[1], colour[2], colour[3] });
        }
        float matrix[16];
        pixel_matrix(static_cast<float>(this->width), static_cast<float>(this->height), &matrix[0]);
        this->flush(GL_TRIANGLES, &matrix[0], 1.0f);
    }

    void renderer::flush(const GLenum mode, const float matrix[16], const float point_size) {
        if (this->vertices.empty()) {
            return;
        }
        glUseProgram(this->colour_program);
        glUniformMatrix4fv(this->colour_matrix, 1, GL_FALSE, matrix);
        glUniform1f(this->colour_point_size, point_size);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glEnableVertexAttribArray(position_attribute);
        glEnableVertexAttribArray(colour_attribute);
        glVertexAttribPointer(position_attribute, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), &this->vertices[0].x);
        glVertexAttribPointer(colour_attribute, 4, GL_FLOAT, GL_FALSE, sizeof(vertex), &this->vertices[0].r);
        glDrawArrays(mode, 0, static_cast<GLsizei>(this->vertices.size()));
        this->vertices.clear();
    }

    button renderer::hit(const float x, const float y) const {
        if (this->reset_button.contains(x, y)) {
            return button::reset;
        }
        if (this->save_button.contains(x, y)) {
            return button::save;
        }
        if (this->record_button.contains(x, y)) {
            return button::record;
        }
        return button::none;
    }

    bool renderer::in_map(const float x, const float y) const {
        return this->map_area.contains(x, y);
    }

    void renderer::orbit(const float dx, const float dy) {
        const float radians_per_pixel = 3.0f / std::max(this->map_area.width, 1.0f);
        this->yaw -= dx * radians_per_pixel;
        this->pitch = std::min(std::max(this->pitch + (dy * radians_per_pixel), -1.5f), 1.5f);
    }

    void renderer::zoom(const float ratio) {
        if (ratio > 0.0f) {
            this->distance = std::min(std::max(this->distance / ratio, 0.2f), 20.0f);
        }
    }
}
