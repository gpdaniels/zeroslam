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
#ifndef ZEROSLAM_TOOLS_GUI_IMAGE_TEXTURE_HPP
#define ZEROSLAM_TOOLS_GUI_IMAGE_TEXTURE_HPP

#include "opengl.hpp"
#include "scene.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace gui {

    class image_texture final {
    private:
        unsigned int handle = 0;
        int texture_width = 0;
        int texture_height = 0;
        long long uploaded_timestamp = -1;
        bool uploaded_any = false;

    public:
        image_texture() = default;
        image_texture(const image_texture&) = delete;
        image_texture& operator=(const image_texture&) = delete;

        image_texture(image_texture&& other) noexcept
            : handle(other.handle)
            , texture_width(other.texture_width)
            , texture_height(other.texture_height)
            , uploaded_timestamp(other.uploaded_timestamp)
            , uploaded_any(other.uploaded_any) {
            other.handle = 0;
            other.uploaded_any = false;
        }

        image_texture& operator=(image_texture&& other) noexcept {
            if (this != &other) {
                this->release();
                this->handle = other.handle;
                this->texture_width = other.texture_width;
                this->texture_height = other.texture_height;
                this->uploaded_timestamp = other.uploaded_timestamp;
                this->uploaded_any = other.uploaded_any;
                other.handle = 0;
                other.uploaded_any = false;
            }
            return *this;
        }

        ~image_texture() {
            this->release();
        }

        void release() {
            if (this->handle != 0) {
                const GLuint name = static_cast<GLuint>(this->handle);
                glDeleteTextures(1, &name);
                this->handle = 0;
            }
            this->uploaded_any = false;
            this->uploaded_timestamp = -1;
        }

        bool is_valid() const {
            return this->uploaded_any && (this->handle != 0);
        }

        int width() const {
            return this->texture_width;
        }

        int height() const {
            return this->texture_height;
        }

        void update(const scene::decoded_image& image) {
            if (!image.is_valid()) {
                return;
            }
            if (this->uploaded_any && (this->uploaded_timestamp == image.timestamp_nanoseconds) && (this->texture_width == static_cast<int>(image.width)) && (this->texture_height == static_cast<int>(image.height))) {
                return;
            }
            const bool resized = (this->texture_width != static_cast<int>(image.width)) || (this->texture_height != static_cast<int>(image.height));
            if (this->handle == 0) {
                GLuint name = 0;
                glGenTextures(1, &name);
                this->handle = name;
                glBindTexture(GL_TEXTURE_2D, name);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            }
            else {
                glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(this->handle));
            }
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            if (resized || !this->uploaded_any) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, static_cast<GLsizei>(image.width), static_cast<GLsizei>(image.height), 0, GL_RGB, GL_UNSIGNED_BYTE, image.rgb.data());
            }
            else {
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(image.width), static_cast<GLsizei>(image.height), GL_RGB, GL_UNSIGNED_BYTE, image.rgb.data());
            }
            this->texture_width = static_cast<int>(image.width);
            this->texture_height = static_cast<int>(image.height);
            this->uploaded_timestamp = image.timestamp_nanoseconds;
            this->uploaded_any = true;
        }

        struct placement {
            int x = 0;
            int y = 0;
            int width = 0;
            int height = 0;
            float scale = 1.0f;
            bool valid = false;
        };

        placement draw(const int x, const int y, const int width, const int height, const float alpha) const {
            placement result;
            if (!this->is_valid() || (this->texture_width <= 0) || (this->texture_height <= 0) || (width <= 0) || (height <= 0)) {
                return result;
            }
            const float scale_x = static_cast<float>(width) / static_cast<float>(this->texture_width);
            const float scale_y = static_cast<float>(height) / static_cast<float>(this->texture_height);
            const float scale = (scale_x < scale_y) ? scale_x : scale_y;
            result.width = static_cast<int>(static_cast<float>(this->texture_width) * scale);
            result.height = static_cast<int>(static_cast<float>(this->texture_height) * scale);
            result.x = x + ((width - result.width) / 2);
            result.y = y + ((height - result.height) / 2);
            result.scale = scale;
            result.valid = true;

            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(this->handle));
            glEnable(GL_TEXTURE_2D);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glColor4f(1.0f, 1.0f, 1.0f, alpha);
            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 1.0f);
            glVertex2f(static_cast<float>(result.x), static_cast<float>(result.y));
            glTexCoord2f(1.0f, 1.0f);
            glVertex2f(static_cast<float>(result.x + result.width), static_cast<float>(result.y));
            glTexCoord2f(1.0f, 0.0f);
            glVertex2f(static_cast<float>(result.x + result.width), static_cast<float>(result.y + result.height));
            glTexCoord2f(0.0f, 0.0f);
            glVertex2f(static_cast<float>(result.x), static_cast<float>(result.y + result.height));
            glEnd();
            glDisable(GL_BLEND);
            glDisable(GL_TEXTURE_2D);
            return result;
        }
    };
}

#endif // ZEROSLAM_TOOLS_GUI_IMAGE_TEXTURE_HPP
